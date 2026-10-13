"""Decision tuning: K-way cross-entropy over each question's options, then a calibrated checkpoint.

Rows are decision rows, ``{"state", "questions"}`` (see ``training/utils/decision.py``). The loop
takes ``sft_loop.Config`` and reuses sft_loop's row path: rows are rendered lazily, a batch at a
time, by the same JSONL datasets, render workers, DataLoader and raw-row resume cursor, and the
same LR schedule, Adam settings, checkpoint intervals and pipelining apply. Decision-specific
rules override a few settings: every step uses ``target_logprob_support="candidates"``, a
calibration split is required, and the final sampler checkpoint is ``step-N-cal`` (its
``decision.json`` holds one fitted temperature per question type), which is what gets promoted.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import random
import tempfile
from collections import deque
from contextlib import ExitStack
from itertools import islice

import tinker
import torch
from fireworks.training.sdk.training_spec import compute_lr, normalize_lr_scheduler_spec
from training.recipes import sft_loop
from training.utils import (
    DEFAULT_ADAM,
    NO_VALID_TRAINING_EXAMPLES_MESSAGE,
    DatasetError,
    RawRowCursor,
    ReconnectableClient,
    RunnerIO,
    RunStatus,
    UserConfigError,
    build_service_client,
    log_metrics_json,
    make_render_dataloader,
    read_api_extra_headers_env,
    setup_wandb,
    validate_config,
    wandb_finish,
    wandb_log,
)
from training.utils.checkpoints import TrainingCheckpoints, validate_warm_start_config
from training.utils.client import DEFAULT_TIMEOUT_S
from training.utils.decision import (
    DECISION_LOSS_FN_CONFIG,
    decision_datums,
    gold_option_index,
    is_decision_row,
    option_logprobs,
)
from training.utils.decision_calibration import calibration_metadata, mean_nll
from training.utils.resource_autosizing import select_render_worker_count
from training.utils.runner_state import (
    start_running,
    write_completed,
    write_running_step,
)
from training.utils.serverless import setup_serverless_training
from training.utils.streaming import dataset_error_for_rendered_row
from training.utils.supervised import RenderedSupervisedDatum

logger = logging.getLogger(__name__)

# Decision tuning takes exactly sft_loop's settings; main() rejects the ones a decision run overrides.
Config = sft_loop.Config
_DEFAULT_TRAIN_ON_WHAT = Config.__dataclass_fields__["train_on_what"].default

# Published session checkpoint ids are ``run-`` + 32 hex + ``-`` + name + ``-`` + 8 hex.
# Resource ids are at most 63 characters, so the caller-supplied name has 17.
_RUN_ID_HEX_LENGTH = 32
_SESSION_SUFFIX_HEX_LENGTH = 8
PROMOTED_CHECKPOINT_NAME_BUDGET = 63 - (len("run-") + _RUN_ID_HEX_LENGTH + 1 + 1 + _SESSION_SUFFIX_HEX_LENGTH)


def calibrated_checkpoint_name(step: int) -> str:
    """Logical name of the final calibrated sampler checkpoint.

    ``step-<n>-cal`` stays within the 17-character promotion budget through
    8-digit step counts. The bare ``step-<n>`` saves beside it are unchanged:
    they are the resume key, and they fit the same budget through 12 digits.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    name = f"step-{step}-cal"
    if len(name) > PROMOTED_CHECKPOINT_NAME_BUDGET:
        raise UserConfigError(
            f"decision checkpoint name {name!r} is {len(name)} characters; "
            f"serverless promotion allows {PROMOTED_CHECKPOINT_NAME_BUDGET}"
        )
    return name


def _render_pairs(row: dict) -> list[tuple[str, tinker.Datum]] | DatasetError:
    """One row's ``(question type, K-way datum)`` pairs, rendered in the sft_loop render worker."""
    if not is_decision_row(row):
        return dataset_error_for_rendered_row(
            row, DatasetError("not a decision row; decision tuning needs state + questions")
        )
    state = sft_loop._worker_state
    try:
        pairs = decision_datums(
            row,
            renderer=state["renderer"],
            renderer_name=state["resolved_renderer_name"],
            max_seq_len=state["max_seq_len"],
        )
    except DatasetError as exc:
        return dataset_error_for_rendered_row(row, exc)
    target = lambda d: int(d.loss_fn_inputs["target_tokens"].data[-1])  # noqa: E731
    sft_loop._write_render_samples(
        row,
        [
            RenderedSupervisedDatum(
                token_ids=[*d.model_input.to_ints(), target(d)],
                token_weights=[0.0] * d.model_input.length + [1.0],
                datum=d,
            )
            for _, d in pairs
        ],
    )
    return pairs


def render_train_row(row: dict) -> list[tinker.Datum] | DatasetError | None:
    pairs = _render_pairs(row)
    if isinstance(pairs, DatasetError):
        return pairs
    return [d for _, d in pairs] or None


def render_eval_row(row: dict) -> list[tuple[str, tinker.Datum]] | DatasetError | None:
    return _render_pairs(row) or None


def _labeled_decision_row(row: dict) -> bool | None:
    """Raw-row sniff for preflight: None for a non-decision row, else whether any question is labeled."""
    if not is_decision_row(row):
        return None
    questions = row["questions"]
    return isinstance(questions, dict) and any(
        isinstance(q, dict) and q.get("label") is not None for q in questions.values()
    )


def _carve_out(cfg: Config, n_rows: int) -> list[int]:
    """The seeded rows held out of the training set as the calibration split when there is no evaluation_dataset."""
    order = list(range(n_rows))
    random.Random(cfg.seed).shuffle(order)
    return order[: sft_loop.compute_eval_carveout(n_rows, cfg.eval_carve_ratio, cfg.max_eval_seqs)]


def preflight(cfg: Config) -> None:
    """Checks that need only raw rows, run before a trainer is provisioned."""
    if cfg.epochs < 1:
        raise UserConfigError("decision tuning requires epochs >= 1 to produce a calibrated checkpoint")
    if not cfg.save_final_checkpoint:
        raise UserConfigError(
            "decision tuning requires save_final_checkpoint=True: the fitted temperatures are written "
            "only into the final calibrated sampler checkpoint"
        )
    if cfg.train_on_what != _DEFAULT_TRAIN_ON_WHAT:
        raise UserConfigError("train_on_what does not apply to decision rows: each labeled question is one target")
    raw = sft_loop._make_jsonl_dataset(
        path=cfg.dataset, render_fn=_labeled_decision_row, cfg=cfg, max_examples=cfg.max_examples
    )
    if len(raw) == 0:
        raise DatasetError(f"No examples found in {cfg.dataset}")
    if raw[0] is None:
        raise DatasetError("dataset row 0 is not a decision row; decision tuning needs state + questions")
    if cfg.evaluation_dataset:
        calibration = sft_loop._make_jsonl_dataset(
            path=cfg.evaluation_dataset, render_fn=_labeled_decision_row, cfg=cfg
        )
        if not len(calibration):
            raise DatasetError(
                f"evaluation_dataset {cfg.evaluation_dataset} has no rows; decision rows need a calibration "
                "split to fit the checkpoint's temperatures"
            )
        if calibration[0] is None:
            raise DatasetError(
                "evaluation_dataset row 0 is not a decision row; decision tuning needs state + questions"
            )
    else:
        held = _carve_out(cfg, len(raw))
        if not held:
            raise DatasetError(
                "decision rows need a calibration split to fit the checkpoint's temperatures: the seeded "
                f"carve-out (eval_carve_ratio={cfg.eval_carve_ratio}) of {len(raw)} rows is empty; "
                "set evaluation_dataset or provide more rows"
            )
        calibration = raw.with_indices(held)
    if not any(calibration[i] for i in range(len(calibration))):
        raise DatasetError("the calibration split has no labeled decision questions")


def prepare_datasets(cfg: Config):
    """The lazily rendered training dataset, and the calibration split rendered once (as sft_loop renders eval)."""
    training_ds = sft_loop._make_jsonl_dataset(
        path=cfg.dataset, render_fn=render_train_row, cfg=cfg, max_examples=cfg.max_examples
    )
    if cfg.evaluation_dataset:
        eval_view = sft_loop._make_jsonl_dataset(path=cfg.evaluation_dataset, render_fn=render_eval_row, cfg=cfg)
    else:
        held_rows = _carve_out(cfg, len(training_ds))
        held = set(held_rows)
        eval_view = sft_loop._make_jsonl_dataset(
            path=cfg.dataset, render_fn=render_eval_row, cfg=cfg, max_examples=cfg.max_examples
        ).with_indices(held_rows)
        training_ds = training_ds.with_indices([i for i in range(len(training_ds)) if i not in held])
    pairs = sft_loop._render_eagerly(eval_view, len(eval_view))
    if not pairs:
        raise DatasetError("the calibration split has no labeled decision questions")
    return training_ds, [t for t, _ in pairs], [d for _, d in pairs]


def evaluate(client, datums: list[tinker.Datum], step: int, runner: RunnerIO) -> None:
    rows = option_logprobs(client, datums)
    labels = [gold_option_index(d) for d in datums]
    loss = mean_nll(rows, labels)
    accuracy = sum(max(range(len(r)), key=r.__getitem__) == y for r, y in zip(rows, labels)) / len(labels)
    logger.info("[Eval] step %d | K-way NLL %.4f | accuracy %.4f", step, loss, accuracy)
    log_metrics_json(step, eval_loss=loss, eval_accuracy=accuracy)
    wandb_log({"eval/loss": loss, "eval/accuracy": accuracy}, step)
    runner.append_metrics(step, {"eval/loss": loss, "eval/accuracy": accuracy})


def main(config: Config) -> dict:
    cfg = config
    validate_config(cfg.base_model, cfg.dataset, output_model_id=cfg.output_model_id)
    if not cfg.tokenizer_model:
        raise ValueError("Config.tokenizer_model is required to render decision rows.")
    validate_warm_start_config(
        warm_start_from_adapter=cfg.warm_start_from_adapter,
        init_from_checkpoint=cfg.init_from_checkpoint,
        lora_rank=cfg.lora_rank,
    )
    preflight(cfg)
    renderer_name = sft_loop._resolved_renderer_name(
        tokenizer_model=cfg.tokenizer_model,
        renderer_name=cfg.renderer_name,
        thinking_trace_history_mode=cfg.thinking_trace_history_mode,
        renderer_name_is_resolved=cfg.renderer_name_is_resolved,
    )
    lr_scheduler = normalize_lr_scheduler_spec(cfg.lr_scheduler, legacy_warmup_steps=cfg.warmup_steps)
    setup_wandb(cfg.wandb, {"lr": cfg.learning_rate, "lr_schedule": lr_scheduler.type, "epochs": cfg.epochs})

    runner = RunnerIO(cfg.runner)
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    headers = read_api_extra_headers_env()
    runner.write_status(
        RunStatus.PENDING, message="attaching to serverless trainer pool" if cfg.serverless else "provisioning"
    )
    with runner, ExitStack() as stack:
        if cfg.serverless:
            service, client, ckpt, job_id, max_seq_len = setup_serverless_training(
                cfg, api_key=api_key, base_url=base_url, additional_headers=headers, stack=stack
            )
            stack.callback(service.close)
            runner.set_accelerator_info(None, None, profile=None)
            runner.mark_serverless()
        else:
            service = build_service_client(
                api_key=api_key,
                base_url=base_url,
                additional_headers=headers,
                base_model=cfg.base_model,
                tokenizer_model=cfg.tokenizer_model,
                max_lora_rank=cfg.lora_rank,
                max_context_length=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                trainer=cfg.trainer,
                cleanup_trainer_on_close=True,
            )
            stack.callback(service.close)
            training_client = service.create_training_client(
                cfg.base_model, lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha
            )
            runner.set_accelerator_info(
                service.accelerator_type, service.accelerator_count, profile=service.training_profile
            )
            job_id, max_seq_len = service.trainer_job_id, service.max_context_length
            client = ReconnectableClient.from_training_client(
                training_client,
                base_model=cfg.base_model,
                lora_rank=cfg.lora_rank,
                job_id=job_id,
                default_timeout=cfg.step_timeout or DEFAULT_TIMEOUT_S,
                service=service,
            )
            ckpt = TrainingCheckpoints(
                client, service, trainer_id=job_id, log_path=cfg.log_path, lora_rank=cfg.lora_rank
            )

        # -- Rows: sft_loop's render workers, render samples and lazy DataLoader ---------------
        base_init_args = (
            cfg.tokenizer_model,
            renderer_name,
            cfg.train_on_what,
            max_seq_len,
            cfg.tokenizer_revision,
            cfg.thinking_trace_history_mode,
            cfg.tokenizer_trust_remote_code,
        )
        sft_loop._init_render_worker(*base_init_args)
        training_ds, eval_types, eval_data = prepare_datasets(cfg)
        samples_dir, samples_limit = "", sft_loop._resolve_render_samples_limit(cfg.render_samples_limit)
        if cfg.render_samples_file and samples_limit != 0:
            samples_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix="decision-render-samples-"))
            stack.callback(sft_loop._finalize_render_samples, samples_dir, cfg.render_samples_file, samples_limit)
        sft_loop._configure_render_sample_state(
            render_samples_local_dir=samples_dir, render_samples_limit=samples_limit
        )
        training_count = len(training_ds)
        batch_size = max(1, min(cfg.batch_size, training_count))
        generator = torch.Generator()
        loader = make_render_dataloader(
            training_ds,
            batch_size=batch_size,
            num_workers=(
                cfg.render_workers
                if cfg.render_workers is not None
                else select_render_worker_count(available_parallel_tasks=training_count)
            ),
            shuffle=True,
            generator=generator,
            worker_init_fn=functools.partial(
                sft_loop._init_render_worker, *base_init_args, samples_dir, samples_limit
            ),
            group_by_length=cfg.group_by_length,
            length_group_factor=cfg.length_group_factor,
            sizes=training_ds.approx_row_sizes() if cfg.group_by_length else None,
        )

        # -- Resume, optimizer and schedule, as sft_loop ------------------------------------------
        resume = ckpt.resume(
            init_from_checkpoint=cfg.init_from_checkpoint,
            warm_start_from_adapter=cfg.warm_start_from_adapter,
            prefer_current_run=cfg.prefer_current_run_checkpoint,
        )
        start_step = step = resume.step if resume else 0
        total_rows = training_count * cfg.epochs
        cursor = RawRowCursor(max_rows=total_rows)
        cursor.resume(resume.data_consumed if resume else None)
        adam = dict(DEFAULT_ADAM, grad_clip_norm=cfg.grad_clip_norm)
        adam.update(
            {k: v for k, v in (("beta2", cfg.adam_beta2), ("weight_decay", cfg.weight_decay)) if v is not None}
        )
        completed_epochs, rows_into_epoch = divmod(cursor.value, training_count)
        total_steps = step + math.ceil(max(0, total_rows - cursor.value) / batch_size)
        timeout = cfg.step_timeout or DEFAULT_TIMEOUT_S
        start_running(runner, total_steps=total_steps)
        if step == 0:
            evaluate(client, eval_data, step, runner)

        in_flight: deque = deque()

        def collect() -> None:
            s, tokens, n, consumed, fb, opt = in_flight.popleft()
            loss = float(fb.result(timeout=timeout).metrics.get("loss:sum", 0.0)) / n
            opt.result(timeout=timeout)
            dcp_due = cfg.dcp_save_interval > 0 and s % cfg.dcp_save_interval == 0
            sampler_due = cfg.sampler_save_interval > 0 and s % cfg.sampler_save_interval == 0
            if dcp_due or sampler_due:
                ckpt.save(f"step-{s}", resumable=dcp_due, promotable=sampler_due, data_consumed=consumed)
            lr = compute_lr(lr_scheduler, step=s, base_lr=cfg.learning_rate, total_steps=total_steps)
            log_metrics_json(s, kway_loss=loss, lr=lr)
            wandb_log({"train/loss": loss, "train/lr": lr}, s)
            write_running_step(runner, step=s, total_steps=total_steps, metrics={"train/loss": loss}, tokens=tokens)

        try:
            for epoch in range(completed_epochs, cfg.epochs):
                generator.manual_seed(cfg.seed + epoch)
                first = rows_into_epoch // batch_size if epoch == completed_epochs else 0
                for raw_index, rendered in enumerate(islice(iter(loader), first, None), start=first):
                    cursor.record(min(batch_size, training_count - raw_index * batch_size))
                    batch = sft_loop._flatten_rendered_batch(rendered)
                    if not batch:
                        continue
                    step += 1
                    lr = compute_lr(lr_scheduler, step=step, base_lr=cfg.learning_rate, total_steps=total_steps)
                    tokens = sum(d.model_input.length for d in batch)
                    fb = client.submit_forward_backward(
                        batch, "cross_entropy", loss_fn_config=dict(DECISION_LOSS_FN_CONFIG)
                    )
                    opt = client.submit_optim_step(tinker.AdamParams(learning_rate=lr, **adam))
                    in_flight.append((step, tokens, len(batch), cursor.value, fb, opt))
                    checkpoint_step = any(
                        i > 0 and step % i == 0 for i in (cfg.dcp_save_interval, cfg.sampler_save_interval)
                    )
                    while in_flight and (len(in_flight) >= cfg.pipeline_depth or checkpoint_step):
                        collect()
                while in_flight:
                    collect()
                if step == 0:
                    raise DatasetError(NO_VALID_TRAINING_EXAMPLES_MESSAGE)
                if step > start_step:
                    evaluate(client, eval_data, step, runner)
        finally:
            while in_flight:
                try:
                    collect()
                except Exception as exc:  # noqa: BLE001 - draining after a failure
                    logger.warning("pipeline drain: %s", exc)

        # -- Calibrated final checkpoint ------------------------------------------------------------
        # A promotable step-N (a periodic sampler save, a weight sync) never stands in for the
        # calibrated checkpoint, and save() refuses to skip a save carrying metadata. Promotion of
        # the same checkpoint is idempotent, so a resume at the final step re-promotes.
        calibrated = calibrated_checkpoint_name(step)
        if step > start_step:
            ckpt.save(f"step-{step}", resumable=True, promotable=False, data_consumed=cursor.value)
        if step > start_step or not ckpt.has_promotable(calibrated):
            labels = [gold_option_index(d) for d in eval_data]
            metadata = calibration_metadata(option_logprobs(client, eval_data), labels, eval_types)
            logger.info("Decision calibration: T=%s", dict(metadata.decision_calibration.temperature))
            ckpt.save(calibrated, resumable=False, promotable=True, sampler_metadata=metadata)
        if cfg.output_model_id:
            ckpt.promote_latest(cfg.output_model_id, cfg.base_model, checkpoint_name=calibrated)
            runner.write_output_model(model_id=cfg.output_model_id, checkpoint=calibrated, job_id=job_id)

        write_completed(runner, step=step, total_steps=step)
    wandb_finish()
    return {"steps": step, "job_id": job_id}
