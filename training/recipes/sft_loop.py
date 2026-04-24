#!/usr/bin/env python3
"""Minimal SFT (Supervised Fine-Tuning) training loop.

A readable, modifiable fine-tuning loop using the Fireworks RLOR API.
Uses a single RLOR trainer job with cross-entropy loss on response tokens.

Dataset format (JSONL, OpenAI chat format):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.sft_loop
"""

from __future__ import annotations

import os
import signal
import logging
import functools
from itertools import islice
from contextlib import ExitStack
from typing import Any, Dict, List
from dataclasses import field, dataclass

import torch
import tinker

import transformers
from dotenv import load_dotenv

from fireworks.training.sdk import TrainerJobManager
from training.utils import (
    DEFAULT_ADAM,
    DEFAULT_RENDER_WORKERS,
    InfraConfig,
    JsonlRenderDataset,
    ResourceCleanup,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    ReconnectableClient,
    make_render_dataloader,
    wandb_log,
    setup_wandb,
    wandb_finish,
    validate_config,
    log_metrics_json,
    create_trainer_job,
    read_api_extra_headers_env,
    build_renderer,
    parse_train_on_what,
    auto_select_training_shape,
    render_messages_to_datum,
    resolve_renderer_name,
)
from training.utils.client import DEFAULT_TIMEOUT_S
from training.utils.checkpoint_utils import (
    resolve_resume,
    save_checkpoint,
    validate_warm_start_config,
    CheckpointKind,
)
from training.utils.losses import make_batch_weighted_sft_loss_fn
from training.utils.timer import timer, flush_timing

load_dotenv()
logger = logging.getLogger(__name__)

# Module-level so DataLoader worker_init_fn can populate it once per
# worker (avoids re-pickling the tokenizer on every batch). The parent
# process also initialises this dict to support eval-set rendering and
# auto carve-out, which run in-process with the same render_fn.
_worker_state: dict = {}


def _init_render_worker(
    tokenizer_model: str,
    renderer_name: str,
    train_on_what_str: str,
    max_seq_len: int,
    _worker_id: int | None = None,
) -> None:
    """Populate ``_worker_state`` with a tokenizer + renderer.

    ``_worker_id`` is accepted (and ignored) so this function can be
    used directly as a DataLoader ``worker_init_fn`` after binding the
    other args via ``functools.partial``.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_model, trust_remote_code=True,
    )
    _worker_state.update(
        renderer=build_renderer(tokenizer, tokenizer_model, renderer_name),
        train_on_what=parse_train_on_what(train_on_what_str),
        max_seq_len=max_seq_len,
    )


def _render_one_worker(row: dict) -> tinker.Datum | None:
    """Render a chat row to a Datum, dropping empty / out-of-range sequences.

    Reads renderer / train_on_what / max_seq_len from the per-process
    ``_worker_state`` populated by ``_init_render_worker``. Top-level
    so spawn workers can pickle it as the DataLoader's render_fn.
    """
    messages = row.get("messages", [])
    if not messages:
        return None
    rendered = render_messages_to_datum(
        messages,
        renderer=_worker_state["renderer"],
        train_on_what=_worker_state["train_on_what"],
    )
    if not 2 <= len(rendered.token_ids) <= _worker_state["max_seq_len"]:
        return None
    return rendered.datum


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

DEFAULT_EVAL_CARVE_RATIO = 0.1
DEFAULT_MAX_EVAL_SEQS = 100


def compute_eval_carveout(
    total_samples: int,
    max_ratio: float = DEFAULT_EVAL_CARVE_RATIO,
    max_seqs: int = DEFAULT_MAX_EVAL_SEQS,
) -> int:
    """Number of head samples to carve out as eval (0 if dataset too small)."""
    if total_samples <= 1:
        return 0
    carveout = min(int(total_samples * max_ratio), max_seqs)
    return 0 if carveout >= total_samples else carveout


def _render_eagerly(ds: JsonlRenderDataset, n: int) -> List[tinker.Datum]:
    """Render the first ``n`` rows of ``ds`` in-process, dropping Nones."""
    return [d for d in (ds[i] for i in range(n)) if d is not None]


def _prepare_datasets(
    cfg: "Config",
) -> tuple[JsonlRenderDataset, List[tinker.Datum]]:
    """Build the training dataset and (optional) eval set.

    Eval can come from an explicit ``cfg.evaluation_dataset`` or be
    carved out from the head of the training dataset. In the carve-out
    case the returned training dataset is sliced past the eval window.
    """
    training_ds = JsonlRenderDataset(
        cfg.dataset, _render_one_worker, max_examples=cfg.max_examples,
    )
    if len(training_ds) == 0:
        raise RuntimeError(f"No examples found in {cfg.dataset}")

    if cfg.evaluation_dataset:
        eval_ds = JsonlRenderDataset(cfg.evaluation_dataset, _render_one_worker)
        eval_data = _render_eagerly(eval_ds, len(eval_ds))
        logger.info(
            "Loaded %d eval examples from %s",
            len(eval_data), cfg.evaluation_dataset,
        )
        return training_ds, eval_data

    if cfg.eval_auto_carveout:
        n = compute_eval_carveout(
            len(training_ds), cfg.eval_carve_ratio, cfg.max_eval_seqs,
        )
        if n > 0:
            eval_data = _render_eagerly(training_ds, n)
            training_ds = training_ds.with_indices(list(range(n, len(training_ds))))
            logger.info(
                "Auto carve-out: %d eval examples, %d training examples",
                len(eval_data), len(training_ds),
            )
            return training_ds, eval_data
        logger.warning("Dataset too small for auto carve-out, skipping eval")

    return training_ds, []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str = ""
    tokenizer_model: str = ""  # HuggingFace model name for chat template, e.g. "Qwen/Qwen3-1.7B"
    renderer_name: str = ""
    train_on_what: str = "all_assistant_messages"

    learning_rate: float = 1e-4
    epochs: int = 3
    batch_size: int = 32
    grad_accum: int = 1
    """Deprecated. Ignored. Use ``batch_size`` to control the effective batch."""
    max_seq_len: int | None = None
    max_examples: int | None = None
    lora_rank: int = 0
    output_model_id: str | None = None
    save_final_checkpoint: bool = True

    dcp_save_interval: int = 0  # save DCP checkpoint every N steps (0 = off)

    init_from_checkpoint: str | None = None
    """Load pretrained DCP weights on a fresh dataset. Supports cross-job
    format ``"job_id:checkpoint_name"``."""

    warm_start_from_adapter: str | None = None
    """GCS URI of an HF PEFT adapter directory. When set, initializes LoRA
    weights from the adapter at training start (weights-only, fresh optimizer).
    Mutually exclusive with ``init_from_checkpoint``. Requires ``lora_rank > 0``."""

    grad_clip_norm: float = 1.0
    """Max gradient norm for clipping. 0 = no clipping."""

    step_timeout: int = 0
    """Timeout in seconds for forward_backward / optim_step calls.
    0 = use DEFAULT_TIMEOUT_S from training.utils.client."""

    trainer_job_id: str | None = None
    """Pre-created RLOR trainer job ID. When set, skips trainer creation."""

    trainer_base_url: str | None = None
    """Deprecated. Kept for back-compat; ignored (the gateway routes all trainer traffic)."""

    evaluation_dataset: str = ""
    """Path to an explicit eval dataset (JSONL).  When set, auto-carveout
    is skipped and this dataset is used for evaluation instead."""

    eval_auto_carveout: bool = False
    """When True and no eval_dataset is provided, automatically carve out
    the first N training examples as an eval set."""

    eval_carve_ratio: float = DEFAULT_EVAL_CARVE_RATIO
    """Max fraction of training data to use for eval carve-out."""

    max_eval_seqs: int = DEFAULT_MAX_EVAL_SEQS
    """Max number of eval sequences for carve-out."""

    render_workers: int | None = None
    """Number of worker processes for streaming dataset rendering. ``None``
    auto-selects ``min(os.cpu_count(), DEFAULT_RENDER_WORKERS)``. Set to 1
    to disable the spawn pool and render in-process (useful for unit tests
    that monkeypatch the renderer)."""

    infra: InfraConfig = field(default_factory=InfraConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="sft-tinker"))
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    """Optional orchestration outputs written during training.

    Paths can be set here or via environment variables:
      COOKBOOK_STATUS_FILE      -- training status + progress (JSON, overwritten each step)
      COOKBOOK_METADATA_FILE    -- tokens processed + accelerator-seconds (JSON)
      COOKBOOK_METRICS_FILE     -- per-step metrics (JSONL, appended each step)
      COOKBOOK_OUTPUT_MODEL_PATH -- final model info written on completion (JSON)

    All paths are optional; unset paths are silently skipped.
    See training/utils/runner.py for file format details.
    """


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def run_eval(
    eval_data: List[tinker.Datum],
    client: ReconnectableClient,
    batch_size: int,
    step: int,
    epoch: int,
) -> float | None:
    """Run evaluation without affecting model weights or optimizer state.

    Uses forward_backward_custom with a zero-gradient loss function: the
    training loss computes correct metrics, but the returned loss is
    multiplied by zero so backward produces zero gradients.  This avoids
    corrupting Adam's momentum/variance estimates.
    """
    if not eval_data:
        return None

    logger.info("[Eval] Running evaluation (%d examples)...", len(eval_data))

    eval_loss_sum = 0.0
    eval_resp_tokens = 0

    def _make_eval_loss_fn():
        train_loss_fn = make_batch_weighted_sft_loss_fn()

        def eval_loss_fn(
            data: List[tinker.Datum],
            logprobs_list: List[torch.Tensor],
        ) -> tuple[torch.Tensor, Dict[str, float]]:
            real_loss, metrics = train_loss_fn(data, logprobs_list)
            return real_loss * 0.0, metrics

        return eval_loss_fn

    batch: List[tinker.Datum] = []
    for item in eval_data:
        batch.append(item)
        if len(batch) >= batch_size:
            result = client.forward_backward_custom(batch, _make_eval_loss_fn())
            m = result.metrics
            eval_loss_sum += m.get("ce_loss_sum", 0.0)
            eval_resp_tokens += int(m.get("response_tokens", 0))
            batch = []

    if batch:
        result = client.forward_backward_custom(batch, _make_eval_loss_fn())
        m = result.metrics
        eval_loss_sum += m.get("ce_loss_sum", 0.0)
        eval_resp_tokens += int(m.get("response_tokens", 0))

    if eval_resp_tokens == 0:
        logger.warning("[Eval] No valid eval tokens, skipping metrics")
        return None

    eval_loss = eval_loss_sum / eval_resp_tokens
    eval_ppl = torch.exp(torch.tensor(eval_loss)).item()

    logger.info(
        "[Eval] Epoch %d | Loss: %.4f | PPL: %.2f | Tokens: %d",
        epoch + 1, eval_loss, eval_ppl, eval_resp_tokens,
    )
    log_metrics_json(step, eval_loss=eval_loss, eval_ppl=eval_ppl)
    wandb_log(
        {"eval/loss": eval_loss, "eval/ppl": eval_ppl, "eval/tokens": eval_resp_tokens},
        step,
    )

    return eval_loss


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
):
    cfg = config
    if cfg.trainer_base_url:
        logger.warning(
            "Config.trainer_base_url is ignored; the gateway routes all trainer "
            "traffic. This field is kept for back-compat and will be removed "
            "in a future release.",
        )
    runner = RunnerIO(cfg.runner)

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(cfg.base_model, cfg.dataset, output_model_id=cfg.output_model_id)
    validate_warm_start_config(
        warm_start_from_adapter=cfg.warm_start_from_adapter,
        init_from_checkpoint=cfg.init_from_checkpoint,
        lora_rank=cfg.lora_rank,
    )

    if cfg.grad_accum > 1:
        logger.warning(
            "grad_accum is deprecated and ignored. "
            "Increase batch_size instead for larger effective batches."
        )

    setup_wandb(
        cfg.wandb,
        {
            "lr": cfg.learning_rate,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
        },
    )

    if not cfg.tokenizer_model:
        raise ValueError(
            "Config.tokenizer_model is required for chat template formatting. "
            "Set it to the HuggingFace model name (e.g. 'Qwen/Qwen3-1.7B')."
        )

    # -- Setup infrastructure ----------------------------------------------

    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
        )

    if cfg.trainer_job_id and cfg.max_seq_len is None:
        raise ValueError(
            "max_seq_len is required when reusing a pre-created trainer "
            "(trainer_job_id is set). The auto-selected training shape may not "
            "match the trainer's actual context length."
        )

    if not cfg.infra.training_shape_id:
        cfg.infra.training_shape_id = auto_select_training_shape(
            rlor_mgr,
            base_model=cfg.base_model,
            trainer_role="policy",
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
        )
        logger.info("Auto-selected training shape: %s", cfg.infra.training_shape_id)

    trainer_profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)
    if cfg.max_seq_len is None:
        cfg.max_seq_len = trainer_profile.max_supported_context_length

    runner.set_accelerator_info(profile=trainer_profile)
    runner.write_status(RunStatus.PENDING, message="provisioning")

    def _on_trainer_status(msg: str) -> None:
        runner.write_status(RunStatus.PENDING, message=msg)

    with runner, ResourceCleanup(rlor_mgr) as cleanup, ExitStack() as stack:
        endpoint = create_trainer_job(
            rlor_mgr,
            base_model=cfg.base_model,
            infra=cfg.infra,
            profile=trainer_profile,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            display_name="sft-trainer",
            job_id=cfg.trainer_job_id,
            cleanup=cleanup,
            on_status=_on_trainer_status,
        )
        job_id = endpoint.job_id
        client = ReconnectableClient(
            rlor_mgr, job_id, cfg.base_model, cfg.lora_rank, fw_api_key=api_key,
            default_timeout=cfg.step_timeout or DEFAULT_TIMEOUT_S,
        )
        if hasattr(client, "close"):
            stack.callback(client.close)

        # -- Prepare data ------------------------------------------------------
        # Render JSONL rows on the fly inside DataLoader workers so peak
        # RAM is O(num_workers * per_worker_render_footprint) instead of
        # O(dataset_size). See docs/engineering/sft-v2-orchestrator-oom-debug.md.
        num_workers = (
            cfg.render_workers
            if cfg.render_workers is not None
            else min(os.cpu_count() or 1, DEFAULT_RENDER_WORKERS)
        )
        # Initialise the parent's _worker_state (used by eval / carve-out
        # rendering in-process) and bind the same args as the DataLoader
        # workers' init_fn.
        init_args = (
            cfg.tokenizer_model, cfg.renderer_name,
            cfg.train_on_what, cfg.max_seq_len,
        )
        _init_render_worker(*init_args)
        worker_init_fn = functools.partial(_init_render_worker, *init_args)

        training_dataset, eval_data = _prepare_datasets(cfg)
        training_count = len(training_dataset)
        effective_batch_size = max(1, min(cfg.batch_size, training_count))
        if effective_batch_size < cfg.batch_size:
            logger.warning(
                "Training examples (%d) < batch_size (%d); reducing effective "
                "batch_size to %d.",
                training_count, cfg.batch_size, effective_batch_size,
            )

        loader = make_render_dataloader(
            training_dataset,
            batch_size=effective_batch_size,
            num_workers=num_workers,
            shuffle=True,
            worker_init_fn=worker_init_fn,
        )
        # Pre-filter upper bound; filtered rows make actual batches
        # smaller but never larger.
        total_batches_per_epoch = (training_count + effective_batch_size - 1) // effective_batch_size
        logger.info(
            "Dataset: %d examples from %s (renderer=%s, train_on_what=%s,"
            " workers=%d) -> ~%d batches/epoch x %d epochs%s",
            training_count, cfg.dataset,
            resolve_renderer_name(cfg.tokenizer_model, cfg.renderer_name),
            cfg.train_on_what, num_workers,
            total_batches_per_epoch, cfg.epochs,
            f" + {len(eval_data)} eval examples" if eval_data else "",
        )

        # -- Resume ---------------------------------------------------------------

        resume_info = resolve_resume(
            client,
            cfg.log_path,
            cfg.init_from_checkpoint,
            cfg.warm_start_from_adapter,
        )

        step = resume_info.step if resume_info else 0
        data_consumed = resume_info.data_consumed if resume_info else 0
        wandb_log({"train/step": step}, step)

        adam_kwargs = dict(DEFAULT_ADAM)
        adam_kwargs["grad_clip_norm"] = cfg.grad_clip_norm
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **adam_kwargs)

        # -- Training loop (batch-indexed) -------------------------------------

        start_batch = data_consumed // effective_batch_size
        total_steps_estimate = total_batches_per_epoch * cfg.epochs

        def _run_train_step(
            batch: list[tinker.Datum],
            step: int,
        ) -> int:

            # Count total tokens for throughput tracking
            step_total_tokens = sum(
                len(chunk.tokens)
                for d in batch
                for chunk in d.model_input.chunks
                if hasattr(chunk, "tokens")
            )

            with timer("fwd_bwd"):
                result = client.forward_backward(batch)
                loss_sum = result.metrics.get("loss:sum", 0.0)
                response_tokens = result.metrics.get("response_tokens")
                if response_tokens is None:
                    response_tokens = sum(sum(d.loss_fn_inputs["weights"].data) for d in batch)

            with timer("optim_step"):
                optim_result = client.optim_step(adam_params)
            step += 1

            if cfg.dcp_save_interval > 0 and step % cfg.dcp_save_interval == 0:
                with timer("dcp_save"):
                    logger.info("Saving DCP checkpoint at step %d", step)
                    save_checkpoint(client, f"step-{step}", cfg.log_path, {
                        "step": step,
                        "data_consumed": data_consumed,
                        "source_job_id": job_id,
                    }, kind=CheckpointKind.STATE,
                    base_model=cfg.base_model,
                    training_shape=cfg.infra.training_shape_id)

            step_metrics: Dict[str, Any] = flush_timing()

            if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
                for k, v in optim_result.metrics.items():
                    step_metrics[f"train/{k}"] = v

            # Compute tokens/sec
            fwd_bwd_time = step_metrics.get("perf/fwd_bwd_time", 0.0)
            step_wall_time = fwd_bwd_time + step_metrics.get("perf/optim_step_time", 0.0)
            if step_wall_time > 0:
                step_metrics["train/tokens_per_sec"] = step_total_tokens / step_wall_time
                step_metrics["train/tokens_per_sec_fwd_bwd"] = step_total_tokens / fwd_bwd_time if fwd_bwd_time > 0 else 0.0
            step_metrics["train/total_tokens"] = step_total_tokens

            if response_tokens > 0:
                avg_loss = loss_sum / response_tokens
                ppl = torch.exp(torch.tensor(avg_loss)).item()
                logger.info(
                    "Step %d/%d | Loss: %.4f | PPL: %.2f | tok/s: %.0f | tokens: %d",
                    step, total_steps_estimate, avg_loss, ppl, step_metrics["train/tokens_per_sec"], step_total_tokens,
                )
                log_metrics_json(step, ce_loss=avg_loss, ppl=ppl)
                step_metrics.update({
                    "train/step": step,
                    "train/ce_loss": avg_loss,
                    "train/loss": avg_loss,  # alias for frontend compatibility
                    "train/ppl": ppl,
                })
                wandb_log(step_metrics, step)

            runner.append_metrics(step, step_metrics, tokens=step_total_tokens)
            runner.write_status(
                RunStatus.RUNNING, step=step, total_steps=total_steps_estimate, message="training",
            )
            runner.write_metadata()

            return step

        runner.start_training()
        runner.write_status(RunStatus.RUNNING, total_steps=total_steps_estimate, message="training")

        for epoch in range(cfg.epochs):
            # On resume, skip the batches already processed in epoch 0.
            # Workers still render skipped batches (wasted but bounded by
            # num_workers * prefetch_factor); resume is rare enough that
            # this is acceptable.
            batch_iter = iter(loader)
            if epoch == 0 and start_batch > 0:
                batch_iter = islice(batch_iter, start_batch, None)
            for batch in batch_iter:
                if not batch:
                    continue  # entire batch was filtered (None render); skip
                data_consumed += len(batch)
                step = _run_train_step(batch, step)

            # Run eval after each epoch
            if eval_data:
                try:
                    eval_loss = run_eval(
                        eval_data=eval_data,
                        client=client,
                        batch_size=cfg.batch_size,
                        step=step,
                        epoch=epoch,
                    )
                    if eval_loss is not None:
                        runner.append_metrics(
                            step,
                            {"eval/loss": eval_loss, "eval/ppl": torch.exp(torch.tensor(eval_loss)).item()},
                        )
                except Exception as e:
                    logger.warning("Eval failed at epoch %d, continuing: %s", epoch + 1, e)

        # -- Final checkpoint --------------------------------------------------

        start_step = resume_info.step if resume_info else 0
        if cfg.save_final_checkpoint and step > start_step:
            logger.info("Saving final checkpoint (step %d)...", step)
            cp_name = f"step-{step}"
            paths = save_checkpoint(client, cp_name, cfg.log_path, {
                "step": step,
                "data_consumed": data_consumed,
                "source_job_id": job_id,
            }, kind=CheckpointKind.BOTH,
            base_model=cfg.base_model,
            training_shape=cfg.infra.training_shape_id)
            if getattr(cfg, "output_model_id", None):
                rlor_mgr.promote_checkpoint(
                    job_id,
                    paths["sampler_path"],
                    cfg.output_model_id,
                    cfg.base_model,
                )
                runner.write_output_model(
                    model_id=cfg.output_model_id, checkpoint=cp_name, job_id=job_id,
                )

        runner.write_status(
            RunStatus.COMPLETED, step=step, total_steps=total_steps_estimate, message="done",
        )
        runner.write_metadata()
        logger.info("Training complete: %d optimizer steps", step)
        wandb_finish()
        return {"steps": step, "job_id": job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        log_path="./sft_logs",
        dataset="kimi2_deid_sample_100_formatted.jsonl",
        tokenizer_model="Qwen/Qwen3-8B",
        max_seq_len=4096,
        max_examples=10,
        infra=InfraConfig(
            training_shape_id="your-training-shape",
        ),
    )
    main(cfg)
