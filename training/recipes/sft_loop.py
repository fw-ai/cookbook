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
from contextlib import ExitStack
from typing import Any, Dict, List
from dataclasses import field, dataclass

import torch
import tinker

import json
import datasets as hf_datasets
import transformers
from dotenv import load_dotenv

from fireworks.training.sdk import TrainerJobManager
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from training.utils import (
    DEFAULT_ADAM,
    InfraConfig,
    ResourceCleanup,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    ReconnectableClient,
    wandb_log,
    setup_wandb,
    wandb_finish,
    validate_config,
    log_metrics_json,
    create_trainer_job,
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
    CheckpointKind,
)
from training.utils.losses import make_batch_weighted_sft_loss_fn
from training.utils.timer import timer, flush_timing

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Eval auto carve-out defaults
# ---------------------------------------------------------------------------

DEFAULT_EVAL_CARVE_RATIO = 0.1
DEFAULT_MAX_EVAL_SEQS = 100


def compute_eval_carveout(
    total_samples: int,
    max_ratio: float = DEFAULT_EVAL_CARVE_RATIO,
    max_seqs: int = DEFAULT_MAX_EVAL_SEQS,
) -> int:
    """Compute number of samples to carve out from training data for eval.

    Mirrors the SFT v1 carve-out logic: take min(total * ratio, max_seqs),
    but return 0 if the dataset is too small to split.

    Args:
        total_samples: Total number of tokenized training examples.
        max_ratio: Max fraction of data to use for eval (default 10%).
        max_seqs: Absolute cap on eval sequences (default 100).

    Returns:
        Number of samples to reserve for eval (first N of the dataset).
    """
    if total_samples <= 1:
        return 0
    carveout = int(total_samples * max_ratio)
    carveout = min(carveout, max_seqs)
    # Need at least 1 sample left for training
    if carveout >= total_samples:
        return 0
    return carveout


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

    grad_clip_norm: float = 1.0
    """Max gradient norm for clipping. 0 = no clipping."""

    step_timeout: int = 0
    """Timeout in seconds for forward_backward / optim_step calls.
    0 = use DEFAULT_TIMEOUT_S from training.utils.client."""

    trainer_job_id: str | None = None
    """Pre-created RLOR trainer job ID. When set, skips trainer creation."""

    trainer_base_url: str | None = None
    """Direct base URL for the trainer (e.g. ``http://localhost:8080``).
    When set together with ``trainer_job_id``, bypasses the gateway and
    connects to the trainer directly."""

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
    runner = RunnerIO(cfg.runner)

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(cfg.base_model, cfg.dataset, output_model_id=cfg.output_model_id)

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

    _precreated_trainer = cfg.trainer_job_id and cfg.trainer_base_url

    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)

    if _precreated_trainer:
        trainer_infra = cfg.infra
        trainer_profile = None
        if cfg.max_seq_len is None:
            raise ValueError("max_seq_len is required when using a pre-created trainer.")
        logger.info(
            "Using pre-created trainer %s at %s",
            cfg.trainer_job_id,
            cfg.trainer_base_url,
        )
    else:
        if not cfg.infra.training_shape_id:
            cfg.infra.training_shape_id = auto_select_training_shape(
                rlor_mgr,
                base_model=cfg.base_model,
                trainer_role="policy",
                lora_rank=cfg.lora_rank,
                max_seq_len=cfg.max_seq_len,
            )
            logger.info("Auto-selected training shape: %s", cfg.infra.training_shape_id)

        profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)
        trainer_profile = profile

        if cfg.max_seq_len is None:
            cfg.max_seq_len = profile.max_supported_context_length

    runner.set_accelerator_info(profile=trainer_profile if not _precreated_trainer else None)
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
            base_url_override=cfg.trainer_base_url,
            cleanup=cleanup,
            on_status=_on_trainer_status,
        )
        job_id = endpoint.job_id
        client = ReconnectableClient(
            rlor_mgr, job_id, cfg.base_model, cfg.lora_rank, fw_api_key=api_key,
            default_timeout=cfg.step_timeout or DEFAULT_TIMEOUT_S,
            endpoint=endpoint if cfg.trainer_base_url else None,
        )
        if hasattr(client, "close"):
            stack.callback(client.close)

        # -- Prepare data ------------------------------------------------------
        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer_model, trust_remote_code=True)
        renderer = build_renderer(tokenizer, cfg.tokenizer_model, cfg.renderer_name)
        train_on_what = parse_train_on_what(cfg.train_on_what)
        logger.info(
            "Using renderer=%s train_on_what=%s",
            resolve_renderer_name(cfg.tokenizer_model, cfg.renderer_name),
            train_on_what.value,
        )

        raw_data: List[Dict[str, Any]] = []
        with open(cfg.dataset) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_data.append(json.loads(line))
                if cfg.max_examples and len(raw_data) >= cfg.max_examples:
                    break
        logger.info("Loaded %d examples from %s", len(raw_data), cfg.dataset)

        max_seq_len = cfg.max_seq_len
        filtered_count = 0

        def _map_fn(row: dict) -> tinker.Datum | None:
            nonlocal filtered_count
            messages = row.get("messages", [])
            if not messages:
                filtered_count += 1
                return None
            rendered = render_messages_to_datum(
                messages,
                renderer=renderer,
                train_on_what=train_on_what,
            )
            if len(rendered.token_ids) > max_seq_len or len(rendered.token_ids) < 2:
                filtered_count += 1
                return None
            return rendered.datum

        training_data = [d for row in raw_data if (d := _map_fn(row)) is not None]
        if filtered_count > 0:
            logger.info(
                "Seq-length filter: %d/%d examples filtered (len > %d or len < 2)",
                filtered_count,
                len(raw_data),
                max_seq_len,
            )
        logger.info("Prepared %d training examples", len(training_data))
        if not training_data:
            raise RuntimeError("No valid training examples after tokenization")

        # -- Eval dataset (explicit or auto carve-out) -------------------------
        eval_data: List[tinker.Datum] = []
        if cfg.evaluation_dataset:
            # Explicit eval dataset
            raw_eval: List[Dict[str, Any]] = []
            with open(cfg.evaluation_dataset) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw_eval.append(json.loads(line))
            eval_data = [d for row in raw_eval if (d := _map_fn(row)) is not None]
            logger.info("Loaded %d eval examples from %s", len(eval_data), cfg.evaluation_dataset)
        elif cfg.eval_auto_carveout:
            # Auto carve-out: split first N examples as eval
            carveout_count = compute_eval_carveout(
                len(training_data), cfg.eval_carve_ratio, cfg.max_eval_seqs,
            )
            if carveout_count > 0:
                eval_data = training_data[:carveout_count]
                training_data = training_data[carveout_count:]
                logger.info(
                    "Auto carve-out: %d eval examples, %d training examples",
                    len(eval_data), len(training_data),
                )
            else:
                logger.warning("Dataset too small for auto carve-out, skipping eval")

        effective_batch_size = cfg.batch_size
        if len(training_data) < effective_batch_size:
            logger.warning(
                "Training examples (%d) < batch_size (%d); reducing effective "
                "batch_size to %d so all examples are trained on.",
                len(training_data),
                effective_batch_size,
                len(training_data),
            )
            effective_batch_size = len(training_data)

        sft_dataset = SupervisedDatasetFromHFDataset(
            hf_datasets.Dataset.from_dict({"datum_idx": list(range(len(training_data)))}),
            batch_size=effective_batch_size,
            map_fn=lambda row: training_data[row["datum_idx"]],
        )
        total_batches_per_epoch = len(sft_dataset)
        logger.info(
            "Dataset: %d examples, %d batches/epoch, %d epochs",
            len(training_data),
            total_batches_per_epoch,
            cfg.epochs,
        )
        if eval_data:
            logger.info("Eval dataset: %d examples (eval after each epoch)", len(eval_data))

        # -- Resume ---------------------------------------------------------------

        resume_info = resolve_resume(client, cfg.log_path, cfg.init_from_checkpoint)
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
            sft_dataset.set_epoch(epoch)
            epoch_start = start_batch if epoch == 0 else 0
            for i_batch in range(epoch_start, total_batches_per_epoch):
                batch = sft_dataset.get_batch(i_batch)
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
