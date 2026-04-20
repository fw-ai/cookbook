#!/usr/bin/env python3
"""ORPO (Odds Ratio Preference Optimization) training loop.

Uses a single RLOR trainer job -- no reference model needed (unlike DPO).

Loss:
    L_ORPO = L_SFT(chosen) + lambda * L_OR
    L_OR   = -log(sigmoid(log(odds_chosen / odds_rejected)))

Dataset format (JSONL, same as DPO):
    {"chosen": {"messages": [...]}, "rejected": {"messages": [...]}}

Usage:
    export FIREWORKS_API_KEY=...
    export FIREWORKS_BASE_URL=...          # optional, defaults to https://api.fireworks.ai
    python -m recipes.orpo_loop

    # override dataset / tokenizer via env vars:
    ORPO_DATASET=/path/to/data.jsonl ORPO_TOKENIZER=Qwen/Qwen3-8B python ...

Config args:
    base_model       Fireworks model ID (default: qwen3-235b-a22b-instruct-2507)
    dataset          Path to preference JSONL file
    tokenizer_model  HuggingFace model name for client-side tokenization
    orpo_lambda      Weight for odds-ratio loss term (default: 1.0)
    learning_rate    Adam learning rate (default: 1e-5)
    epochs           Number of passes over the dataset (default: 1)
    batch_size       Number of preference pairs per optimizer step (default: 4)
    max_seq_len      Max token length per sequence (auto from training shape)
    lora_rank        LoRA rank, 0 for full fine-tuning (default: 0)

Infrastructure defaults target Qwen3-235B on 2 nodes with CP=16, EP=8.
"""

from __future__ import annotations

import math
import os
import time
import signal
import random
import logging
from contextlib import ExitStack
from dataclasses import field, dataclass
import torch

import tinker

from fireworks.training.sdk import TrainerJobManager
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
    log_metrics_json,
    make_batch_orpo_loss_fn,
    create_trainer_job,
    read_api_extra_headers_env,
    load_preference_dataset,
    build_renderer,
    auto_select_training_shape,
    render_preference_pair,
    resolve_renderer_name,
    validate_config,
)
from training.utils.client import DEFAULT_TIMEOUT_S
from training.utils.checkpoint_utils import resolve_resume, save_checkpoint, CheckpointKind

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    base_model: str = "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"
    dataset: str = ""
    tokenizer_model: str = ""
    renderer_name: str = ""

    orpo_lambda: float = 1.0
    learning_rate: float = 1e-5
    epochs: int = 1
    batch_size: int = 4
    """Number of preference pairs per optimizer step."""
    grad_accum: int = 1
    """Deprecated. Ignored. Use ``batch_size`` to control the effective batch."""
    max_seq_len: int | None = None
    max_pairs: int | None = None
    lora_rank: int = 0
    job_id: str | None = None
    output_model_id: str | None = None

    warmup_ratio: float = 0.0
    """Fraction of total steps used for linear LR warmup (0.0 = no warmup)."""

    min_lr_ratio: float = 0.0
    """Minimum LR as a fraction of ``learning_rate``. Cosine/linear decay
    anneals to ``learning_rate * min_lr_ratio``."""

    lr_schedule: str = "constant"
    """LR schedule after warmup: ``"constant"``, ``"cosine"``, or ``"linear"``."""

    grad_accumulation_normalization: str | None = None
    """Server-side gradient normalization mode passed to optim_step.
    ``None``: no server normalization (default). The ORPO loss function
    already computes per-pair means client-side, so server-side
    normalization would double-normalize."""

    infra: InfraConfig = field(
        default_factory=lambda: InfraConfig()
    )
    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(
            project="dsv3-training",
        )
    )
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
    save_final_checkpoint: bool = True
    dcp_save_interval: int = 0  # save DCP checkpoint every N steps (0 = off)

    step_timeout: int = 0
    """Timeout in seconds for forward_backward / optim_step calls.
    0 = use DEFAULT_TIMEOUT_S from training.utils.client."""

    init_from_checkpoint: str | None = None


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _compute_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_ratio: float,
    min_lr_ratio: float,
    schedule: str,
) -> float:
    """Linear warmup then cosine/linear/constant decay."""
    min_lr = peak_lr * min_lr_ratio
    warmup_steps = int(total_steps * warmup_ratio)

    if step < warmup_steps:
        return min_lr + (peak_lr - min_lr) * step / max(warmup_steps, 1)

    if schedule == "constant":
        return peak_lr

    decay_step = step - warmup_steps
    decay_total = max(total_steps - warmup_steps, 1)
    if schedule == "cosine":
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * decay_step / decay_total))
    if schedule == "linear":
        return peak_lr - (peak_lr - min_lr) * decay_step / decay_total

    raise ValueError(f"Unknown lr_schedule: {schedule!r}. Use 'constant', 'cosine', or 'linear'.")


# ---------------------------------------------------------------------------
# Main
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

    if not cfg.tokenizer_model:
        raise ValueError(
            "Config.tokenizer_model is required for client-side tokenization. "
            "Set it to the HuggingFace model name "
            "(e.g. 'Qwen/Qwen3-235B-A22B-Instruct-2507')."
        )
    setup_wandb(
        cfg.wandb,
        {
            "orpo_lambda": cfg.orpo_lambda,
            "lr": cfg.learning_rate,
            "epochs": cfg.epochs,
        },
    )

    # -- Infrastructure ------------------------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
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

    profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)
    trainer_profile = profile

    if cfg.max_seq_len is None:
        cfg.max_seq_len = profile.max_supported_context_length

    runner.set_accelerator_info(profile=profile)
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
            display_name="orpo-trainer",
            job_id=cfg.job_id,
            cleanup=cleanup if not cfg.job_id else None,
            on_status=_on_trainer_status,
        )
        job_id = endpoint.job_id
        client = ReconnectableClient(
            rlor_mgr, endpoint.job_id, cfg.base_model, cfg.lora_rank,
            default_timeout=cfg.step_timeout or DEFAULT_TIMEOUT_S,
        )
        if hasattr(client, "close"):
            stack.callback(client.close)
        resume_info = resolve_resume(client, cfg.log_path, cfg.init_from_checkpoint)
        step_offset = resume_info.step if resume_info else 0

        # -- Data ----------------------------------------------------------------

        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.tokenizer_model, trust_remote_code=True
        )
        renderer = build_renderer(tokenizer, cfg.tokenizer_model, cfg.renderer_name)
        logger.info(
            "Using renderer=%s for preference tokenization",
            resolve_renderer_name(cfg.tokenizer_model, cfg.renderer_name),
        )

        raw_data = load_preference_dataset(cfg.dataset, cfg.max_pairs)
        if not raw_data:
            raise RuntimeError(f"No data loaded from {cfg.dataset}")

        total_raw = len(raw_data)
        log_interval = max(1, total_raw // 20)  # ~5% increments
        logger.info("Tokenizing %d preference pairs...", total_raw)
        pair_cache: list[dict] = []
        filtered_count = 0

        for i, example in enumerate(raw_data):
            pair = render_preference_pair(
                example["chosen"],
                example["rejected"],
                renderer=renderer,
                tokenizer=tokenizer,
            )
            if pair is None:
                continue

            if (
                len(pair.chosen_tokens) > cfg.max_seq_len
                or len(pair.rejected_tokens) > cfg.max_seq_len
            ):
                filtered_count += 1
                continue
            pair_cache.append(
                {
                    "chosen_tokens": pair.chosen_tokens,
                    "rejected_tokens": pair.rejected_tokens,
                    "response_start": pair.response_start,
                    "chosen_datum": pair.chosen_datum,
                    "rejected_datum": pair.rejected_datum,
                }
            )
            if (i + 1) % log_interval == 0 or (i + 1) == total_raw:
                runner.report_rendering_progress(i + 1, total_raw)

        if filtered_count > 0:
            logger.info(
                "Seq-length filter: %d/%d pairs filtered (chosen or rejected > %d tokens)",
                filtered_count,
                len(raw_data),
                cfg.max_seq_len,
            )
        logger.info("Prepared %d preference pairs", len(pair_cache))
        if not pair_cache:
            raise RuntimeError("No valid pairs after tokenization")

        # -- Training loop -------------------------------------------------------

        step = step_offset
        total_steps = ((len(pair_cache) + cfg.batch_size - 1) // cfg.batch_size) * cfg.epochs

        has_lr_schedule = cfg.warmup_ratio > 0 or cfg.lr_schedule != "constant"
        if has_lr_schedule:
            logger.info(
                "LR schedule: %s | warmup_ratio=%.2f (%d steps) | "
                "peak_lr=%g | min_lr=%g",
                cfg.lr_schedule,
                cfg.warmup_ratio,
                int(total_steps * cfg.warmup_ratio),
                cfg.learning_rate,
                cfg.learning_rate * cfg.min_lr_ratio,
            )

        def _run_train_step(epoch: int, step_pairs: list[dict], step_started_at: float) -> float:
            nonlocal step

            datums: list[tinker.Datum] = []
            response_starts: list[int] = []
            step_tokens = 0
            for pair in step_pairs:
                datums.extend([pair["chosen_datum"], pair["rejected_datum"]])
                response_starts.append(pair["response_start"])
                step_tokens += len(pair["chosen_tokens"]) + len(pair["rejected_tokens"])

            current_lr = _compute_lr(
                step, total_steps, cfg.learning_rate,
                cfg.warmup_ratio, cfg.min_lr_ratio, cfg.lr_schedule,
            )
            adam_params = tinker.AdamParams(learning_rate=current_lr, **DEFAULT_ADAM)

            loss_fn = make_batch_orpo_loss_fn(response_starts, cfg.orpo_lambda)
            result = client.forward_backward_custom(datums, loss_fn)
            client.optim_step(adam_params)
            step += 1

            if cfg.dcp_save_interval > 0 and step % cfg.dcp_save_interval == 0:
                logger.info("Saving DCP checkpoint at step %d", step)
                save_checkpoint(client, f"step-{step}", cfg.log_path, {
                    "step": step,
                    "data_consumed": (step - step_offset) * cfg.batch_size,
                    "source_job_id": job_id,
                }, kind=CheckpointKind.STATE,
                base_model=cfg.base_model,
                training_shape=cfg.infra.training_shape_id)

            step_elapsed = time.monotonic() - step_started_at
            tokens_per_sec = step_tokens / step_elapsed if step_elapsed > 0 else 0.0
            metrics = result.metrics

            logger.info(
                "Step %d/%d | lr=%.2e | ORPO: %.4f | SFT: %.4f | OR: %.4f | "
                "LogOR: %+.4f | Acc: %.1f%% | %.1f tok/s (%.1fs)",
                step,
                total_steps,
                current_lr,
                metrics["orpo_loss"],
                metrics["sft_loss"],
                metrics["or_loss"],
                metrics["log_odds_ratio"],
                metrics["accuracy"] * 100,
                tokens_per_sec,
                step_elapsed,
            )
            log_metrics_json(step, tokens_per_sec=tokens_per_sec, **metrics)
            step_metrics = {
                "train/step": step,
                "train/orpo_loss": metrics["orpo_loss"],
                "train/sft_loss": metrics["sft_loss"],
                "train/or_loss": metrics["or_loss"],
                "train/log_odds_ratio": metrics["log_odds_ratio"],
                "train/accuracy": metrics["accuracy"],
                "train/tokens_per_sec": tokens_per_sec,
                "train/step_time_sec": step_elapsed,
                "train/step_tokens": step_tokens,
                "train/epoch": epoch + 1,
            }
            wandb_log(step_metrics, step)
            runner.append_metrics(step, step_metrics, tokens=step_tokens)
            runner.write_status(RunStatus.RUNNING, step=step, total_steps=total_steps, message="training")
            runner.write_metadata()
            return time.monotonic()

        runner.start_training()
        runner.write_status(RunStatus.RUNNING, total_steps=total_steps, message="training")

        for epoch in range(cfg.epochs):
            random.shuffle(pair_cache)
            step_t0 = time.monotonic()
            batch_buffer: list[dict] = []
            for pair in pair_cache:
                batch_buffer.append(pair)
                if len(batch_buffer) >= cfg.batch_size:
                    step_t0 = _run_train_step(epoch, batch_buffer, step_t0)
                    batch_buffer = []

            if batch_buffer:
                _run_train_step(epoch, batch_buffer, step_t0)

        # -- Final checkpoint ------------------------------------------------

        if cfg.save_final_checkpoint and step > step_offset:
            logger.info("Saving final checkpoint (step %d)...", step)
            cp_name = f"step-{step}"
            paths = save_checkpoint(client, cp_name, cfg.log_path, {
                "step": step,
                "data_consumed": (step - step_offset) * cfg.batch_size,
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

        runner.write_status(RunStatus.COMPLETED, step=step, total_steps=total_steps, message="done")
        runner.write_metadata()
        logger.info(
            "Training complete: %d optimizer steps (%d new)", step, step - step_offset
        )
        wandb_finish()
        return {"steps": step, "job_id": job_id}


if __name__ == "__main__":
    import os

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    cfg = Config(
        log_path="./orpo_logs",
        dataset=os.environ.get("ORPO_DATASET_PATH"),
        tokenizer_model=os.environ.get("ORPO_TOKENIZER", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    )
    main(cfg)
