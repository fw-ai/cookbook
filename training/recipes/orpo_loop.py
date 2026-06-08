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
    tokenizer_revision Optional HuggingFace revision for client-side tokenization
    orpo_lambda      Weight for odds-ratio loss term (default: 1.0)
    learning_rate    Adam learning rate (default: 1e-5)
    epochs           Number of passes over the dataset (default: 1)
    batch_size       Number of preference pairs per optimizer step (default: 4)
    max_seq_len      Max token length per sequence (auto from training shape)
    lora_rank        LoRA rank, 0 for full fine-tuning (default: 0)

Infrastructure defaults target Qwen3-235B on 2 nodes with CP=16, EP=8.
"""

from __future__ import annotations

import logging
import math
import os
import random
import signal
import time
from contextlib import ExitStack
from dataclasses import dataclass, field

import tinker
from fireworks.training.sdk.training_spec import (
    LRSchedulerSpec,
    compute_lr,
    default_constant_schedule,
    normalize_lr_scheduler_spec,
)

from training.utils import (
    DEFAULT_ADAM,
    TrainerConfig,
    ReconnectableClient,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    build_renderer,
    build_service_client,
    load_preference_dataset,
    load_tokenizer,
    log_metrics_json,
    make_batch_orpo_loss_fn,
    read_api_extra_headers_env,
    render_preference_pair,
    resolve_renderer_name,
    setup_wandb,
    validate_config,
    wandb_finish,
    wandb_log,
)
from training.utils.checkpoints import TrainingCheckpoints, validate_warm_start_config
from training.utils.client import DEFAULT_TIMEOUT_S
from training.utils.runner_state import start_running, write_completed, write_running_step

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
    tokenizer_revision: str = ""
    renderer_name: str = ""

    orpo_lambda: float = 1.0
    learning_rate: float = 1e-5
    lr_scheduler: LRSchedulerSpec = field(default_factory=default_constant_schedule)
    """Per-step LR scheduler spec. Legacy flat scheduler fields below remain accepted."""

    epochs: int = 1
    batch_size: int = 4
    """Number of preference pairs per optimizer step. For managed (V2) jobs
    this is set from ``BaseTrainingConfig.batch_size_samples`` via the
    cookbook orchestrator."""
    seed: int = 0
    """Seed for deterministic per-epoch shuffling of preference pairs."""
    max_seq_len: int | None = None
    max_pairs: int | None = None
    lora_rank: int = 0
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

    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig()
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
    warm_start_from_adapter: str | None = None
    """GCS URI of an HF PEFT adapter directory. When set, initializes LoRA
    weights from the adapter at training start (weights-only, fresh optimizer).
    Mutually exclusive with ``init_from_checkpoint``. Requires ``lora_rank > 0``."""


def _is_default_lr_scheduler(data: object) -> bool:
    if data is None:
        return True
    if isinstance(data, dict):
        return (
            data.get("type", "constant") == "constant"
            and (data.get("warmup_steps") or 0) == 0
            and data.get("warmup_ratio") is None
        )
    return (
        getattr(data, "type", None) == "constant"
        and getattr(data, "warmup_steps", 0) == 0
        and getattr(data, "warmup_ratio", None) is None
    )


def _uses_legacy_orpo_lr_schedule(cfg: Config) -> bool:
    return _is_default_lr_scheduler(cfg.lr_scheduler) and (
        cfg.lr_schedule != "constant" or cfg.warmup_ratio > 0
    )


def _compute_legacy_orpo_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_ratio: float,
    min_lr_ratio: float,
    schedule: str,
) -> float:
    """Legacy ORPO flat-field schedule; ``step`` is zero-indexed."""

    min_lr = peak_lr * min_lr_ratio
    warmup_steps = int(total_steps * warmup_ratio)

    if step < warmup_steps:
        return min_lr + (peak_lr - min_lr) * step / max(warmup_steps, 1)

    if schedule == "constant":
        return peak_lr

    decay_step = step - warmup_steps
    decay_total = max(total_steps - warmup_steps, 1)
    if schedule == "cosine":
        return min_lr + 0.5 * (peak_lr - min_lr) * (
            1 + math.cos(math.pi * decay_step / decay_total)
        )
    if schedule == "linear":
        return peak_lr - (peak_lr - min_lr) * decay_step / decay_total

    raise ValueError(
        f"Unknown lr_schedule: {schedule!r}. Use 'constant', 'cosine', or 'linear'."
    )


def _shuffled_pair_cache(
    pair_cache: list[dict],
    seed: int,
    epoch: int,
) -> list[dict]:
    epoch_pairs = list(pair_cache)
    random.Random(seed + epoch).shuffle(epoch_pairs)
    return epoch_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
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
    validate_warm_start_config(
        warm_start_from_adapter=cfg.warm_start_from_adapter,
        init_from_checkpoint=cfg.init_from_checkpoint,
        lora_rank=cfg.lora_rank,
    )
    use_legacy_orpo_lr_schedule = _uses_legacy_orpo_lr_schedule(cfg)
    lr_scheduler = normalize_lr_scheduler_spec(
        cfg.lr_scheduler,
        legacy_lr_schedule=cfg.lr_schedule,
        legacy_warmup_ratio=cfg.warmup_ratio,
        legacy_min_lr_ratio=cfg.min_lr_ratio,
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
            "lr_schedule": lr_scheduler.type,
            "epochs": cfg.epochs,
        },
    )

    # -- SDK-managed Tinker client -----------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    runner.write_status(RunStatus.PENDING, message="provisioning")

    with runner, ExitStack() as stack:
        service = build_service_client(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
            base_model=cfg.base_model,
            tokenizer_model=cfg.tokenizer_model,
            lora_rank=cfg.lora_rank,
            max_context_length=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            trainer=cfg.trainer,
        )
        stack.callback(service.close)
        training_client = service.create_training_client(cfg.base_model, lora_rank=cfg.lora_rank)
        runner.set_accelerator_info(
            service.accelerator_type,
            service.accelerator_count,
            profile=service.training_profile,
        )
        job_id = service.trainer_job_id
        max_seq_len = service.max_context_length
        client = ReconnectableClient.from_training_client(
            training_client,
            base_model=cfg.base_model,
            lora_rank=cfg.lora_rank,
            job_id=job_id,
            default_timeout=cfg.step_timeout or DEFAULT_TIMEOUT_S,
            service=service,
        )

        ckpt = TrainingCheckpoints(
            client,
            service,
            trainer_id=job_id,
            log_path=cfg.log_path,
            lora_rank=cfg.lora_rank,
        )

        resume_info = ckpt.resume(
            init_from_checkpoint=cfg.init_from_checkpoint,
            warm_start_from_adapter=cfg.warm_start_from_adapter,
        )
        step_offset = resume_info.step if resume_info else 0

        # -- Data ----------------------------------------------------------------

        tokenizer = load_tokenizer(cfg.tokenizer_model, cfg.tokenizer_revision)
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
                len(pair.chosen_tokens) > max_seq_len
                or len(pair.rejected_tokens) > max_seq_len
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
                max_seq_len,
            )
        logger.info("Prepared %d preference pairs", len(pair_cache))
        if not pair_cache:
            raise RuntimeError("No valid pairs after tokenization")

        # -- Training loop -------------------------------------------------------

        step = step_offset
        total_steps = ((len(pair_cache) + cfg.batch_size - 1) // cfg.batch_size) * cfg.epochs

        logger.info(
            "LR schedule: %s | warmup_steps=%s | warmup_ratio=%s | peak_lr=%g",
            lr_scheduler.type,
            lr_scheduler.warmup_steps,
            lr_scheduler.warmup_ratio,
            cfg.learning_rate,
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

            if use_legacy_orpo_lr_schedule:
                current_lr = _compute_legacy_orpo_lr(
                    step,
                    total_steps,
                    cfg.learning_rate,
                    cfg.warmup_ratio,
                    cfg.min_lr_ratio,
                    cfg.lr_schedule,
                )
            else:
                current_lr = compute_lr(
                    lr_scheduler,
                    step=step + 1,
                    base_lr=cfg.learning_rate,
                    total_steps=total_steps,
                )
            adam_params = tinker.AdamParams(learning_rate=current_lr, **DEFAULT_ADAM)

            loss_fn = make_batch_orpo_loss_fn(response_starts, cfg.orpo_lambda)
            result = client.forward_backward_custom(datums, loss_fn)
            client.optim_step(adam_params)
            step += 1

            if cfg.dcp_save_interval > 0 and step % cfg.dcp_save_interval == 0:
                logger.info("Saving DCP checkpoint at step %d", step)
                ckpt.save(
                    f"step-{step}",
                    resumable=True,
                    promotable=False,
                    data_consumed=(step - step_offset) * cfg.batch_size,
                )

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
            log_metrics_json(step, tokens_per_sec=tokens_per_sec, lr=current_lr, **metrics)
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
                "train/lr": current_lr,
            }
            wandb_log(step_metrics, step)
            write_running_step(
                runner,
                step=step,
                total_steps=total_steps,
                metrics=step_metrics,
                tokens=step_tokens,
            )
            return time.monotonic()

        start_running(runner, total_steps=total_steps)

        for epoch in range(cfg.epochs):
            epoch_pairs = _shuffled_pair_cache(pair_cache, cfg.seed, epoch)
            step_t0 = time.monotonic()
            batch_buffer: list[dict] = []
            for pair in epoch_pairs:
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
            ckpt.save(
                cp_name,
                resumable=True,
                promotable=True,
                data_consumed=(step - step_offset) * cfg.batch_size,
            )
            if getattr(cfg, "output_model_id", None):
                ckpt.promote_latest(cfg.output_model_id, cfg.base_model)
                runner.write_output_model(
                    model_id=cfg.output_model_id, checkpoint=cp_name, job_id=job_id,
                )

        write_completed(runner, step=step, total_steps=total_steps)
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
