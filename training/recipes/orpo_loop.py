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
    export FIREWORKS_ACCOUNT_ID=...
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
    grad_accum       Gradient accumulation steps (default: 4)
    max_seq_len      Max token length per sequence (auto from training shape)
    lora_rank        LoRA rank, 0 for full fine-tuning (default: 0)

Infrastructure defaults target Qwen3-235B on 2 nodes with CP=16, EP=8.
"""

from __future__ import annotations

import os
import time
import signal
import random
import logging
from dataclasses import field, dataclass

import tinker

from fireworks.training.sdk import TrainerJobManager
from training.utils import (
    DEFAULT_ADAM,
    InfraConfig,
    WandBConfig,
    ResumeConfig,
    ReconnectableClient,
    wandb_log,
    setup_wandb,
    setup_resume,
    wandb_finish,
    log_metrics_json,
    make_orpo_loss_fn,
    create_trainer_job,
    load_preference_dataset,
    build_renderer,
    render_preference_pair,
    resolve_renderer_name,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    base_model: str = "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"
    dataset: str = ""
    tokenizer_model: str = ""
    renderer_name: str = ""

    orpo_lambda: float = 1.0
    learning_rate: float = 1e-5
    epochs: int = 1
    grad_accum: int = 4
    max_seq_len: int | None = None
    max_pairs: int | None = None
    lora_rank: int = 0

    infra: InfraConfig = field(
        default_factory=lambda: InfraConfig()
    )
    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(
            project="dsv3-training",
        )
    )
    resume: ResumeConfig = field(default_factory=ResumeConfig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
):
    cfg = config

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    if not cfg.base_model or not cfg.base_model.startswith("accounts/"):
        raise ValueError(f"Invalid base_model: '{cfg.base_model}' (expected accounts/...)")
    if not cfg.dataset:
        raise ValueError("Config.dataset is required.")
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
    account = os.environ.get("FIREWORKS_ACCOUNT_ID", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(
            api_key=api_key, account_id=account, base_url=base_url
        )

    profile = None
    if cfg.infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)

    if profile and cfg.max_seq_len is None:
        cfg.max_seq_len = profile.max_supported_context_length
        logger.info("max_seq_len from training shape: %d", cfg.max_seq_len)

    if cfg.max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )

    endpoint = create_trainer_job(
        rlor_mgr,
        base_model=cfg.base_model,
        infra=cfg.infra,
        profile=profile,
        lora_rank=cfg.lora_rank,
        max_seq_len=cfg.max_seq_len,
        learning_rate=cfg.learning_rate,
        display_name="orpo-trainer",
    )
    client = ReconnectableClient(
        rlor_mgr, endpoint.job_id, cfg.base_model, cfg.lora_rank
    )

    job_id = endpoint.job_id
    step_offset, _ = setup_resume(client, cfg.resume)
    adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)

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

    logger.info("Tokenizing %d preference pairs...", len(raw_data))
    pair_cache: list[dict] = []
    filtered_count = 0

    for example in raw_data:
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
    total_steps = len(pair_cache) * cfg.epochs // cfg.grad_accum
    accum_count = 0
    agg = {
        "orpo_loss": 0.0,
        "sft_loss": 0.0,
        "or_loss": 0.0,
        "log_odds_ratio": 0.0,
        "accuracy": 0.0,
        "tokens": 0,
        "count": 0,
    }

    try:
        for epoch in range(cfg.epochs):
            random.shuffle(pair_cache)
            step_t0 = time.monotonic()
            for pair in pair_cache:
                chosen_tokens = pair["chosen_tokens"]
                rejected_tokens = pair["rejected_tokens"]
                loss_fn = make_orpo_loss_fn(pair["response_start"], cfg.orpo_lambda)
                result = client.forward_backward_custom(
                    [pair["chosen_datum"], pair["rejected_datum"]], loss_fn
                )

                metrics = result.metrics
                agg["orpo_loss"] += metrics["orpo_loss"]
                agg["sft_loss"] += metrics["sft_loss"]
                agg["or_loss"] += metrics["or_loss"]
                agg["log_odds_ratio"] += metrics["log_odds_ratio"]
                agg["accuracy"] += metrics["accuracy"]
                agg["tokens"] += len(chosen_tokens) + len(rejected_tokens)
                agg["count"] += 1
                accum_count += 1

                if accum_count >= cfg.grad_accum:
                    client.optim_step(adam_params)
                    step += 1
                    accum_count = 0

                    step_elapsed = time.monotonic() - step_t0
                    step_tokens = agg["tokens"]
                    tokens_per_sec = step_tokens / step_elapsed if step_elapsed > 0 else 0.0

                    n = agg["count"]
                    if n > 0:
                        avg = {k: agg[k] / n for k in agg if k not in ("count", "tokens")}
                        logger.info(
                            "Step %d/%d | ORPO: %.4f | SFT: %.4f | OR: %.4f | "
                            "LogOR: %+.4f | Acc: %.1f%% | %.1f tok/s (%.1fs)",
                            step,
                            total_steps,
                            avg["orpo_loss"],
                            avg["sft_loss"],
                            avg["or_loss"],
                            avg["log_odds_ratio"],
                            avg["accuracy"] * 100,
                            tokens_per_sec,
                            step_elapsed,
                        )
                        log_metrics_json(step, tokens_per_sec=tokens_per_sec, **avg)
                        wandb_log(
                            {
                                "train/orpo_loss": avg["orpo_loss"],
                                "train/sft_loss": avg["sft_loss"],
                                "train/or_loss": avg["or_loss"],
                                "train/log_odds_ratio": avg["log_odds_ratio"],
                                "train/accuracy": avg["accuracy"],
                                "train/tokens_per_sec": tokens_per_sec,
                                "train/step_time_sec": step_elapsed,
                                "train/step_tokens": step_tokens,
                                "train/epoch": epoch + 1,
                            },
                            step,
                        )

                    agg = {k: 0.0 for k in agg}
                    step_t0 = time.monotonic()

            if accum_count > 0:
                client.optim_step(adam_params).result()
                step += 1
                accum_count = 0

        # -- Final checkpoint ------------------------------------------------

        if step > step_offset:
            logger.info("Saving final base checkpoint (step %d)...", step)
            result = client.inner.save_weights_for_sampler_ext(
                f"final-step-{step}", checkpoint_type="base"
            )
            logger.info("Final base checkpoint saved: %s", result.path)

        logger.info(
            "Training complete: %d optimizer steps (%d new)", step, step - step_offset
        )
        return {"steps": step, "job_id": job_id}
    finally:
        wandb_finish()
        try:
            logger.info("Cleanup: deleting trainer job %s", job_id)
            rlor_mgr.delete(job_id)
        except Exception as e:
            logger.warning("Cleanup: failed to delete trainer job %s: %s", job_id, e)


if __name__ == "__main__":
    import os
    import pathlib

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    cfg = Config(
        dataset=os.environ.get("ORPO_DATASET_PATH"), 
        tokenizer_model=os.environ.get("ORPO_TOKENIZER", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    )
    main(cfg)
