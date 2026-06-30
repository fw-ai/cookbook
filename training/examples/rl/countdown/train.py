#!/usr/bin/env python3
"""Run async GRPO on the Countdown arithmetic task."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from training.examples.rl.countdown.rollout import make_rollout_fn
from training.recipes import async_rl_loop
from training.utils import DeployConfig, RunnerConfig, TrainerConfig, WandBConfig
from training.utils.rl import TISConfig

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).with_name("data") / "countdown.jsonl"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _load_rows(path: str, *, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"Countdown row must be a JSON object: {text[:80]}")
            rows.append(row)
            if len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"No Countdown rows loaded from {path}")
    return rows


def _bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Countdown async GRPO training")
    parser.add_argument("--variant", choices=("text", "vision"), default="vision")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument(
        "--base-model",
        default="accounts/fireworks/models/qwen3-vl-8b-instruct",
    )
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--training-shape",
        default="accounts/fireworks/trainingShapes/qwen3-vl-8b-256k-h200-lora",
    )
    parser.add_argument(
        "--deployment-shape",
        default="accounts/fireworks/deploymentShapes/rft-qwen3-vl-8b-instruct",
    )
    parser.add_argument("--deployment-id", default="")
    parser.add_argument("--policy-job-id", default="")
    parser.add_argument("--output-model-id", default="")
    parser.add_argument("--lora-rank", type=_nonnegative_int, default=16)
    parser.add_argument("--max-rows", type=_positive_int, default=16)
    parser.add_argument("--epochs", type=_positive_int, default=1)
    parser.add_argument("--completions-per-prompt", type=_positive_int, default=4)
    parser.add_argument("--prompt-groups-per-step", type=_positive_int, default=2)
    parser.add_argument("--max-completion-tokens", type=_positive_int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-head-offpolicy-versions", type=_nonnegative_int, default=0)
    parser.add_argument("--pipeline-chunks-per-step", type=_positive_int, default=1)
    parser.add_argument("--max-concurrency-rollout-sample", type=_positive_int, default=8)
    parser.add_argument("--deployment-replica-count", type=_positive_int, default=None)
    parser.add_argument("--weight-sync-timeout", type=_positive_int, default=900)
    parser.add_argument("--sample-timeout", type=_positive_int, default=900)
    parser.add_argument("--dcp-save-interval", type=_nonnegative_int, default=25)
    parser.add_argument("--save-final-checkpoint", type=_bool_flag, default=True)
    parser.add_argument("--cleanup", type=_bool_flag, default=True)
    parser.add_argument("--require-positive-reward", type=_bool_flag, default=False)
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "grpo-countdown-public"),
    )
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--log-path", default="./countdown_logs")
    return parser.parse_args()


def _read_metric_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _assert_positive_reward(metrics_file: str) -> None:
    records = _read_metric_records(metrics_file)
    rewards = [
        float(value)
        for record in records
        if isinstance((value := record.get("rollout/reward")), (int, float))
    ]
    accuracies = [
        float(value)
        for record in records
        if isinstance((value := record.get("rollout/accuracy")), (int, float))
    ]
    if not rewards:
        raise RuntimeError(
            "Countdown run wrote no rollout reward metrics; inspect rollouts and filters."
        )
    if max(rewards) <= 0.0 and (not accuracies or max(accuracies) <= 0.0):
        raise RuntimeError(
            "Countdown run produced no positive reward or accuracy. "
            "This is not a valid end-to-end proof."
        )


def run() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    rows = _load_rows(args.dataset_path, max_rows=args.max_rows)
    logger.info(
        "Countdown %s run: rows=%d model=%s training_shape=%s",
        args.variant,
        len(rows),
        args.base_model,
        args.training_shape or "auto",
    )

    config = async_rl_loop.Config(
        log_path=args.log_path,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        epochs=args.epochs,
        shuffle=False,
        max_rows=len(rows),
        lora_rank=args.lora_rank,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_head_offpolicy_versions=args.max_head_offpolicy_versions,
        max_concurrency_rollout_sample=args.max_concurrency_rollout_sample,
        min_group_size=2 if args.completions_per_prompt > 1 else 1,
        pipeline_chunks_per_step=args.pipeline_chunks_per_step,
        tis=TISConfig(cap=1.0, level="token"),
        trainer=TrainerConfig(
            job_id=args.policy_job_id or None,
            training_shape_id=args.training_shape or None,
        ),
        deployment=DeployConfig(
            deployment_id=args.deployment_id or None,
            deployment_shape=args.deployment_shape or None,
            tokenizer_model=args.tokenizer_model,
            replica_count=args.deployment_replica_count,
            sample_timeout=args.sample_timeout,
        ),
        weight_sync_before_training=True,
        weight_sync_timeout=args.weight_sync_timeout,
        dcp_save_interval=args.dcp_save_interval,
        cleanup_on_exit=args.cleanup,
        save_final_checkpoint=args.save_final_checkpoint,
        output_model_id=args.output_model_id or None,
        wandb=WandBConfig(
            entity=args.wandb_entity or None,
            project=args.wandb_project or None,
            run_name=args.wandb_run_name or f"countdown-{args.variant}-{int(time.time()) % 100000}",
        ),
        runner=RunnerConfig(),
    )

    async_rl_loop.main(
        config,
        rollout_fn_factory=make_rollout_fn,
        dynamic_filter_fn=lambda group: len(set(group.rewards)) > 1,
        rows=rows,
        rollout_extras={"variant": args.variant},
    )

    if args.require_positive_reward:
        _assert_positive_reward(os.environ.get("COOKBOOK_METRICS_FILE", ""))


if __name__ == "__main__":
    run()
