#!/usr/bin/env python3
"""Train GRPO with async_rl_loop and Eval Protocol RemoteRolloutProcessor."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Iterator

from training.examples.rl.grpo_remote_rollout.rollout import make_rollout_fn
from training.recipes.async_rl_loop import Config, main
from training.utils import DeployConfig, InfraConfig, WandBConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "train.jsonl")


def _iter_rows(path: str, max_rows: int | None) -> Iterator[dict]:
    count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)
            count += 1
            if max_rows is not None and count >= max_rows:
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO with RemoteRolloutProcessor")
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3-1p5b-instruct")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--remote-rollout-base-url", default=os.environ.get("REMOTE_ROLLOUT_BASE_URL"))
    parser.add_argument("--tracing-base-url", default=os.environ.get("EP_MODEL_BASE_URL"))
    parser.add_argument("--output-model-id", default=None)
    parser.add_argument("--max-rows", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--completions-per-prompt", type=int, default=2)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompt-groups-per-step", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--remote-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--remote-poll-interval", type=float, default=1.0)
    parser.add_argument("--max-head-offpolicy-versions", type=int, default=0)
    parser.add_argument("--ppo-n-minibatches", type=int, default=1)
    parser.add_argument("--max-concurrency-rollout-sample", type=int, default=None)
    parser.add_argument("--training-shape-id", default=None)
    parser.add_argument("--deployment-shape", default=None)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--replica-count", type=int, default=None)
    parser.add_argument("--log-path", default="./grpo_remote_rollout_logs")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "grpo-remote-rollout"))
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--filter-constant-reward", action="store_true")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if not args.remote_rollout_base_url:
        raise ValueError("Set --remote-rollout-base-url or REMOTE_ROLLOUT_BASE_URL.")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. Run `python prepare_data.py` first."
        )

    rows = list(_iter_rows(args.dataset_path, args.max_rows))
    logger.info("Loaded %d rows from %s", len(rows), args.dataset_path)

    cfg = Config(
        log_path=args.log_path,
        base_model=args.base_model,
        policy_loss="grpo",
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=args.max_rows,
        lora_rank=args.lora_rank,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_head_offpolicy_versions=args.max_head_offpolicy_versions,
        ppo_n_minibatches=args.ppo_n_minibatches,
        max_concurrency_rollout_sample=args.max_concurrency_rollout_sample,
        output_model_id=args.output_model_id,
        infra=InfraConfig(training_shape_id=args.training_shape_id),
        deployment=DeployConfig(
            deployment_shape=args.deployment_shape,
            tokenizer_model=args.tokenizer_model,
            replica_count=args.replica_count,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"grpo-remote-{int(time.time()) % 100000}",
        ),
    )
    dynamic_filter_fn = (
        (lambda pg: len(set(pg.rewards)) > 1)
        if args.filter_constant_reward else None
    )
    main(
        cfg,
        rollout_fn_factory=make_rollout_fn,
        rows=rows,
        rollout_extras={
            "remote_rollout_base_url": args.remote_rollout_base_url,
            "tracing_base_url": args.tracing_base_url,
            "max_turns": args.max_turns,
            "remote_timeout_seconds": args.remote_timeout_seconds,
            "remote_poll_interval": args.remote_poll_interval,
        },
        dynamic_filter_fn=dynamic_filter_fn,
    )


if __name__ == "__main__":
    run()
