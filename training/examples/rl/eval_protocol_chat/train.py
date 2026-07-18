#!/usr/bin/env python3
"""Train with async_rl_loop and Eval Protocol chat rollouts."""

from __future__ import annotations

import argparse
import logging
import os
import time

from training.examples.rl.eval_protocol_chat.rollout import make_rollout_fn
from training.recipes.async_rl_loop import Config, main
from training.utils import DeployConfig, TrainerConfig, WandBConfig, load_jsonl_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "train.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async RL with Eval Protocol chat rollouts")
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3-1p5b-instruct")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--renderer-name", default="")
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
    parser.add_argument("--remote-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--remote-poll-interval", type=float, default=1.0)
    parser.add_argument("--training-shape-id", default=None)
    parser.add_argument("--deployment-shape", default=None)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--replica-count", type=int, default=None)
    parser.add_argument("--log-path", default="./eval_protocol_chat_logs")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "eval-protocol-chat"))
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if not args.remote_rollout_base_url:
        raise ValueError("Set --remote-rollout-base-url or REMOTE_ROLLOUT_BASE_URL.")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}.")

    rows = load_jsonl_dataset(args.dataset_path, args.max_rows)

    cfg = Config(
        log_path=args.log_path,
        base_model=args.base_model,
        kl_beta=0.0,
        learning_rate=args.learning_rate,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=args.max_rows,
        lora_rank=args.lora_rank,
        prompt_groups_per_step=args.prompt_groups_per_step,
        output_model_id=args.output_model_id,
        trainer=TrainerConfig(training_shape_id=args.training_shape_id),
        deployment=DeployConfig(
            deployment_shape=args.deployment_shape,
            tokenizer_model=args.tokenizer_model,
            replica_count=args.replica_count,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"eval-protocol-chat-{int(time.time()) % 100000}",
        ),
    )

    main(
        cfg,
        rollout_fn_factory=make_rollout_fn,
        rows=rows,
        rollout_extras={
            "remote_rollout_base_url": args.remote_rollout_base_url,
            "tracing_base_url": args.tracing_base_url,
            "renderer_name": args.renderer_name,
            "remote_timeout_seconds": args.remote_timeout_seconds,
            "remote_poll_interval": args.remote_poll_interval,
        },
    )


if __name__ == "__main__":
    run()
