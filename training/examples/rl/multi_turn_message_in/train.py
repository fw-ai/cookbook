#!/usr/bin/env python3
"""Multi-turn GSM8K async RL training.

Run ``prepare_data.py`` first to materialize ``train.jsonl`` (and
``test.jsonl``) from HuggingFace ``openai/gsm8k`` (config ``main``).
Then::

    python train.py \\
        --base-model accounts/fireworks/models/qwen3-1p5b-instruct \\
        --tokenizer-model Qwen/Qwen2.5-1.5B-Instruct \\
        --output-model-id accounts/<acct>/models/gsm8k-mt

The rollout (see ``rollout.py``) does up to ``max_turns`` (default 2)
LLM calls per row: if the boxed answer is wrong, a fixed user-feedback
message is appended and the model retries.  Reward is the verified
boxed-answer match for the last turn.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Iterator

from training.examples.rl.multi_turn_message_in.rollout import make_rollout_fn
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
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if max_rows is not None and n >= max_rows:
                return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-turn GSM8K async RL")
    p.add_argument("--base-model", default="accounts/fireworks/models/qwen3-1p5b-instruct")
    p.add_argument("--tokenizer-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset-path", default=DEFAULT_DATASET)
    p.add_argument("--output-model-id", required=False, default=None)
    p.add_argument("--max-rows", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--completions-per-prompt", type=int, default=4)
    p.add_argument("--max-completion-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--prompt-groups-per-step", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1.7e-5)
    p.add_argument("--kl-beta", type=float, default=0.0)
    p.add_argument("--max-head-offpolicy-versions", type=int, default=0,
                   help="Off-policy staleness budget in weight-sync (policy) versions; "
                        "one weight-sync per outer rollout batch, regardless of ppo-n-minibatches "
                        "(0 = strict on-policy).")
    p.add_argument("--ppo-n-minibatches", type=int, default=1,
                   help="Inner PPO minibatches per rollout batch (1 = legacy 1:1).")
    p.add_argument("--filter-constant-reward", action="store_true",
                   help="Drop prompt groups whose samples all share the same reward (zero GRPO advantage).")
    p.add_argument("--max-turns", type=int, default=2,
                   help="Max LLM calls per trajectory (AReaL default: 2)")
    p.add_argument("--log-path", default="./gsm8k_mt_logs")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "gsm8k-mt"))
    p.add_argument("--training-shape-id", default=None,
                   help="Full training shape resource name (accounts/.../trainingShapes/...).")
    p.add_argument("--lora-rank", type=int, default=0,
                   help="LoRA rank (0 = full-parameter training).")
    p.add_argument("--synchronous-training", action="store_true",
                   help="Force fully-synchronous mode (no rollout/train overlap). "
                        "Drains in-flight rollouts before each train_step and "
                        "marks the rollout side blocked-on-trainer; "
                        "perf/sampler_wait_for_trainer_time then reflects train+sync wall time.")
    p.add_argument("--max-concurrency-rollout-sample", type=int, default=None,
                   help="Cap in-flight LLM calls against the inference deployment "
                        "(maps to deployment max_batch_size; e.g. 64 for a 64-slot "
                        "shape with completions_per_prompt=8 = 8 rows in flight). "
                        "Must be >= completions_per_prompt.  Default unbounded.")
    p.add_argument("--wandb-run-name", default=None,
                   help="Override the WandB run name.")
    p.add_argument("--replica-count", type=int, default=None,
                   help="Pin the inference deployment to a fixed replica count "
                        "(default 1). Higher values fan out rollout sampling.")
    return p.parse_args()


def run():
    args = parse_args()
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. "
            "Run `python prepare_data.py` first."
        )

    rows = list(_iter_rows(args.dataset_path, args.max_rows))
    logger.info("Loaded %d GSM8K rows from %s", len(rows), args.dataset_path)

    cfg = Config(
        log_path=args.log_path,
        base_model=args.base_model,
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
        synchronous_training=args.synchronous_training,
        output_model_id=args.output_model_id,
        infra=InfraConfig(training_shape_id=args.training_shape_id),
        deployment=DeployConfig(
            tokenizer_model=args.tokenizer_model,
            replica_count=args.replica_count,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"gsm8k-mt-{int(time.time()) % 100000}",
        ),
    )
    # Drop prompt groups whose rewards are constant across all samples --
    # GRPO z-score advantage is 0 there, so the optimizer step is a no-op.
    dynamic_filter_fn = (
        (lambda pg: len(set(pg.rewards)) > 1)
        if args.filter_constant_reward else None
    )
    main(
        cfg,
        rollout_fn_factory=make_rollout_fn,
        rows=rows,
        rollout_extras={"max_turns": args.max_turns},
        dynamic_filter_fn=dynamic_filter_fn,
    )


if __name__ == "__main__":
    run()
