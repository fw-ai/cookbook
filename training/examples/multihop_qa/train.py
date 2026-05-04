#!/usr/bin/env python3
"""Multi-hop QA async RL training with optional IGPO turn-level scoring.

Wires :func:`training.recipes.async_rl_loop.main` (gate-native rollout/
train overlap, PPO inner minibatches, weight-sync hotload) to the
multi-hop QA rollout (search + submit_answer tool loop).  When
``--ig-weight > 0`` the rollout fires per-turn Information-Gain scoring
in parallel with sampling and folds the IG signal into the trajectory's
scalar reward as ``outcome + ig_weight * sum(ig_per_turn)``.

Usage:
    python prepare_data.py                           # writes dataset.jsonl
    python train.py --output-model-id <id>           # uses defaults
    python train.py --ig-weight 0 --output-model-id <id>   # GRPO baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Iterator

from training.examples.multihop_qa.rollout import make_rollout_fn
from training.recipes.async_rl_loop import Config, main
from training.utils import DeployConfig, InfraConfig, WandBConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "dataset.jsonl")


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
    p = argparse.ArgumentParser(
        description="Multi-hop QA async RL training (with optional IGPO)",
    )
    p.add_argument("--base-model", default="accounts/fireworks/models/qwen3-8b")
    p.add_argument("--tokenizer-model", default="Qwen/Qwen3-8B")
    p.add_argument("--dataset-path", default=DEFAULT_DATASET)
    p.add_argument("--output-model-id", default=None)
    p.add_argument("--max-rows", type=int, default=500)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--completions-per-prompt", type=int, default=8)
    p.add_argument("--max-completion-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--prompt-groups-per-step", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.001)
    p.add_argument(
        "--max-head-offpolicy-versions",
        type=int,
        default=1,
        help="Off-policy staleness budget in weight-sync versions; "
             "0 = strict on-policy.",
    )
    p.add_argument("--ppo-n-minibatches", type=int, default=1)
    p.add_argument(
        "--max-concurrency-rollout-sample",
        type=int,
        default=None,
        help="Cap in-flight rollout invocations against the deployment's "
             "max_batch_size; default unbounded.",
    )
    p.add_argument(
        "--filter-constant-reward",
        action="store_true",
        help="Drop prompt groups whose IG-augmented rewards are identical "
             "across all completions (zero GRPO advantage).",
    )
    p.add_argument(
        "--synchronous-training",
        action="store_true",
        help="Drain in-flight rollouts before each train_step (no overlap).",
    )

    # IGPO knobs
    p.add_argument(
        "--ig-weight",
        type=float,
        default=1.0,
        help="Folds turn-level IG into the scalar reward "
             "(outcome + ig_weight * sum(ig_per_turn)). 0 = pure GRPO.",
    )
    p.add_argument("--skip-ig-last-turn", action="store_true", default=True)
    p.add_argument(
        "--no-skip-ig-last-turn", dest="skip_ig_last_turn", action="store_false",
    )
    p.add_argument("--scoring-workers", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=8,
                   help="Max search/submit turns per trajectory.")
    p.add_argument("--search-top-k", type=int, default=2)

    # Infra
    p.add_argument("--training-shape-id", default=None)
    p.add_argument("--lora-rank", type=int, default=0)
    p.add_argument(
        "--replica-count",
        type=int,
        default=None,
        help="Pin the inference deployment replica count (default 1).",
    )

    p.add_argument("--log-path", default="./multihop_qa_async_logs")
    p.add_argument("--wandb-entity",
                   default=os.environ.get("WANDB_ENTITY", ""))
    p.add_argument("--wandb-project",
                   default=os.environ.get("WANDB_PROJECT", "multihop-qa-async"))
    p.add_argument("--wandb-run-name", default=None)
    return p.parse_args()


def run():
    args = parse_args()
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. "
            "Run `python prepare_data.py` first."
        )

    rows = list(_iter_rows(args.dataset_path, args.max_rows))
    logger.info("Loaded %d multi-hop QA rows from %s", len(rows), args.dataset_path)

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
            sample_timeout=1200,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=(
                args.wandb_run_name
                or f"multihop-qa-{int(time.time()) % 100000}"
            ),
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
            "ig_weight": args.ig_weight,
            "skip_ig_last_turn": args.skip_ig_last_turn,
            "scoring_workers": args.scoring_workers,
            "max_steps": args.max_turns,
            "search_top_k": args.search_top_k,
        },
        dynamic_filter_fn=dynamic_filter_fn,
    )


if __name__ == "__main__":
    run()
