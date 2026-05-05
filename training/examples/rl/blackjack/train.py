#!/usr/bin/env python3
"""Blackjack GRPO training — thin wrapper over the async RL recipe.

Usage:
    export FIREWORKS_API_KEY=...
    python train.py --output-model-id accounts/<acct>/models/blackjack-v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

from training.examples.rl.blackjack.rollout import DEFAULT_SYSTEM_PROMPT, make_rollout_fn
from training.recipes.async_rl_loop import Config, main
from training.utils import DeployConfig, InfraConfig, WandBConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_SEEDS_PATH = os.path.join(os.path.dirname(__file__), "seeds.jsonl")


def _load_rows(path: str, max_seeds: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rows.append({
                "seed": int(entry["seed"]),
                "player_cards": entry["player_cards"],
                "dealer_cards": entry["dealer_cards"],
            })
            if len(rows) >= max_seeds:
                break
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blackjack GRPO via async RL recipe")
    p.add_argument("--base-model", default="accounts/fireworks/models/qwen3p5-9b")
    p.add_argument("--tokenizer-model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--seeds-path", default=DEFAULT_SEEDS_PATH)
    p.add_argument("--max-seeds", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--output-model-id", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--completions-per-prompt", type=int, default=8)
    p.add_argument("--max-completion-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--prompt-groups-per-step", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-6)
    p.add_argument("--kl-beta", type=float, default=0.01)
    p.add_argument("--ppo-n-minibatches", type=int, default=1)
    p.add_argument("--max-head-offpolicy-versions", type=int, default=0)
    p.add_argument("--max-concurrency-rollout-sample", type=int, default=None)
    p.add_argument("--no-thinking", action="store_true",
                   help="Disable Qwen3 thinking mode (faster, fewer tokens needed)")
    p.add_argument("--natural", action="store_true",
                   help="Award +1.5 for natural blackjack win")
    p.add_argument("--sab", action="store_true",
                   help="Sutton-Barto rules: player natural always beats dealer non-natural")
    p.add_argument("--log-path", default="./blackjack_logs")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "blackjack-grpo"))
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--training-shape-id", default=None)
    p.add_argument("--lora-rank", type=int, default=0)
    p.add_argument("--filter-constant-reward", action="store_true",
                   help="Drop groups where all completions share the same reward (zero GRPO advantage)")
    p.add_argument("--synchronous-training", action="store_true",
                   help="Drain rollouts before each train step (no rollout/train overlap)")
    return p.parse_args()


def run() -> None:
    args = parse_args()
    rows = _load_rows(args.seeds_path, args.max_seeds)
    logger.info("Loaded %d seed rows from %s", len(rows), args.seeds_path)

    cfg = Config(
        log_path=args.log_path,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=args.max_seeds,
        lora_rank=args.lora_rank,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_head_offpolicy_versions=args.max_head_offpolicy_versions,
        ppo_n_minibatches=args.ppo_n_minibatches,
        max_concurrency_rollout_sample=args.max_concurrency_rollout_sample,
        synchronous_training=args.synchronous_training,
        output_model_id=args.output_model_id,
        infra=InfraConfig(training_shape_id=args.training_shape_id),
        deployment=DeployConfig(tokenizer_model=args.tokenizer_model),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"blackjack-{int(time.time()) % 100000}",
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
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "max_steps": args.max_steps,
            "natural": args.natural,
            "sab": args.sab,
            "no_thinking": args.no_thinking,
        },
        dynamic_filter_fn=dynamic_filter_fn,
    )


if __name__ == "__main__":
    run()
