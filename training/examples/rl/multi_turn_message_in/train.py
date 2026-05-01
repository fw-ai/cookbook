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
from training.utils import DeployConfig, WandBConfig, WeightSyncConfig

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
    p.add_argument("--max-turns", type=int, default=2,
                   help="Max LLM calls per trajectory (AReaL default: 2)")
    p.add_argument("--log-path", default="./gsm8k_mt_logs")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "gsm8k-mt"))
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
        prompt_groups_per_step=args.prompt_groups_per_step,
        output_model_id=args.output_model_id,
        deployment=DeployConfig(tokenizer_model=args.tokenizer_model),
        weight_sync=WeightSyncConfig(weight_sync_interval=1),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=f"gsm8k-mt-{int(time.time()) % 100000}",
        ),
    )
    main(
        cfg,
        rollout_fn_factory=make_rollout_fn,
        rows=rows,
        rollout_extras={"max_turns": args.max_turns},
    )


if __name__ == "__main__":
    run()
