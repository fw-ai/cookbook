#!/usr/bin/env python3
"""Black-box coding-agent async RL training.

Runs the real claude-code CLI in a SWE-Gym Docker runtime behind an
Anthropic->Fireworks TITO shim and trains the policy on the captured trajectory
with ``async_rl_loop``.

SWE-Gym/ProRL-parity subset::

    export FIREWORKS_API_KEY=...
    export WANDB_API_KEY=...
    export AGENT_HEAD_HOST=127.0.0.1
    python examples/rl/coding_agent/make_swegym_data.py \
        --output /tmp/cagent_swegym_50.jsonl --max-rows 50

    python examples/rl/coding_agent/train.py \
        --dataset-path /tmp/cagent_swegym_50.jsonl

Dataset rows are generated from the public ``NovaSky-AI/SkyRL-v0-293-data``
split and carry the SWE-Gym runtime image plus SWE-bench instance metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Iterator

from training.examples.rl.coding_agent.rollout import make_rollout_fn
from training.recipes.async_rl_loop import Config, main
from training.utils import DeployConfig, TrainerConfig, WandBConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora"
DEFAULT_BASE_MODEL = "accounts/fireworks/models/qwen3p5-9b"
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3.5-27B"
MIN_COMPLETIONS_PER_PROMPT = 2
MIN_PROMPT_GROUPS_PER_STEP = 1
MIN_ROLLOUT_CONCURRENCY = 1
MIN_REPLICA_COUNT = 1
MAX_TRAIN_CONTEXT_TOKENS = 65_536


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
    p = argparse.ArgumentParser(description="Black-box coding-agent async RL")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output-model-id", default=None)
    p.add_argument("--max-rows", type=int, default=293)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--completions-per-prompt", type=int, default=16,
                   help="GRPO group size (>=2). Each is one full agent run.")
    p.add_argument("--max-completion-tokens", type=int, default=16000,
                   help="Per-turn generation cap.")
    p.add_argument("--max-seq-len", type=int, default=64000,
                   help="Per-segment training context budget. Must not exceed "
                        f"{MAX_TRAIN_CONTEXT_TOKENS}, the current trainer service limit.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--prompt-groups-per-step", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--kl-beta", type=float, default=0.001)
    p.add_argument("--max-head-offpolicy-versions", type=int, default=4,
                   help="Off-policy staleness budget; black-box rollouts are slow, so allow some.")
    p.add_argument("--max-concurrency-rollout-sample", type=int, default=64,
                   help="In-flight LLM calls cap (>= completions_per_prompt). Sandboxes are "
                        "slow, so this also bounds concurrent sandboxes loosely.")
    p.add_argument("--filter-constant-reward", action="store_true",
                   help="Drop groups whose samples all share one reward (zero GRPO advantage).")
    p.add_argument("--log-path", default="./coding_agent_logs")
    p.add_argument("--training-shape-id", default=DEFAULT_TRAINING_SHAPE)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "coding-agent-rl"))
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--replica-count", type=int, default=4)
    p.add_argument("--deployment-extra-value", action="append", default=[],
                   help="Extra deployment Helm value as key=value. Repeatable. "
                        "Do not use with pinned deployment shapes.")
    return p.parse_args()


def _parse_extra_values(items: list[str]) -> dict[str, str]:
    extra_values: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--deployment-extra-value must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"--deployment-extra-value key cannot be empty, got {item!r}")
        extra_values[key] = value
    return extra_values


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.completions_per_prompt < MIN_COMPLETIONS_PER_PROMPT:
        raise ValueError(f"--completions-per-prompt must be >= {MIN_COMPLETIONS_PER_PROMPT}")
    if args.prompt_groups_per_step < MIN_PROMPT_GROUPS_PER_STEP:
        raise ValueError(f"--prompt-groups-per-step must be >= {MIN_PROMPT_GROUPS_PER_STEP}")
    if args.max_concurrency_rollout_sample < MIN_ROLLOUT_CONCURRENCY:
        raise ValueError(f"--max-concurrency-rollout-sample must be >= {MIN_ROLLOUT_CONCURRENCY}")
    if args.replica_count < MIN_REPLICA_COUNT:
        raise ValueError(f"--replica-count must be >= {MIN_REPLICA_COUNT}")
    if args.max_completion_tokens < 1:
        raise ValueError("--max-completion-tokens must be >= 1")
    if args.max_seq_len < 1:
        raise ValueError("--max-seq-len must be >= 1")
    if args.max_seq_len > MAX_TRAIN_CONTEXT_TOKENS:
        raise ValueError(f"--max-seq-len must be <= {MAX_TRAIN_CONTEXT_TOKENS}")
    if args.lora_rank < 0:
        raise ValueError("--lora-rank must be >= 0")


def run() -> None:
    args = parse_args()
    _validate_args(args)
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}.")
    rows = list(_iter_rows(args.dataset_path, args.max_rows))
    logger.info("Loaded %d coding-agent rows from %s", len(rows), args.dataset_path)

    cfg = Config(
        log_path=args.log_path,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=args.max_rows,
        lora_rank=args.lora_rank,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_head_offpolicy_versions=args.max_head_offpolicy_versions,
        max_concurrency_rollout_sample=args.max_concurrency_rollout_sample,
        output_model_id=args.output_model_id,
        trainer=TrainerConfig(training_shape_id=args.training_shape_id),
        deployment=DeployConfig(
            tokenizer_model=args.tokenizer_model,
            replica_count=args.replica_count,
            extra_values=_parse_extra_values(args.deployment_extra_value),
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"coding-agent-{int(time.time()) % 100000}",
        ),
    )
    dynamic_filter_fn = (
        (lambda pg: len(set(pg.rewards)) > 1) if args.filter_constant_reward else None
    )

    main(
        cfg,
        rollout_fn_factory=make_rollout_fn,
        rows=rows,
        dynamic_filter_fn=dynamic_filter_fn,
    )


if __name__ == "__main__":
    run()
