#!/usr/bin/env python3
"""Train OpenCode on Terminal-Bench with the serverless async RL loop.

Assembly: Terminal-Bench 2.0 Harbor rows split into a cycled training pool and
eight fixed evaluation tasks + the OpenCode harness
(``opencode.rollout.make_rollout_fn``) +
``recipes.experiment.async_rl_loop_serverless.main``. ``--sampling-only``
calibrates five tasks through ``run_sampling_preflight`` first.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import random
import time
from functools import partial
from pathlib import Path
from typing import Any

from training.examples.harbor_rl.harnesses.opencode.constants import DEFAULT_OPENCODE_VERSION
from training.examples.harbor_rl.harnesses.opencode.rollout import make_rollout_fn
from training.examples.harbor_rl import common
from training.examples.harbor_rl.records import select_rows, write_json
from training.examples.harbor_rl.tito.evaluate import make_fixed_evaluation
from training.examples.harbor_rl.tito.trial import load_harbor_rows
from training.recipes.experiment.async_rl_loop_serverless import (
    Config,
    main,
    run_sampling_preflight,
)
from training.utils import WandBConfig

logger = logging.getLogger(__name__)

COMPLETIONS_PER_PROMPT = 8
PROMPT_GROUPS_PER_STEP = 8
PIPELINE_CHUNKS_PER_STEP = 2
MAX_COMPLETION_TOKENS = 32768
MAX_SEQ_LEN = 196608
TASK_SEED = 20260808

# Fixed evaluation membership makes step-to-step reward directly comparable.
# These tasks span coding, systems, security, and scientific workloads in the
# pinned Terminal-Bench 2.0 release.
DEFAULT_EVAL_TASKS = (
    "break-filter-js-from-html",
    "build-pmars",
    "cancel-async-tasks",
    "git-leak-recovery",
    "largest-eigenval",
    "log-summary-date-ranges",
    "portfolio-optimization",
    "write-compressor",
)
DEFAULT_CALIBRATION_TASKS = (
    "break-filter-js-from-html",
    "build-pmars",
    "git-leak-recovery",
    "largest-eigenval",
    "write-compressor",
)


def split_rows(
    rows: list[dict[str, Any]], eval_task_names: tuple[str, ...] | list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return name-sorted training rows and a fixed, disjoint evaluation set."""
    eval_rows = select_rows(rows, eval_task_names)
    eval_names = {row["task_name"] for row in eval_rows}
    train_rows = sorted(
        (dict(row) for row in rows if row["task_name"] not in eval_names),
        key=lambda row: str(row["task_name"]),
    )
    if not train_rows:
        raise ValueError("Terminal-Bench training split is empty")
    return train_rows, eval_rows


def training_rows(
    rows: list[dict[str, Any]], *, max_rows: int, seed: int
) -> list[dict[str, Any]]:
    """Shuffle once, then cycle tasks to the requested number of prompt groups."""
    if max_rows < 1 or not rows:
        raise ValueError("training rows and max_rows must be non-empty")
    shuffled = [dict(row) for row in rows]
    random.Random(seed).shuffle(shuffled)
    return [dict(row) for row in itertools.islice(itertools.cycle(shuffled), max_rows)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_model_args(parser)
    common.add_harness_args(parser)
    common.add_wandb_args(parser, project="harbor-rl-opencode")
    parser.add_argument("--harbor-dataset", required=True)
    parser.add_argument("--harbor-trials-dir", required=True, type=Path)
    parser.add_argument("--opencode-version", default=DEFAULT_OPENCODE_VERSION)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(f"./harbor_terminal_bench_serverless_{int(time.time())}"),
    )
    parser.add_argument("--sampling-only", action="store_true")
    parser.add_argument("--calibration-task", action="append", default=[])
    parser.add_argument("--calibration-completions", type=int, default=1)
    parser.add_argument("--calibration-concurrency", type=int, default=5)
    parser.add_argument("--eval-task", action="append", default=[])
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--evaluation-concurrency", type=int, default=24)
    parser.add_argument("--max-rows", type=int, default=80)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-12)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-head-offpolicy-versions", type=int, default=2)
    parser.add_argument("--sample-timeout", type=float, default=2400.0)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace, *, run_dir: Path, run_name: str) -> Config:
    return Config(
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        tokenizer_revision=args.tokenizer_revision,
        completions_per_prompt=COMPLETIONS_PER_PROMPT,
        prompt_groups_per_step=PROMPT_GROUPS_PER_STEP,
        pipeline_chunks_per_step=PIPELINE_CHUNKS_PER_STEP,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        max_seq_len=args.max_seq_len,
        max_rows=args.max_rows,
        lora_rank=args.lora_rank,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        max_head_offpolicy_versions=args.max_head_offpolicy_versions,
        sample_timeout=args.sample_timeout,
        snapshot_prefix=run_name,
        metrics_file=str(run_dir / "metrics.jsonl"),
        wandb=WandBConfig(
            entity=args.wandb_entity, project=args.wandb_project, run_name=run_name
        ),
    )


def run() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args()
    all_rows = load_harbor_rows(args.harbor_dataset)
    train_pool, eval_rows = split_rows(
        all_rows, tuple(args.eval_task or DEFAULT_EVAL_TASKS)
    )
    rows = training_rows(train_pool, max_rows=args.max_rows, seed=TASK_SEED)
    mode = "calibration" if args.sampling_only else "train"
    run_name = args.wandb_run_name or (
        f"harbor-terminal-bench-serverless-{mode}-{int(time.time()) % 100000}"
    )
    run_dir = args.run_dir.expanduser().resolve()
    trials_dir = args.harbor_trials_dir.expanduser().resolve()
    trials_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "task-split.json",
        {
            "dataset": args.harbor_dataset,
            "seed": TASK_SEED,
            "train_pool": [row["task_name"] for row in train_pool],
            "training_rows": [row["task_name"] for row in rows],
            "eval_tasks": [row["task_name"] for row in eval_rows],
        },
    )
    config = build_config(args, run_dir=run_dir, run_name=run_name)
    factory = partial(
        make_rollout_fn,
        harbor=common.harbor_settings(
            args, run_dir=run_dir, environment="docker", trials_dir=trials_dir
        ),
        version=args.opencode_version,
    )

    if args.sampling_only:
        calibration_tasks = args.calibration_task or DEFAULT_CALIBRATION_TASKS
        result = run_sampling_preflight(
            config,
            rollout_fn_factory=factory,
            evaluation_fn=make_fixed_evaluation(
                select_rows(all_rows, calibration_tasks),
                completions_per_prompt=args.calibration_completions,
                metric_prefix="calibration",
                max_concurrency=args.calibration_concurrency,
            ),
        )
    else:
        wandb_run_id = common.pin_wandb_run_id(args.wandb_entity)
        result = main(
            config,
            rollout_fn_factory=factory,
            evaluation_fn=make_fixed_evaluation(
                eval_rows,
                completions_per_prompt=COMPLETIONS_PER_PROMPT,
                max_concurrency=args.evaluation_concurrency,
            ),
            evaluation_interval=args.evaluation_interval,
            rows=rows,
        )
        result["wandb_run_id"] = wandb_run_id
    path = write_json(
        run_dir / "run-state.json",
        {"mode": mode, "trials_dir": str(trials_dir), **result},
    )
    logger.info("Terminal-Bench %s complete; run state in %s", mode, path)


if __name__ == "__main__":
    run()
