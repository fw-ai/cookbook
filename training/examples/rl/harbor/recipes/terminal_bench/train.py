#!/usr/bin/env python3
"""Train OpenCode on Terminal-Bench with the serverless async-RL recipe."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

from training.examples.rl.harbor.tito.evaluate import (
    evaluate_rows,
    make_fixed_evaluation,
)
from training.examples.rl.harbor.opencode.constants import DEFAULT_OPENCODE_VERSION
from training.examples.rl.harbor.opencode.rollout import make_rollout_fn
from training.examples.rl.harbor.tito.trial import (
    DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
    load_harbor_rows,
)
from training.recipes.experiment.async_rl_loop_serverless import (
    Config,
    main,
    run_sampling_preflight,
)
from training.utils import WandBConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

COMPLETIONS_PER_PROMPT = 8
PROMPT_GROUPS_PER_STEP = 8
PIPELINE_CHUNKS_PER_STEP = 2
MAX_COMPLETION_TOKENS = 32768
MAX_SEQ_LEN = 196608
DEFAULT_MAX_ROWS = 80
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


def _rows_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_name = {str(row.get("task_name") or ""): row for row in rows}
    if "" in by_name:
        raise ValueError("Terminal-Bench row is missing task_name")
    if len(by_name) != len(rows):
        raise ValueError("Terminal-Bench task names must be unique")
    return by_name


def select_rows(
    rows: list[dict[str, Any]],
    task_names: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    """Select explicitly named tasks in caller-provided order."""
    names = list(task_names)
    if len(names) != len(set(names)):
        raise ValueError("Terminal-Bench task selection contains duplicates")
    by_name = _rows_by_name(rows)
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Terminal-Bench dataset is missing tasks: {missing}")
    return [dict(by_name[name]) for name in names]


def split_rows(
    rows: list[dict[str, Any]],
    eval_task_names: tuple[str, ...] | list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return sorted training rows and a fixed, disjoint evaluation set."""
    eval_rows = select_rows(rows, eval_task_names)
    eval_names = {str(row["task_name"]) for row in eval_rows}
    train_rows = [
        dict(row)
        for row in sorted(rows, key=lambda row: str(row.get("task_name") or ""))
        if str(row.get("task_name")) not in eval_names
    ]
    if not train_rows:
        raise ValueError("Terminal-Bench training split is empty")
    return train_rows, eval_rows


def training_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Shuffle once, then cycle tasks to the requested number of prompt groups."""
    if max_rows < 1:
        raise ValueError("max_rows must be >= 1")
    if not rows:
        raise ValueError("Terminal-Bench training rows must not be empty")
    shuffled = [dict(row) for row in rows]
    random.Random(seed).shuffle(shuffled)
    return [dict(row) for row in itertools.islice(itertools.cycle(shuffled), max_rows)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument(
        "--renderer-name",
        required=True,
        help="Production-certified sidecar renderer matching the model/tokenizer",
    )
    parser.add_argument(
        "--tito-prompt-mode",
        choices=("full_history", "incremental"),
        default="full_history",
        help=(
            "Prompt construction used by the TITO sidecar; incremental is an "
            "experimental model-specific opt-in"
        ),
    )
    parser.add_argument("--harbor-dataset", required=True)
    parser.add_argument("--harbor-trial-config", default=None)
    parser.add_argument("--harbor-trials-dir", required=True)
    parser.add_argument("--opencode-version", default=DEFAULT_OPENCODE_VERSION)
    parser.add_argument("--sampling-only", action="store_true")
    parser.add_argument(
        "--router-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay inference MoE routes during training when the pool supports it",
    )
    parser.add_argument("--calibration-task", action="append", default=[])
    parser.add_argument("--calibration-completions", type=int, default=1)
    parser.add_argument("--eval-task", action="append", default=[])
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-12)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--max-head-offpolicy-versions", type=int, default=2)
    parser.add_argument("--evaluation-concurrency", type=int, default=24)
    parser.add_argument("--calibration-concurrency", type=int, default=5)
    parser.add_argument("--sample-timeout", type=float, default=2400.0)
    parser.add_argument(
        "--harness-tool-timeout-seconds",
        type=int,
        default=DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--terminal-failure-reward",
        type=float,
        default=None,
        help=(
            "Optional reward for terminal agent failures without a verifier "
            "reward; the default retries and discards them"
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=f"./harbor_terminal_bench_serverless_{int(time.time())}",
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "harbor-rl-opencode"),
    )
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if args.calibration_completions < 1:
        raise ValueError("--calibration-completions must be >= 1")
    if args.evaluation_interval < 1:
        raise ValueError("--evaluation-interval must be >= 1")
    all_rows = load_harbor_rows(args.harbor_dataset, n_tasks=None)
    eval_tasks = tuple(args.eval_task or DEFAULT_EVAL_TASKS)
    train_pool, eval_rows = split_rows(all_rows, eval_tasks)
    rows = training_rows(train_pool, max_rows=args.max_rows, seed=TASK_SEED)

    run_name = args.wandb_run_name or (
        "harbor-terminal-bench-serverless-"
        f"{'calibration' if args.sampling_only else 'train'}-"
        f"{int(time.time()) % 100000}"
    )
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = Path(args.harbor_trials_dir).expanduser().resolve()
    trials_dir.mkdir(parents=True, exist_ok=True)
    wandb_run_id: str | None = None
    if not args.sampling_only and args.wandb_entity:
        wandb_run_id = uuid.uuid4().hex[:8]
        os.environ["WANDB_RUN_ID"] = wandb_run_id
        os.environ["WANDB_RESUME"] = "never"

    config = Config(
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
        router_replay=args.router_replay,
        sample_timeout=args.sample_timeout,
        snapshot_prefix=run_name,
        metrics_file=str(run_dir / "metrics.jsonl"),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=run_name,
        ),
    )
    extras: dict[str, object] = {
        "renderer_name": args.renderer_name,
        "tito_prompt_mode": args.tito_prompt_mode,
        "rollout_retries": 3,
        "terminal_failure_reward": args.terminal_failure_reward,
        "opencode_version": args.opencode_version,
        "harness_tool_timeout_seconds": args.harness_tool_timeout_seconds,
        "harbor_trial_config": args.harbor_trial_config,
        "harbor_trials_dir": str(trials_dir),
    }

    split_path = run_dir / "task-split.json"
    split_path.write_text(
        json.dumps(
            {
                "dataset": args.harbor_dataset,
                "seed": TASK_SEED,
                "train_pool": [row["task_name"] for row in train_pool],
                "training_rows": [row["task_name"] for row in rows],
                "eval_tasks": [row["task_name"] for row in eval_rows],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.sampling_only:
        calibration_tasks = tuple(args.calibration_task or DEFAULT_CALIBRATION_TASKS)
        calibration_rows = select_rows(all_rows, calibration_tasks)

        async def evaluate_calibration(step, rollout_fn):
            return await evaluate_rows(
                rollout_fn,
                calibration_rows,
                completions_per_prompt=args.calibration_completions,
                metric_prefix="calibration",
                step=step,
                max_concurrency=args.calibration_concurrency,
            )

        metrics = run_sampling_preflight(
            config,
            rollout_fn_factory=make_rollout_fn,
            evaluation_fn=evaluate_calibration,
            rollout_extras=extras,
        )
        result = {
            "mode": "sampling_only",
            "metrics": metrics,
            "task_split": str(split_path),
            "trials_dir": str(trials_dir),
        }
        result_path = run_dir / "run-state.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True))
        return

    evaluation_fn = make_fixed_evaluation(
        eval_rows,
        completions_per_prompt=COMPLETIONS_PER_PROMPT,
        max_concurrency=args.evaluation_concurrency,
    )
    result = main(
        config,
        rollout_fn_factory=make_rollout_fn,
        evaluation_fn=evaluation_fn,
        evaluation_interval=args.evaluation_interval,
        rows=rows,
        rollout_extras=extras,
    )
    result.update(
        task_split=str(split_path),
        trials_dir=str(trials_dir),
    )
    if wandb_run_id is not None:
        result["wandb_run_id"] = wandb_run_id
        result["wandb_url"] = (
            f"https://wandb.ai/{args.wandb_entity}/{args.wandb_project}/runs/"
            f"{wandb_run_id}"
        )
    result_path = run_dir / "run-state.json"
    result["run_state"] = str(result_path)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Serverless Terminal-Bench training complete: %s", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run()
