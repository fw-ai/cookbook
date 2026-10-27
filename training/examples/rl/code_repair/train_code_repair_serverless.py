#!/usr/bin/env python3
"""Serverless GRPO on single-function bug repair, graded by hidden unit tests.

Fireworks samples completions and runs the optimizer; this process extracts
each reply's code and grades it locally in :mod:`sandbox` (reward: share of
test cases passing, see :mod:`rewards`), then the recipe turns the rewards
into group-relative advantages for the trainer. Layout follows
``rl/deepmath/train_deepmath_serverless.py``.

``--sampling-only`` scores every row of ``--dataset`` with the untrained base
model through :func:`run_sampling_preflight` and writes one line per row to
``<run-dir>/sampled_rows.jsonl`` for ``build_dataset``.

    python -m training.examples.rl.code_repair.train_code_repair_serverless \\
        --sampling-only --dataset .../data/eval.jsonl --run-dir .../runs/baseline_eval
    python -m training.examples.rl.code_repair.train_code_repair_serverless --run-dir .../runs/full
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.examples.rl.code_repair.prepare_data import DATA_DIR
from training.examples.rl.code_repair.rewards import code_repair_reward
from training.recipes.async_rl_loop import RolloutFn, RolloutSetup
from training.recipes.experiment import async_rl_loop_serverless
from training.recipes.experiment.async_rl_loop_serverless import Config, run_sampling_preflight
from training.renderer import get_text_content
from training.utils import WandBConfig, load_jsonl_dataset
from training.utils.rl.heldout import HeldoutRuns, make_heldout_evaluation
from training.utils.rl.losses import has_reward_variance
from training.utils.rl.rollout import single_turn_renderer_rollout
from training.utils.supervised import build_renderer

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def make_rollout_fn(setup: RolloutSetup, *, renderer_name: str = "") -> RolloutFn:
    """One repair completion per call, graded in the local sandbox."""
    renderer = build_renderer(setup.tokenizer, setup.tokenizer_id, renderer_name)

    async def reward(row: dict, message: Any, _parsed: bool) -> float:
        return await code_repair_reward(get_text_content(message), row["ground_truth"])

    async def rollout_fn(row: dict):
        return await single_turn_renderer_rollout(
            row,
            renderer=renderer,
            sample_with_prompt_tokens=setup.sampler.sample_with_prompt_tokens,
            message_builder=lambda r: r["messages"],
            reward_fn=reward,
            sample_kwargs=setup.sample_kwargs,
            tokenizer=setup.tokenizer,
        )

    return rollout_fn


def summarize_row(row: Mapping[str, Any], row_runs: Sequence[Any]) -> dict[str, Any]:
    """Per-row sample stats; dropped samples count as wrong."""
    samples = [run.segments[0] for run in row_runs if run is not None]
    rewards = [s.reward for s in samples]
    buggy = row["ground_truth"]["buggy_pass_rate"]
    return {
        "id": row.get("id"),
        "samples": len(row_runs),
        "returned": len(samples),
        "buggy_pass_rate": buggy,
        "rewards": [round(r, 4) for r in rewards],
        "exact": sum(r >= 1.0 for r in rewards),
        "zero": sum(r == 0.0 for r in rewards),
        "regressed": sum(r < buggy - 1e-9 for r in rewards),
        "truncated": sum(s.finish_reason == "length" for s in samples),
        "completion_tokens": [sum(s.loss_mask) for s in samples],
    }


def repair_metrics(
    rows: Sequence[Mapping[str, Any]],
    runs: HeldoutRuns,
    *,
    prefix: str = "eval",
    rows_out: Path | None = None,
) -> dict[str, float]:
    per_row = [summarize_row(row, row_runs) for row, row_runs in zip(rows, runs)]
    if rows_out is not None:
        rows_out.parent.mkdir(parents=True, exist_ok=True)
        with rows_out.open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in per_row)
    expected = sum(r["samples"] for r in per_row) or 1
    returned = sum(r["returned"] for r in per_row)
    tokens = [t for r in per_row for t in r["completion_tokens"]]
    gains = [reward - r["buggy_pass_rate"] for r in per_row for reward in r["rewards"]]
    return {
        f"{prefix}/pass_rate": sum(sum(r["rewards"]) for r in per_row) / expected,
        f"{prefix}/fully_fixed": sum(r["exact"] for r in per_row) / expected,
        f"{prefix}/gain_over_buggy": sum(gains) / len(gains) if gains else 0.0,
        f"{prefix}/regressed": sum(r["regressed"] for r in per_row) / expected,
        f"{prefix}/zero_reward": sum(r["zero"] for r in per_row) / expected,
        f"{prefix}/truncation_rate": sum(r["truncated"] for r in per_row) / expected,
        f"{prefix}/return_ratio": returned / expected,
        f"{prefix}/buggy_pass_rate": statistics.mean(r["buggy_pass_rate"] for r in per_row) if per_row else 0.0,
        f"{prefix}/completion_tokens_median": statistics.median(tokens) if tokens else 0.0,
        f"{prefix}/completion_tokens_max": max(tokens, default=0),
    }


def train(
    config: Config,
    *,
    renderer_name: str = "",
    eval_rows: int = 100,
    eval_samples_per_row: int = 4,
    evaluation_interval: int = 5,
) -> dict[str, Any]:
    """Hold out the first ``eval_rows`` rows, train on the next ``config.max_rows``."""
    rows = load_jsonl_dataset(config.dataset)
    held_out, train_rows = rows[:eval_rows], rows[eval_rows : eval_rows + config.max_rows]
    if not train_rows:
        raise ValueError(f"{config.dataset} has no rows left for training after {eval_rows} eval rows")
    evaluation_fn = None
    if held_out:
        evaluation_fn = make_heldout_evaluation(
            held_out,
            samples_per_row=eval_samples_per_row,
            metrics_fn=repair_metrics,
            max_concurrency=config.max_concurrency_rollout_sample,
        )
    return async_rl_loop_serverless.main(
        config,
        rollout_fn_factory=functools.partial(make_rollout_fn, renderer_name=renderer_name),
        dynamic_filter_fn=has_reward_variance,
        evaluation_fn=evaluation_fn,
        evaluation_interval=evaluation_interval,
        rows=train_rows,
    )


def sample_only(config: Config, *, renderer_name: str, run_dir: Path, max_rows: int | None) -> dict[str, Any]:
    """Score dataset rows with the zero-update snapshot (the base model)."""
    rows = load_jsonl_dataset(config.dataset)[:max_rows]
    evaluation_fn = make_heldout_evaluation(
        rows,
        samples_per_row=config.completions_per_prompt,
        metrics_fn=functools.partial(repair_metrics, prefix="baseline", rows_out=run_dir / "sampled_rows.jsonl"),
        max_concurrency=config.max_concurrency_rollout_sample,
    )
    return run_sampling_preflight(
        config,
        rollout_fn_factory=functools.partial(make_rollout_fn, renderer_name=renderer_name),
        evaluation_fn=evaluation_fn,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serverless GRPO on single-function bug repair",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3p8-27b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--renderer-name", default="", help="Empty resolves it from the tokenizer.")
    parser.add_argument("--dataset", default=str(DATA_DIR / "dataset.jsonl"))
    parser.add_argument("--sampling-only", action="store_true", help="Score --dataset with the base model; no training.")
    parser.add_argument("--sample-rows", type=int, default=None, help="With --sampling-only, score only the first N rows.")
    parser.add_argument("--eval-rows", type=int, default=100, help="Leading rows held out for eval (0 disables).")
    parser.add_argument("--eval-samples-per-row", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=500, help="Training prompts per epoch.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--prompt-groups-per-step", type=int, default=16)
    parser.add_argument("--completions-per-prompt", type=int, default=8)
    parser.add_argument("--max-concurrency-rollout-sample", type=int, default=64)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--sample-timeout", type=float, default=1800.0)
    parser.add_argument("--snapshot-prefix", default="code-repair")
    parser.add_argument("--dcp-save-interval", type=int, default=0)
    parser.add_argument("--init-from-checkpoint", default=None, help="Training-state path from result.json.")
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--resolved-rows-offset", type=int, default=0)
    parser.add_argument("--run-dir", default=str(RUNS_DIR / f"run_{int(time.time())}"))
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="serverless-rl-code-repair")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        dataset=args.dataset,
        learning_rate=args.learning_rate,
        completions_per_prompt=args.completions_per_prompt,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_completion_tokens=args.max_completion_tokens,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        epochs=args.epochs,
        # Keep file order so the row cursor stays stable on resume.
        shuffle=False,
        max_rows=args.max_rows,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        weight_decay=0.0,
        max_concurrency_rollout_sample=args.max_concurrency_rollout_sample,
        min_group_size=2,
        sample_timeout=args.sample_timeout,
        snapshot_prefix=args.snapshot_prefix,
        metrics_file=str(Path(args.run_dir) / "metrics.jsonl"),
        init_from_checkpoint=args.init_from_checkpoint,
        step_offset=args.step_offset,
        resolved_rows_offset=args.resolved_rows_offset,
        dcp_save_interval=args.dcp_save_interval,
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
        ),
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = config_from_args(args)
    if args.sampling_only:
        result = sample_only(config, renderer_name=args.renderer_name, run_dir=run_dir, max_rows=args.sample_rows)
        (run_dir / "baseline.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        result = train(
            config,
            renderer_name=args.renderer_name,
            eval_rows=args.eval_rows,
            eval_samples_per_row=args.eval_samples_per_row,
            evaluation_interval=args.evaluation_interval,
        )
        (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
