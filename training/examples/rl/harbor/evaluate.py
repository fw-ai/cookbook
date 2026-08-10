"""Sampling-only metrics over the same Harbor rollout function used for RL."""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Sequence
from typing import Any

from training.recipes.async_rl_loop import RolloutEvaluationFn, RolloutFn
from training.utils.rl.rollout import RolloutRun, count_trainable_tokens


async def evaluate_rows(
    rollout_fn: RolloutFn,
    rows: Sequence[dict[str, Any]],
    *,
    completions_per_prompt: int,
    metric_prefix: str,
    step: int,
    max_concurrency: int | None = None,
) -> dict[str, float | int]:
    """Run fixed rows without assembling a train batch or mutating weights."""
    if completions_per_prompt < 1:
        raise ValueError("completions_per_prompt must be >= 1")
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def run_one(row_index: int, row: dict[str, Any], sample_index: int):
        async def invoke():
            result = await rollout_fn(
                row,
                cursor_index=row_index,
                row_index=row_index,
                epoch=step,
                rollout_idx=sample_index,
                sample_index=sample_index,
            )
            if result is not None and not isinstance(result, RolloutRun):
                raise TypeError(
                    "rollout_fn must return RolloutRun or None, got "
                    f"{type(result).__name__}"
                )
            if isinstance(result, RolloutRun) and not result.segments:
                raise ValueError("rollout_fn returned a RolloutRun without segments")
            return result

        if semaphore is None:
            return await invoke()
        async with semaphore:
            return await invoke()

    calls = [
        run_one(row_index, row, sample_index)
        for row_index, row in enumerate(rows)
        for sample_index in range(completions_per_prompt)
    ]
    results = await asyncio.gather(*calls, return_exceptions=True)
    runs = [result for result in results if isinstance(result, RolloutRun)]
    failures = [result for result in results if isinstance(result, BaseException)]
    no_trajectory = sum(result is None for result in results)
    rewards = [float(run.segments[0].reward) for run in runs]
    trainable_tokens = [count_trainable_tokens(run) for run in runs]

    metrics: dict[str, float | int] = {
        f"{metric_prefix}/attempted_trajectories": len(results),
        f"{metric_prefix}/completed_trajectories": len(runs),
        f"{metric_prefix}/failed_trajectories": len(failures),
        f"{metric_prefix}/no_trajectory": no_trajectory,
        f"{metric_prefix}/reward": statistics.fmean(rewards) if rewards else 0.0,
        f"{metric_prefix}/history_wipes": sum(
            int((run.metadata or {}).get("history_wipes") or 0) for run in runs
        ),
        f"{metric_prefix}/append_token_mismatches": sum(
            int((run.metadata or {}).get("append_token_mismatches") or 0)
            for run in runs
        ),
    }
    failure_types: dict[str, int] = {}
    for failure in failures:
        name = type(failure).__name__
        failure_types[name] = failure_types.get(name, 0) + 1
    for name, count in failure_types.items():
        metrics[f"{metric_prefix}/failure/{name}"] = count
    if trainable_tokens:
        metrics.update(
            {
                f"{metric_prefix}/trainable_tokens_mean": statistics.fmean(
                    trainable_tokens
                ),
                f"{metric_prefix}/trainable_tokens_max": max(trainable_tokens),
                f"{metric_prefix}/trainable_tokens_min": min(trainable_tokens),
            }
        )

    task_rewards: dict[str, list[float]] = {}
    for run in runs:
        metadata = run.metadata or {}
        task_name = str(metadata.get("task_name") or "unknown-task")
        task_rewards.setdefault(task_name, []).append(float(run.segments[0].reward))
    for task_name, values in task_rewards.items():
        metrics[f"{metric_prefix}/task/{task_name}"] = statistics.fmean(values)
    return metrics


def make_fixed_evaluation(
    rows: Sequence[dict[str, Any]],
    *,
    completions_per_prompt: int,
    metric_prefix: str = "eval",
    max_concurrency: int | None = None,
) -> RolloutEvaluationFn:
    """Build evaluation over one immutable set of rows."""
    if not rows:
        raise ValueError("evaluation rows must not be empty")
    if completions_per_prompt < 1:
        raise ValueError("completions_per_prompt must be >= 1")
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    fixed_rows = tuple(dict(row) for row in rows)

    async def evaluate(step: int, rollout_fn: RolloutFn):
        return await evaluate_rows(
            rollout_fn,
            fixed_rows,
            completions_per_prompt=completions_per_prompt,
            metric_prefix=metric_prefix,
            step=step,
            max_concurrency=max_concurrency,
        )

    return evaluate


__all__ = ["evaluate_rows", "make_fixed_evaluation"]
