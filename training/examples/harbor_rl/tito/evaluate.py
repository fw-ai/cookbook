"""Sampling-only metrics over the same Harbor-backed TITO rollout used for RL."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from training.utils.rl.heldout import RolloutEvaluationFn, rollout_heldout_rows
from training.utils.rl.metrics import publish_tito_sidecar_metrics
from training.utils.rl.rollout import count_trainable_tokens
from training.utils.rl.rollout.setup import RolloutFn


async def evaluate_rows(
    rollout_fn: RolloutFn,
    rows: Sequence[dict[str, Any]],
    *,
    completions_per_prompt: int,
    metric_prefix: str,
    step: int,
    max_concurrency: int | None = None,
) -> dict[str, float | int]:
    """Run fixed rows without assembling a train batch or mutating weights.

    Each call passes ``epoch=step`` and ``cursor_index=row_index`` so every
    evaluation gets its own Harbor rollout group and trial identity. A failed
    rollout counts as ``no_trajectory``.
    """

    async def rollout_at_step(row: dict[str, Any], *, row_index: int, **context: Any):
        return await rollout_fn(
            row, row_index=row_index, cursor_index=row_index, epoch=step, **context
        )

    results = [
        run
        for runs in await rollout_heldout_rows(
            rollout_at_step,
            rows,
            samples_per_row=completions_per_prompt,
            max_concurrency=max_concurrency,
        )
        for run in runs
    ]
    runs = [run for run in results if run is not None]
    rewards = [float(run.segments[0].reward) for run in runs]
    trainable_tokens = [count_trainable_tokens(run) for run in runs]

    metrics: dict[str, float | int] = {
        f"{metric_prefix}/attempted_trajectories": len(results),
        f"{metric_prefix}/completed_trajectories": len(runs),
        f"{metric_prefix}/no_trajectory": len(results) - len(runs),
        f"{metric_prefix}/reward": statistics.fmean(rewards) if rewards else 0.0,
    }
    tito_summaries: list[Mapping[str, Any]] = []
    tito_debug_values: set[bool] = set()
    for run in runs:
        value = (run.metadata or {}).get("tito_metrics")
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError("run metadata tito_metrics must be a mapping")
        tito_summaries.append(value)
        tito_debug_values.add(
            bool((run.metadata or {}).get("tito_debug_enabled", False))
        )
    if len(tito_debug_values) > 1:
        raise ValueError("an evaluation cannot mix TITO debug modes")
    publish_tito_sidecar_metrics(
        metrics,
        tito_summaries,
        debug_enabled=tito_debug_values == {True},
    )
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

    # Per-task reward detail lives in the run-dir trial artifacts; publishing
    # one series per task floods WandB at full-split scale (450+ series).
    task_rewards: dict[str, list[float]] = {}
    for run in runs:
        metadata = run.metadata or {}
        task_name = str(metadata.get("task_name") or "unknown-task")
        task_rewards.setdefault(task_name, []).append(float(run.segments[0].reward))
    if task_rewards:
        task_means = [statistics.fmean(values) for values in task_rewards.values()]
        metrics[f"{metric_prefix}/task_count"] = len(task_means)
        metrics[f"{metric_prefix}/task_reward_mean"] = statistics.fmean(task_means)
        metrics[f"{metric_prefix}/task_reward_min"] = min(task_means)
        metrics[f"{metric_prefix}/task_reward_max"] = max(task_means)
    return metrics


def make_fixed_evaluation(
    rows: Sequence[dict[str, Any]],
    *,
    completions_per_prompt: int,
    metric_prefix: str = "eval",
    max_concurrency: int | None = None,
) -> RolloutEvaluationFn:
    """Build the recipes' ``evaluation_fn(step, rollout_fn)`` over fixed rows."""
    if not rows:
        raise ValueError("evaluation rows must not be empty")
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
