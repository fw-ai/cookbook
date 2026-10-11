"""Sampling-only evaluation of a rollout function on a fixed held-out row set.

The RL recipes accept ``evaluation_fn(step, rollout_fn)`` and call it at the
start, every ``evaluation_interval`` optimizer steps, and at the end, through
:func:`run_evaluation`. The ``rollout_fn`` they pass is the training rollout
(one object, so stateful rollouts keep one concurrency budget) wrapped to
receive ``evaluation=True``; evaluation samples the current policy without
touching the trainer. :func:`make_heldout_evaluation` builds that callable from fixed
rows and a task-specific metrics function::

    evaluation_fn = make_heldout_evaluation(
        eval_rows, samples_per_row=4, metrics_fn=my_metrics, max_concurrency=32
    )
    async_rl_loop.main(cfg, rollout_fn_factory=..., evaluation_fn=evaluation_fn)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from training.utils.rl.rollout.setup import RolloutFn, call_rollout_fn, rollout_context_names
from training.utils.rl.rollout.types import RolloutRun
from training.utils.timer import wall_timer

logger = logging.getLogger(__name__)

__all__ = [
    "HeldoutEvaluationFn",
    "HeldoutMetricsFn",
    "HeldoutRuns",
    "RolloutEvaluationFn",
    "evaluation_rollout_fn",
    "make_heldout_evaluation",
    "rollout_heldout_rows",
    "run_evaluation",
]

HeldoutRuns = list[list[RolloutRun | None]]
"""Per row, ``samples_per_row`` results; ``None`` is a dropped or failed rollout."""

HeldoutMetricsFn = Callable[[Sequence[Mapping[str, Any]], HeldoutRuns], dict[str, float]]
"""``(rows, runs_by_row) -> metrics``; called once per evaluation."""

HeldoutEvaluationFn = Callable[[int, Callable[..., Awaitable[Any]]], Awaitable[dict[str, float]]]
"""Matches the recipes' ``evaluation_fn(step, rollout_fn)`` hook."""

RolloutEvaluationFn = HeldoutEvaluationFn


def evaluation_rollout_fn(rollout_fn: RolloutFn) -> RolloutFn:
    """Wrap ``rollout_fn`` to receive ``evaluation=True`` plus the context it accepts.

    Evaluation inherits ``max_tokens`` and ``max_seq_len`` from the training
    ``sample_kwargs``; there is intentionally no second length config.
    """
    names = rollout_context_names(rollout_fn)

    async def evaluate_one(row: dict, **context: Any):
        return await call_rollout_fn(rollout_fn, row, {**context, "evaluation": True}, names)

    return evaluate_one


async def run_evaluation(
    step: int, evaluation_fn: RolloutEvaluationFn, rollout_fn: RolloutFn
) -> dict[str, float]:
    """Run one evaluation; returns its metrics plus ``eval/wall_time`` and ``eval/failed``.

    An evaluation that raises is logged and reported as ``eval/failed = 1``
    instead of stopping training.
    """
    with wall_timer("evaluation", category="evaluation", attributes={"step": step}) as span:
        try:
            metrics = await evaluation_fn(step, evaluation_rollout_fn(rollout_fn))
        except Exception:
            logger.exception("evaluation failed at policy step %d", step)
            metrics, failed = None, 1.0
        else:
            failed = 0.0
    return {"eval/wall_time": span.elapsed, **(metrics or {}), "eval/failed": failed}


async def rollout_heldout_rows(
    rollout_fn: Callable[..., Awaitable[RolloutRun | None]],
    rows: Sequence[Mapping[str, Any]],
    *,
    samples_per_row: int,
    max_concurrency: int | None = None,
) -> HeldoutRuns:
    """Run every row ``samples_per_row`` times and group the results by row.

    ``rollout_fn`` is the recipe-provided evaluation rollout, which accepts the
    loop's context keywords (``row_index``, ``sample_index``, ...). A rollout
    that raises is logged and reported as ``None`` so one failed trajectory
    does not void the whole evaluation.
    """
    if samples_per_row < 1:
        raise ValueError("samples_per_row must be >= 1")
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def run_one(row_index: int, row: Mapping[str, Any], sample_index: int):
        call = rollout_fn(
            dict(row),
            row_index=row_index,
            sample_index=sample_index,
            rollout_idx=sample_index,
        )
        if semaphore is None:
            return await call
        async with semaphore:
            return await call

    results = await asyncio.gather(
        *(
            run_one(row_index, row, sample_index)
            for row_index, row in enumerate(rows)
            for sample_index in range(samples_per_row)
        ),
        return_exceptions=True,
    )
    runs: list[RolloutRun | None] = []
    for result in results:
        if isinstance(result, BaseException):
            if not isinstance(result, Exception):
                raise result
            logger.warning(
                "held-out rollout failed: %s: %s", type(result).__name__, result
            )
            result = None
        runs.append(result)
    return [
        runs[index * samples_per_row : (index + 1) * samples_per_row]
        for index in range(len(rows))
    ]


def make_heldout_evaluation(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples_per_row: int,
    metrics_fn: HeldoutMetricsFn,
    max_concurrency: int | None = None,
) -> HeldoutEvaluationFn:
    """Build a recipe ``evaluation_fn`` over one immutable held-out row set."""
    if not rows:
        raise ValueError("held-out rows must not be empty")
    if samples_per_row < 1:
        raise ValueError("samples_per_row must be >= 1")
    fixed_rows = tuple(dict(row) for row in rows)

    async def evaluate(step: int, rollout_fn: Callable[..., Awaitable[Any]]):
        del step
        runs = await rollout_heldout_rows(
            rollout_fn,
            fixed_rows,
            samples_per_row=samples_per_row,
            max_concurrency=max_concurrency,
        )
        return metrics_fn(fixed_rows, runs)

    return evaluate
