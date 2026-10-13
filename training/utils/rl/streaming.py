"""Submit complete RL groups while retaining one optimizer barrier."""

from __future__ import annotations

import asyncio
import logging
import json
import time
from dataclasses import dataclass, field, fields
from collections.abc import AsyncIterator, Coroutine, Sequence
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

from training.utils.phase_tracing import phase_span
from training.utils.rl.common import inference_drift_metrics
from training.utils.rl.losses import PolicyBatch

logger = logging.getLogger(__name__)


@dataclass
class SubmittedGroup:
    batch: PolicyBatch
    future: Any
    ready_at: float
    submitted_at: float
    ready_utc: float
    submitted_utc: float
    completed_at: float = 0.0
    completed_utc: float = 0.0
    waiter: asyncio.Task | None = None
    context: dict[str, Any] = field(default_factory=dict)
    timeout: float | None = None
    api_submit_duration: float = 0.0
    abort_task: asyncio.Task | None = None
    group_ready_times: tuple[float, ...] = ()


def track_group(
    future: Any,
    batch: PolicyBatch,
    *,
    ready_at: float,
    ready_utc: float,
    submit_started_at: float,
    timeout: float | None = None,
    context: dict[str, Any] | None = None,
    abort_task: asyncio.Task | None = None,
    group_ready_times: Sequence[float] = (),
) -> SubmittedGroup:
    """Record an already-submitted native future; the recipe owns its objective."""
    return SubmittedGroup(
        batch=batch,
        future=future,
        ready_at=ready_at,
        ready_utc=ready_utc,
        submitted_at=time.monotonic(),
        submitted_utc=time.time(),
        api_submit_duration=time.monotonic() - submit_started_at,
        timeout=timeout,
        context=context or {},
        abort_task=abort_task,
        group_ready_times=tuple(group_ready_times),
    )


async def retain_submission(
    submissions: list[SubmittedGroup], submission: Coroutine[Any, Any, SubmittedGroup]
) -> SubmittedGroup:
    """Retain a worker's native future even when cancellation races submission."""
    worker = asyncio.create_task(submission)
    cancelled = False
    try:
        group = await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancelled = True
        group = await asyncio.shield(worker)
    submissions.append(group)
    group.waiter = asyncio.create_task(_wait_one(group))
    if cancelled:
        raise asyncio.CancelledError
    return group


@asynccontextmanager
async def submission_window() -> AsyncIterator[list[SubmittedGroup]]:
    """Drain retained futures on every error; a training error wins over cancellation."""
    submissions: list[SubmittedGroup] = []
    try:
        yield submissions
    except BaseException as original_error:
        if submissions:
            try:
                await drain_groups(submissions)
            except BaseException as training_error:
                if isinstance(original_error, asyncio.CancelledError) and not isinstance(
                    training_error, asyncio.CancelledError
                ):
                    raise training_error
        raise


async def _wait_one(group: SubmittedGroup) -> Any:
    status = "error"
    with phase_span(
        "fwd_bwd_result",
        category="training",
        attributes={"groups": len(group.batch.groups)},
    ):
        try:
            result = await group.future.result_async(timeout=group.timeout)
            status = "success"
            return result
        except Exception:
            if group.abort_task is not None and not group.abort_task.done() and not group.abort_task.cancelling():
                group.abort_task.cancel()
            raise
        finally:
            group.completed_at, group.completed_utc = time.monotonic(), time.time()
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    logger.debug(
                        "RL group training %s",
                        json.dumps(
                            {
                                **group.context,
                                "ready_utc": group.ready_utc,
                                "submit_utc": group.submitted_utc,
                                "complete_utc": group.completed_utc,
                                "groups": len(group.batch.groups),
                                "datums": len(group.batch.data),
                                "ready_to_submit_s": group.submitted_at - group.ready_at,
                                "api_submit_s": group.api_submit_duration,
                                "submit_to_complete_s": group.completed_at - group.submitted_at,
                                "status": status,
                            },
                            sort_keys=True,
                            default=str,
                        ),
                    )
                except Exception:
                    logger.debug("RL group diagnostic serialization failed", exc_info=True)


async def drain_groups(submissions: list[SubmittedGroup]) -> list[Any]:
    """Join every submitted request before propagating failure or cancellation."""

    async def join_all() -> list[Any]:
        return await asyncio.gather(*(group.waiter for group in submissions), return_exceptions=True)

    joined = asyncio.create_task(join_all())
    try:
        results = await asyncio.shield(joined)
    except asyncio.CancelledError:
        await asyncio.shield(joined)
        raise
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


def merge_policy_batches(batches: Sequence[PolicyBatch]) -> PolicyBatch:
    """Concatenate prepared groups without recomputing group-local objectives."""
    values: dict[str, Any] = {}
    for descriptor in fields(PolicyBatch):
        rows = [getattr(batch, descriptor.name) for batch in batches]
        if all(row is None for row in rows):
            values[descriptor.name] = None
        elif any(row is None for row in rows):
            raise ValueError(f"Prepared groups must consistently set {descriptor.name}")
        else:
            values[descriptor.name] = [value for row in rows for value in row]
    return PolicyBatch(**values)


def add_batch_diagnostics(submissions: list[SubmittedGroup], results: list[Any], config: Any, objective: Any) -> None:
    """Compute means once over all outputs, independent of request group size."""
    batch = merge_policy_batches([submission.batch for submission in submissions])
    merged = SimpleNamespace(loss_fn_outputs=[output for result in results for output in result.loss_fn_outputs])
    diagnostics = objective.metrics(batch, merged, config)
    diagnostics.update(inference_drift_metrics(batch, merged))
    for result in results:
        for name in diagnostics:
            result.metrics.pop(name, None)
    results[0].metrics.update(diagnostics)


def streaming_metrics(submissions: list[SubmittedGroup]) -> dict[str, float]:
    """Client request windows; these include queueing and transport, not only compute."""
    first = min(group.submitted_at for group in submissions)
    last = max(group.completed_at for group in submissions)
    latencies = sorted(group.completed_at - group.submitted_at for group in submissions)
    pending_time, covered_until = 0.0, first
    for group in sorted(submissions, key=lambda group: group.submitted_at):
        pending_time += max(0.0, group.completed_at - max(group.submitted_at, covered_until))
        covered_until = max(covered_until, group.completed_at)
    return {
        "perf/forward_backward_pending_time": pending_time,
        "perf/fwd_bwd_time": max(0.0, last - first),
        "perf/fwd_bwd_submit_time": sum(group.api_submit_duration for group in submissions),
        "perf/group_ready_to_submit_time_sum": sum(
            group.submitted_at - ready_at
            for group in submissions
            for ready_at in (group.group_ready_times or (group.ready_at,) * len(group.batch.groups))
        ),
        "perf/group_request_latency_mean": sum(latencies) / len(latencies),
        "perf/group_request_latency_max": max(latencies),
        "train/forward_backward_requests": len(submissions),
        "train/groups_per_request_mean": sum(len(group.batch.groups) for group in submissions) / len(submissions),
        "train/groups_per_request_max": max(len(group.batch.groups) for group in submissions),
    }
