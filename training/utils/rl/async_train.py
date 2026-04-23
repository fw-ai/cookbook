"""Gate-native async RL training loop.

One function, one extension point.  Unlike the windowed :func:`run_rl_loop`
this loop takes an explicit off-policy budget (``max_head_offpolicy_versions``)
and never conflates hotload cadence with staleness.  Inspired by AReaL's
``StalenessManager``: the deployment version is tagged on each rollout at
submit time and capacity is computed per-version, not per-window.

Capacity formula (per AReaL)::

    staleness_cap = (max_head_offpolicy_versions + current_version + 1) * gpb
                    - (total_accepted + in_flight)

where ``gpb`` is ``prompt_groups_per_step``.  When ``max_head_offpolicy_versions``
is ``0`` the loop is strict on-policy: a new rollout can only be submitted when
the prior step's rollouts have cleared.  Higher values let the sampler run
ahead at the cost of policy-version drift, which the existing TIS / decoupled
IS correction compensates for.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Iterable, Iterator

from training.utils.rl.losses import PromptGroup
from training.utils.rl.train import DynamicFilterFn, TrainStepFns

logger = logging.getLogger(__name__)

__all__ = ["run_async_rl_loop"]


async def run_async_rl_loop(
    sample_fns: Iterable[Coroutine[Any, Any, PromptGroup | None]],
    *,
    train_fns: TrainStepFns,
    prompt_groups_per_step: int,
    max_head_offpolicy_versions: int,
    weight_sync_fn: Callable[[int], None] | None = None,
    weight_sync_interval: int = 1,
    max_concurrent: int | None = None,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    global_step: int = 0,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Run the gate-native async RL loop.

    Args:
        sample_fns: Iterable of rollout coroutines.  Each must resolve to a
            :class:`PromptGroup` or ``None`` (sample failure).  The iterator is
            consumed lazily as gate capacity opens; exhausting it signals
            end-of-data.
        train_fns: Training callbacks (see :class:`TrainStepFns`).  The
            ``train_step(step, batch, extra)`` callable is invoked once per
            filled batch.
        prompt_groups_per_step: How many accepted groups make up one optimizer
            step.
        max_head_offpolicy_versions: Off-policy budget.  ``0`` is strict
            on-policy; ``N`` allows rollouts up to ``N`` versions behind the
            current policy.
        weight_sync_fn: Called after every ``weight_sync_interval`` optimizer
            steps.  Must bump the deployment version; the loop increments its
            internal ``current_version`` counter on return.
        weight_sync_interval: Fire ``weight_sync_fn`` every N steps.  ``1``
            matches AReaL's typical async setup (sync every step).
        max_concurrent: Hard cap on in-flight rollout tasks.  ``None`` lets
            the gate alone bound concurrency.
        dynamic_filter_fn: Post-sampling filter; ``False`` return drops the
            group from the trainer buffer (without charging the gate).
        global_step: Initial optimizer-step counter.  Returned value is the
            step after the final accepted batch trained.
        metrics_callback: Receives a metrics dict per training step.

    Returns:
        Final ``global_step`` after all possible batches have trained.
    """
    if prompt_groups_per_step < 1:
        raise ValueError("prompt_groups_per_step must be >= 1")
    if max_head_offpolicy_versions < 0:
        raise ValueError("max_head_offpolicy_versions must be >= 0")
    if weight_sync_interval < 1:
        raise ValueError("weight_sync_interval must be >= 1")
    if max_concurrent is not None and max_concurrent < 1:
        raise ValueError("max_concurrent, if set, must be >= 1")

    current_version: int = global_step
    total_accepted: int = 0
    sample_fail_count: int = 0
    filter_drop_count: int = 0
    in_flight: dict[asyncio.Task, int] = {}
    buffer: list[tuple[PromptGroup, int]] = []

    sample_iter: Iterator[Coroutine[Any, Any, PromptGroup | None]] = iter(sample_fns)
    iterator_exhausted = False

    def _capacity() -> int:
        # Pending = rollouts that have been submitted but not yet trained on.
        # Counts both in-flight tasks and completed groups sitting in the buffer
        # so completions don't silently open extra gate capacity before the
        # next training step arrives.
        pending = len(in_flight) + len(buffer)
        staleness_cap = (
            (max_head_offpolicy_versions + current_version + 1) * prompt_groups_per_step
            - (total_accepted + pending)
        )
        if max_concurrent is None:
            concurrency_cap = staleness_cap
        else:
            concurrency_cap = max_concurrent - len(in_flight)
        return max(0, min(staleness_cap, concurrency_cap))

    def _refill() -> None:
        nonlocal iterator_exhausted
        if iterator_exhausted:
            return
        slots = _capacity()
        for _ in range(slots):
            try:
                coro = next(sample_iter)
            except StopIteration:
                iterator_exhausted = True
                return
            task = asyncio.ensure_future(coro)
            in_flight[task] = current_version

    def _can_make_batch() -> bool:
        return len(buffer) >= prompt_groups_per_step

    def _has_outstanding_work() -> bool:
        return bool(in_flight) or _can_make_batch() or not iterator_exhausted

    _refill()

    while _has_outstanding_work():
        if not _can_make_batch():
            if not in_flight:
                if iterator_exhausted:
                    break
                _refill()
                if not in_flight:
                    # Gate permanently closed (no version bump will arrive)
                    # and iterator can't produce because of gate. Exit cleanly.
                    logger.warning(
                        "Async loop stalled: capacity=0 with no in-flight "
                        "tasks and %d rows remaining. Increase "
                        "max_head_offpolicy_versions or supply a "
                        "weight_sync_fn so versions advance.",
                        sum(1 for _ in sample_iter),
                    )
                    iterator_exhausted = True
                    break
                continue

            done, _ = await asyncio.wait(
                in_flight.keys(), return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                submit_version = in_flight.pop(task)
                exc = task.exception()
                if exc is not None:
                    logger.warning("sample_fn task failed: %s", exc)
                    sample_fail_count += 1
                    continue
                pg = task.result()
                if pg is None:
                    sample_fail_count += 1
                    continue
                if dynamic_filter_fn is not None and not dynamic_filter_fn(pg):
                    filter_drop_count += 1
                    continue
                buffer.append((pg, submit_version))
            _refill()
            continue

        batch_pairs = buffer[:prompt_groups_per_step]
        buffer = buffer[prompt_groups_per_step:]
        batch = [pg for pg, _ in batch_pairs]
        versions = [v for _, v in batch_pairs]
        offsets = [current_version - v for v in versions]

        step_start = time.time()
        extra_metrics: dict[str, Any] = {
            "async/version_offset_mean": sum(offsets) / len(offsets),
            "async/version_offset_max": max(offsets),
            "async/version_offset_min": min(offsets),
            "async/in_flight": len(in_flight),
            "async/sample_fails": sample_fail_count,
            "async/filter_drops": filter_drop_count,
            "all_raw_rewards": [r for pg in batch for r in pg.rewards],
            "valid_prompt_groups": len(batch),
            "total_sampled": len(batch) + sample_fail_count + filter_drop_count,
            "filter_drops": filter_drop_count,
            "sample_fails": sample_fail_count,
            "sample_wait_time": 0.0,
        }

        global_step, step_metrics = await asyncio.to_thread(
            train_fns.train_step, global_step, batch, extra_metrics,
        )
        total_accepted += len(batch)
        extra_metrics["async/step_wall_time"] = time.time() - step_start

        if metrics_callback is not None:
            merged = dict(step_metrics or {})
            merged.update(extra_metrics)
            merged["train/step"] = global_step
            metrics_callback(merged)

        if weight_sync_fn is not None and global_step % weight_sync_interval == 0:
            await asyncio.to_thread(weight_sync_fn, global_step)
            current_version += 1

        _refill()

    for task in in_flight:
        task.cancel()
    for task in list(in_flight):
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    in_flight.clear()

    return global_step
