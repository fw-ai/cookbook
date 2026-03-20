"""Rollout scheduling infrastructure for RL training.

Provides sync batch collection (``collect_sync_batch``) and an async
rollout scheduler (``AsyncRolloutScheduler``) that mirrors AReaL's
``BatchTaskDispatcher`` + ``StalenessManager`` pattern.

No imports from ``train.py`` -- the dependency flows the other way.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Coroutine, Iterator

from training.utils.rl.losses import PromptGroup

logger = logging.getLogger(__name__)

__all__ = [
    "DynamicFilterFn",
    "RolloutStats",
    "collect_sync_batch",
    "AsyncRolloutScheduler",
]

DynamicFilterFn = Callable[[PromptGroup], bool]
"""Filter callback applied after sampling, before training.

Return ``True`` to accept the group into the training buffer,
``False`` to discard it.
"""


@dataclass
class RolloutStats:
    """Statistics from one batch collection round."""

    valid_groups: int = 0
    total_sampled: int = 0
    filter_drops: int = 0
    sample_fails: int = 0
    wall_time: float = 0.0
    raw_rewards: list[float] = field(default_factory=list)
    version_offsets: list[int] = field(default_factory=list)
    """Per-accepted-group staleness: ``current_version - sample_version``."""


async def collect_sync_batch(
    coros: list[Coroutine[Any, Any, PromptGroup | None]],
    filter_fn: DynamicFilterFn | None = None,
    target: int = 1,
) -> tuple[list[PromptGroup], RolloutStats]:
    """Submit coroutines concurrently, collect up to *target* accepted groups.

    This is the extracted sync collection logic formerly inline in
    ``run_rl_loop``.  Each coroutine is wrapped in a worker task that
    pushes results to a queue; the caller waits for all workers.
    """
    queue: asyncio.Queue[PromptGroup | None] = asyncio.Queue()
    worker_error: BaseException | None = None

    async def _worker(coro: Coroutine) -> None:
        nonlocal worker_error
        try:
            result = await coro
            queue.put_nowait(result)
        except BaseException as exc:
            if worker_error is None:
                worker_error = exc
            queue.put_nowait(None)

    for c in coros:
        asyncio.create_task(_worker(c))

    stats = RolloutStats()
    accepted: list[PromptGroup] = []
    t0 = time.time()

    for _ in range(len(coros)):
        item = await queue.get()

        if worker_error is not None:
            raise RuntimeError(f"Sampling worker failed: {worker_error}") from worker_error

        if item is None:
            stats.sample_fails += 1
            stats.total_sampled += 1
            continue

        stats.total_sampled += 1
        stats.raw_rewards.extend(item.rewards)

        if filter_fn is not None and not filter_fn(item):
            stats.filter_drops += 1
            continue

        accepted.append(item)

    stats.valid_groups = len(accepted)
    stats.wall_time = time.time() - t0
    return accepted, stats


class AsyncRolloutScheduler:
    """Async rollout scheduler with AReaL-style capacity gating.

    Manages in-flight asyncio tasks across ``collect_batch`` calls,
    enabling overlap between rollout and training.  Capacity is gated
    by a cumulative staleness formula and a concurrency cap — the same
    two-level gate AReaL uses (without pause/resume).

    The scheduler does NOT own the training loop — the recipe
    (``rl_loop.py``) calls ``collect_batch``, runs training, then calls
    ``bump_version``.
    """

    def __init__(
        self,
        step_target: int,
        max_head_offpolicy_versions: int,
        filter_fn: DynamicFilterFn | None = None,
        global_step: int = 0,
        total_accepted: int = 0,
        total_rejected: int = 0,
        rows_submitted: int = 0,
        max_concurrent: int | None = None,
    ):
        self._step_target = step_target
        self._max_offpolicy = max_head_offpolicy_versions
        self._filter_fn = filter_fn

        self._current_version = global_step
        self._total_accepted = total_accepted
        self._total_rejected = total_rejected
        self._rows_submitted = rows_submitted

        policy_window = (max_head_offpolicy_versions + 1) * step_target
        self._max_concurrent = min(max_concurrent, policy_window) if max_concurrent is not None else policy_window

        self._in_flight: set[asyncio.Task] = set()
        self._result_queue: asyncio.Queue[tuple[PromptGroup | None, int]] = asyncio.Queue()
        self._rows_exhausted = False

    # -- public properties -----------------------------------------------------

    @property
    def current_version(self) -> int:
        return self._current_version

    @property
    def data_exhausted(self) -> bool:
        """True when row iterator is exhausted and no in-flight tasks remain."""
        return self._rows_exhausted and len(self._in_flight) == 0

    # -- capacity --------------------------------------------------------------

    def _staleness_cap(self) -> int:
        budget = (
            (self._max_offpolicy + self._current_version + 1) * self._step_target
            - (self._total_accepted + len(self._in_flight))
        )
        return max(budget, 0)

    def _concurrency_cap(self) -> int:
        return max(self._max_concurrent - len(self._in_flight), 0)

    def _capacity(self) -> int:
        return min(self._staleness_cap(), self._concurrency_cap())

    # -- task management -------------------------------------------------------

    def _submit_one(
        self,
        sample_fn_factory: Callable[[dict], Coroutine[Any, Any, PromptGroup | None]],
        row: dict,
    ) -> None:
        version = self._current_version
        coro = sample_fn_factory(row)

        async def _worker():
            try:
                result = await coro
            except Exception as exc:
                logger.warning("Rollout task failed: %s", exc)
                result = None
            self._result_queue.put_nowait((result, version))

        task = asyncio.create_task(_worker())
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)
        self._rows_submitted += 1

    # -- collect_batch ---------------------------------------------------------

    async def collect_batch(
        self,
        sample_fn_factory: Callable[[dict], Coroutine[Any, Any, PromptGroup | None]],
        rows: Iterator[dict],
    ) -> tuple[list[PromptGroup], RolloutStats]:
        """Collect ``step_target`` accepted results, submitting new rollouts
        as capacity allows.

        In-flight tasks from previous calls are harvested first.  When the
        row iterator is exhausted, remaining in-flight tasks are drained.
        """
        accepted: list[PromptGroup] = []
        stats = RolloutStats()
        t0 = time.time()

        def _process_one(item: PromptGroup | None, version: int) -> None:
            if item is None:
                stats.sample_fails += 1
                stats.total_sampled += 1
                return
            stats.total_sampled += 1
            stats.raw_rewards.extend(item.rewards)
            if self._filter_fn is not None and not self._filter_fn(item):
                stats.filter_drops += 1
                self._total_rejected += 1
                return
            accepted.append(item)
            self._total_accepted += 1
            stats.version_offsets.append(self._current_version - version)

        def _drain_and_process() -> None:
            while not self._result_queue.empty():
                pair = self._result_queue.get_nowait()
                _process_one(*pair)
                if len(accepted) >= self._step_target:
                    break

        _drain_and_process()

        while len(accepted) < self._step_target:
            if not self._rows_exhausted:
                cap = self._capacity()
                submitted = 0
                for _ in range(cap):
                    try:
                        row = next(rows)
                    except StopIteration:
                        self._rows_exhausted = True
                        break
                    self._submit_one(sample_fn_factory, row)
                    submitted += 1
                if submitted > 0:
                    logger.debug("Submitted %d rollouts (capacity=%d)", submitted, cap)

            if not self._in_flight and self._result_queue.empty():
                break

            try:
                item, version = await asyncio.wait_for(self._result_queue.get(), timeout=0.1)
                _process_one(item, version)
                if len(accepted) < self._step_target:
                    _drain_and_process()
            except asyncio.TimeoutError:
                _drain_and_process()

        stats.valid_groups = len(accepted)
        stats.wall_time = time.time() - t0
        return accepted, stats

    # -- stream_groups ---------------------------------------------------------

    async def stream_groups(
        self,
        sample_fn_factory: Callable[[dict], Coroutine[Any, Any, PromptGroup | None]],
        rows: Iterator[dict],
    ) -> AsyncIterator[tuple[PromptGroup, int]]:
        """Yield accepted ``(PromptGroup, version)`` one at a time.

        Like ``collect_batch`` but yields each accepted group immediately
        instead of buffering.  The caller can process each group
        incrementally (e.g. ref_forward + fwd_bwd with server-side grad
        accumulation) while sampling continues in the background.

        Yields exactly ``step_target`` accepted groups then returns.
        """
        accepted = 0

        def _try_submit() -> None:
            if self._rows_exhausted:
                return
            cap = self._capacity()
            for _ in range(cap):
                try:
                    row = next(rows)
                except StopIteration:
                    self._rows_exhausted = True
                    break
                self._submit_one(sample_fn_factory, row)

        _try_submit()

        while accepted < self._step_target:
            if not self._in_flight and self._result_queue.empty():
                break

            try:
                item, version = await asyncio.wait_for(
                    self._result_queue.get(), timeout=0.1,
                )
            except asyncio.TimeoutError:
                _try_submit()
                continue

            if item is None:
                _try_submit()
                continue

            if self._filter_fn is not None and not self._filter_fn(item):
                self._total_rejected += 1
                _try_submit()
                continue

            self._total_accepted += 1
            accepted += 1
            yield item, version
            _try_submit()

    # -- version management ----------------------------------------------------

    def bump_version(self) -> None:
        """Increment current version after optim_step. Opens capacity budget."""
        self._current_version += 1
        logger.debug(
            "Version bumped to %d (accepted=%d, in_flight=%d)",
            self._current_version,
            self._total_accepted,
            len(self._in_flight),
        )

    # -- state persistence -----------------------------------------------------

    def get_state(self) -> dict:
        """Snapshot cumulative counters for checkpointing."""
        return {
            "rows_submitted": self._rows_submitted,
            "total_accepted": self._total_accepted,
            "total_rejected": self._total_rejected,
        }
