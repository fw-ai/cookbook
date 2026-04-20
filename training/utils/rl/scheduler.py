"""Async rollout scheduler with two-level capacity gating.

This is an opt-in utility. Recipes that need overlapping sampling and
training can wire it up; recipes that just want sync GRPO shouldn't
import it.

The scheduler manages in-flight sampling tasks against two budgets:

1. **Staleness cap (policy window):** how far ahead in (rows submitted -
   rows trained on) we let sampling run. Rolls forward as the trainer
   completes optimizer steps via :meth:`bump_version`.

2. **Concurrency cap (resource window):** max in-flight HTTP requests
   against the deployment. Independent of the policy window.

Use :meth:`stream_groups` for true streaming (yield each accepted group
one-at-a-time so the trainer can start ``ref_fwd_bwd`` while sampling
continues), or :meth:`collect_batch` for "fill a step's worth and return
the whole batch" semantics.

Inspired by AReaL's ``BatchTaskDispatcher`` + ``StalenessManager``
(https://github.com/inclusionAI/AReaL).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator

from training.utils.rl.losses import PromptGroup

logger = logging.getLogger(__name__)

__all__ = [
    "RolloutStats",
    "AsyncRolloutScheduler",
]


@dataclass
class RolloutStats:
    """Per-batch sampling/filtering counters used by metrics callbacks."""

    valid_groups: int = 0
    total_sampled: int = 0
    sample_fails: int = 0
    filter_drops: int = 0
    raw_rewards: list[float] = field(default_factory=list)
    version_offsets: list[int] = field(default_factory=list)
    wall_time: float = 0.0


SampleFn = Callable[[dict], Awaitable[PromptGroup | None]]
FilterFn = Callable[[PromptGroup], bool]


class AsyncRolloutScheduler:
    """Async rollout scheduler with two-level capacity gating.

    Args:
        step_target: Target number of accepted groups per training step.
        max_head_offpolicy_versions: Maximum staleness (in optimizer steps)
            an accepted rollout can lag behind the current weight version.
            ``0`` = strict on-policy.
        filter_fn: Optional predicate run on each completed group; groups
            returning ``False`` are dropped (and not retried).
        global_step: Current weight version (matches optimizer step count).
        total_accepted: Resume counter — accepted groups so far.
        total_rejected: Resume counter — rejected groups so far.
        rows_submitted: Resume counter — rows submitted so far.
        max_concurrent: Max in-flight HTTP requests. ``None`` = unlimited
            (still capped by the policy window).
    """

    def __init__(
        self,
        *,
        step_target: int,
        max_head_offpolicy_versions: int,
        filter_fn: FilterFn | None = None,
        global_step: int = 0,
        total_accepted: int = 0,
        total_rejected: int = 0,
        rows_submitted: int = 0,
        max_concurrent: int | None = None,
    ):
        if step_target < 1:
            raise ValueError("step_target must be >= 1")
        if max_head_offpolicy_versions < 0:
            raise ValueError("max_head_offpolicy_versions must be >= 0")

        self._step_target = step_target
        self._max_offpolicy = max_head_offpolicy_versions
        self._filter_fn = filter_fn

        self._current_version = global_step
        self._total_accepted = total_accepted
        self._total_rejected = total_rejected
        self._rows_submitted = rows_submitted

        # Policy window = (offpolicy + 1) * step_target rows in flight at most.
        policy_window = (max_head_offpolicy_versions + 1) * step_target
        self._max_concurrent = (
            min(max_concurrent, policy_window)
            if max_concurrent is not None
            else policy_window
        )

        self._in_flight: set[asyncio.Task] = set()
        self._result_queue: asyncio.Queue[tuple[PromptGroup | None, int]] = asyncio.Queue()
        self._rows_exhausted = False

    # -- Properties ----------------------------------------------------------

    @property
    def current_version(self) -> int:
        return self._current_version

    @property
    def step_target(self) -> int:
        return self._step_target

    @property
    def data_exhausted(self) -> bool:
        """True when the row iterator is empty AND no rollouts are in-flight."""
        return self._rows_exhausted and len(self._in_flight) == 0

    # -- Capacity calculation -----------------------------------------------

    def _staleness_cap(self) -> int:
        """How many more rows we can submit without exceeding the policy window."""
        budget = (
            (self._max_offpolicy + self._current_version + 1) * self._step_target
            - (self._total_accepted + len(self._in_flight))
        )
        return max(budget, 0)

    def _concurrency_cap(self) -> int:
        return max(self._max_concurrent - len(self._in_flight), 0)

    def _capacity(self) -> int:
        return min(self._staleness_cap(), self._concurrency_cap())

    # -- Submission & draining ----------------------------------------------

    def _submit_one(self, sample_fn: SampleFn, row: dict) -> None:
        """Spawn a single rollout task tagged with the current version."""
        version = self._current_version
        coro = sample_fn(row)

        async def _worker() -> None:
            try:
                result = await coro
            except Exception as exc:
                logger.warning(
                    "Rollout task failed (%s): %s",
                    type(exc).__name__, exc or repr(exc),
                )
                result = None
            self._result_queue.put_nowait((result, version))

        task = asyncio.create_task(_worker())
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)
        self._rows_submitted += 1

    def _refill(self, sample_fn: SampleFn, rows: Iterator[dict]) -> None:
        """Submit as many rows as current capacity allows."""
        if self._rows_exhausted:
            return
        for _ in range(self._capacity()):
            try:
                row = next(rows)
            except StopIteration:
                self._rows_exhausted = True
                break
            self._submit_one(sample_fn, row)

    # -- Public API ---------------------------------------------------------

    async def stream_groups(
        self,
        sample_fn: SampleFn,
        rows: Iterator[dict],
    ) -> AsyncIterator[tuple[PromptGroup, int]]:
        """Yield ``(group, version)`` one at a time, up to ``step_target``.

        The streaming entry point: each accepted group is yielded as soon
        as it arrives so the caller can fire ``ref_fwd_bwd`` immediately
        while sampling for later groups continues in the background.

        Stops when ``step_target`` accepted groups have been yielded, or
        when no more rows can produce results.
        """
        accepted = 0
        self._refill(sample_fn, rows)

        while accepted < self._step_target:
            if not self._in_flight and self._result_queue.empty():
                break
            try:
                item, version = await asyncio.wait_for(
                    self._result_queue.get(), timeout=0.1,
                )
            except asyncio.TimeoutError:
                self._refill(sample_fn, rows)
                continue

            if item is None:
                self._refill(sample_fn, rows)
                continue
            if self._filter_fn is not None and not self._filter_fn(item):
                self._total_rejected += 1
                self._refill(sample_fn, rows)
                continue

            self._total_accepted += 1
            accepted += 1
            yield item, version
            self._refill(sample_fn, rows)

    async def collect_batch(
        self,
        sample_fn: SampleFn,
        rows: Iterator[dict],
    ) -> tuple[list[PromptGroup], RolloutStats]:
        """Block until ``step_target`` groups are accepted; return the batch.

        The non-streaming entry point: useful when the trainer wants the
        full batch up front (sync GRPO) rather than per-group fwd_bwd.
        """
        accepted: list[PromptGroup] = []
        stats = RolloutStats()
        t0 = time.time()

        def _process(item: PromptGroup | None, version: int) -> None:
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

        def _drain_ready() -> None:
            while not self._result_queue.empty() and len(accepted) < self._step_target:
                _process(*self._result_queue.get_nowait())

        _drain_ready()

        while len(accepted) < self._step_target:
            self._refill(sample_fn, rows)
            if not self._in_flight and self._result_queue.empty():
                break
            try:
                item, version = await asyncio.wait_for(
                    self._result_queue.get(), timeout=0.1,
                )
                _process(item, version)
                _drain_ready()
            except asyncio.TimeoutError:
                _drain_ready()

        stats.valid_groups = len(accepted)
        stats.wall_time = time.time() - t0
        return accepted, stats

    def bump_version(self) -> None:
        """Advance the current weight version after an optimizer step.

        This opens new staleness capacity, allowing more rollouts to be
        submitted under the policy window.
        """
        self._current_version += 1

    def get_state(self) -> dict[str, Any]:
        """Return resumable counters for checkpointing."""
        return {
            "rows_submitted": self._rows_submitted,
            "total_accepted": self._total_accepted,
            "total_rejected": self._total_rejected,
        }
