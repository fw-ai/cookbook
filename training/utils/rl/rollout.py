"""Sync rollout collection for the RL training loop.

Provides ``collect_sync_batch`` used by the sync path in ``run_rl_loop``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from training.utils.rl.losses import PromptGroup

logger = logging.getLogger(__name__)

__all__ = [
    "DynamicFilterFn",
    "RolloutStats",
    "collect_sync_batch",
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
