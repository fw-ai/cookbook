"""Batch collection for the synchronous RL recipe."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, TypeVar

from training.utils.dataloader import CursorItem
from training.utils.rl.losses import PromptGroup

RowT = TypeVar("RowT")

SamplePromptFn = Callable[..., Awaitable[PromptGroup | None]]
PromptGroupFilter = Callable[[PromptGroup], bool]


async def collect_prompt_groups(
    rows: Iterator[CursorItem[RowT]],
    *,
    target_size: int,
    sample_prompt: SamplePromptFn,
    should_accept: PromptGroupFilter | None,
) -> tuple[list[PromptGroup], list[int], dict[str, Any]]:
    """Collect one training batch, refilling rows rejected by the filter.

    At most ``target_size`` prompt groups are sampled at once. Sampling
    exceptions propagate; a rollout function returns ``None`` only for a
    recoverable per-row failure it intentionally chose to drop.
    """
    if target_size < 1:
        raise ValueError("target_size must be >= 1")

    started = time.monotonic()
    groups: list[PromptGroup] = []
    consumed_indices: list[int] = []
    raw_rewards: list[float] = []
    filter_drops = 0
    sample_fails = 0

    while len(groups) < target_size:
        window: list[CursorItem[RowT]] = []
        for _ in range(target_size - len(groups)):
            try:
                window.append(next(rows))
            except StopIteration:
                break
        if not window:
            break

        consumed_indices.extend(item.index for item in window)
        results = await asyncio.gather(
            *(
                sample_prompt(item.value, cursor_index=item.index)
                for item in window
            )
        )
        for result in results:
            if result is None:
                sample_fails += 1
                continue
            raw_rewards.extend(result.rewards)
            if should_accept is not None and not should_accept(result):
                filter_drops += 1
                continue
            groups.append(result)

    return groups, consumed_indices, {
        "valid_prompt_groups": len(groups),
        "total_sampled": len(consumed_indices),
        "filter_drops": filter_drops,
        "sample_fails": sample_fails,
        "trainer_wait_for_sampler_time": 0.0,
        "rollout_batch_wall_time": time.monotonic() - started,
        "all_raw_rewards": raw_rewards,
    }
