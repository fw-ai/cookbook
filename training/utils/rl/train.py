"""RL training loop: rollout -> filter -> shuffle -> train.

Each iteration samples ``rollout_batch_size`` groups concurrently,
applies an optional filter, shuffles for mini-batch diversity, and
trains in chunks of ``prompt_groups_per_step``.  IS correction
(``ISConfig`` / ``compute_tis_weight``) handles the off-policy gap
between the sampling policy and the current training policy
automatically, following the AReaL / slime decoupled pattern.
"""

from __future__ import annotations

import time
import random
import asyncio
import logging
import itertools
from typing import Any, Callable, Iterable, Coroutine
from dataclasses import dataclass, field

from tqdm import tqdm

from training.utils.rl.losses import PromptGroup

logger = logging.getLogger(__name__)

__all__ = [
    "RolloutStats",
    "run_rl_loop",
]

FilterFn = Callable[[PromptGroup], bool]
TrainStepFn = Callable[[int, list[PromptGroup], "RolloutStats"], tuple[int, dict]]


@dataclass
class RolloutStats:
    """Stats from one rollout iteration, passed to train_step and metrics."""

    accepted: int = 0
    sampled: int = 0
    failed: int = 0
    filtered: int = 0
    completions: int = 0
    rewards: list[float] = field(default_factory=list)
    wall_time: float = 0.0
    wait_time: float = 0.0
    iteration: int = 0
    train_step_in_rollout: int = 0
    total_groups: int = 0


async def _collect_rollout(
    coros: list,
    filter_fn: FilterFn | None,
    rollout_id: int,
) -> tuple[list[PromptGroup], RolloutStats]:
    """Fire all sampling coroutines concurrently and collect accepted groups."""
    stats = RolloutStats(iteration=rollout_id)
    groups: list[PromptGroup] = []
    t0 = time.time()

    results = await asyncio.gather(*coros, return_exceptions=True)

    for result in results:
        if isinstance(result, BaseException):
            stats.failed += 1
            stats.sampled += 1
            logger.warning("Sampling failed for one prompt: %s", result)
            continue

        if result is None:
            stats.sampled += 1
            continue

        stats.sampled += 1
        stats.completions += len(result.rewards)
        stats.rewards.extend(result.rewards)

        if filter_fn is not None and not filter_fn(result):
            stats.filtered += 1
            continue

        groups.append(result)

    stats.accepted = len(groups)
    stats.wall_time = time.time() - t0
    return groups, stats


async def run_rl_loop(
    sample_fns: Iterable[Coroutine[Any, Any, PromptGroup | None]],
    *,
    train_step: TrainStepFn,
    prompt_groups_per_step: int,
    rollout_batch_size: int,
    shuffle: bool = True,
    filter_fn: FilterFn | None = None,
    global_step: int = 0,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """RL training loop: rollout -> filter -> shuffle -> train.

    Each iteration samples ``rollout_batch_size`` groups concurrently,
    applies ``filter_fn``, shuffles, and trains in chunks of
    ``prompt_groups_per_step``.  IS correction handles the off-policy
    gap automatically.

    Set ``rollout_batch_size == prompt_groups_per_step`` for on-policy
    (1:1 ratio).  Set higher for off-policy.

    Tip: for off-policy, set ``weight_sync_interval`` to
    ``rollout_batch_size // prompt_groups_per_step`` so deployment
    weights update once per rollout iteration.
    """
    if rollout_batch_size < prompt_groups_per_step:
        raise ValueError(
            f"rollout_batch_size ({rollout_batch_size}) must be >= "
            f"prompt_groups_per_step ({prompt_groups_per_step})"
        )

    logger.info(
        "RL loop: rollout_batch_size=%d, prompt_groups_per_step=%d "
        "(%.1fx train steps per rollout)",
        rollout_batch_size, prompt_groups_per_step,
        rollout_batch_size / prompt_groups_per_step,
    )

    coro_iter = iter(list(sample_fns))

    for rollout_id in itertools.count():
        batch = list(itertools.islice(coro_iter, rollout_batch_size))
        if not batch:
            break

        groups, stats = await _collect_rollout(batch, filter_fn, rollout_id)

        if not groups:
            logger.warning("[rollout %d] no groups survived filtering", rollout_id)
            continue

        logger.info(
            "Rollout %d: %d/%d groups (%d completions) in %.1fs",
            rollout_id, stats.accepted, stats.sampled,
            stats.completions, stats.wall_time,
        )

        if shuffle and len(groups) > prompt_groups_per_step:
            random.shuffle(groups)

        stats.total_groups = len(groups)

        for i in range(0, len(groups), prompt_groups_per_step):
            chunk = groups[i:i + prompt_groups_per_step]
            stats.train_step_in_rollout = i // prompt_groups_per_step

            global_step, _ = await asyncio.to_thread(
                train_step, global_step, chunk, stats,
            )

            if metrics_callback is not None:
                metrics_callback({
                    "train/step": global_step,
                    "rollout/iteration": rollout_id,
                    "rollout/failed": stats.failed,
                    "rollout/filtered": stats.filtered,
                    "rollout/train_step_in_rollout": stats.train_step_in_rollout,
                })

    return global_step
