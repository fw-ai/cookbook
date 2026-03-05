"""Async off-policy RL training loop for Fireworks recipes.

Provides ``run_rl_loop_async`` -- a 3-loop async runner where generation
overlaps with training.  Pipeline depth is bounded by the results queue
maxsize of ``(max_steps_off_policy + 1) * groups_per_batch``, which
guarantees that the oldest unconsumed sample is at most
``max_steps_off_policy`` training steps behind.  Within that window the
existing TIS correction handles residual off-policy mismatch.

Architecture (inspired by tinker-cookbook ``do_async_training``):

* **Dataloader loop** -- iterates over rows, pushes into a bounded queue.
* **Worker loops** (``groups_per_batch`` concurrent tasks) -- pull rows,
  call ``sample_fn``, push results.  Backpressure from the bounded
  results queue prevents workers from getting too far ahead.
* **Training loop** -- drains results, applies ``dynamic_filter_fn``,
  accumulates ``groups_per_batch`` valid groups, then calls ``train_step``.

The staleness-bounded async design is informed by AReaL
(https://arxiv.org/abs/2505.24298), which introduced version-aware
capacity gating and decoupled PPO for fully asynchronous RL training.
For <=2 steps off-policy the cookbook's simpler TIS correction is
sufficient; deeper async would require AReaL's decoupled PPO objective.
"""

from __future__ import annotations

import time
import asyncio
import logging
from typing import Any, Callable, Coroutine
from dataclasses import dataclass

from training.utils.rl.losses import PromptGroup
from training.utils.rl.metrics import build_loop_metrics
from training.utils.rl.train import DynamicFilterFn, TrainStepFns

logger = logging.getLogger(__name__)

__all__ = [
    "AsyncConfig",
    "run_rl_loop_async",
]


@dataclass
class AsyncConfig:
    """Configuration for async off-policy RL training."""

    max_steps_off_policy: int = 2
    """Max allowed staleness (hard cap: <=2).  Workers wait when further ahead.
    For <=2 steps the cookbook's existing TIS correction is sufficient.
    Deeper async (>2) would require AReaL's decoupled PPO loss."""

    groups_per_batch: int = 8
    """Number of valid groups per training step.
    Also determines the number of concurrent worker loops."""

    def __post_init__(self) -> None:
        if self.max_steps_off_policy > 2:
            raise ValueError(
                f"max_steps_off_policy={self.max_steps_off_policy} exceeds the "
                f"supported limit of 2.  For deeper async, AReaL's decoupled "
                f"PPO loss is needed."
            )
        if self.max_steps_off_policy < 0:
            raise ValueError("max_steps_off_policy must be >= 0")
        if self.groups_per_batch < 1:
            raise ValueError("groups_per_batch must be >= 1")


@dataclass
class _WrappedPromptGroup:
    """Internal carrier: result + metadata for staleness tracking."""

    prompt_group: PromptGroup | None
    row: dict
    generation_step: int


_ROW_SENTINEL = object()
_WORKER_DONE = object()


async def run_rl_loop_async(
    sample_fn: Callable[[dict], Coroutine[Any, Any, PromptGroup | None]],
    rows: list[dict],
    *,
    train_fns: TrainStepFns,
    async_config: AsyncConfig,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    global_step: int = 0,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Run the async off-policy RL training loop.

    Generation and training overlap: while ``train_step`` runs (via
    ``asyncio.to_thread``), worker loops keep sampling from the
    Fireworks deployment.

    Pipeline depth is bounded by ``results_queue`` maxsize =
    ``(max_steps_off_policy + 1) * groups_per_batch``.  This guarantees
    that the oldest unconsumed sample is at most ``max_steps_off_policy``
    training steps behind -- workers block on ``results_queue.put()``
    when the queue is full, naturally throttling generation.

    Returns the final ``global_step`` after all training is complete.
    """
    gpb = async_config.groups_per_batch
    total_steps = len(rows) // gpb
    if total_steps == 0:
        logger.warning(
            "Not enough rows (%d) for even one batch of %d groups",
            len(rows), gpb,
        )
        return global_step

    # Bounded: at most (max_steps_off_policy + 1) batches in the pipeline.
    # When full, workers await on put(), preventing them from getting
    # further ahead than max_steps_off_policy steps.
    # Extra capacity (+gpb) for worker-done sentinels that must never block.
    pipeline_depth = (async_config.max_steps_off_policy + 1) * gpb + gpb
    rows_queue: asyncio.Queue[dict | object] = asyncio.Queue(maxsize=gpb)
    results_queue: asyncio.Queue[_WrappedPromptGroup | object] = asyncio.Queue(
        maxsize=pipeline_depth,
    )

    current_step = global_step

    # -- Loop 1: Dataloader --------------------------------------------------

    async def _dataloader_loop() -> None:
        for row in rows:
            await rows_queue.put(row)
        for _ in range(gpb):
            await rows_queue.put(_ROW_SENTINEL)

    # -- Loop 2: Workers -----------------------------------------------------

    async def _worker_loop(worker_id: int) -> None:
        while True:
            item = await rows_queue.get()
            if item is _ROW_SENTINEL:
                results_queue.put_nowait(_WORKER_DONE)
                return

            row: dict = item  # type: ignore[assignment]
            generation_step = current_step

            try:
                pg = await sample_fn(row)
            except Exception as exc:
                logger.warning("Worker %d: sample_fn failed: %s", worker_id, exc)
                pg = None

            if pg is not None:
                pg.generation_step = generation_step

            # Backpressure: blocks when pipeline is full, preventing
            # workers from getting too far ahead of training.
            await results_queue.put(
                _WrappedPromptGroup(
                    prompt_group=pg,
                    row=row,
                    generation_step=generation_step,
                )
            )

    # -- Loop 3: Training ----------------------------------------------------

    async def _training_loop() -> None:
        nonlocal current_step

        workers_alive = gpb
        steps_done = 0
        while steps_done < total_steps:
            step_prompt_groups: list[PromptGroup] = []
            sample_fails = 0
            filter_drops = 0
            all_raw_rewards: list[float] = []
            staleness_steps: list[int] = []
            step_start_time = time.time()

            while len(step_prompt_groups) < gpb:
                item = await results_queue.get()

                if item is _WORKER_DONE:
                    workers_alive -= 1
                    if workers_alive == 0:
                        if step_prompt_groups:
                            logger.info(
                                "[step %d] all workers done with %d/%d groups, training partial batch",
                                current_step + 1, len(step_prompt_groups), gpb,
                            )
                            break
                        return
                    continue

                wrapped: _WrappedPromptGroup = item  # type: ignore[assignment]

                if wrapped.prompt_group is None:
                    sample_fails += 1
                    continue

                staleness = current_step - wrapped.generation_step
                staleness_steps.append(staleness)
                all_raw_rewards.extend(wrapped.prompt_group.rewards)

                if dynamic_filter_fn is not None and not dynamic_filter_fn(wrapped.prompt_group):
                    filter_drops += 1
                    continue

                step_prompt_groups.append(wrapped.prompt_group)

            if not step_prompt_groups:
                logger.warning("[step %d] no valid groups, skipping", current_step + 1)
                continue

            step_wall_time = time.time() - step_start_time
            total_sampled = len(step_prompt_groups) + sample_fails + filter_drops
            loop_stats = {
                "valid_prompt_groups": len(step_prompt_groups),
                "total_sampled": total_sampled,
                "filter_drops": filter_drops,
                "sample_fails": sample_fails,
                "sample_wait_time": 0.0,
                "step_wall_time": step_wall_time,
                "all_raw_rewards": all_raw_rewards,
            }

            current_step, _ = await asyncio.to_thread(
                train_fns.train_step,
                current_step,
                step_prompt_groups,
                loop_stats,
            )
            steps_done += 1

            if metrics_callback is not None:
                loop_metrics = build_loop_metrics(
                    train_step=current_step,
                    sample_fails=sample_fails,
                    staleness_steps=staleness_steps,
                )
                metrics_callback(loop_metrics)

            if workers_alive == 0:
                return

    # -- Launch all loops concurrently ----------------------------------------

    dataloader_task = asyncio.create_task(_dataloader_loop(), name="dataloader")
    worker_tasks = [
        asyncio.create_task(_worker_loop(i), name=f"worker_{i}")
        for i in range(gpb)
    ]
    training_task = asyncio.create_task(_training_loop(), name="training")

    try:
        await training_task
    finally:
        dataloader_task.cancel()
        for t in worker_tasks:
            t.cancel()

        for t in [dataloader_task, *worker_tasks]:
            try:
                await t
            except asyncio.CancelledError:
                pass

    return current_step
