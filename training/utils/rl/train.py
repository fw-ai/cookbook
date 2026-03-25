"""RL training loop orchestration for Fireworks recipes.

Provides ``run_rl_loop`` -- a pipelined on-policy loop that samples
``prompt_groups_per_step`` prompts per optimizer step, then runs a single
``train_step`` callback (1:1 ratio).

``weight_sync_interval`` controls both hotload frequency *and* pipeline
overlap depth:

- **0**: no syncs, one big window, full overlap (deployment never changes)
- **1**: sync every step, 1-step windows, no overlap (strict on-policy)
- **N > 1**: sync every N steps, N-step windows, overlap within windows
"""

from __future__ import annotations

import time
import asyncio
import logging
import itertools
from typing import Any, Callable, Iterable, Coroutine
from dataclasses import dataclass

from tqdm import tqdm

from training.utils.rl.losses import PromptGroup

logger = logging.getLogger(__name__)

__all__ = [
    "TrainStepFns",
    "DynamicFilterFn",
    "run_rl_loop",
]

DynamicFilterFn = Callable[[PromptGroup], bool]
"""Filter callback applied after sampling, before training.

Return ``True`` to accept the group into the training buffer,
``False`` to discard it.
"""

_DONE = object()


@dataclass
class TrainStepFns:
    """Training callbacks for the 1:1 loop.

    A single ``train_step`` callback receives all prompt groups for one
    optimizer step and is responsible for ref_forward + fwd_bwd + optim_step.
    """

    train_step: Callable[[int, list[PromptGroup], dict | None], tuple[int, dict]]


async def _run_pipeline_window(
    window_coros: list[Coroutine],
    *,
    train_fns: TrainStepFns,
    prompt_groups_per_step: int,
    dynamic_filter_fn: DynamicFilterFn | None,
    global_step: int,
    metrics_callback: Callable[[dict[str, Any]], None] | None,
    max_concurrent: int = 0,
) -> int:
    """Process one policy window with pipelined sampling/training.

    All sampling coroutines fire at once (capped by ``max_concurrent``
    if > 0).  As results arrive they are filtered and accumulated; every
    ``prompt_groups_per_step`` valid groups are sent to the trainer via
    an unbounded queue so the sampler can pre-build batches while the
    trainer is busy.  Training for batch K overlaps with sampling still
    in-flight for later batches -- early arrivals get trained first.
    """
    pipe: asyncio.Queue = asyncio.Queue()
    results_q: asyncio.Queue[PromptGroup | None] = asyncio.Queue()
    sem = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
    worker_error: BaseException | None = None

    async def _worker(coro: Coroutine) -> None:
        nonlocal worker_error
        try:
            if sem is not None:
                async with sem:
                    result = await coro
            else:
                result = await coro
            results_q.put_nowait(result)
        except BaseException as exc:
            if worker_error is None:
                worker_error = exc
            results_q.put_nowait(None)

    def _make_stats(
        n_groups: int, total_sampled: int, filter_drops: int,
        sample_fails: int, all_raw_rewards: list[float], wall: float,
    ) -> dict:
        return {
            "valid_prompt_groups": n_groups,
            "total_sampled": total_sampled,
            "filter_drops": filter_drops,
            "sample_fails": sample_fails,
            "sample_wait_time": 0.0,
            "step_wall_time": wall,
            "all_raw_rewards": list(all_raw_rewards),
        }

    async def _sampler() -> None:
        for c in window_coros:
            asyncio.create_task(_worker(c))

        buffer: list[PromptGroup] = []
        filter_drops = 0
        sample_fails = 0
        total_sampled = 0
        total_completions = 0
        all_raw_rewards: list[float] = []
        step_start = time.time()

        pbar = tqdm(
            total=len(window_coros), desc="sampling", unit="group", dynamic_ncols=True,
        )
        try:
            for _ in range(len(window_coros)):
                item = await results_q.get()

                if worker_error is not None:
                    raise RuntimeError(
                        f"Sampling worker failed: {worker_error}"
                    ) from worker_error

                if item is None:
                    sample_fails += 1
                    total_sampled += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        groups=f"{len(buffer)}/{prompt_groups_per_step}",
                        failed=sample_fails, filtered=filter_drops,
                    )
                    continue

                total_sampled += 1
                total_completions += len(item.rewards)
                all_raw_rewards.extend(item.rewards)

                if dynamic_filter_fn is not None and not dynamic_filter_fn(item):
                    filter_drops += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        completions=total_completions,
                        failed=sample_fails, filtered=filter_drops,
                    )
                    continue

                buffer.append(item)
                pbar.update(1)
                pbar.set_postfix(
                    completions=total_completions,
                    failed=sample_fails, filtered=filter_drops,
                )

                if len(buffer) >= prompt_groups_per_step:
                    step_groups = buffer[:prompt_groups_per_step]
                    buffer = buffer[prompt_groups_per_step:]
                    wall = time.time() - step_start
                    logger.info(
                        "Batch ready: %d groups (%d completions) in %.1fs "
                        "(failed=%d, filtered=%d)",
                        len(step_groups), total_completions, wall,
                        sample_fails, filter_drops,
                    )
                    await pipe.put((
                        step_groups,
                        _make_stats(
                            len(step_groups), total_sampled, filter_drops,
                            sample_fails, all_raw_rewards, wall,
                        ),
                    ))
                    filter_drops = 0
                    sample_fails = 0
                    total_sampled = 0
                    total_completions = 0
                    all_raw_rewards = []
                    step_start = time.time()

            if buffer:
                wall = time.time() - step_start
                logger.info(
                    "Batch ready (partial): %d groups in %.1fs "
                    "(failed=%d, filtered=%d)",
                    len(buffer), wall, sample_fails, filter_drops,
                )
                await pipe.put((
                    buffer,
                    _make_stats(
                        len(buffer), total_sampled, filter_drops,
                        sample_fails, all_raw_rewards, wall,
                    ),
                ))
        finally:
            pbar.close()
            await pipe.put(_DONE)

    async def _trainer() -> None:
        nonlocal global_step
        while True:
            batch = await pipe.get()
            if batch is _DONE:
                break
            groups, stats = batch
            global_step, _ = await asyncio.to_thread(
                train_fns.train_step, global_step, groups, stats,
            )
            if metrics_callback is not None:
                metrics_callback({
                    "train/step": global_step,
                    "rollout/sample_fails": stats["sample_fails"],
                    "rollout/filter_drops": stats["filter_drops"],
                })

    await asyncio.gather(
        asyncio.create_task(_sampler()),
        asyncio.create_task(_trainer()),
    )
    return global_step


async def run_rl_loop(
    sample_fns: Iterable[Coroutine[Any, Any, PromptGroup | None]],
    *,
    train_fns: TrainStepFns,
    prompt_groups_per_step: int = 1,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    global_step: int = 0,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
    weight_sync_fn: Callable[[int], None] | None = None,
    weight_sync_interval: int = 0,
    max_concurrent: int = 0,
) -> int:
    """Run the pipelined RL training loop.

    Coroutines are grouped into **policy windows** of
    ``weight_sync_interval * prompt_groups_per_step`` coroutines each.
    All coroutines in a window fire concurrently (capped by
    ``max_concurrent`` if > 0).  Results stream back in completion
    order -- early arrivals get trained first while slower rollouts are
    still in-flight.  Each ``prompt_groups_per_step`` valid groups form
    one training step.

    At window boundaries the pipeline drains, ``weight_sync_fn`` fires
    (hotload) if at least one training step ran, and new sampling starts
    with the updated deployment.  Windows where all groups are filtered
    out do not trigger a sync (matching pre-pipeline behavior).

    ``weight_sync_interval=0`` puts all coroutines in a single window
    (full overlap, no syncs).  ``weight_sync_interval=1`` creates
    single-step windows (no overlap, strict on-policy).
    """
    coros = list(sample_fns)
    if not coros:
        return global_step

    if weight_sync_interval > 0:
        window_size = weight_sync_interval * prompt_groups_per_step
    else:
        window_size = len(coros)

    coro_iter = iter(coros)
    while True:
        window = list(itertools.islice(coro_iter, window_size))
        if not window:
            break

        step_before = global_step
        global_step = await _run_pipeline_window(
            window,
            train_fns=train_fns,
            prompt_groups_per_step=prompt_groups_per_step,
            dynamic_filter_fn=dynamic_filter_fn,
            global_step=global_step,
            metrics_callback=metrics_callback,
            max_concurrent=max_concurrent,
        )

        if weight_sync_fn is not None and weight_sync_interval > 0 and global_step > step_before:
            await asyncio.to_thread(weight_sync_fn, global_step)

    return global_step
