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


async def _collect_samples(
    coros: list[Coroutine],
    prompt_groups_per_step: int,
    dynamic_filter_fn: DynamicFilterFn | None,
    step_label: int,
) -> tuple[list[PromptGroup], dict] | None:
    """Fire sampling coroutines concurrently, filter, return groups + stats.

    Returns ``None`` when all groups are filtered out (caller should skip
    the training step).
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

    total_wait_time = 0.0
    filter_drops = 0
    sample_fails = 0
    all_raw_rewards: list[float] = []
    total_sampled = 0
    total_completions = 0
    step_start = time.time()
    groups: list[PromptGroup] = []

    pbar = tqdm(total=len(coros), desc="sampling", unit="group", dynamic_ncols=True)

    for _ in range(len(coros)):
        t = time.time()
        item = await queue.get()
        total_wait_time += time.time() - t

        if worker_error is not None:
            pbar.close()
            raise RuntimeError(f"Sampling worker failed: {worker_error}") from worker_error

        if item is None:
            sample_fails += 1
            total_sampled += 1
            pbar.set_postfix(
                groups=f"{len(groups)}/{prompt_groups_per_step}",
                failed=sample_fails, filtered=filter_drops,
            )
            continue

        total_sampled += 1
        total_completions += len(item.rewards)
        all_raw_rewards.extend(item.rewards)

        if dynamic_filter_fn is not None and not dynamic_filter_fn(item):
            filter_drops += 1
            pbar.update(1)
            pbar.set_postfix(completions=total_completions, failed=sample_fails, filtered=filter_drops)
            continue

        groups.append(item)
        pbar.update(1)
        pbar.set_postfix(completions=total_completions, failed=sample_fails, filtered=filter_drops)
        if len(groups) % 5 == 0 or len(groups) == prompt_groups_per_step:
            logger.info(
                "Sampling %d/%d groups (%d completions, failed=%d, filtered=%d, %.0fs elapsed)",
                len(groups), prompt_groups_per_step,
                total_completions, sample_fails, filter_drops,
                time.time() - step_start,
            )

    pbar.close()

    if not groups:
        logger.warning("[step %d] no valid prompt groups after filtering, skipping", step_label)
        return None

    wall = time.time() - step_start
    logger.info(
        "Sampling complete: %d/%d groups (%d completions) in %.1fs (failed=%d, filtered=%d)",
        len(groups), prompt_groups_per_step,
        total_completions, wall, sample_fails, filter_drops,
    )
    stats = {
        "valid_prompt_groups": len(groups),
        "total_sampled": total_sampled,
        "filter_drops": filter_drops,
        "sample_fails": sample_fails,
        "sample_wait_time": total_wait_time,
        "step_wall_time": wall,
        "all_raw_rewards": list(all_raw_rewards),
    }
    return groups, stats


async def _run_pipeline_window(
    window_coros: list[Coroutine],
    *,
    train_fns: TrainStepFns,
    prompt_groups_per_step: int,
    dynamic_filter_fn: DynamicFilterFn | None,
    global_step: int,
    metrics_callback: Callable[[dict[str, Any]], None] | None,
) -> int:
    """Process one policy window with pipelined sampling/training.

    The sampler and trainer run as independent coroutines connected by a
    bounded queue (``maxsize=1``).  Sampling for step K+1 overlaps with
    training for step K.  When the window's coroutines are exhausted the
    sampler sends ``_DONE`` and the trainer drains.
    """
    pipe: asyncio.Queue = asyncio.Queue(maxsize=1)

    async def _sampler() -> None:
        coro_iter = iter(window_coros)
        try:
            while True:
                batch = list(itertools.islice(coro_iter, prompt_groups_per_step))
                if not batch:
                    break
                result = await _collect_samples(
                    batch, prompt_groups_per_step, dynamic_filter_fn, global_step + 1,
                )
                if result is not None:
                    await pipe.put(result)
        finally:
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
) -> int:
    """Run the pipelined RL training loop.

    Coroutines are grouped into **policy windows** of
    ``weight_sync_interval * prompt_groups_per_step`` coroutines each.
    Within a window, sampling and training overlap via a bounded queue.
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
        )

        if weight_sync_fn is not None and weight_sync_interval > 0 and global_step > step_before:
            await asyncio.to_thread(weight_sync_fn, global_step)

    return global_step
