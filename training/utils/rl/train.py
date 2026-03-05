"""RL training loop orchestration for Fireworks recipes.

Provides ``run_rl_loop`` -- a streaming on-policy loop that samples
``prompt_groups_per_step`` prompts per optimizer step, fires ``fwd_bwd``
as minibatches fill, then runs ``optim_step`` + hotload before sampling
the next step.  Only the current step's prompts are in-flight at any time.
"""

from __future__ import annotations

import time
import asyncio
import logging
import itertools
from typing import Any, Callable, Iterable, Coroutine
from dataclasses import dataclass

from tqdm import tqdm

from training.utils.timer import Timer
from training.utils.rl.losses import PromptGroup

logger = logging.getLogger(__name__)

__all__ = [
    "MinibatchTrainFns",
    "DynamicFilterFn",
    "run_rl_loop",
]

# -- Types -------------------------------------------------------------------

DynamicFilterFn = Callable[[PromptGroup], bool]
"""Filter callback applied after sampling, before training.

Return ``True`` to accept the group into the training buffer,
``False`` to discard it.  This runs after sampling and before training.
"""


@dataclass
class MinibatchTrainFns:
    """Split training callbacks for streaming minibatch mode.

    Instead of one monolithic ``train_step_fn`` that receives all prompt
    groups at once, these three callbacks are invoked incrementally as
    data arrives.
    """

    ref_forward_batch: Callable[[list[PromptGroup]], None]
    """Compute reference logprobs for a batch of prompt groups (one HTTP call)."""

    fwd_bwd_one: Callable[[list[PromptGroup]], Any]
    """Run forward_backward_custom on one micro-batch.  Fired when the buffer
    reaches ``min_prompt_groups_per_fwd_bwd``."""

    finish_step: Callable[[int, list[PromptGroup], list, int, dict], tuple[int, dict]]
    """optim_step + hotload + metrics. Called after all fwd_bwd calls for a
    step complete. Signature: (step, all_groups, fwd_bwd_results, n_accum,
    loop_stats) -> (new_step, metrics).  ``loop_stats`` contains
    valid_prompt_groups, total_sampled, filter_drops, sample_fails."""


# -- Main entry point -------------------------------------------------------


async def run_rl_loop(
    sample_fns: Iterable[Coroutine[Any, Any, PromptGroup | None]],
    *,
    minibatch_fns: MinibatchTrainFns,
    prompt_groups_per_step: int = 1,
    max_concurrent: int = 32,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    global_step: int = 0,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
    min_prompt_groups_per_fwd_bwd: int | None = None,
    completions_per_prompt: int = 1,
) -> int:
    """Run the streaming RL training loop.

    Launches all sampling coroutines concurrently (capped by
    ``max_concurrent``).  Prompt groups accumulate until
    ``min_prompt_groups_per_fwd_bwd`` is reached, then ``fwd_bwd_one`` fires
    with all available groups.
    ``optim_step`` runs after ``prompt_groups_per_step`` groups are collected.
    """
    coros = list(sample_fns)

    if min_prompt_groups_per_fwd_bwd is None:
        min_prompt_groups_per_fwd_bwd = prompt_groups_per_step

    return await _stream_loop(
        coros, minibatch_fns, prompt_groups_per_step, global_step,
        min_prompt_groups_per_fwd_bwd,
        completions_per_prompt, max_concurrent,
        dynamic_filter_fn, metrics_callback,
    )


# -- Stream mode (greedy batching with max_batch_size cap) -------------------


async def _stream_loop(
    coros: list[Coroutine],
    fns: MinibatchTrainFns,
    prompt_groups_per_step: int,
    global_step: int,
    min_prompt_groups_per_fwd_bwd: int,
    completions_per_prompt: int,
    max_concurrent: int,
    dynamic_filter_fn: DynamicFilterFn | None,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """On-policy streaming loop with greedy fwd_bwd batching.

    Samples ``prompt_groups_per_step`` prompts per optimizer step (on-policy).
    Within a step, ``fwd_bwd_one`` fires as soon as the minibatch buffer
    reaches ``min_prompt_groups_per_fwd_bwd``.  After all prompts for the
    step complete, flushes remaining buffer and runs ``optim_step`` + hotload
    before launching the next step's prompts.
    """
    queue: asyncio.Queue[PromptGroup | None] = asyncio.Queue()
    sem = asyncio.Semaphore(max_concurrent)
    worker_error: BaseException | None = None

    async def _worker(coro: Coroutine) -> None:
        nonlocal worker_error
        try:
            async with sem:
                result = await coro
            queue.put_nowait(result)
        except BaseException as exc:
            if worker_error is None:
                worker_error = exc
            queue.put_nowait(None)

    total_wait_time = 0.0
    filter_drops = 0
    sample_fails = 0
    all_raw_rewards: list[float] = []
    total_sampled = 0
    step_start_time = time.time()

    minibatch_prompt_groups: list[PromptGroup] = []
    step_prompt_groups: list[PromptGroup] = []
    fwd_bwd_results: list = []
    fwd_bwd_prompt_group_counts: list[int] = []
    fwd_bwd_call_count = 0

    async def _flush_and_finish_step() -> None:
        """Flush remaining buffer, run optim_step, reset state."""
        nonlocal global_step, fwd_bwd_call_count
        nonlocal total_wait_time, filter_drops, sample_fails
        nonlocal total_sampled, step_start_time
        nonlocal minibatch_prompt_groups, step_prompt_groups, fwd_bwd_results, fwd_bwd_prompt_group_counts
        nonlocal all_raw_rewards

        idle_start = time.time()

        step_num = global_step + 1
        if minibatch_prompt_groups:
            n_datums = sum(len(pg.ref_data) for pg in minibatch_prompt_groups)
            fwd_bwd_prompt_group_counts.append(len(minibatch_prompt_groups))

            logger.info("[step %d] ref_forward_batch: %d groups, %d datums...", step_num, len(minibatch_prompt_groups), n_datums)
            t0 = time.time()
            await asyncio.to_thread(fns.ref_forward_batch, minibatch_prompt_groups)
            Timer().add("ref_forward", time.time() - t0)
            logger.info("[step %d] ref_forward_batch: done (%.1fs)", step_num, time.time() - t0)

            logger.info("[step %d] fwd_bwd %d: %d groups, %d samples...", step_num, fwd_bwd_call_count + 1, len(minibatch_prompt_groups), len(minibatch_prompt_groups) * completions_per_prompt)
            t0 = time.time()
            result = await asyncio.to_thread(fns.fwd_bwd_one, minibatch_prompt_groups)
            Timer().add("fwd_bwd", time.time() - t0)
            logger.info("[step %d] fwd_bwd %d: done (%.1fs)", step_num, fwd_bwd_call_count + 1, time.time() - t0)
            fwd_bwd_results.append(result)
            minibatch_prompt_groups = []
            fwd_bwd_call_count += 1

        logger.info("[step %d] optim_step...", step_num)
        step_wall_time = time.time() - step_start_time
        loop_stats = {
            "valid_prompt_groups": len(step_prompt_groups),
            "total_sampled": total_sampled,
            "filter_drops": filter_drops,
            "sample_fails": sample_fails,
            "sample_wait_time": total_wait_time,
            "step_wall_time": step_wall_time,
            "all_raw_rewards": list(all_raw_rewards),
            "fwd_bwd_group_counts": fwd_bwd_prompt_group_counts,
        }
        sample_idle_time = time.time() - idle_start
        loop_stats["sample_idle_time"] = sample_idle_time
        global_step, step_metrics = await asyncio.to_thread(
            fns.finish_step, global_step, step_prompt_groups,
            fwd_bwd_results, fwd_bwd_call_count, loop_stats,
        )

        minibatch_prompt_groups = []
        step_prompt_groups = []
        fwd_bwd_results = []
        fwd_bwd_prompt_group_counts = []
        fwd_bwd_call_count = 0
        total_wait_time = 0.0
        filter_drops = 0
        sample_fails = 0
        all_raw_rewards = []
        total_sampled = 0
        step_start_time = time.time()

    coro_iter = iter(coros)
    while True:
        step_coros = list(itertools.islice(coro_iter, prompt_groups_per_step))
        if not step_coros:
            break

        for c in step_coros:
            asyncio.create_task(_worker(c))

        pbar = tqdm(total=prompt_groups_per_step, desc="sampling", unit="prompt_grp", dynamic_ncols=True)

        for _ in range(len(step_coros)):
            t_wait = time.time()
            item = await queue.get()
            total_wait_time += time.time() - t_wait

            if worker_error is not None:
                pbar.close()
                raise RuntimeError(f"Sampling worker failed: {worker_error}") from worker_error

            if item is None:
                sample_fails += 1
                total_sampled += 1
                pbar.set_postfix(sampled=total_sampled, failed=sample_fails, filtered=filter_drops)
                continue

            total_sampled += 1
            all_raw_rewards.extend(item.rewards)
            if dynamic_filter_fn is not None and not dynamic_filter_fn(item):
                filter_drops += 1
                pbar.set_postfix(sampled=total_sampled, failed=sample_fails, filtered=filter_drops)
                continue

            minibatch_prompt_groups.append(item)
            step_prompt_groups.append(item)
            pbar.update(1)
            pbar.set_postfix(sampled=total_sampled, failed=sample_fails, filtered=filter_drops)

            if len(minibatch_prompt_groups) >= min_prompt_groups_per_fwd_bwd:
                batch = minibatch_prompt_groups
                minibatch_prompt_groups = []
                n_datums = sum(len(pg.ref_data) for pg in batch)
                step_num = global_step + 1

                logger.info("[step %d] ref_forward_batch: %d groups, %d datums...", step_num, len(batch), n_datums)
                t0 = time.time()
                await asyncio.to_thread(fns.ref_forward_batch, batch)
                Timer().add("ref_forward", time.time() - t0)
                logger.info("[step %d] ref_forward_batch: done (%.1fs)", step_num, time.time() - t0)

                fwd_bwd_prompt_group_counts.append(len(batch))
                logger.info("[step %d] fwd_bwd %d: %d groups, %d samples...", step_num, fwd_bwd_call_count + 1, len(batch), len(batch) * completions_per_prompt)
                t0 = time.time()
                result = await asyncio.to_thread(fns.fwd_bwd_one, batch)
                Timer().add("fwd_bwd", time.time() - t0)
                logger.info("[step %d] fwd_bwd %d: done (%.1fs)", step_num, fwd_bwd_call_count + 1, time.time() - t0)
                fwd_bwd_results.append(result)
                fwd_bwd_call_count += 1

        pbar.close()

        if step_prompt_groups:
            await _flush_and_finish_step()
        else:
            logger.warning("[step %d] no valid prompt groups after filtering, skipping", global_step + 1)
            if metrics_callback is not None and all_raw_rewards:
                avg_reward = sum(all_raw_rewards) / len(all_raw_rewards)
                metrics_callback({
                    "train/step": global_step,
                    "rollout/reward": avg_reward,
                    "rollout/reward_std": float(
                        (sum((r - avg_reward) ** 2 for r in all_raw_rewards) / len(all_raw_rewards)) ** 0.5
                    ),
                    "rollout/n_raw_samples": len(all_raw_rewards),
                    "rollout/filter_drops": filter_drops,
                    "rollout/sample_fails": sample_fails,
                    "rollout/skipped_step": 1,
                })
            step_start_time = time.time()

    return global_step
