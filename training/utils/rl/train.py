"""RL training loop orchestration for Fireworks recipes.

Provides ``run_rl_loop`` -- an on-policy loop that samples
``prompt_groups_per_step`` prompts per optimizer step, then runs a single
``train_step`` callback (1:1 ratio).
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


@dataclass
class TrainStepFns:
    """Training callbacks for the 1:1 loop.

    A single ``train_step`` callback receives all prompt groups for one
    optimizer step and is responsible for ref_forward + fwd_bwd + optim_step.
    """

    train_step: Callable[[int, list[PromptGroup], dict | None], tuple[int, dict]]
    """Execute one training step.  Signature:
    (step, prompt_groups, loop_stats) -> (new_step, metrics)."""


async def run_rl_loop(
    sample_fns: Iterable[Coroutine[Any, Any, PromptGroup | None]],
    *,
    train_fns: TrainStepFns,
    prompt_groups_per_step: int = 1,
    max_concurrent: int = 32,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    global_step: int = 0,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Run the on-policy RL training loop.

    Samples ``prompt_groups_per_step`` prompts concurrently (capped by
    ``max_concurrent``), collects valid groups, then calls
    ``train_fns.train_step`` once per optimizer step.

    Each prompt fires ``completions_per_prompt`` individual n=1 requests
    internally (via ``sample_with_tokens``), so the actual in-flight
    request count is up to ``max_concurrent * completions_per_prompt``.
    """
    coros = list(sample_fns)

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

    coro_iter = iter(coros)
    while True:
        step_coros = list(itertools.islice(coro_iter, prompt_groups_per_step))
        if not step_coros:
            break

        for c in step_coros:
            asyncio.create_task(_worker(c))

        total_wait_time = 0.0
        filter_drops = 0
        sample_fails = 0
        all_raw_rewards: list[float] = []
        total_sampled = 0
        step_start_time = time.time()
        step_prompt_groups: list[PromptGroup] = []

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

            step_prompt_groups.append(item)
            pbar.update(1)
            pbar.set_postfix(sampled=total_sampled, failed=sample_fails, filtered=filter_drops)
            if len(step_prompt_groups) % 5 == 0 or len(step_prompt_groups) == prompt_groups_per_step:
                logger.info(
                    "Sampling %d/%d groups (sampled=%d, failed=%d, filtered=%d, %.0fs elapsed)",
                    len(step_prompt_groups), prompt_groups_per_step,
                    total_sampled, sample_fails, filter_drops,
                    time.time() - step_start_time,
                )

        pbar.close()

        if not step_prompt_groups:
            logger.warning("[step %d] no valid prompt groups after filtering, skipping", global_step + 1)
            continue

        step_wall_time = time.time() - step_start_time
        logger.info(
            "Sampling complete: %d/%d groups in %.1fs (failed=%d, filtered=%d)",
            len(step_prompt_groups), prompt_groups_per_step,
            step_wall_time, sample_fails, filter_drops,
        )
        loop_stats = {
            "valid_prompt_groups": len(step_prompt_groups),
            "total_sampled": total_sampled,
            "filter_drops": filter_drops,
            "sample_fails": sample_fails,
            "sample_wait_time": total_wait_time,
            "step_wall_time": step_wall_time,
            "all_raw_rewards": list(all_raw_rewards),
        }

        global_step, _ = await asyncio.to_thread(
            train_fns.train_step, global_step, step_prompt_groups, loop_stats,
        )

        if metrics_callback is not None:
            metrics_callback({
                "train/step": global_step,
                "rollout/sample_fails": sample_fails,
                "rollout/filter_drops": filter_drops,
            })

    return global_step
