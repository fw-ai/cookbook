"""RL training loop orchestration for Fireworks recipes.

Provides ``run_rl_loop`` -- a policy-windowed loop that samples up to
``prompt_groups_per_step`` prompts per optimizer step, then runs a single
``train_step`` callback (1:1 ratio). Setting
``prompt_groups_per_policy == prompt_groups_per_step`` yields on-policy
behavior; larger policy windows allow stale-rollout training with overlap.
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


async def run_rl_loop(
    sample_fns: Iterable[Coroutine[Any, Any, PromptGroup | None]],
    *,
    train_fns: TrainStepFns,
    prompt_groups_per_step: int = 1,
    prompt_groups_per_policy: int | None = None,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    global_step: int = 0,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
    policy_boundary_fn: Callable[[int], None] | None = None,
) -> int:
    """Run the policy-windowed RL training loop.

    All ``prompt_groups_per_step`` sampling coroutines fire concurrently.
    Request-level throttling is handled by the ``request_semaphore``
    passed into ``sample_with_tokens`` by the caller -- this loop does
    not limit concurrency itself.

    When ``prompt_groups_per_policy`` is greater than ``prompt_groups_per_step``,
    the next step's sampling is fired before the current step trains, but only
    within the same policy window. The caller can use ``policy_boundary_fn`` to
    hotload a new policy between windows.
    """
    coros = list(sample_fns)
    policy_target = prompt_groups_per_step if prompt_groups_per_policy is None else prompt_groups_per_policy
    if policy_target <= 0:
        raise ValueError("prompt_groups_per_policy must be positive")

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

    def _launch_batch(start_idx: int, batch_size: int) -> int:
        end_idx = min(len(coros), start_idx + batch_size)
        for c in coros[start_idx:end_idx]:
            asyncio.create_task(_worker(c))
        return end_idx - start_idx

    def _collect_step_stats(n_expected: int) -> tuple[list[PromptGroup], dict]:
        total_wait_time = 0.0
        filter_drops = 0
        sample_fails = 0
        all_raw_rewards: list[float] = []
        total_sampled = 0
        total_completions = 0
        step_start_time = time.time()
        step_prompt_groups: list[PromptGroup] = []

        pbar = tqdm(total=n_expected, desc="sampling", unit="group", dynamic_ncols=True)

        async def _collect() -> tuple[list[PromptGroup], dict]:
            nonlocal total_wait_time, filter_drops, sample_fails
            nonlocal all_raw_rewards, total_sampled, total_completions
            for _ in range(n_expected):
                t_wait = time.time()
                item = await queue.get()
                total_wait_time += time.time() - t_wait

                if worker_error is not None:
                    pbar.close()
                    raise RuntimeError(f"Sampling worker failed: {worker_error}") from worker_error

                if item is None:
                    sample_fails += 1
                    total_sampled += 1
                    pbar.set_postfix(
                        groups=f"{len(step_prompt_groups)}/{prompt_groups_per_step}",
                        failed=sample_fails, filtered=filter_drops,
                    )
                    continue

                total_sampled += 1
                n_completions = len(item.rewards)
                total_completions += n_completions
                all_raw_rewards.extend(item.rewards)
                if dynamic_filter_fn is not None and not dynamic_filter_fn(item):
                    filter_drops += 1
                    pbar.update(1)
                    pbar.set_postfix(completions=total_completions, failed=sample_fails, filtered=filter_drops)
                    continue

                step_prompt_groups.append(item)
                pbar.update(1)
                pbar.set_postfix(completions=total_completions, failed=sample_fails, filtered=filter_drops)
                if len(step_prompt_groups) % 5 == 0 or len(step_prompt_groups) == prompt_groups_per_step:
                    logger.info(
                        "Sampling %d/%d groups (%d completions, failed=%d, filtered=%d, %.0fs elapsed)",
                        len(step_prompt_groups), prompt_groups_per_step,
                        total_completions, sample_fails, filter_drops,
                        time.time() - step_start_time,
                    )
            pbar.close()

            if not step_prompt_groups:
                logger.warning("[step %d] no valid prompt groups after filtering, skipping", global_step + 1)
                return [], {}

            step_wall_time = time.time() - step_start_time
            logger.info(
                "Sampling complete: %d/%d groups (%d completions) in %.1fs (failed=%d, filtered=%d)",
                len(step_prompt_groups), prompt_groups_per_step,
                total_completions, step_wall_time, sample_fails, filter_drops,
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
            return step_prompt_groups, loop_stats

        return _collect()

    next_idx = 0
    while True:
        policy_sampled = 0
        current_batch_size = _launch_batch(next_idx, min(prompt_groups_per_step, policy_target))
        if current_batch_size == 0:
            break
        next_idx += current_batch_size
        policy_sampled += current_batch_size

        while current_batch_size > 0:
            step_prompt_groups, loop_stats = await _collect_step_stats(current_batch_size)

            next_batch_size = 0
            remaining_in_policy = policy_target - policy_sampled
            if remaining_in_policy > 0:
                next_batch_size = _launch_batch(
                    next_idx,
                    min(prompt_groups_per_step, remaining_in_policy),
                )
                next_idx += next_batch_size
                policy_sampled += next_batch_size
                if next_batch_size > 0:
                    # Give prefetched tasks a chance to start before the current
                    # train step enters its blocking work on a background thread.
                    await asyncio.sleep(0)

            if step_prompt_groups:
                global_step, _ = await asyncio.to_thread(
                    train_fns.train_step, global_step, step_prompt_groups, loop_stats,
                )

                if metrics_callback is not None:
                    metrics_callback({
                        "train/step": global_step,
                        "rollout/sample_fails": loop_stats["sample_fails"],
                        "rollout/filter_drops": loop_stats["filter_drops"],
                    })

            current_batch_size = next_batch_size

        if policy_boundary_fn is not None and next_idx < len(coros):
            await asyncio.to_thread(policy_boundary_fn, global_step)

    return global_step
