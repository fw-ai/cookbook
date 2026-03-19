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
from collections import deque
from typing import Any, Callable, Iterable, Coroutine
from dataclasses import dataclass, field

from tqdm import tqdm

from training.utils.rl.losses import PromptGroup

logger = logging.getLogger(__name__)

__all__ = [
    "TrainStepFns",
    "DynamicFilterFn",
    "AsyncRLState",
    "run_rl_loop",
    "async_rl_loop",
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


@dataclass
class AsyncRLState:
    """Observable state of the async RL scheduler."""

    global_step: int
    current_launch_version: int
    rows_submitted: int
    accepted_total: int
    buffered_accepted: int
    running: int
    oldest_unfinished_version: int
    consumed_in_current_policy: int


async def run_rl_loop(
    sample_fns: Iterable[Coroutine[Any, Any, PromptGroup | None]],
    *,
    train_fns: TrainStepFns,
    prompt_groups_per_step: int = 1,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    global_step: int = 0,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Run the on-policy RL training loop.

    All ``prompt_groups_per_step`` sampling coroutines fire concurrently.
    Request-level throttling is handled by the ``request_semaphore``
    passed into ``sample_with_tokens`` by the caller -- this loop does
    not limit concurrency itself.
    """
    coros = list(sample_fns)

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
        total_completions = 0
        step_start_time = time.time()
        step_prompt_groups: list[PromptGroup] = []

        pbar = tqdm(total=len(step_coros), desc="sampling", unit="group", dynamic_ncols=True)

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
            continue

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


async def async_rl_loop(
    sample_fns: Iterable[Coroutine[Any, Any, PromptGroup | None]],
    *,
    train_fns: TrainStepFns,
    prompt_groups_per_step: int = 1,
    prompt_groups_per_policy: int | None = None,
    max_head_offpolicy_versions: int = 0,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    global_step: int = 0,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
    current_launch_version: int = 0,
    rows_submitted: int = 0,
    accepted_total: int = 0,
    consumed_by_version: dict[int, int] | None = None,
    post_step_fn: Callable[[AsyncRLState], None] | None = None,
    policy_boundary_fn: Callable[[AsyncRLState], None] | None = None,
) -> AsyncRLState:
    """Run an AReaL-style async RL loop.

    The scheduler continuously tops up sampling requests while:
    - the accepted buffer for the next step is not yet full,
    - in-flight work is below the concurrency target,
    - the AReaL-style count gate is open, and
    - the extra oldest-version hard stop is open.

    Training starts whenever ``prompt_groups_per_step`` accepted prompt groups
    are buffered. The caller is responsible for handling any microbatching
    inside ``train_fns.train_step``.
    """
    if prompt_groups_per_step <= 0:
        raise ValueError("prompt_groups_per_step must be positive")
    policy_target = prompt_groups_per_step if prompt_groups_per_policy is None else prompt_groups_per_policy
    if policy_target <= 0:
        raise ValueError("prompt_groups_per_policy must be positive")
    if max_head_offpolicy_versions < 0:
        raise ValueError("max_head_offpolicy_versions must be non-negative")

    coros = iter(sample_fns)
    queue: asyncio.Queue[tuple[int, PromptGroup | None]] = asyncio.Queue()
    worker_error: BaseException | None = None
    exhausted = False

    accepted_buffer: deque[tuple[int, PromptGroup]] = deque()
    pending_tasks: set[asyncio.Task[None]] = set()
    task_versions: dict[asyncio.Task[None], int] = {}
    consumed = dict(consumed_by_version or {})

    step_filter_drops = 0
    step_sample_fails = 0
    step_total_sampled = 0
    step_total_completions = 0
    step_sample_wait_time = 0.0
    step_all_raw_rewards: list[float] = []
    step_start_time = time.time()

    def oldest_unfinished_version() -> int:
        if not task_versions:
            return current_launch_version
        return min(task_versions.values())

    def count_gate_capacity() -> int:
        allowed = (current_launch_version + max_head_offpolicy_versions + 1) * policy_target
        return allowed - (accepted_total + len(pending_tasks))

    def build_state() -> AsyncRLState:
        return AsyncRLState(
            global_step=global_step,
            current_launch_version=current_launch_version,
            rows_submitted=rows_submitted,
            accepted_total=accepted_total,
            buffered_accepted=len(accepted_buffer),
            running=len(pending_tasks),
            oldest_unfinished_version=oldest_unfinished_version(),
            consumed_in_current_policy=consumed.get(current_launch_version, 0),
        )

    async def _worker(coro: Coroutine[Any, Any, PromptGroup | None], launch_version: int) -> None:
        nonlocal worker_error
        try:
            result = await coro
            queue.put_nowait((launch_version, result))
        except BaseException as exc:
            if worker_error is None:
                worker_error = exc
            queue.put_nowait((launch_version, None))

    def _forget_task(task: asyncio.Task[None]) -> None:
        pending_tasks.discard(task)
        task_versions.pop(task, None)

    def reap_finished_tasks() -> None:
        for task in list(pending_tasks):
            if task.done():
                _forget_task(task)

    def try_submit_more() -> None:
        nonlocal exhausted, rows_submitted
        reap_finished_tasks()
        while True:
            concurrency_capacity = prompt_groups_per_step - (len(accepted_buffer) + len(pending_tasks))
            version_gap = current_launch_version - oldest_unfinished_version()
            if (
                exhausted
                or concurrency_capacity <= 0
                or count_gate_capacity() <= 0
                or version_gap > max_head_offpolicy_versions
            ):
                return
            try:
                coro = next(coros)
            except StopIteration:
                exhausted = True
                return
            task = asyncio.create_task(_worker(coro, current_launch_version))
            pending_tasks.add(task)
            task_versions[task] = current_launch_version
            task.add_done_callback(_forget_task)
            rows_submitted += 1

    def finalize_step_stats() -> dict[str, Any]:
        return {
            "valid_prompt_groups": prompt_groups_per_step,
            "total_sampled": step_total_sampled,
            "filter_drops": step_filter_drops,
            "sample_fails": step_sample_fails,
            "sample_wait_time": step_sample_wait_time,
            "step_wall_time": time.time() - step_start_time,
            "all_raw_rewards": list(step_all_raw_rewards),
        }

    def reset_step_stats() -> None:
        nonlocal step_filter_drops, step_sample_fails, step_total_sampled
        nonlocal step_total_completions, step_sample_wait_time, step_all_raw_rewards, step_start_time
        step_filter_drops = 0
        step_sample_fails = 0
        step_total_sampled = 0
        step_total_completions = 0
        step_sample_wait_time = 0.0
        step_all_raw_rewards = []
        step_start_time = time.time()

    try_submit_more()

    while True:
        reap_finished_tasks()
        while len(accepted_buffer) >= prompt_groups_per_step:
            step_groups = [accepted_buffer.popleft() for _ in range(prompt_groups_per_step)]
            prompt_groups = [pg for _, pg in step_groups]
            loop_stats = finalize_step_stats()
            logger.info(
                "Async sampling complete: %d accepted groups (%d sampled, failed=%d, filtered=%d, %.1fs)",
                len(prompt_groups),
                step_total_sampled,
                step_sample_fails,
                step_filter_drops,
                loop_stats["step_wall_time"],
            )

            global_step, _ = await asyncio.to_thread(
                train_fns.train_step, global_step, prompt_groups, loop_stats,
            )

            for version, _ in step_groups:
                consumed[version] = consumed.get(version, 0) + 1

            if metrics_callback is not None:
                metrics_callback({
                    "train/step": global_step,
                    "rollout/sample_fails": step_sample_fails,
                    "rollout/filter_drops": step_filter_drops,
                })

            if post_step_fn is not None:
                await asyncio.to_thread(post_step_fn, build_state())

            reset_step_stats()

            while consumed.get(current_launch_version, 0) >= policy_target:
                if policy_boundary_fn is not None:
                    await asyncio.to_thread(policy_boundary_fn, build_state())
                current_launch_version += 1

            try_submit_more()

        if exhausted and not pending_tasks:
            break

        if not pending_tasks:
            try_submit_more()
            if exhausted and not pending_tasks:
                break

        t_wait = time.time()
        launch_version, item = await queue.get()
        step_sample_wait_time += time.time() - t_wait

        if worker_error is not None:
            raise RuntimeError(f"Sampling worker failed: {worker_error}") from worker_error

        step_total_sampled += 1

        if item is None:
            step_sample_fails += 1
            try_submit_more()
            continue

        step_total_completions += len(item.rewards)
        step_all_raw_rewards.extend(item.rewards)

        if dynamic_filter_fn is not None and not dynamic_filter_fn(item):
            step_filter_drops += 1
            try_submit_more()
            continue

        accepted_total += 1
        accepted_buffer.append((launch_version, item))
        try_submit_more()

    if accepted_buffer:
        logger.info(
            "Async loop finished with %d buffered accepted groups below the step size; dropping tail.",
            len(accepted_buffer),
        )

    return build_state()
