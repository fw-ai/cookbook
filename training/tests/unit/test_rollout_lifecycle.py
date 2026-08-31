from __future__ import annotations

import asyncio

from training.utils.rl.rollout.lifecycle import ActiveRolloutTasks, close_rollout_fn


def test_plain_rollout_function_needs_no_close() -> None:
    async def rollout(_row):
        return None

    asyncio.run(close_rollout_fn(rollout))


def test_stateful_rollout_callable_is_closed_once() -> None:
    class Rollout:
        def __init__(self) -> None:
            self.closed = 0

        async def __call__(self, _row):
            return None

        async def aclose(self) -> None:
            self.closed += 1

    rollout = Rollout()
    asyncio.run(close_rollout_fn(rollout))
    assert rollout.closed == 1


def test_active_rollouts_are_cancelled_and_drained_before_shared_shutdown() -> None:
    async def exercise() -> None:
        tracker = ActiveRolloutTasks()
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def rollout() -> None:
            async with tracker.track():
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    cleaned.set()

        task = asyncio.create_task(rollout())
        await started.wait()
        await tracker.cancel_and_wait()

        assert task.cancelled()
        assert cleaned.is_set()

    asyncio.run(exercise())
