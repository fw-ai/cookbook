from __future__ import annotations

import asyncio

import pytest

from training.utils.rl.async_rl import OverlappedEvaluation


def test_evaluation_overlaps_work_and_deduplicates_completed_step() -> None:
    async def exercise() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        completed: list[int] = []

        async def evaluate(step: int) -> None:
            started.set()
            await release.wait()
            completed.append(step)

        evaluations = OverlappedEvaluation(evaluate, interval=2)
        assert evaluations.start(0, force=True)
        await started.wait()

        # The caller remains free to run rollout and trainer work.
        assert evaluations.active_step == 0
        release.set()
        await evaluations.join()

        assert completed == [0]
        assert evaluations.active_step is None
        assert not evaluations.start(0, force=True)
        assert not evaluations.start(1)
        assert evaluations.start(2)
        await evaluations.join()
        assert completed == [0, 2]

    asyncio.run(exercise())


def test_evaluation_requires_join_before_a_different_step() -> None:
    async def exercise() -> None:
        release = asyncio.Event()

        async def evaluate(_step: int) -> None:
            await release.wait()

        evaluations = OverlappedEvaluation(evaluate, interval=1)
        assert evaluations.start(3)
        with pytest.raises(RuntimeError, match="step 3 must finish before step 4"):
            evaluations.start(4)
        await evaluations.cancel()

    asyncio.run(exercise())


def test_disabled_evaluation_never_creates_a_task() -> None:
    async def exercise() -> None:
        evaluations = OverlappedEvaluation(None, interval=1)
        assert not evaluations.start(0, force=True)
        assert evaluations.active_step is None
        await evaluations.join()

    asyncio.run(exercise())


def test_join_propagates_failure_and_clears_state() -> None:
    async def exercise() -> None:
        async def evaluate(_step: int) -> None:
            raise RuntimeError("evaluation failed")

        evaluations = OverlappedEvaluation(evaluate, interval=1)
        assert evaluations.start(3)
        with pytest.raises(RuntimeError, match="evaluation failed"):
            await evaluations.join()
        assert evaluations.active_step is None

        # A failed attempt is not marked completed, so an explicit retry of the
        # same policy step remains possible for callers that want it.
        assert evaluations.start(3)
        await evaluations.cancel()

    asyncio.run(exercise())


def test_cancel_retrieves_an_already_failed_evaluation() -> None:
    async def exercise() -> None:
        failed = asyncio.Event()

        async def evaluate(_step: int) -> None:
            failed.set()
            raise RuntimeError("evaluation failed")

        evaluations = OverlappedEvaluation(evaluate, interval=1)
        assert evaluations.start(4)
        await failed.wait()
        await asyncio.sleep(0)

        # Shutdown is already handling another outcome; retrieving the failed
        # background task must not replace it with the evaluation exception.
        await evaluations.cancel()
        assert evaluations.active_step is None

    asyncio.run(exercise())


def test_cancelled_join_does_not_orphan_evaluation() -> None:
    async def exercise() -> None:
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def evaluate(_step: int) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        evaluations = OverlappedEvaluation(evaluate, interval=1)
        assert evaluations.start(5)
        await started.wait()

        joining = asyncio.create_task(evaluations.join())
        await asyncio.sleep(0)
        joining.cancel()
        with pytest.raises(asyncio.CancelledError):
            await joining

        await stopped.wait()
        assert evaluations.active_step is None

    asyncio.run(exercise())
