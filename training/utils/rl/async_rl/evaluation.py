"""Single-task evaluation overlap for asynchronous RL loops."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


class OverlappedEvaluation:
    """Run at most one policy-version evaluation outside the trainer path."""

    def __init__(
        self,
        evaluate: Callable[[int], Awaitable[None]] | None,
        *,
        interval: int,
    ) -> None:
        if interval < 1:
            raise ValueError("evaluation interval must be >= 1")
        self._evaluate = evaluate
        self._interval = interval
        self._task: asyncio.Task[None] | None = None
        self._active_step: int | None = None
        self._last_completed_step: int | None = None

    @property
    def active_step(self) -> int | None:
        return self._active_step

    def start(self, step: int, *, force: bool = False) -> bool:
        """Start an eligible evaluation without waiting for it."""

        if self._evaluate is None:
            return False
        if step == self._last_completed_step:
            return False
        if not force and step % self._interval:
            return False
        if self._task is not None:
            raise RuntimeError(
                f"evaluation for step {self._active_step} must finish before step {step}"
            )

        self._active_step = step
        self._task = asyncio.create_task(
            self._run(step),
            name=f"async-rl-evaluation-step-{step}",
        )
        return True

    async def join(self) -> None:
        """Wait for the active evaluation and propagate its failure."""

        task = self._task
        if task is None:
            return
        try:
            await task
        finally:
            self._task = None
            self._active_step = None

    async def cancel(self) -> None:
        """Cancel and retrieve the active task without masking another failure."""

        task = self._task
        if task is None:
            return
        self._task = None
        self._active_step = None
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _run(self, step: int) -> None:
        evaluate = self._evaluate
        if evaluate is None:
            raise RuntimeError("evaluation callback is unavailable")
        await evaluate(step)
        self._last_completed_step = step


__all__ = ["OverlappedEvaluation"]
