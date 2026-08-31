"""Optional lifecycle hook for stateful rollout-function facades."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


class ActiveRolloutTasks:
    """Track facade calls so shutdown can drain them before shared state."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    @asynccontextmanager
    async def track(self) -> AsyncIterator[None]:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        try:
            yield
        finally:
            if task is not None:
                self._tasks.discard(task)

    async def cancel_and_wait(self) -> None:
        current = asyncio.current_task()
        pending = [task for task in self._tasks if task is not current]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def close_rollout_fn(rollout_fn: Any) -> None:
    """Close a stateful callable without changing the rollout function API."""
    close = getattr(rollout_fn, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


__all__ = ["ActiveRolloutTasks", "close_rollout_fn"]
