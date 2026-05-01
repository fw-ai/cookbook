"""Small request gates for rollout-side inference calls."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping, Protocol


DEFAULT_REQUEST_GATE_CONCURRENCY = 32


class RequestGate(Protocol):
    """Minimal request throttle consumed by rollout samplers/engines."""

    async def acquire(self) -> None: ...

    def release(self, server_metrics: Mapping[str, Any] | Any | None = None) -> None: ...


class FixedRequestGate:
    """Fixed-width request gate for rollout inference calls."""

    def __init__(self, max_concurrency: int = DEFAULT_REQUEST_GATE_CONCURRENCY) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def acquire(self) -> None:
        await self._semaphore.acquire()

    def release(self, server_metrics: Mapping[str, Any] | Any | None = None) -> None:
        self._semaphore.release()
