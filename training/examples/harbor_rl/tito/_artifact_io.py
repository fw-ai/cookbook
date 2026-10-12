"""Bound artifact CPU work while keeping rollout and training event loops responsive."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

_Result = TypeVar("_Result")
_ARTIFACT_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="harbor-artifact"
)
_DRAIN_TIMEOUT_SECONDS = 300.0
_DEFAULT_MAX_TASKS_PER_WORKER = 64


class ArtifactProcessPool:
    """Own a bounded CPU process for complete artifact-to-rollout conversion.

    Submit paths and small trial metadata, not decoded artifacts. The worker
    releases captured prompts and intermediate objects before sending back the
    normal RolloutRun. Submissions are serialized so a worker crash cannot fail
    every queued rollout, and workers are recycled to bound allocator growth.
    Spawn avoids inheriting the live SDK's threads/sockets.
    """

    def __init__(
        self, *, max_tasks_per_worker: int = _DEFAULT_MAX_TASKS_PER_WORKER
    ) -> None:
        if max_tasks_per_worker < 1:
            raise ValueError("max_tasks_per_worker must be positive")
        self._executor: ProcessPoolExecutor | None = None
        self._run_lock = asyncio.Lock()
        self._max_tasks_per_worker = max_tasks_per_worker
        self._tasks_on_executor = 0
        self._closed = False

    async def run(
        self, function: Callable[..., _Result], *args: Any, **kwargs: Any
    ) -> _Result:
        async with self._run_lock:
            return await self._run_serialized(function, *args, **kwargs)

    async def _run_serialized(
        self, function: Callable[..., _Result], *args: Any, **kwargs: Any
    ) -> _Result:
        broken_retries = 0
        while True:
            if self._closed:
                raise RuntimeError("artifact process pool is closed")
            executor = self._ensure_executor()
            try:
                value = await self._run_once(executor, function, *args, **kwargs)
            except BrokenProcessPool:
                self._discard_broken_executor(executor)
                if broken_retries:
                    raise
                broken_retries += 1
                continue
            self._tasks_on_executor += 1
            if self._tasks_on_executor >= self._max_tasks_per_worker:
                await self._retire_executor(executor)
            return value

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )
            self._tasks_on_executor = 0
        return self._executor

    async def _run_once(
        self,
        executor: ProcessPoolExecutor,
        function: Callable[..., _Result],
        *args: Any,
        **kwargs: Any,
    ) -> _Result:
        pending = executor.submit(partial(function, *args, **kwargs))
        result = asyncio.wrap_future(pending)
        try:
            return await asyncio.shield(result)
        except asyncio.CancelledError:
            pending.cancel()
            # A running process still owns input files inside trial_workspace.
            # Drain it before the caller removes that directory — but bound the
            # drain: a wedged worker must not hang the producer slot forever.
            # Cancellation remains cancellation, including if the worker
            # subsequently fails.
            deadline = asyncio.get_running_loop().time() + _DRAIN_TIMEOUT_SECONDS
            while not result.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(result),
                        timeout=max(1.0, deadline - asyncio.get_running_loop().time()),
                    )
                except asyncio.CancelledError:
                    continue
                except asyncio.TimeoutError:
                    # The worker outlived the drain bound. It still holds this
                    # pool's only process, so every later trial would queue
                    # behind it: uninstall it and let the next call spawn a
                    # fresh one. The orphan exits on its own once its task
                    # finishes or its input files disappear.
                    logger.warning(
                        "Artifact worker exceeded the %.0fs drain bound; "
                        "discarding it so later trials get a fresh process",
                        _DRAIN_TIMEOUT_SECONDS,
                    )
                    self._discard_broken_executor(executor)
                    break
                except Exception:
                    break
            if result.done() and not result.cancelled():
                error = result.exception()
                if isinstance(error, BrokenProcessPool):
                    self._discard_broken_executor(executor)
            raise

    def _discard_broken_executor(self, executor: ProcessPoolExecutor) -> None:
        if self._executor is executor:
            self._executor = None
            self._tasks_on_executor = 0
        executor.shutdown(wait=False, cancel_futures=True)

    async def _retire_executor(self, executor: ProcessPoolExecutor) -> None:
        if self._executor is executor:
            self._executor = None
            self._tasks_on_executor = 0
        await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    async def aclose(self) -> None:
        async with self._run_lock:
            self._closed = True
            executor, self._executor = self._executor, None
            self._tasks_on_executor = 0
            if executor is not None:
                await asyncio.to_thread(
                    executor.shutdown, wait=True, cancel_futures=True
                )


async def run_artifact_task(
    function: Callable[..., _Result], *args: Any, **kwargs: Any
) -> _Result:
    context = contextvars.copy_context()
    call = partial(function, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(
        _ARTIFACT_EXECUTOR, context.run, call
    )
