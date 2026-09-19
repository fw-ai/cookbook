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


class ArtifactProcessPool:
    """Own a bounded CPU process for complete artifact-to-rollout conversion.

    Submit paths and small trial metadata, not decoded artifacts. The worker
    releases captured prompts and intermediate objects before sending back the
    normal RolloutRun. Spawn avoids inheriting the live SDK's threads/sockets.
    """

    def __init__(self) -> None:
        self._executor: ProcessPoolExecutor | None = None
        self._closed = False

    async def run(
        self, function: Callable[..., _Result], *args: Any, **kwargs: Any
    ) -> _Result:
        if self._closed:
            raise RuntimeError("artifact process pool is closed")
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )
        executor = self._executor
        try:
            pending = executor.submit(partial(function, *args, **kwargs))
        except BrokenProcessPool:
            self._discard_broken_executor(executor)
            raise
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
        except BrokenProcessPool:
            # A failed worker must not poison every subsequent rollout retry.
            self._discard_broken_executor(executor)
            raise

    def _discard_broken_executor(self, executor: ProcessPoolExecutor) -> None:
        if self._executor is executor:
            self._executor = None
        executor.shutdown(wait=False, cancel_futures=True)

    async def aclose(self) -> None:
        self._closed = True
        executor, self._executor = self._executor, None
        if executor is not None:
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)


async def run_artifact_task(
    function: Callable[..., _Result], *args: Any, **kwargs: Any
) -> _Result:
    context = contextvars.copy_context()
    call = partial(function, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(
        _ARTIFACT_EXECUTOR, context.run, call
    )
