"""Signal-driven termination of training recipes."""

from __future__ import annotations

import asyncio
import os
import signal
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from functools import wraps
from typing import Any, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


def with_signals(function: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, T]:
    """Run the whole recipe lifecycle under cooperative signal handling."""

    @wraps(function)
    def run(*args: P.args, **kwargs: P.kwargs) -> T:
        return run_async_with_signals(function(*args, **kwargs))

    return run


class TerminatedBySignal(SystemExit):
    """A signal unwinds recipe cleanup like SystemExit."""

    def __init__(self, signal_name: str) -> None:
        super().__init__(f"Terminated by {signal_name}")
        self.signal_name = signal_name


async def run_in_thread(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Yield before dispatch, then join a native call before resource cleanup."""
    # Let the signal wakeup-fd reader schedule its loop-safe stop callback.
    await asyncio.sleep(0)
    owner = asyncio.current_task()

    async def invoke() -> T:
        # Deliver loop-scheduled stops before issuing the next native mutation.
        await asyncio.sleep(0)
        if owner is not None and owner.cancelling():
            raise asyncio.CancelledError
        return await asyncio.to_thread(function, *args, **kwargs)

    return await _join_task(asyncio.create_task(invoke()))


async def close_in_thread(function: Callable[[], T]) -> T:
    """Join cleanup even when the owning recipe has already been cancelled."""
    return await _join_task(asyncio.create_task(asyncio.to_thread(function)))


async def _join_task(worker: asyncio.Task[T]) -> T:
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        await asyncio.shield(worker)
        raise


def _loop_is_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _run_beside_running_loop(awaitable: Coroutine[Any, Any, T]) -> T:
    """Run a recipe on its own loop in a worker thread, for kernels that already run a loop.

    Jupyter and IPython run an event loop on the main thread, where asyncio.run()
    raises. A kernel interrupt arrives on the main thread as KeyboardInterrupt: it
    deletes still-provisioning trainers, cancels the recipe and waits for its
    cleanup. A second interrupt stops waiting and may leave remote work running.
    """
    started: Future[tuple[asyncio.AbstractEventLoop, asyncio.Task[Any]]] = Future()

    async def run() -> T:
        task = asyncio.current_task()
        assert task is not None
        started.set_result((asyncio.get_running_loop(), task))
        return await awaitable

    def wait(result: Future[T]) -> T:
        # Short timeouts keep the main thread able to take the interrupt while it waits.
        while True:
            try:
                return result.result(timeout=0.1)
            except TimeoutError:
                if result.done():
                    return result.result()  # Finished meanwhile, or the recipe raised TimeoutError.
                continue

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recipe-loop")
    result = executor.submit(asyncio.run, run())
    try:
        return wait(result)
    except KeyboardInterrupt:
        if result.done():
            raise
        from training.utils.service import abandon_provisioning_trainers

        abandon_provisioning_trainers()
        loop, task = started.result()
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            pass  # The loop closed after the recipe finished.
        try:
            wait(result)
        except (asyncio.CancelledError, Exception):
            pass
        raise
    finally:
        executor.shutdown(wait=False)


def run_async_with_signals(awaitable: Coroutine[Any, Any, T]) -> T:
    """First signal cancels training cooperatively; a second exits immediately.

    Only the root task is cancelled, so cleanup can join submitted work. The
    force exit deliberately skips cleanup and may leave remote work running.
    """
    if threading.current_thread() is not threading.main_thread():
        return asyncio.run(awaitable)
    if _loop_is_running():
        return _run_beside_running_loop(awaitable)

    received_signal: str | None = None
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    async def run() -> T:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()

        def stop(signum: int) -> None:
            nonlocal received_signal
            if received_signal is not None:
                os._exit(128 + signum)
            received_signal = signal.Signals(signum).name
            if task is not None and not task.done():
                task.cancel()

        installed = []
        for signum in previous:
            try:
                loop.add_signal_handler(signum, stop, signum)
                installed.append(signum)
            except NotImplementedError:
                # Platforms without Unix signal handlers still wake the loop.
                signal.signal(signum, lambda signum, _frame: loop.call_soon_threadsafe(stop, signum))
        try:
            return await awaitable
        finally:
            for signum in installed:
                loop.remove_signal_handler(signum)

    try:
        try:
            return asyncio.run(run())
        except asyncio.CancelledError:
            if received_signal is not None:
                raise TerminatedBySignal(received_signal) from None
            raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
