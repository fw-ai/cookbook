import pytest

from training.utils.termination import TerminatedBySignal

def test_async_signal_stop_joins_pending_work_and_restores_handlers():
    import asyncio
    import signal
    from types import SimpleNamespace
    from training.utils.termination import run_async_with_signals
    from training.utils.rl.streaming import drain_groups

    original = signal.getsignal(signal.SIGTERM)
    completed = []

    async def scenario():
        release = asyncio.Event()

        async def waiter():
            await release.wait()
            completed.append(True)

        groups = [SimpleNamespace(waiter=asyncio.create_task(waiter())) for _ in range(2)]
        asyncio.get_running_loop().call_later(0.01, signal.raise_signal, signal.SIGTERM)
        asyncio.get_running_loop().call_later(0.02, release.set)
        try:
            await drain_groups(groups)
        finally:
            assert all(group.waiter.done() for group in groups)
            assert all(not group.waiter.cancelled() for group in groups)

    with pytest.raises(TerminatedBySignal, match="SIGTERM"):
        run_async_with_signals(scenario())
    assert completed == [True, True]
    assert signal.getsignal(signal.SIGTERM) is original


@pytest.mark.parametrize("signum", [2, 15])
def test_second_signal_forces_exit_during_stalled_cleanup(signum):
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import asyncio, signal
        from training.utils.termination import run_async_with_signals
        async def run():
            print('ready', flush=True)
            try:
                await asyncio.Event().wait()
            finally:
                print('cleaning', flush=True)
                await asyncio.Event().wait()
        run_async_with_signals(run())
    """)
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(signum)
        assert process.stdout.readline().strip() == "cleaning"
        process.send_signal(signum)
        process.communicate(timeout=5)
        assert process.returncode == 128 + signum
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_runs_inside_an_already_running_event_loop():
    """Notebook kernels run an event loop on the main thread; recipes must still run there."""
    import asyncio

    from training.utils.termination import run_async_with_signals

    async def recipe():
        await asyncio.sleep(0)
        return "done"

    async def notebook_cell():
        return run_async_with_signals(recipe())

    assert asyncio.run(notebook_cell()) == "done"


def test_kernel_interrupt_cancels_notebook_recipe_and_waits_for_cleanup(monkeypatch):
    """A kernel interrupt must stop the recipe, run its cleanup and delete provisioning trainers."""
    import _thread
    import asyncio
    import signal
    import threading
    import time

    from training.utils import service
    from training.utils.termination import run_async_with_signals

    abandoned = []
    monkeypatch.setattr(service, "abandon_provisioning_trainers", lambda: abandoned.append(True))
    cleaned_up = threading.Event()

    async def recipe():
        try:
            await asyncio.sleep(30)
        finally:
            await asyncio.sleep(0.1)
            cleaned_up.set()

    async def notebook_cell():
        # Like ipykernel, raise KeyboardInterrupt on SIGINT instead of asyncio.run's task cancel.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        threading.Timer(0.2, _thread.interrupt_main).start()
        return run_async_with_signals(recipe())

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(notebook_cell())

    assert time.monotonic() - started < 5
    assert cleaned_up.is_set()
    assert abandoned == [True]


def test_notebook_recipe_timeout_error_propagates():
    """A TimeoutError raised by the recipe must surface, not look like a pending wait."""
    import asyncio

    from training.utils.termination import run_async_with_signals

    async def recipe():
        await asyncio.sleep(0)
        raise TimeoutError("trainer never became ready")

    async def notebook_cell():
        return run_async_with_signals(recipe())

    with pytest.raises(TimeoutError, match="never became ready"):
        asyncio.run(notebook_cell())
