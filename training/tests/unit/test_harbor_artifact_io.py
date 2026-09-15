"""The artifact thread keeps trial context outside the event-loop thread."""

import asyncio
import contextvars
import threading

from training.examples.rl.harbor.tito._artifact_io import run_artifact_task


def test_artifact_worker_preserves_context():
    context = contextvars.ContextVar("trial", default="unset")
    context.set("trial-context")
    thread, value = asyncio.run(
        run_artifact_task(lambda: (threading.get_ident(), context.get()))
    )
    assert thread != threading.get_ident()
    assert value == "trial-context"
