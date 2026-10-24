"""Run long harness commands without depending on a long-lived exec stream."""

from __future__ import annotations

import asyncio
import json
import shlex
import uuid
from typing import Any, Awaitable, Callable

import httpcore
import httpx
import pyqwest

from .sidecar import SIDECAR_PYTHON

_POLL_SECONDS = 5.0
_RPC_TIMEOUT_SECONDS = 20
_RPC_ATTEMPTS = 3
_RPC_TRANSPORT_ERRORS = (
    httpx.TransportError,
    httpcore.NetworkError,
    httpcore.TimeoutException,
    httpcore.ProtocolError,
    ConnectionError,
    TimeoutError,
    pyqwest.ReadError,
    pyqwest.WriteError,
    pyqwest.StreamError,
)

# One per-invocation lock deduplicates an ambiguous launch response. The worker
# stays alive after publishing its exit code so cleanup can verify PID identity
# before killing its entire process group (including leftover tool processes).
_CONTROL = r"""
import fcntl, json, os, signal, subprocess, sys, time
from pathlib import Path

mode, directory = sys.argv[1:3]
root = Path(directory)
root.mkdir(mode=0o700, exist_ok=True)

def identity(pid, live_only=False):
    try:
        fields = Path('/proc/%s/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return None if live_only and fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None

def publish(name, value):
    temporary = root / (name + '.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(root / name)

if mode == 'run':
    # Register before executing anything, even if the spawning RPC was killed.
    # The same lock orders this handshake against cancellation's tombstone.
    with (root / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / 'cancelled').exists():
            sys.exit(0)
        publish('pid', {'pid': os.getpid(), 'start': identity(os.getpid())})
    code = subprocess.call(['/bin/bash', '-c', sys.argv[3]])
    publish('exit', {'return_code': code if code >= 0 else 128 - code})
    while True:
        time.sleep(60)

with (root / 'lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if mode == 'launch':
        if (root / 'cancelled').exists():
            raise RuntimeError('harness launch was cancelled')
        if not (root / 'claimed').exists():
            publish('claimed', {'started': time.monotonic()})
            with open(os.devnull, 'rb') as stdin, open(os.devnull, 'wb') as output:
                child = subprocess.Popen(
                    [sys.executable, '-I', '-c', sys.argv[4], 'run', directory, sys.argv[3]],
                    stdin=stdin, stdout=output, stderr=output, start_new_session=True,
                )
        print('{}')
    elif mode == 'status':
        if (root / 'exit').exists():
            print((root / 'exit').read_text())
        elif not (root / 'pid').exists():
            claim = json.loads((root / 'claimed').read_text())
            if time.monotonic() - claim['started'] > 30:
                raise RuntimeError('harness worker did not register within 30 seconds')
            print('{}')
        else:
            record = json.loads((root / 'pid').read_text())
            if record['start'] is None or identity(record['pid'], live_only=True) != record['start']:
                raise RuntimeError('harness worker disappeared without an exit status')
            print('{}')
    elif mode == 'stop':
        (root / 'cancelled').touch()
        if (root / 'pid').exists():
            record = json.loads((root / 'pid').read_text())
            current = identity(record['pid'])
            if record['start'] is None or (current is not None and current != record['start']):
                raise RuntimeError('harness process identity changed; refusing to signal it')
            if current is None or current == record['start']:
                try:
                    os.killpg(record['pid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
                # Zombies are already dead; waiting for their reaper is unrelated
                # to whether an agent or tool can still mutate the workspace.
                for attempt in range(100):
                    active = False
                    for entry in Path('/proc').glob('[0-9]*/stat'):
                        try:
                            fields = entry.read_text().rsplit(')', 1)[1].split()
                        except (FileNotFoundError, ProcessLookupError):
                            continue
                        if int(fields[2]) == record['pid'] and fields[0] != 'Z':
                            active = True
                            break
                    if not active:
                        break
                    time.sleep(0.01)
                else:
                    raise RuntimeError('harness process group did not stop')
        print('{}')
    else:
        raise ValueError('unknown harness control operation')
"""


async def exec_harness(
    exec_as_agent: Callable[..., Awaitable[Any]],
    environment: Any,
    *,
    command: str,
    log_path: str,
    env: dict[str, str] | None = None,
    timeout_sec: float | None = None,
) -> Any:
    """Detach once, poll short RPCs, and stop the owned group on every exit.

    The installed agent's executor preserves its user, scoped environment and
    nonzero-exit classification. Only idempotent lifecycle RPCs are retried;
    the harness itself is never re-executed after a transport failure.
    """
    directory = f"/tmp/fireworks-harness-{uuid.uuid4().hex}"
    prefix = f"{shlex.quote(SIDECAR_PYTHON)} -I -c {shlex.quote(_CONTROL)}"

    async def control(mode: str) -> dict[str, Any]:
        args = [mode, directory]
        if mode == "launch":
            args.extend([command, _CONTROL])
        rpc = prefix + " " + " ".join(shlex.quote(arg) for arg in args)
        for attempt in range(_RPC_ATTEMPTS):
            try:
                result = await exec_as_agent(
                    environment,
                    command=rpc,
                    env=env,
                    timeout_sec=_RPC_TIMEOUT_SECONDS,
                )
                return json.loads(result.stdout)
            except Exception as error:
                # E2B's RPC adapter preserves transport failures as the cause
                # of ConnectError; decoding and command errors are not retries.
                if not isinstance(error, _RPC_TRANSPORT_ERRORS) and not isinstance(
                    error.__cause__, _RPC_TRANSPORT_ERRORS
                ):
                    raise
                if attempt + 1 == _RPC_ATTEMPTS:
                    raise
                await asyncio.sleep(attempt + 1)
        raise AssertionError("unreachable")

    try:
        async with asyncio.timeout(timeout_sec):
            await control("launch")
            while True:
                status = await control("status")
                if "return_code" in status:
                    code = status["return_code"]
                    await control("stop")
                    if code == 0:
                        return None
                    # Failure classification uses bounded evidence; successful
                    # completion never waits for optional agent log collection.
                    return await exec_as_agent(
                        environment,
                        command=f"tail -c 8192 {shlex.quote(log_path)}; exit {int(code)}",
                        timeout_sec=_RPC_TIMEOUT_SECONDS,
                    )
                await asyncio.sleep(_POLL_SECONDS)
    except BaseException as error:
        try:
            await asyncio.shield(control("stop"))
        except Exception as cleanup_error:
            error.add_note(f"harness process cleanup failed: {cleanup_error}")
        raise
