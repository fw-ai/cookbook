"""Exercise real detached processes, including ambiguous transport outcomes."""

import asyncio
import json
import os
import shlex
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import httpcore
import httpx
import pyqwest
import pytest

from training.examples.harbor_rl.tito import harness_exec, sidecar


class LocalExec:
    def __init__(self) -> None:
        self.fail_after_launch = False
        self.fail_status = False
        self.transport_error: Exception = httpx.ReadError("response lost after dispatch")
        self.commands: list[str] = []

    async def __call__(
        self,
        environment: object,
        *,
        command: str,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> SimpleNamespace:
        self.commands.append(command)
        child = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **(env or {})},
        )
        stdout, stderr = await asyncio.wait_for(child.communicate(), timeout_sec)
        if child.returncode:
            raise RuntimeError(f"exit={child.returncode}: {stdout.decode()}{stderr.decode()}")
        if self.fail_after_launch and shlex.split(command)[4] == "launch":
            self.fail_after_launch = False
            raise self.transport_error
        if self.fail_status and shlex.split(command)[4] == "status":
            self.fail_status = False
            raise self.transport_error
        return SimpleNamespace(stdout=stdout.decode(), stderr=stderr.decode(), return_code=0)


@pytest.fixture
def control(monkeypatch: pytest.MonkeyPatch) -> LocalExec:
    monkeypatch.setattr(harness_exec, "SIDECAR_PYTHON", sys.executable)
    monkeypatch.setattr(harness_exec, "_POLL_SECONDS", 0.01)
    return LocalExec()


def wrapped(tmp_path: Path, command: str) -> str:
    return sidecar.harness_command(command, status_path=str(tmp_path / "status"), log_path=str(tmp_path / "log"))


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [0, 7])
async def test_quiet_command_preserves_log_env_and_exit(control: LocalExec, tmp_path: Path, code: int) -> None:
    command = wrapped(tmp_path, f"sleep 0.05; printf '%s' \"$HARNESS_TEST\"; (exit {code})")
    call = harness_exec.exec_harness(
        control,
        None,
        command=command,
        log_path=str(tmp_path / "log"),
        env={"HARNESS_TEST": "original"},
    )
    if code:
        with pytest.raises(RuntimeError, match="exit=7: original"):
            await call
    else:
        assert await call is None
        assert all(shlex.split(command)[0] == sys.executable for command in control.commands)
    assert (tmp_path / "log").read_text() == "original"
    assert (tmp_path / "status").read_text() == f"{code}\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadError("response lost"),
        httpcore.ReadError("response lost"),
        pyqwest.ReadError("response lost"),
    ],
)
async def test_transport_retry_does_not_duplicate_command(
    control: LocalExec, tmp_path: Path, error: Exception
) -> None:
    control.transport_error = error
    control.fail_after_launch = control.fail_status = True
    await harness_exec.exec_harness(
        control,
        None,
        command=wrapped(tmp_path, f"echo once >> {tmp_path / 'effects'}; sleep 0.1"),
        log_path=str(tmp_path / "log"),
    )
    assert (tmp_path / "effects").read_text() == "once\n"
    assert sum(shlex.split(command)[4] == "launch" for command in control.commands) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_failure", [True, False])
async def test_e2b_mapped_errors_retry_only_transport_causes(
    control: LocalExec, tmp_path: Path, transport_failure: bool
) -> None:
    rpc = pytest.importorskip("e2b.envd.rpc")
    codes = pytest.importorskip("connectrpc.code")
    errors = pytest.importorskip("connectrpc.errors")
    wrapped_error = errors.ConnectError(codes.Code.INTERNAL, "request interrupted")
    wrapped_error.__cause__ = pyqwest.ReadError("response lost") if transport_failure else ValueError("invalid body")

    async def running() -> bool:
        return True

    control.transport_error = await rpc.ahandle_rpc_exception_with_health(wrapped_error, running)
    control.fail_after_launch = control.fail_status = True
    execute = harness_exec.exec_harness(
        control,
        None,
        command=wrapped(tmp_path, f"echo once >> {tmp_path / 'effects'}; sleep 0.1"),
        log_path=str(tmp_path / "log"),
    )
    if transport_failure:
        await execute
        assert (tmp_path / "effects").read_text() == "once\n"
        assert sum(shlex.split(command)[4] == "launch" for command in control.commands) == 2
    else:
        with pytest.raises(type(control.transport_error)) as failure:
            await execute
        assert failure.value is control.transport_error
        assert sum(shlex.split(command)[4] == "launch" for command in control.commands) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_or_cancel_stops_tools(control: LocalExec, tmp_path: Path, cancel: bool) -> None:
    marker = tmp_path / "late"
    command = wrapped(tmp_path, f"sleep 2; touch {marker}")
    task = asyncio.create_task(
        harness_exec.exec_harness(
            control,
            None,
            command=command,
            log_path=str(tmp_path / "log"),
            timeout_sec=None if cancel else 0.2,
        )
    )
    if cancel:
        await asyncio.sleep(0.2)
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    await asyncio.sleep(2.1)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_independent_invocations_do_not_reuse_exit(control: LocalExec, tmp_path: Path) -> None:
    for _ in range(2):
        await harness_exec.exec_harness(
            control,
            None,
            command=wrapped(tmp_path, f"echo once >> {tmp_path / 'effects'}"),
            log_path=str(tmp_path / "log"),
        )
    assert (tmp_path / "effects").read_text() == "once\nonce\n"


@pytest.mark.asyncio
async def test_missing_worker_is_an_error(control: LocalExec, tmp_path: Path) -> None:
    directory = str(tmp_path / "control")

    async def rpc(mode: str, *args: str) -> SimpleNamespace:
        return await control(
            None,
            command=f"{shlex.quote(sys.executable)} -I -c {shlex.quote(harness_exec._CONTROL)} "
            + " ".join(shlex.quote(x) for x in [mode, directory, *args]),
        )

    try:
        await rpc("launch", f"sleep 1; touch {tmp_path / 'late'}", harness_exec._CONTROL)
        for _ in range(100):
            if (Path(directory) / "pid").exists():
                break
            await asyncio.sleep(0.01)
        record = json.loads((Path(directory) / "pid").read_text())
        os.kill(record["pid"], signal.SIGKILL)
        await asyncio.sleep(0.05)
        with pytest.raises(RuntimeError, match="disappeared"):
            await rpc("status")
    finally:
        await rpc("stop")
    await asyncio.sleep(1.1)
    assert not (tmp_path / "late").exists()


@pytest.mark.asyncio
async def test_permanent_poll_failure_stops_command(control: LocalExec, tmp_path: Path) -> None:
    marker = tmp_path / "late"

    async def broken(environment: object, *, command: str, **kwargs: object) -> SimpleNamespace:
        if shlex.split(command)[4] == "status":
            raise httpx.ReadError("disconnected")
        return await control(environment, command=command, **kwargs)

    with pytest.raises(httpx.ReadError):
        await harness_exec.exec_harness(
            broken,
            None,
            command=wrapped(tmp_path, f"sleep 8; touch {marker}"),
            log_path=str(tmp_path / "log"),
        )
    await asyncio.sleep(5.1)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_cancel_before_launch_records_tombstone(control: LocalExec, tmp_path: Path) -> None:
    directory = str(tmp_path / "control")
    for mode in ["stop", "launch"]:
        args = [mode, directory]
        if mode == "launch":
            args += [f"touch {tmp_path / 'unexpected'}", harness_exec._CONTROL]
        command = f"{shlex.quote(sys.executable)} -I -c {shlex.quote(harness_exec._CONTROL)} " + " ".join(
            shlex.quote(x) for x in args
        )
        if mode == "launch":
            with pytest.raises(RuntimeError, match="launch was cancelled"):
                await control(None, command=command)
        else:
            await control(None, command=command)
    assert not (tmp_path / "unexpected").exists()


@pytest.mark.asyncio
async def test_killed_launcher_cannot_orphan_an_unregistered_worker(control: LocalExec, tmp_path: Path) -> None:
    directory = str(tmp_path / "control")
    marker = tmp_path / "late"
    # Kill only the spawning RPC immediately after fork. The actual worker
    # receives the unmodified controller and must register itself or cancel.
    fault = harness_exec._CONTROL.replace(
        "stdin=stdin, stdout=output, stderr=output, start_new_session=True,\n                )",
        "stdin=stdin, stdout=output, stderr=output, start_new_session=True,\n                )\n            os.kill(os.getpid(), signal.SIGKILL)",
    )
    assert fault != harness_exec._CONTROL
    command = f"{shlex.quote(sys.executable)} -I -c {shlex.quote(fault)} " + " ".join(
        shlex.quote(x)
        for x in [
            "launch",
            directory,
            f"sleep 1; touch {marker}",
            harness_exec._CONTROL,
        ]
    )
    with pytest.raises(RuntimeError, match="exit=-9"):
        await control(None, command=command)
    await control(
        None,
        command=f"{shlex.quote(sys.executable)} -I -c {shlex.quote(harness_exec._CONTROL)} stop {shlex.quote(directory)}",
    )
    await asyncio.sleep(1.1)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_claim_without_spawn_has_bounded_handshake(control: LocalExec, tmp_path: Path) -> None:
    directory = tmp_path / "control"
    directory.mkdir()
    (directory / "claimed").write_text(json.dumps({"started": 0}))
    with pytest.raises(RuntimeError, match="did not register"):
        await control(
            None,
            command=f"{shlex.quote(sys.executable)} -I -c {shlex.quote(harness_exec._CONTROL)} status {shlex.quote(str(directory))}",
        )
