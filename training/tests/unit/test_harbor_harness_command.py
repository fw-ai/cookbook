"""Exercise the actual shell wrapper used by installed Harbor harnesses."""

import subprocess
from pathlib import Path

import pytest

from training.examples.harbor_rl.tito import sidecar


@pytest.mark.parametrize("driver_status,overflow,expected", [(0, False, 0), (7, False, 7), (0, True, 43)])
def test_harness_keepalive_preserves_exit_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    driver_status: int,
    overflow: bool,
    expected: int,
) -> None:
    marker = tmp_path / "overflow"
    monkeypatch.setattr(sidecar, "SIDECAR_CONTEXT_OVERFLOW_PATH", str(marker))
    if overflow:
        marker.write_text("{}")
    command = sidecar.harness_command(
        f"exit {driver_status}",
        status_path=str(tmp_path / "status"),
        log_path=str(tmp_path / "agent.log"),
    )
    result = subprocess.run(["/bin/bash", "-c", command], capture_output=True, timeout=10)
    assert result.returncode == expected
    assert (tmp_path / "status").read_text() == f"{driver_status}\n"
