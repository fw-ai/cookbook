"""Opt-in recovery for the known torch-tensor-parallelism verifier deadlock.

Some candidate solutions leave ``test_outputs.py`` blocked in a spawned
``torch.multiprocessing`` child after pytest has already emitted its first test
results.  Waiting for the outer verifier timeout holds a synchronous RL group
for another ten minutes without doing useful work.

This guard is intentionally fail-closed.  It requires two unchanged observer
samples, the exact task and verifier phase, a quiet verifier log, stable
sleeping parent/child identities and zero CPU progress.  It then revalidates
the process tree, argv, cwd, test file and log identity inside the sandbox
before sending SIGTERM only to the spawned test child.  It never signals the
agent, Harbor, trainer or rollout processes.
"""

import inspect
import json
import os
from pathlib import Path
import re
import signal
import time


TASK_PREFIX = "harbor-opencode-torch-tensor-parallelism-"
MIN_QUIET_S = 600
MIN_CHILD_RSS_BYTES = 128 * 2**20
VERIFIER_LOG = Path("/logs/verifier/test-stdout.txt")
TEST_FILE = Path("/tests/test_outputs.py")
LOG_MARKER = b"test_outputs.py ...."


def _stable(before, after, observation_s):
    cpu_before = before.get("cpu_seconds")
    cpu_after = after.get("cpu_seconds")
    return (
        observation_s > 0
        and after.get("start_ticks") is not None
        and after["start_ticks"] == before.get("start_ticks")
        and after.get("PPid") == before.get("PPid")
        and isinstance(cpu_before, (int, float))
        and isinstance(cpu_after, (int, float))
        and 0 <= cpu_after - cpu_before <= max(1.0, observation_s * 0.02)
        and after.get("State", "").startswith("S ")
        and before.get("State", "").startswith("S ")
    )


def recovery_candidates(previous, current):
    """Return stable spawned-test leaves; sandbox revalidation remains final."""
    if (
        not previous
        or not current.get("trial", "").startswith(TASK_PREFIX)
        or current.get("phase") != "verification_or_finalization"
        or previous.get("phase") != current.get("phase")
        or not current.get("sandbox_id")
        or previous.get("sandbox_id") != current["sandbox_id"]
    ):
        return []
    before = previous.get("remote", {})
    after = current.get("remote", {})
    observation_s = (
        (after.get("observed_at_ms", 0) - before.get("observed_at_ms", 0)) / 1000
    )
    old_log = before.get("verifier_log", {})
    log = after.get("verifier_log", {})
    if not (
        (log.get("age_s") or 0) >= MIN_QUIET_S
        and all(
            log.get(key) is not None and log[key] == old_log.get(key)
            for key in ("bytes", "mtime_ns", "inode")
        )
    ):
        return []

    old = {process["Pid"]: process for process in before.get("processes", [])}
    new = {process["Pid"]: process for process in after.get("processes", [])}
    candidates = []
    for child in new.values():
        parent = new.get(child.get("PPid"))
        if (
            child.get("Name") == "python"
            and parent
            and parent.get("Name") == "python"
            and (child.get("VmRSS_bytes") or 0) >= MIN_CHILD_RSS_BYTES
            and (child.get("elapsed_s") or 0) >= MIN_QUIET_S
            and _stable(old.get(child["Pid"], {}), child, observation_s)
            and _stable(old.get(parent["Pid"], {}), parent, observation_s)
        ):
            candidates.append(
                {
                    "pid": int(child["Pid"]),
                    "start_ticks": child["start_ticks"],
                    "parent_pid": int(parent["Pid"]),
                    "parent_start_ticks": parent["start_ticks"],
                    "log_bytes": int(log["bytes"]),
                    "log_mtime_ns": int(log["mtime_ns"]),
                    "log_inode": int(log["inode"]),
                }
            )
    return candidates


def _proc_stat(proc_root, pid):
    fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return fields


def is_known_deadlock(expected, proc_root=Path("/proc")):
    """Revalidate the exact blocked verifier leaf and its pytest parent."""
    child = proc_root / str(expected["pid"])
    parent = proc_root / str(expected["parent_pid"])
    child_stat = _proc_stat(proc_root, expected["pid"])
    parent_stat = _proc_stat(proc_root, expected["parent_pid"])
    if (
        child_stat[19] != expected["start_ticks"]
        or parent_stat[19] != expected["parent_start_ticks"]
        or child_stat[0] == "Z"
        or parent_stat[0] == "Z"
        or int(child_stat[1]) != expected["parent_pid"]
        or not child_stat[0].startswith("S")
        or not parent_stat[0].startswith("S")
    ):
        return False
    uptime = float((proc_root / "uptime").read_text().split()[0])
    elapsed = uptime - int(child_stat[19]) / os.sysconf("SC_CLK_TCK")
    if elapsed < MIN_QUIET_S:
        return False
    if child.joinpath("comm").read_text().strip() != "python":
        return False
    if parent.joinpath("comm").read_text().strip() != "python":
        return False
    child_argv = child.joinpath("cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    parent_argv = parent.joinpath("cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    spawn = child_argv[2] if len(child_argv) >= 3 else b""
    if not (
        child_argv[1:2] == [b"-c"]
        and re.fullmatch(
            rb"from multiprocessing[.]spawn import spawn_main; "
            rb"spawn_main[(]tracker_fd=[0-9]+, pipe_handle=[0-9]+[)]",
            spawn,
        )
        and b"--multiprocessing-fork" in child_argv[3:]
    ):
        return False
    if not (
        len(parent_argv) == 6
        and parent_argv[1].endswith(b"/bin/pytest")
        and parent_argv[2:]
        == [b"--ctrf", b"/logs/verifier/ctrf.json", b"/tests/test_outputs.py", b"-rA"]
    ):
        return False
    if os.readlink(child / "cwd") != "/app" or os.readlink(parent / "cwd") != "/app":
        return False
    if not TEST_FILE.is_file() or not VERIFIER_LOG.is_file():
        return False
    log = VERIFIER_LOG.stat()
    if (
        log.st_size != expected["log_bytes"]
        or log.st_mtime_ns != expected["log_mtime_ns"]
        or log.st_ino != expected["log_inode"]
        or time.time() - log.st_mtime < MIN_QUIET_S
        or LOG_MARKER not in VERIFIER_LOG.read_bytes()[-256:]
    ):
        return False
    return True


def signal_known_deadlock(expected, proc_root=Path("/proc")):
    descriptor = None
    try:
        descriptor = os.pidfd_open(expected["pid"])
        if not is_known_deadlock(expected, proc_root):
            return {"action": "none", "reason": "evidence_changed"}
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        return {
            "action": "SIGTERM",
            "pid": expected["pid"],
            "start_ticks": expected["start_ticks"],
            "reason": "torch_tensor_parallelism_spawned_verifier_deadlock",
        }
    except (OSError, ValueError, KeyError, IndexError, AttributeError) as error:
        return {"action": "none", "reason": type(error).__name__}
    finally:
        if descriptor is not None:
            os.close(descriptor)


def recovery_command(expected):
    """Serialize only process/log identity; revalidate inside the sandbox."""
    payload = {
        key: expected[key]
        for key in (
            "pid",
            "start_ticks",
            "parent_pid",
            "parent_start_ticks",
            "log_bytes",
            "log_mtime_ns",
            "log_inode",
        )
    }
    return (
        "python3 - <<'RECOVER'\nimport os, signal, json, time, re\nfrom pathlib import Path\n"
        + f"MIN_QUIET_S = {MIN_QUIET_S!r}\n"
        + f"VERIFIER_LOG = Path({str(VERIFIER_LOG)!r})\n"
        + f"TEST_FILE = Path({str(TEST_FILE)!r})\n"
        + f"LOG_MARKER = {LOG_MARKER!r}\n"
        + inspect.getsource(_proc_stat)
        + "\n"
        + inspect.getsource(is_known_deadlock)
        + "\n"
        + inspect.getsource(signal_known_deadlock)
        + f"\nprint(json.dumps(signal_known_deadlock(json.loads({json.dumps(payload)!r}))))\nRECOVER"
    )
