from copy import deepcopy
import os
import signal
import time
from unittest.mock import patch

import pytest

from training.examples.rl.harbor.recipes.terminal_bench import verifier_deadlock_guard as guard


def _stat(parent, start, state="S"):
    fields = [state] + ["0"] * 19
    fields[1], fields[19] = str(parent), str(start)
    return fields


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    hz = os.sysconf("SC_CLK_TCK")
    proc = tmp_path / "proc"
    parent = proc / "20"
    child = proc / "21"
    parent.mkdir(parents=True)
    child.mkdir()
    (proc / "uptime").write_text("2000 0")
    parent_stat = _stat(10, 100 * hz)
    child_stat = _stat(20, 100 * hz)
    (parent / "stat").write_text("20 (python) " + " ".join(parent_stat))
    (child / "stat").write_text("21 (python) " + " ".join(child_stat))
    for process in (parent, child):
        (process / "comm").write_text("python")
        (process / "cwd").symlink_to("/app")
    (parent / "cmdline").write_bytes(
        b"/cache/bin/python\0/cache/bin/pytest\0--ctrf\0"
        b"/logs/verifier/ctrf.json\0/tests/test_outputs.py\0-rA\0"
    )
    (child / "cmdline").write_bytes(
        b"/cache/bin/python\0-c\0from multiprocessing.spawn import spawn_main; "
        b"spawn_main(tracker_fd=16, pipe_handle=22)\0--multiprocessing-fork\0"
    )
    verifier_log = tmp_path / "test-stdout.txt"
    verifier_log.write_bytes(b"test_outputs.py ......")
    old = time.time() - 700
    os.utime(verifier_log, (old, old))
    test_file = tmp_path / "test_outputs.py"
    test_file.write_text("def test_placeholder(): pass\n")
    monkeypatch.setattr(guard, "VERIFIER_LOG", verifier_log)
    monkeypatch.setattr(guard, "TEST_FILE", test_file)
    log = verifier_log.stat()
    record = {
        "trial": "harbor-opencode-torch-tensor-parallelism-0-test",
        "sandbox_id": "exact",
        "phase": "verification_or_finalization",
        "remote": {
            "observed_at_ms": 1_000_000,
            "verifier_log": {
                "bytes": log.st_size,
                "mtime_ns": log.st_mtime_ns,
                "inode": log.st_ino,
                "age_s": 700,
            },
            "processes": [
                {
                    "Pid": "20", "PPid": "10", "Name": "python",
                    "State": "S (sleeping)", "VmRSS_bytes": 400 * 2**20,
                    "start_ticks": parent_stat[19], "elapsed_s": 700,
                    "cpu_seconds": 4.5,
                },
                {
                    "Pid": "21", "PPid": "20", "Name": "python",
                    "State": "S (sleeping)", "VmRSS_bytes": 400 * 2**20,
                    "start_ticks": child_stat[19], "elapsed_s": 700,
                    "cpu_seconds": 8.5,
                },
            ],
        },
    }
    expected = {
        "pid": 21,
        "start_ticks": child_stat[19],
        "parent_pid": 20,
        "parent_start_ticks": parent_stat[19],
        "log_bytes": log.st_size,
        "log_mtime_ns": log.st_mtime_ns,
        "log_inode": log.st_ino,
    }
    return proc, expected, record


def test_positive_and_command_serialization(evidence):
    proc, expected, record = evidence
    current = deepcopy(record)
    current["remote"]["observed_at_ms"] += 60_000
    assert guard.recovery_candidates(record, current) == [expected]
    assert guard.is_known_deadlock(expected, proc)
    command = guard.recovery_command(expected)
    compile(command.split("\n", 1)[1].rsplit("\nRECOVER", 1)[0], "<remote>", "exec")


@pytest.mark.parametrize("field,value", [
    ("trial", "harbor-opencode-other-task"),
    ("sandbox_id", "changed"),
    ("phase", "agent_or_setup"),
])
def test_trial_identity_changes_fail_closed(evidence, field, value):
    _, _, record = evidence
    changed = deepcopy(record)
    changed[field] = value
    changed["remote"]["observed_at_ms"] += 60_000
    assert guard.recovery_candidates(record, changed) == []


@pytest.mark.parametrize("mutation", ["log", "cpu", "child", "parent", "rss"])
def test_observation_changes_fail_closed(evidence, mutation):
    _, _, record = evidence
    changed = deepcopy(record)
    if mutation == "log":
        changed["remote"]["verifier_log"]["mtime_ns"] += 1
    elif mutation == "cpu":
        changed["remote"]["processes"][1]["cpu_seconds"] += 10
    elif mutation == "child":
        changed["remote"]["processes"][1]["start_ticks"] = "999"
    elif mutation == "parent":
        changed["remote"]["processes"][0]["start_ticks"] = "999"
    else:
        changed["remote"]["processes"][1]["VmRSS_bytes"] = 1
    changed["remote"]["observed_at_ms"] += 60_000
    assert guard.recovery_candidates(record, changed) == []


@pytest.mark.parametrize("mutation", [
    "child_argv", "parent_argv", "cwd", "log", "test", "young", "state",
])
def test_sandbox_revalidation_changes_fail_closed(evidence, mutation):
    proc, expected, _ = evidence
    child = proc / "21"
    parent = proc / "20"
    if mutation == "child_argv":
        (child / "cmdline").write_bytes(b"python\0-c\0other\0")
    elif mutation == "parent_argv":
        (parent / "cmdline").write_bytes(b"python\0pytest\0other\0")
    elif mutation == "cwd":
        (child / "cwd").unlink()
        (child / "cwd").symlink_to("/tmp")
    elif mutation == "log":
        guard.VERIFIER_LOG.write_bytes(b"different")
    elif mutation == "test":
        guard.TEST_FILE.unlink()
    elif mutation == "young":
        (proc / "uptime").write_text("101 0")
    else:
        fields = _stat(20, expected["start_ticks"], state="R")
        (child / "stat").write_text("21 (python) " + " ".join(fields))
    assert not guard.is_known_deadlock(expected, proc)


def test_only_spawned_leaf_receives_sigterm(evidence):
    proc, expected, _ = evidence
    with patch.object(guard.os, "pidfd_open", return_value=99) as opened, \
            patch.object(guard.signal, "pidfd_send_signal") as sent, \
            patch.object(guard.os, "close") as closed:
        result = guard.signal_known_deadlock(expected, proc)
        assert result["action"] == "SIGTERM"
        opened.assert_called_once_with(21)
        sent.assert_called_once_with(99, signal.SIGTERM)
        closed.assert_called_once_with(99)
