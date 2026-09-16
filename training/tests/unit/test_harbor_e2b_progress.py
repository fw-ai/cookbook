from copy import deepcopy
from types import SimpleNamespace

import pytest

from training.examples.rl.harbor.recipes.terminal_bench import monitor_e2b_progress as monitor


@pytest.mark.parametrize('total,available,warn', [
    (1000, 100, True), (1000, 0, True), (1000, 101, False),
    (1000, -1, False), (0, 0, False), (None, 0, False),
    (1000, None, False), (1000, 1001, False),
])
def test_guest_memory_warning_is_inspection_only(total, available, warn):
    current = {'remote': {'guest_memory': {
        'MemTotal': total, 'MemAvailable': available}}}
    original = deepcopy(current)
    warnings = monitor.memory_warnings(current)
    assert bool(warnings) is warn
    assert current == original
    if warn:
        assert warnings[0]['code'] == 'sandbox_memory_pressure'
        assert 'do not kill, retry or increase task resources' in warnings[0]['action']


def test_memory_warning_allows_missing_remote_observation():
    assert monitor.memory_warnings({}) == []
    compile(monitor.REMOTE.removeprefix("python3 - <<'REMOTE'\n").removesuffix('\nREMOTE'), '<remote-probe>', 'exec')


@pytest.mark.parametrize("phase", ["list", "connect", "command"])
@pytest.mark.parametrize("finalized", [False, True])
def test_inspection_failure_rechecks_finalization(tmp_path, monkeypatch, phase, finalized):
    trial = {"trial": "sample", "phase": "agent_or_setup", "trial_age_s": 200}
    result_path = tmp_path / "trials/sample/result.json"

    def fail():
        if finalized:
            result_path.parent.mkdir(parents=True)
            # Finalized does not imply success; do not inspect credentials or
            # fabricate a score from this marker.
            result_path.write_text('{"exception_info": {"exception_type": "AgentTimeoutError"}}')
        raise TimeoutError("remote observation failed")

    def list_sandboxes(**kwargs):
        if phase == "list":
            fail()
        return SimpleNamespace(next_items=lambda: [SimpleNamespace(
            metadata={"session_id": "sample__env"}, sandbox_id="sandbox-1")])

    def connect(sandbox_id):
        assert sandbox_id == "sandbox-1"
        if phase == "connect":
            fail()
        return SimpleNamespace(commands=SimpleNamespace(run=lambda *args, **kwargs: fail()))

    monkeypatch.setattr(monitor, "Sandbox", SimpleNamespace(list=list_sandboxes, connect=connect))
    observed = monitor.inspect_trial(tmp_path, trial)
    if finalized:
        assert observed["observation"] == "already_finalized"
        assert "observation_error" not in observed
    else:
        assert observed["observation_error"] == "TimeoutError"
        assert "observation" not in observed


@pytest.mark.parametrize("finalized", [False, True])
def test_missing_sandbox_is_not_assumed_finished(tmp_path, monkeypatch, finalized):
    def list_sandboxes(**kwargs):
        if finalized:
            result_path = tmp_path / "trials/sample/result.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text("{}")
        return SimpleNamespace(next_items=lambda: [])

    monkeypatch.setattr(monitor, "Sandbox", SimpleNamespace(list=list_sandboxes))
    observed = monitor.inspect_trial(tmp_path, {"trial": "sample", "phase": "agent_or_setup"})
    if finalized:
        assert observed["observation"] == "already_finalized"
    else:
        assert observed["matching_sandboxes"] == 0
        assert "observation" not in observed


def _apt_observation():
    return {
        "sandbox_id": "sandbox-1", "phase": "verification_or_finalization",
        "remote": {
            "verifier_log": {"bytes": 425, "mtime_ns": 123, "inode": 7, "age_s": 301},
            "processes": [{"Name": "apt-get", "Pid": "20", "PPid": "19",
                           "start_ticks": "100", "cpu_seconds": 0.06}],
        },
    }


def test_quiet_verifier_apt_warns_before_generic_fifteen_minute_alert():
    previous = _apt_observation()
    current = deepcopy(previous)
    warnings = monitor.stall_warnings(previous, current)
    assert [w["code"] for w in warnings] == ["suspected_verifier_apt_wait"]
    assert warnings[0]["pid"] == "20"
    assert "Never terminate automatically" in warnings[0]["action"]


@pytest.mark.parametrize("change", [
    "no_previous", "different_sandbox", "agent_phase", "young_log", "new_bytes",
    "new_mtime", "new_inode", "new_cpu", "pid_reused", "new_parent", "missing_cpu",
    "missing_process", "not_apt",
])
def test_apt_warning_requires_repeated_matching_observations(change):
    previous = _apt_observation()
    current = deepcopy(previous)
    log = current["remote"]["verifier_log"]
    process = current["remote"]["processes"][0]
    if change == "no_previous":
        previous = None
    elif change == "different_sandbox":
        current["sandbox_id"] = "sandbox-2"
    elif change == "agent_phase":
        current["phase"] = "agent_or_setup"
    elif change == "young_log":
        log["age_s"] = 299
    elif change == "new_bytes":
        log["bytes"] += 1
    elif change == "new_mtime":
        log["mtime_ns"] += 1
    elif change == "new_inode":
        log["inode"] += 1
    elif change == "new_cpu":
        process["cpu_seconds"] += 0.1
    elif change == "pid_reused":
        process["start_ticks"] = "101"
    elif change == "new_parent":
        process["PPid"] = "18"
    elif change == "missing_cpu":
        del process["cpu_seconds"]
    elif change == "missing_process":
        current["remote"]["processes"] = []
    elif change == "not_apt":
        process["Name"] = "pytest"
    assert "suspected_verifier_apt_wait" not in {
        w["code"] for w in monitor.stall_warnings(previous, current)
    }
