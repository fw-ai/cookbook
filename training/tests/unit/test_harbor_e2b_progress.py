from copy import deepcopy
from types import SimpleNamespace

import pytest

from training.examples.rl.harbor.recipes.terminal_bench import monitor_e2b_progress as monitor


def _tool_observation(start=1000, now=901000, part_id='part-1'):
    return {'trial': 'trial-1', 'sandbox_id': 'sandbox-1', 'phase': 'agent_or_setup',
            'remote': {'observed_at_ms': now, 'running_tools': [
                {'part_id': part_id, 'tool': 'bash', 'start_ms': start,
                 'elapsed_s': (now - start) / 1000}]}}


def test_tool_metadata_refresh_does_not_hide_long_running_tool():
    previous = _tool_observation(now=301000)
    monitor.retain_tool_start(None, previous)
    saved = deepcopy(previous)
    current = _tool_observation(start=900000)
    monitor.retain_tool_start(previous, current)
    tool = current['remote']['running_tools'][0]
    assert tool['start_ms'] == 900000  # Preserve the raw, reset timestamp.
    assert tool['elapsed_s'] == 1
    assert tool['earliest_recorded_start_ms'] == 1000
    assert tool['observed_elapsed_s_lower_bound'] == 900
    assert previous == saved
    warnings = monitor.sampling_budget_warnings(current)
    assert warnings[0]['code'] == 'long_tool_call'
    assert warnings[0]['elapsed_s'] == 900
    assert 'does not prove a hang' in warnings[0]['action']
    latest = _tool_observation(start=950000, now=960000)
    monitor.retain_tool_start(current, latest)
    assert latest['remote']['running_tools'][0]['observed_elapsed_s_lower_bound'] == 959


@pytest.mark.parametrize('field', ['trial', 'sandbox_id', 'part_id'])
def test_tool_start_never_crosses_trial_sandbox_or_call(field):
    previous = _tool_observation(now=301000)
    monitor.retain_tool_start(None, previous)
    current = _tool_observation(start=900000)
    if field == 'part_id':
        current['remote']['running_tools'][0][field] = 'part-2'
    else:
        current[field] = 'different'
    monitor.retain_tool_start(previous, current)
    assert current['remote']['running_tools'][0]['observed_elapsed_s_lower_bound'] == 1
    assert monitor.sampling_budget_warnings(current) == []


@pytest.mark.parametrize('value', [None, -1, True, '1000', float('nan'), float('inf'), 9999999])
def test_invalid_tool_start_is_not_used(value):
    current = _tool_observation()
    tool = current['remote']['running_tools'][0]
    tool['start_ms'] = value
    monitor.retain_tool_start(None, current)
    assert 'earliest_recorded_start_ms' not in tool


def test_missing_history_and_legacy_observations_are_safe():
    monitor.retain_tool_start(None, {})
    current = _tool_observation(part_id=None)
    monitor.retain_tool_start(None, current)
    assert 'observed_elapsed_s_lower_bound' not in current['remote']['running_tools'][0]
    current = _tool_observation(start=900000)
    monitor.retain_tool_start({'observation_error': 'TimeoutError'}, current)
    assert current['remote']['running_tools'][0]['observed_elapsed_s_lower_bound'] == 1


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


@pytest.mark.parametrize('count,warn', [(4, True), (1, True), (0, False),
                                     (-1, False), (None, False), ('4', False),
                                     (True, False)])
def test_guest_oom_history_survives_recovered_memory(count, warn):
    current = {'remote': {'guest_oom_kills': count,
                          'guest_memory': {'MemTotal': 1000, 'MemAvailable': 900}}}
    original = deepcopy(current)
    assert monitor.memory_warnings(current) == []
    warnings = monitor.guest_oom_warnings(current)
    assert bool(warnings) is warn
    assert current == original
    if warn:
        assert warnings[0]['guest_lifetime_kills'] == count
        assert 'Do not rewrite rewards' in warnings[0]['action']


def test_guest_oom_history_allows_missing_observation():
    assert monitor.guest_oom_warnings({}) == []


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


def test_quiet_verifier_apt_warns_with_generic_five_minute_alert():
    previous = _apt_observation()
    current = deepcopy(previous)
    warnings = monitor.stall_warnings(previous, current)
    assert [w["code"] for w in warnings] == ["verifier_log_unchanged", "suspected_verifier_apt_wait"]
    assert warnings[1]["pid"] == "20"
    assert all("Never terminate automatically" in w["action"] for w in warnings)


@pytest.mark.parametrize("age,warn", [(299, False), (300, True), (899, True)])
def test_quiet_verifier_alert_does_not_require_apt_or_fifteen_minutes(age, warn):
    previous = _apt_observation()
    previous["remote"]["processes"] = []
    current = deepcopy(previous)
    current["remote"]["verifier_log"]["age_s"] = age
    warnings = monitor.stall_warnings(previous, current)
    assert bool(warnings) is warn
    if warn:
        assert warnings[0]["code"] == "verifier_log_unchanged"
        assert "quiet output alone is not failure" in warnings[0]["action"]


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
