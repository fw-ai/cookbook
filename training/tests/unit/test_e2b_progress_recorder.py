import importlib
from copy import deepcopy
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("e2b", reason="The read-only E2B recorder requires the optional sandbox SDK")
recorder = importlib.import_module(
    "training.examples.rl.harbor.recipes.terminal_bench.monitor_e2b_progress"
)


def test_pid_identity_and_remote_probe_syntax():
    assert recorder.identity(os.getpid()) is not None
    assert recorder.identity(2147483647) is None
    code = recorder.REMOTE.split("python3 - <<'REMOTE'\n", 1)[1].rsplit("\nREMOTE", 1)[0]
    compile(code, "<read-only remote probe>", "exec")


def test_finalized_trial_never_contacts_e2b(tmp_path, monkeypatch):
    trial = tmp_path / "trials" / "done"
    trial.mkdir(parents=True)
    (trial / "result.json").touch()
    api = Mock()
    monkeypatch.setattr(recorder, "Sandbox", api)
    result = recorder.inspect_trial(tmp_path, {"trial": "done", "phase": "agent_or_setup"})
    assert result["observation"] == "already_finalized"
    api.list.assert_not_called()
    api.connect.assert_not_called()


@pytest.mark.parametrize("matching_count", [0, 2])
def test_no_connection_without_unique_exact_session(tmp_path, monkeypatch, matching_count):
    other = SimpleNamespace(metadata={"session_id": "another-run__env"}, sandbox_id="other")
    matching = [SimpleNamespace(metadata={"session_id": "ours__env"}, sandbox_id=str(i))
                for i in range(matching_count)]
    api = Mock()
    api.list.return_value.next_items.return_value = [other, *matching]
    monkeypatch.setattr(recorder, "Sandbox", api)
    result = recorder.inspect_trial(tmp_path, {"trial": "ours", "phase": "agent_or_setup"})
    assert result["matching_sandboxes"] == matching_count
    api.connect.assert_not_called()


def test_unique_exact_session_uses_bounded_read_only_command(tmp_path, monkeypatch):
    api = Mock()
    api.list.return_value.next_items.return_value = [
        SimpleNamespace(metadata={"session_id": "other__env"}, sandbox_id="other"),
        SimpleNamespace(metadata={"session_id": "ours__env"}, sandbox_id="ours-id"),
    ]
    api.connect.return_value.commands.run.return_value.stdout = '{"activity": []}'
    monkeypatch.setattr(recorder, "Sandbox", api)
    result = recorder.inspect_trial(tmp_path, {"trial": "ours", "phase": "agent_or_setup"})
    api.connect.assert_called_once_with("ours-id")
    api.connect.return_value.commands.run.assert_called_once_with(recorder.REMOTE, timeout=15)
    assert result["remote"] == {"activity": []}


def test_observation_error_does_not_log_credentials_or_retry(tmp_path, monkeypatch):
    api = Mock()
    api.list.side_effect = RuntimeError("credential-bearing error content")
    monkeypatch.setattr(recorder, "Sandbox", api)
    result = recorder.inspect_trial(tmp_path, {"trial": "ours", "phase": "agent_or_setup"})
    assert result == {"trial": "ours", "phase": "agent_or_setup",
                      "observation_error": "RuntimeError"}
    assert api.list.call_count == 1
    api.connect.assert_not_called()


def traced_observation():
    return {'sandbox_id': 'ours', 'remote': {
        'running_tools': [{'tool': 'bash', 'start_ms': 1234}],
        'processes': [
            {'Pid': '11', 'PPid': '10', 'TracerPid': '0', 'State': 'S (sleeping)',
             'start_ticks': '100', 'cpu_seconds': 0.03},
            {'Pid': '12', 'PPid': '11', 'TracerPid': '11', 'State': 't (tracing stop)',
             'start_ticks': '101', 'cpu_seconds': 0.0},
        ],
    }}


def test_traced_child_stall_requires_two_matching_observations():
    current = traced_observation()
    assert recorder.stall_warnings(None, current) == []
    warnings = recorder.stall_warnings(deepcopy(current), current)
    assert len(warnings) == 1
    assert warnings[0]['code'] == 'suspected_traced_child_stall'
    assert warnings[0]['child_pid'] == '12'


@pytest.mark.parametrize('change', ['sandbox', 'tool', 'cpu', 'pid_reuse', 'resumed', 'missing'])
def test_progress_or_changed_identity_is_not_reported_as_stall(change):
    previous = traced_observation()
    current = deepcopy(previous)
    if change == 'sandbox':
        current['sandbox_id'] = 'replacement'
    elif change == 'tool':
        current['remote']['running_tools'][0]['start_ms'] += 1
    elif change == 'cpu':
        current['remote']['processes'][0]['cpu_seconds'] += 0.01
    elif change == 'pid_reuse':
        current['remote']['processes'][1]['start_ticks'] = '999'
    elif change == 'resumed':
        current['remote']['processes'][1]['State'] = 'R (running)'
    elif change == 'missing':
        current['remote'] = {}
    assert recorder.stall_warnings(previous, current) == []
