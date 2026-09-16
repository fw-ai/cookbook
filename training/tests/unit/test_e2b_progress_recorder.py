import importlib
import contextlib
from copy import deepcopy
import io
import json
import os
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("e2b", reason="The read-only E2B recorder requires the optional sandbox SDK")
recorder = importlib.import_module(
    "training.examples.rl.harbor.recipes.terminal_bench.monitor_e2b_progress"
)


@pytest.mark.parametrize('interval', [0, 29, 3601])
def test_rejects_unsafe_poll_interval(monkeypatch, interval):
    monkeypatch.setattr(sys, 'argv', ['monitor', '--pid', '1', '--interval-seconds', str(interval)])
    with pytest.raises(SystemExit) as error:
        recorder.main()
    assert error.value.code == 2


@pytest.mark.parametrize('interval', [30, 60, 180, 3600])
def test_interval_preserves_live_client_gate(monkeypatch, interval):
    monkeypatch.setattr(sys, 'argv', ['monitor', '--pid', '1', '--interval-seconds', str(interval)])
    monkeypatch.setattr(recorder, 'identity', lambda _: None)
    with pytest.raises(SystemExit, match='Harness PID is not live'):
        recorder.main()


def test_pid_identity_and_remote_probe_syntax():
    assert recorder.identity(os.getpid()) is not None
    assert recorder.identity(2147483647) is None
    code = recorder.REMOTE.split("python3 - <<'REMOTE'\n", 1)[1].rsplit("\nREMOTE", 1)[0]
    compile(code, "<read-only remote probe>", "exec")


@pytest.mark.parametrize('scenario', ['finalized', 'changed_sandbox', 'matching'])
def test_recovery_rechecks_finished_trial_and_exact_sandbox(tmp_path, monkeypatch, scenario):
    current = {'trial': 'our-trial', 'sandbox_id': 'original'}
    expected = {'pid': 1727, 'parent_pid': 1694, 'start_ticks': '100',
                'parent_start_ticks': '50', 'cpu_seconds': 0.1}
    monkeypatch.setattr(recorder, 'recovery_candidates', lambda *_: [expected])
    sandbox_id = 'replacement' if scenario == 'changed_sandbox' else 'original'
    api = Mock()
    api.list.return_value.next_items.return_value = [SimpleNamespace(
        sandbox_id=sandbox_id, metadata={'session_id': 'our-trial__env'})]
    api.connect.return_value.commands.run.return_value.stdout = json.dumps({'action': 'SIGTERM'})
    monkeypatch.setattr(recorder, 'Sandbox', api)
    if scenario == 'finalized':
        path = tmp_path / 'trials' / 'our-trial'
        path.mkdir(parents=True)
        (path / 'result.json').write_text('{}')
    actions = recorder.recover_trial_search(tmp_path, {}, current)
    if scenario == 'finalized':
        assert actions == []
        api.list.assert_not_called()
    elif scenario == 'changed_sandbox':
        assert actions == [{'action': 'none', 'reason': 'sandbox_identity_changed'}]
        api.connect.assert_not_called()
    else:
        assert actions == [{'action': 'SIGTERM'}]
        api.connect.assert_called_once_with('original')
        api.connect.return_value.commands.run.assert_called_once_with(
            recorder.recovery_command(expected), timeout=15)


@pytest.mark.parametrize('tool,elapsed,expected', [
    ('grep', 300, True), ('grep', 900, True), ('grep', 299, False),
    ('grep', None, False), ('bash', 900, False),
])
def test_long_search_warning_is_read_only_and_not_a_failure(tool, elapsed, expected):
    current = {'remote': {'running_tools': [
        {'tool': tool, 'elapsed_s': elapsed, 'private': 'secret-pattern'},
    ]}}
    before = deepcopy(current)
    warnings = recorder.search_wait_warnings(current)
    assert bool(warnings) == expected
    assert current == before
    assert 'secret-pattern' not in json.dumps(warnings)
    if expected:
        assert warnings[0]['code'] == 'long_grep_wait'
        assert 'do not terminate automatically' in warnings[0]['action']
    assert recorder.search_wait_warnings({}) == []


def test_scored_exception_audit_is_read_only_and_redacted(tmp_path):
    trial = tmp_path / 'trial-1'
    trial.mkdir()
    path = trial / 'result.json'
    payload = json.dumps({
        'finished_at': '2026-09-15T12:05:50Z',
        'exception_info': {'exception_type': 'NonZeroAgentExitCodeError',
                           'exception_message': 'private command and secret'},
        'verifier_result': {'rewards': {'reward': 0.0}},
        'config': {'secret': 'private config'},
    })
    path.write_text(payload)
    result = recorder.scored_exception_inventory(tmp_path)
    assert result['warnings'][0]['exception_type'] == 'NonZeroAgentExitCodeError'
    assert result['warnings'][0]['trial'] == 'trial-1'
    assert 'private' not in json.dumps(result)
    assert path.read_text() == payload


@pytest.mark.parametrize('finished,exception,reward', [
    (None, {'exception_type': 'Error'}, 0),
    ('now', None, 1),
    ('now', {'exception_type': 'VerifierTimeoutError'}, None),
])
def test_scored_exception_audit_does_not_conflate_pending_or_unscored(tmp_path, finished, exception, reward):
    trial = tmp_path / 'trial-1'
    trial.mkdir()
    (trial / 'result.json').write_text(json.dumps({
        'finished_at': finished, 'exception_info': exception,
        'verifier_result': {'rewards': {'reward': reward}},
    }))
    assert recorder.scored_exception_inventory(tmp_path)['warnings'] == []


def test_scored_exception_audit_tolerates_partial_json(tmp_path):
    trial = tmp_path / 'trial-1'
    trial.mkdir()
    (trial / 'result.json').write_text('{')
    result = recorder.scored_exception_inventory(tmp_path)
    assert result['warnings'] == []
    assert result['unreadable_results'] == 1


@pytest.mark.parametrize('completed', [None, 5000])
@pytest.mark.parametrize('duration_args,expected_duration', [
    (['60'], 60), (['2m'], 120), (['1.5s'], 1.5), (['0'], 0),
    (['--kill-after=5', '60'], None), (['invalid'], None),
])
def test_remote_probe_reports_message_timing_not_model_text(tmp_path, completed, duration_args, expected_duration):
    database = tmp_path / 'opencode.db'
    with sqlite3.connect(database) as c:
        c.execute('create table part (time_updated integer, data text)')
        c.execute('create table message (time_updated integer, data text)')
        c.execute('insert into message values (?, ?)', (5000, json.dumps({
            'role': 'assistant', 'time': {'created': 1000, 'completed': completed},
            'finish': 'stop' if completed is not None else None,
            'text': 'private-model-text', 'tokens': {'output': 999},
        })))
    log = tmp_path / 'opencode.txt'
    log.write_text('private-model-text')
    verifier_log = tmp_path / 'verifier.txt'
    verifier_log.write_text('private-verifier-output')
    proc = tmp_path / 'proc'
    proc.mkdir()
    timeout = proc / '2001'
    timeout.mkdir()
    (timeout / 'status').write_text('Name:\ttimeout\nPid:\t2001\nPPid:\t1\nState:\tS (sleeping)\nTracerPid:\t0\n')
    fields = ['S'] + ['0'] * 19
    fields[19] = '100'
    (timeout / 'stat').write_text('2001 (timeout) ' + ' '.join(fields))
    (timeout / 'cmdline').write_bytes(bytes([0]).join(
        s.encode() for s in ['timeout', *duration_args, 'node', 'private-command', '']))
    code = recorder.REMOTE.split("python3 - <<'REMOTE'\n", 1)[1].rsplit("\nREMOTE", 1)[0]
    code = code.replace('/logs/agent/opencode/xdg-data/opencode/opencode.db', str(database))
    code = code.replace('/logs/agent/opencode.txt', str(log))
    code = code.replace('/logs/verifier/test-stdout.txt', str(verifier_log))
    code = code.replace('/tmp/fireworks-tito-opencode/agent-status', str(tmp_path / 'absent-status'))
    code = code.replace("Path('/proc')", f"Path({str(proc)!r})")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, '<read-only remote probe>', 'exec'), {})
    result = json.loads(output.getvalue())
    assert result['latest_assistant']['completed_ms'] == completed
    assert result['latest_assistant']['duration_s'] == (4.0 if completed is not None else None)
    assert result['agent_log']['bytes'] == log.stat().st_size
    assert result['verifier_log']['bytes'] == verifier_log.stat().st_size
    assert result['verifier_log']['mtime_ns'] == verifier_log.stat().st_mtime_ns
    assert result['running_tools'] == []
    assert result['processes'][0].get('declared_timeout_s') == expected_duration
    assert 'private-model-text' not in output.getvalue()
    assert 'private-verifier-output' not in output.getvalue()
    assert 'private-command' not in output.getvalue()
    assert 'tokens' not in output.getvalue()


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


@pytest.mark.parametrize('duration,elapsed,expected', [(60, 91, True), (60, 90, False),
    (60, 59, False), (0, 900, False)])
def test_inner_timeout_warning_never_terminates(duration, elapsed, expected):
    current = {'remote': {'processes': [{'Name': 'timeout', 'Pid': '2001',
        'declared_timeout_s': duration, 'elapsed_s': elapsed}]}}
    warnings = recorder.timeout_warnings(current)
    assert bool(warnings) == expected
    if expected:
        assert warnings[0]['code'] == 'inner_timeout_overrun'
        assert 'Do not kill automatically' in warnings[0]['action']


def test_missing_timeout_metadata_does_not_guess():
    assert recorder.timeout_warnings({'remote': {'processes': [{'Name': 'timeout'}]}}) == []


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
                      "trial_age_s": None,
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


@pytest.mark.parametrize('change,expected', [
    (None, True), ('recovered', False), ('first_error', False),
    ('replacement', False), ('unidentified', False),
])
def test_repeated_observation_errors_are_not_silent_or_terminal(change, expected):
    previous = {'sandbox_id': 'ours', 'observation_error': 'TimeoutException'}
    current = deepcopy(previous)
    if change == 'recovered':
        current.pop('observation_error')
        current['remote'] = {}
    elif change == 'first_error':
        previous.pop('observation_error')
    elif change == 'replacement':
        current['sandbox_id'] = 'new'
    elif change == 'unidentified':
        previous.pop('sandbox_id')
        current.pop('sandbox_id')
    warnings = recorder.stall_warnings(previous, current)
    assert bool(warnings) == expected
    if expected:
        assert warnings[0]['code'] == 'repeated_sandbox_observation_error'
        assert 'Do not restart automatically' in warnings[0]['action']
        assert 'TimeoutException' not in json.dumps(warnings)


def test_traced_child_stall_requires_two_matching_observations():
    current = traced_observation()
    assert recorder.stall_warnings(None, current) == []
    warnings = recorder.stall_warnings(deepcopy(current), current)
    assert len(warnings) == 1
    assert warnings[0]['code'] == 'suspected_traced_child_stall'
    assert warnings[0]['child_pid'] == '12'


@pytest.mark.parametrize('change', [None, 'bytes', 'mtime_ns', 'inode', 'phase', 'young', 'missing'])
def test_verifier_quiet_log_is_only_an_inspection_warning(change):
    previous = {'sandbox_id': 'ours', 'phase': 'verification_or_finalization',
                'remote': {'verifier_log': {'bytes': 123, 'mtime_ns': 100, 'inode': 1, 'age_s': 901}}}
    current = deepcopy(previous)
    if change in ('bytes', 'mtime_ns', 'inode'):
        current['remote']['verifier_log'][change] += 1
    elif change == 'phase':
        current['phase'] = 'agent_or_setup'
    elif change == 'young':
        current['remote']['verifier_log']['age_s'] = 899
    elif change == 'missing':
        current['remote'] = {}
    warnings = recorder.stall_warnings(previous, current)
    assert [w['code'] for w in warnings] == (['verifier_log_unchanged'] if change is None else [])


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
