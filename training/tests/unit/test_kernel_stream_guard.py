from copy import deepcopy
import json
import os
import signal

import pytest

from training.examples.rl.harbor.recipes.terminal_bench import kernel_stream_guard as guard


def process_stat(pid, parent, start, cpu_ticks=0, state='S'):
    fields = [state] + ['0'] * 19
    fields[1], fields[11], fields[19] = str(parent), str(cpu_ticks), str(start)
    return f'{pid} (process) ' + ' '.join(fields)


@pytest.fixture
def blocked(tmp_path):
    proc = tmp_path / 'proc'
    process, parent = proc / '1727', proc / '1694'
    (process / 'fd').mkdir(parents=True)
    (process / 'task' / '1728').mkdir(parents=True)
    parent.mkdir()
    (proc / 'uptime').write_text('1000 0')
    (process / 'stat').write_text(process_stat(1727, 1694, 100, 10))
    (parent / 'stat').write_text(process_stat(1694, 1, 50))
    (process / 'comm').write_text('rg\n')
    (parent / 'comm').write_text('opencode\n')
    (process / 'cwd').symlink_to('/')
    (process / 'fd' / '3').symlink_to('/proc/kmsg')
    (process / 'task' / '1728' / 'wchan').write_text('syslog_print')
    expected = {'pid': 1727, 'parent_pid': 1694, 'start_ticks': '100',
                'parent_start_ticks': '50', 'cpu_seconds': 10 / os.sysconf('SC_CLK_TCK')}
    return proc, process, parent, expected


def test_confirmed_stream_requires_both_descriptor_and_blocked_wait(blocked):
    proc, process, _, expected = blocked
    assert guard.is_blocked_kernel_grep(expected, proc)
    (process / 'task' / '1728' / 'wchan').write_text('futex_wait_queue')
    assert not guard.is_blocked_kernel_grep(expected, proc)


@pytest.mark.parametrize('change', ['pid_reuse', 'parent_reuse', 'cpu_progress', 'running',
                                   'different_parent', 'not_rg', 'not_opencode', 'cwd',
                                   'ordinary_fd', 'young'])
def test_guard_rejects_changed_or_insufficient_evidence(blocked, change):
    proc, process, parent, expected = blocked
    if change == 'pid_reuse':
        expected['start_ticks'] = '999'
    elif change == 'parent_reuse':
        expected['parent_start_ticks'] = '999'
    elif change == 'cpu_progress':
        expected['cpu_seconds'] = 0
    elif change == 'running':
        (process / 'stat').write_text(process_stat(1727, 1694, 100, 10, 'R'))
    elif change == 'different_parent':
        (process / 'stat').write_text(process_stat(1727, 1, 100, 10))
    elif change == 'not_rg':
        (process / 'comm').write_text('python')
    elif change == 'not_opencode':
        (parent / 'comm').write_text('bash')
    elif change == 'cwd':
        (process / 'cwd').unlink()
        (process / 'cwd').symlink_to('/app')
    elif change == 'ordinary_fd':
        (process / 'fd' / '3').unlink()
        (process / 'fd' / '3').symlink_to('/app/log.txt')
    else:
        (proc / 'uptime').write_text('10 0')
    assert not guard.is_blocked_kernel_grep(expected, proc)


def test_signal_uses_pidfd_only_after_fresh_validation(blocked, monkeypatch):
    proc, _, _, expected = blocked
    calls = []
    monkeypatch.setattr(os, 'pidfd_open', lambda pid: calls.append(('open', pid)) or 555)
    monkeypatch.setattr(os, 'close', lambda fd: calls.append(('close', fd)))
    monkeypatch.setattr(signal, 'pidfd_send_signal', lambda fd, sig: calls.append(('signal', fd, sig)))
    assert guard.signal_blocked_kernel_grep(expected, proc)['action'] == 'SIGTERM'
    assert calls == [('open', 1727), ('signal', 555, signal.SIGTERM), ('close', 555)]
    calls.clear()
    expected['start_ticks'] = '999'
    assert guard.signal_blocked_kernel_grep(expected, proc)['action'] == 'none'
    assert calls == [('open', 1727), ('close', 555)]


def test_vanished_process_fails_closed(blocked, monkeypatch):
    def absent(_):
        raise ProcessLookupError()
    monkeypatch.setattr(os, 'pidfd_open', absent)
    assert guard.signal_blocked_kernel_grep(blocked[-1], blocked[0]) == {
        'action': 'none', 'reason': 'ProcessLookupError'}


def observation():
    return {'phase': 'agent_or_setup', 'sandbox_id': 'exact-sandbox', 'remote': {
        'running_tools': [{'tool': 'grep', 'start_ms': 1234, 'elapsed_s': 400}],
        'processes': [{'Pid': '1727', 'PPid': '1694', 'Name': 'rg', 'State': 'S (sleeping)',
                       'start_ticks': '100', 'cpu_seconds': 0.1},
                      {'Pid': '1694', 'Name': 'opencode', 'start_ticks': '50'}]}}


@pytest.mark.parametrize('change', [None, 'sandbox', 'pid', 'cpu', 'tool', 'age', 'phase', 'missing'])
def test_recovery_candidates_require_two_consistent_observations(change):
    previous = observation()
    current = deepcopy(previous)
    if change == 'sandbox':
        current['sandbox_id'] = 'other'
    elif change == 'pid':
        current['remote']['processes'][0]['start_ticks'] = '999'
    elif change == 'cpu':
        current['remote']['processes'][0]['cpu_seconds'] += 0.01
    elif change == 'tool':
        current['remote']['running_tools'][0]['start_ms'] += 1
    elif change == 'age':
        current['remote']['running_tools'][0]['elapsed_s'] = 299
    elif change == 'phase':
        current['phase'] = 'verification_or_finalization'
    elif change == 'missing':
        previous = None
    assert bool(guard.recovery_candidates(previous, current)) == (change is None)


def test_remote_command_serializes_only_numeric_identity(blocked):
    expected = {**blocked[-1], 'private': 'secret-tool-pattern'}
    command = guard.recovery_command(expected)
    assert 'secret-tool-pattern' not in command
    code = command.split("python3 - <<'RECOVER'\n", 1)[1].rsplit('\nRECOVER', 1)[0]
    compile(code, '<remote recovery>', 'exec')
    # Inspect the exact generated function definitions without signalling.
    namespace = {}
    exec(code.rsplit('print(', 1)[0], namespace)
    assert namespace['is_blocked_kernel_grep'](expected, blocked[0])
    assert json.loads(json.dumps(expected)) == expected
