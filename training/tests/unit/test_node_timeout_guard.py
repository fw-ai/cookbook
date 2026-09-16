import copy
import os
import signal
from unittest.mock import patch

import pytest

from training.examples.rl.harbor.recipes.terminal_bench import node_timeout_guard as guard


@pytest.fixture
def evidence(tmp_path):
    hz = os.sysconf('SC_CLK_TCK')
    (tmp_path / 'uptime').write_text('1000 0')
    chain = []
    processes = []
    for pid, name in enumerate(('node', 'timeout', 'bash', 'opencode'), 10):
        p = tmp_path / str(pid)
        p.mkdir()
        stat = ['0'] * 20
        stat[0], stat[1], stat[19] = 'S', str(pid + 1), str(700 * hz)
        (p / 'stat').write_text(f'{pid} ({name}) ' + ' '.join(stat))
        (p / 'comm').write_text(name)
        chain.append({'pid': pid, 'start_ticks': str(700 * hz)})
        processes.append({'Pid': str(pid), 'PPid': str(pid + 1), 'Name': name,
                          'start_ticks': str(700 * hz), 'elapsed_s': 300,
                          'declared_timeout_s': 120 if name == 'timeout' else 0})
    (tmp_path / '11' / 'cmdline').write_bytes(b'timeout\0' + b'120\0node\0candidate.js\0')
    expected = {'chain': chain, 'duration_s': 120}
    record = {'sandbox_id': 'one', 'phase': 'agent_or_setup', 'remote': {
        'processes': processes, 'running_tools': [{'tool': 'bash', 'start_ms': 1}]}}
    return tmp_path, expected, record


def test_positive_and_remote_serialization(evidence):
    root, expected, record = evidence
    assert guard.overdue_node(expected, root)
    assert guard.recovery_candidates(record, record) == [expected]
    command = guard.recovery_command(expected)
    assert 'candidate.js' not in command
    compile(command.split('\n', 1)[1].rsplit('\nRECOVER', 1)[0], '<remote>', 'exec')


def test_explicit_sigint_timeout_preserves_child_only_revalidation(evidence):
    root, expected, _ = evidence
    (root / '11' / 'cmdline').write_bytes(b'timeout\0-s\0INT\0' + b'120\0node\0candidate.js\0')
    assert guard.overdue_node(expected, root)
    with patch.object(guard.os, 'pidfd_open', return_value=99), \
            patch.object(guard.signal, 'pidfd_send_signal') as sent, patch.object(guard.os, 'close'):
        assert guard.signal_overdue_node(expected, root)['action'] == 'SIGKILL'
        sent.assert_called_once_with(99, signal.SIGKILL)
    (root / 'uptime').write_text('849 0')
    assert not guard.overdue_node(expected, root)


@pytest.mark.parametrize('change', ['sandbox', 'phase', 'tool', 'child', 'timer', 'shell', 'agent', 'parent', 'deadline'])
def test_candidate_rejects_changed_evidence(evidence, change):
    _, _, before = evidence
    after = copy.deepcopy(before)
    if change == 'sandbox':
        after['sandbox_id'] = 'different'
    elif change == 'phase':
        after['phase'] = 'verification_or_finalization'
    elif change == 'tool':
        after['remote']['running_tools'][0]['start_ms'] = 2
    elif change == 'parent':
        after['remote']['processes'][0]['PPid'] = '99'
    elif change == 'deadline':
        after['remote']['processes'][1]['elapsed_s'] = 149
    else:
        after['remote']['processes'][{'child': 0, 'timer': 1, 'shell': 2, 'agent': 3}[change]]['start_ticks'] = '1'
    assert guard.recovery_candidates(before, after) == []


@pytest.mark.parametrize('args', [b'timeout\0--preserve-status\0node', b'timeout\0nan\0node',
                                  b'timeout\0' + b'0\0node', b'timeout\0' + b'120\0python', b'timeout',
                                  b'timeout\0-s\0TERM\0' + b'120\0node',
                                  b'timeout\0-s\0INT\0--foreground\0' + b'120\0node',
                                  b'timeout\0-s\0INT',
                                  b'timeout\0-s\0INT\0' + b'120\0python'])
def test_rejects_unknown_command(evidence, args):
    root, expected, _ = evidence
    (root / '11' / 'cmdline').write_bytes(args)
    assert not guard.overdue_node(expected, root)


@pytest.mark.parametrize('index', range(4))
def test_remote_checks_all_start_times(evidence, index):
    root, expected, _ = evidence
    expected['chain'][index]['start_ticks'] = '1'
    assert not guard.overdue_node(expected, root)


def test_pidfd_targets_child_only(evidence):
    root, expected, _ = evidence
    with patch.object(guard.os, 'pidfd_open', return_value=99) as opened, \
            patch.object(guard.signal, 'pidfd_send_signal') as sent, patch.object(guard.os, 'close') as closed:
        assert guard.signal_overdue_node(expected, root)['action'] == 'SIGKILL'
        opened.assert_called_once_with(10)
        sent.assert_called_once_with(99, signal.SIGKILL)
        closed.assert_called_once_with(99)


def test_revalidation_prevents_signal(evidence):
    root, expected, _ = evidence
    (root / 'uptime').write_text('800 0')
    with patch.object(guard.os, 'pidfd_open', return_value=99), \
            patch.object(guard.signal, 'pidfd_send_signal') as sent, patch.object(guard.os, 'close'):
        assert guard.signal_overdue_node(expected, root)['action'] == 'none'
        sent.assert_not_called()


@pytest.mark.parametrize('change', ['parent', 'name', 'duration', 'missing'])
def test_remote_rejects_other_evidence_changes(evidence, change):
    root, expected, _ = evidence
    if change == 'parent':
        path = root / '10' / 'stat'
        path.write_text(path.read_text().replace('S 11 ', 'S 99 '))
    elif change == 'name':
        (root / '13' / 'comm').write_text('pytest')
    elif change == 'duration':
        expected['duration_s'] = 60
    else:
        (root / '10' / 'stat').unlink()
    with patch.object(guard.os, 'pidfd_open', return_value=99), \
            patch.object(guard.signal, 'pidfd_send_signal') as sent, patch.object(guard.os, 'close'):
        assert guard.signal_overdue_node(expected, root)['action'] == 'none'
        sent.assert_not_called()
