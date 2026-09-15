from copy import deepcopy
import os
import signal
from unittest.mock import patch

import pytest

from training.examples.rl.harbor.recipes.terminal_bench import kernel_core_guard as guard


@pytest.fixture
def evidence(tmp_path):
    hz = os.sysconf('SC_CLK_TCK')
    (tmp_path / 'uptime').write_text('1000 0')
    chain, processes = [], []
    for pid, name in enumerate(('grep', 'bash', 'opencode'), 10):
        p = tmp_path / str(pid)
        (p / 'fd').mkdir(parents=True)
        stat = ['R'] + ['0'] * 19
        stat[1], stat[19] = str(pid + 1), str(100 * hz)
        (p / 'stat').write_text(f'{pid} ({name}) ' + ' '.join(stat))
        (p / 'comm').write_text(name)
        chain.append({'pid': pid, 'start_ticks': stat[19]})
        processes.append({'Pid': str(pid), 'PPid': str(pid + 1), 'Name': name, 'start_ticks': stat[19]})
    (tmp_path / '10/fd/4').symlink_to('/proc/kcore')
    record = {'sandbox_id': 'exact', 'phase': 'agent_or_setup', 'remote': {
        'running_tools': [{'tool': 'bash', 'start_ms': 10, 'elapsed_s': 900}], 'processes': processes}}
    return tmp_path, {'chain': chain}, record


def test_positive_and_serialization(evidence):
    root, expected, record = evidence
    assert guard.is_kcore_grep(expected, root)
    assert guard.recovery_candidates(record, record) == [expected]
    command = guard.recovery_command(expected)
    compile(command.split('\n', 1)[1].rsplit('\nRECOVER', 1)[0], '<remote>', 'exec')


@pytest.mark.parametrize('index', range(3))
def test_recycled_chain_rejected(evidence, index):
    root, expected, record = evidence
    changed = deepcopy(record)
    changed['remote']['processes'][index]['start_ticks'] = '999'
    assert guard.recovery_candidates(record, changed) == []
    expected['chain'][index]['start_ticks'] = '999'
    assert not guard.is_kcore_grep(expected, root)


@pytest.mark.parametrize('target', ['/app/data', '/proc/kcore-copy', '/dev/zero', '/proc/kmsg'])
def test_only_exact_kcore(evidence, target):
    root, expected, _ = evidence
    (root / '10/fd/4').unlink()
    (root / '10/fd/4').symlink_to(target)
    assert not guard.is_kcore_grep(expected, root)


def test_short_run_or_wrong_sandbox_rejected(evidence):
    root, expected, record = evidence
    changed = deepcopy(record)
    changed['sandbox_id'] = 'other'
    assert guard.recovery_candidates(record, changed) == []
    (root / 'uptime').write_text('699 0')
    assert not guard.is_kcore_grep(expected, root)


@pytest.mark.parametrize('index', range(3))
def test_wrong_process_name_rejected(evidence, index):
    root, expected, _ = evidence
    (root / str(10 + index) / 'comm').write_text('other')
    assert not guard.is_kcore_grep(expected, root)


def test_phase_and_tool_change_rejected(evidence):
    _, _, record = evidence
    changed = deepcopy(record)
    changed['phase'] = 'verification_or_finalization'
    assert guard.recovery_candidates(record, changed) == []
    changed = deepcopy(record)
    changed['remote']['running_tools'][0]['start_ms'] = 11
    assert guard.recovery_candidates(record, changed) == []


def test_only_child_signalled_after_revalidation(evidence):
    root, expected, _ = evidence
    with patch.object(guard.os, 'pidfd_open', return_value=99) as opened, \
            patch.object(guard.signal, 'pidfd_send_signal') as sent, patch.object(guard.os, 'close') as closed:
        assert guard.signal_kcore_grep(expected, root)['action'] == 'SIGTERM'
        opened.assert_called_once_with(10)
        sent.assert_called_once_with(99, signal.SIGTERM)
        closed.assert_called_once_with(99)
        sent.reset_mock()
        expected['chain'][0]['start_ticks'] = '999'
        assert guard.signal_kcore_grep(expected, root)['action'] == 'none'
        sent.assert_not_called()
