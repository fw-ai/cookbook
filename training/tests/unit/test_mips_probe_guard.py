import copy
import os
import signal
from unittest.mock import patch

import pytest

from training.examples.rl.harbor.recipes.terminal_bench import mips_probe_guard as guard


@pytest.fixture
def evidence(tmp_path):
    hz = os.sysconf('SC_CLK_TCK')
    (tmp_path / 'uptime').write_text('1000 0')
    chain = []
    processes = []
    for pid, name in zip(range(10, 13), ('node', 'bash', 'opencode'), strict=True):
        process = tmp_path / str(pid)
        process.mkdir()
        stat = ['0'] * 20
        stat[0], stat[1], stat[19] = 'S', str(pid + 1), str(900 * hz)
        (process / 'stat').write_text(f'{pid} ({name}) ' + ' '.join(stat))
        (process / 'comm').write_text(name)
        (process / 'cmdline').write_bytes(name.encode() + b'\0')
        chain.append({'pid': pid, 'start_ticks': str(900 * hz), 'name': name})
        processes.append({'Pid': str(pid), 'PPid': str(pid + 1), 'Name': name,
                          'start_ticks': str(900 * hz), 'elapsed_s': 100})
    (tmp_path / '10' / 'cmdline').write_bytes(b'node\0/app/vm.js\0/app/doomgeneric_mips\0')
    (tmp_path / '11' / 'cmdline').write_bytes(
        b'bash\0-c\0node /app/vm.js /app/doomgeneric_mips | tail -3\0')
    expected = {'chain': chain}
    record = {'trial': guard.TASK_PREFIX + '0-1-0-id', 'sandbox_id': 'one',
              'phase': 'agent_or_setup', 'remote': {'processes': processes,
              'running_tools': [{'tool': 'bash', 'start_ms': 1}]}}
    return tmp_path, expected, record


def test_tail_pipeline_is_recovered(evidence):
    root, expected, record = evidence
    assert guard.overdue_mips_node(expected, root)
    assert guard.recovery_candidates(record, record) == [expected]
    command = guard.recovery_command(expected)
    assert 'doomgeneric_mips' not in command
    compile(command.split('\n', 1)[1].rsplit('\nRECOVER', 1)[0], '<remote>', 'exec')


def test_short_sigint_probe_uses_sleep_plus_grace(evidence):
    root, expected, _ = evidence
    (root / '10' / 'cmdline').write_bytes(b'node\0vm.js\0')
    (root / '11' / 'cmdline').write_bytes(
        b'bash\0-c\0node vm.js & P=$!; sleep 8; kill -INT $P; wait $P\0')
    assert guard.overdue_mips_node(expected, root)
    (root / 'uptime').write_text('930 0')
    assert not guard.overdue_mips_node(expected, root)


def test_direct_foreground_vm_is_recovered(evidence):
    root, expected, _ = evidence
    (root / '11' / 'cmdline').write_bytes(
        b'bash\0-c\0node /app/vm.js >/tmp/opencode/exit.log 2>&1; echo code=$?\0')
    assert guard.overdue_mips_node(expected, root)


@pytest.mark.parametrize('change', ['task', 'sandbox', 'phase', 'tool', 'child', 'parent'])
def test_candidate_fails_closed(evidence, change):
    _, _, before = evidence
    after = copy.deepcopy(before)
    if change == 'task':
        after['trial'] = 'harbor-opencode-other-task-0'
    elif change == 'sandbox':
        after['sandbox_id'] = 'other'
    elif change == 'phase':
        after['phase'] = 'verification_or_finalization'
    elif change == 'tool':
        after['remote']['running_tools'][0]['start_ms'] = 2
    elif change == 'child':
        after['remote']['processes'][0]['start_ticks'] = '1'
    else:
        after['remote']['processes'][0]['PPid'] = '99'
    assert guard.recovery_candidates(before, after) == []


def test_only_exact_leaf_is_signaled(evidence):
    root, expected, _ = evidence
    with patch.object(guard.os, 'pidfd_open', return_value=99) as opened, \
            patch.object(guard.signal, 'pidfd_send_signal') as sent, \
            patch.object(guard.os, 'close') as closed:
        assert guard.signal_overdue_mips_node(expected, root)['action'] == 'SIGKILL'
        opened.assert_called_once_with(10)
        sent.assert_called_once_with(99, signal.SIGKILL)
        closed.assert_called_once_with(99)


@pytest.mark.parametrize('child,parent', [
    (b'node\0server.js\0', b'bash\0-c\0node server.js | tail -3\0'),
    (b'python\0/app/vm.js\0', b'bash\0-c\0python /app/vm.js | tail -3\0'),
    (b'node\0/app/vm.js\0', b'bash\0-c\0timeout 120 node /app/vm.js\0'),
])
def test_unknown_commands_are_not_recovered(evidence, child, parent):
    root, expected, _ = evidence
    (root / '10' / 'cmdline').write_bytes(child)
    (root / '11' / 'cmdline').write_bytes(parent)
    assert not guard.overdue_mips_node(expected, root)
