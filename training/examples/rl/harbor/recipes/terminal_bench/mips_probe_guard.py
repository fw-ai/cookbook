"""Opt-in recovery for unbounded MIPS task probes authored by OpenCode.

The ``make-mips-interpreter`` task legitimately runs persistent programs, but
an exploratory shell probe must still return to the agent.  This guard handles
only the observed Node probes whose shell requested a short SIGINT shutdown or
piped an endless VM into ``tail``.  It revalidates the exact process ancestry
inside the exact E2B sandbox and signals only the Node leaf.  It never signals
OpenCode, the sandbox, the trainer, the rollout, or a process group.
"""
import inspect
import json
import os
from pathlib import Path
import re
import signal


TASK_PREFIX = 'harbor-opencode-make-mips-interpreter-'


def _chain(expected, proc_root):
    records = []
    for index, item in enumerate(expected['chain']):
        process = proc_root / str(item['pid'])
        stat = (process / 'stat').read_text().rsplit(')', 1)[1].split()
        name = (process / 'comm').read_text().strip()
        if stat[19] != item['start_ticks'] or stat[0] == 'Z' or name != item['name']:
            return []
        if index + 1 < len(expected['chain']) and int(stat[1]) != expected['chain'][index + 1]['pid']:
            return []
        records.append((process, stat))
    return records


def overdue_mips_node(expected, proc_root=Path('/proc')):
    """Confirm a known unbounded Node probe after its intended short window."""
    records = _chain(expected, proc_root)
    names = [item['name'] for item in expected['chain']]
    if not records or names[0] != 'node' or names[-1] != 'opencode' or any(
            name != 'bash' for name in names[1:-1]):
        return False
    child_args = records[0][0].joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    parent_args = records[1][0].joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    if not child_args or Path(os.fsdecode(child_args[0])).name != 'node' or len(parent_args) < 3:
        return False
    child = b' '.join(child_args)
    command = parent_args[-1]
    known_vm = (b'vm.js' in child or (b'node -e' in child and b'while(true)' in child))
    if not known_vm:
        return False
    elapsed = (float((proc_root / 'uptime').read_text().split()[0])
               - int(records[0][1][19]) / os.sysconf('SC_CLK_TCK'))
    # The agent explicitly requested a short run and SIGINT, but the synchronous
    # VM prevented Node from servicing the signal.  Preserve 30 seconds of grace.
    sleep = re.search(rb'(?:^|[;&|] *)sleep +([0-9]+(?:[.][0-9]+)?)', command)
    if b'kill -INT' in command and b'wait' in command and sleep:
        return elapsed > float(sleep[1]) + 30
    # The task's VM executes a persistent Doom/MIPS program.  A direct foreground
    # invocation (including output pipelines) has no natural EOF; explicit GNU
    # timeout wrappers are handled by node_timeout_guard instead.
    if b'vm.js' in command and b'timeout ' not in command:
        return elapsed > 60
    return False


def signal_overdue_mips_node(expected, proc_root=Path('/proc')):
    descriptor = None
    try:
        descriptor = os.pidfd_open(expected['chain'][0]['pid'])
        if not overdue_mips_node(expected, proc_root):
            return {'action': 'none', 'reason': 'evidence_changed'}
        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        return {'action': 'SIGKILL', 'pid': expected['chain'][0]['pid'],
                'start_ticks': expected['chain'][0]['start_ticks'],
                'reason': 'unbounded_make_mips_node_probe'}
    except (OSError, ValueError, KeyError, IndexError, AttributeError) as error:
        return {'action': 'none', 'reason': type(error).__name__}
    finally:
        if descriptor is not None:
            os.close(descriptor)


def recovery_candidates(previous, current):
    """Require the same task, tool and process ancestry on two observations."""
    if (not previous or not current.get('trial', '').startswith(TASK_PREFIX)
            or previous.get('sandbox_id') != current.get('sandbox_id')
            or previous.get('phase') != current.get('phase')
            or current.get('phase') != 'agent_or_setup'):
        return []
    before, after = previous.get('remote', {}), current.get('remote', {})
    starts = {tool.get('start_ms') for tool in before.get('running_tools', [])
              if tool.get('tool') == 'bash' and tool.get('start_ms') is not None}
    if not any(tool.get('tool') == 'bash' and tool.get('start_ms') in starts
               for tool in after.get('running_tools', [])):
        return []
    old = {process['Pid']: process for process in before.get('processes', [])}
    new = {process['Pid']: process for process in after.get('processes', [])}
    candidates = []
    for process in new.values():
        if process.get('Name') != 'node' or (process.get('elapsed_s') or 0) <= 30:
            continue
        chain = [process]
        for _ in range(3):
            parent = new.get(chain[-1].get('PPid'))
            if not parent:
                break
            chain.append(parent)
            if parent.get('Name') == 'opencode':
                break
        names = [item.get('Name') for item in chain]
        if names[-1:] != ['opencode'] or any(name != 'bash' for name in names[1:-1]):
            continue
        if not all(item.get('start_ticks') is not None
                   and item['start_ticks'] == old.get(item['Pid'], {}).get('start_ticks')
                   and item.get('PPid') == old.get(item['Pid'], {}).get('PPid')
                   for item in chain):
            continue
        candidates.append({'chain': [
            {'pid': int(item['Pid']), 'start_ticks': item['start_ticks'], 'name': item['Name']}
            for item in chain]})
    return candidates


def recovery_command(expected):
    """Send identity evidence only; command text remains inside the sandbox."""
    payload = {'chain': [{'pid': int(item['pid']),
                          'start_ticks': str(int(item['start_ticks'])),
                          'name': item['name']} for item in expected['chain']]}
    return ("python3 - <<'RECOVER'\nimport os, re, signal, json\nfrom pathlib import Path\n"
            + inspect.getsource(_chain) + '\n' + inspect.getsource(overdue_mips_node) + '\n'
            + inspect.getsource(signal_overdue_mips_node) + '\n'
            + f'print(json.dumps(signal_overdue_mips_node(json.loads({json.dumps(payload)!r}))))\nRECOVER')
