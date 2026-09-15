"""Opt-in recovery of Node children exceeding a model-authored GNU timeout.

This honors an existing command deadline, not a sampling-time target. Only the
exact Node child may receive SIGKILL, after 30 seconds of termination grace.
Unknown timeout options, verifier processes and changed identities fail closed.
Never signal the agent, a process group, or the timeout parent itself.
"""
import inspect
import json
import os
from pathlib import Path
import re
import signal


def overdue_node(expected, proc_root=Path('/proc')):
    """Revalidate the entire Node -> timeout -> bash -> OpenCode ancestry."""
    records = []
    for item, name in zip(expected['chain'], ('node', 'timeout', 'bash', 'opencode'), strict=True):
        path = proc_root / str(item['pid'])
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if (stat[19] != item['start_ticks'] or stat[0] == 'Z'
                or (path / 'comm').read_text().strip() != name):
            return False
        records.append((path, stat))
    if any(int(records[i][1][1]) != expected['chain'][i + 1]['pid'] for i in range(3)):
        return False
    args = (records[1][0] / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    match = re.fullmatch(rb'([0-9]+(?:[.][0-9]+)?)([smhd]?)', args[1]) if len(args) >= 3 else None
    if not match or Path(os.fsdecode(args[2])).name != 'node':
        return False
    duration = float(match[1]) * {b'': 1, b's': 1, b'm': 60, b'h': 3600, b'd': 86400}[match[2]]
    elapsed = float((proc_root / 'uptime').read_text().split()[0]) - int(records[1][1][19]) / os.sysconf('SC_CLK_TCK')
    return 0 < duration == expected['duration_s'] and elapsed > duration + 30


def signal_overdue_node(expected, proc_root=Path('/proc')):
    descriptor = None
    try:
        descriptor = os.pidfd_open(expected['chain'][0]['pid'])
        if not overdue_node(expected, proc_root):
            return {'action': 'none', 'reason': 'evidence_changed'}
        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        return {'action': 'SIGKILL', 'pid': expected['chain'][0]['pid'],
                'start_ticks': expected['chain'][0]['start_ticks'],
                'requested_timeout_s': expected['duration_s'],
                'reason': 'node_exceeded_own_command_deadline'}
    except (OSError, ValueError, KeyError, IndexError, AttributeError) as error:
        return {'action': 'none', 'reason': type(error).__name__}
    finally:
        if descriptor is not None:
            os.close(descriptor)


def recovery_candidates(previous, current):
    """Require the same live tool and four process identities on two polls."""
    if (not previous or not current.get('sandbox_id')
            or previous.get('sandbox_id') != current['sandbox_id']
            or previous.get('phase') != 'agent_or_setup'
            or current.get('phase') != 'agent_or_setup'):
        return []
    before, after = previous.get('remote', {}), current.get('remote', {})
    old_tools = {t['start_ms'] for t in before.get('running_tools', [])
                 if t.get('tool') == 'bash' and t.get('start_ms') is not None}
    if not any(t.get('tool') == 'bash' and t.get('start_ms') in old_tools
               for t in after.get('running_tools', [])):
        return []
    old = {p['Pid']: p for p in before.get('processes', [])}
    new = {p['Pid']: p for p in after.get('processes', [])}
    result = []
    for child in new.values():
        chain = [child]
        for _ in range(3):
            chain.append(new.get(chain[-1].get('PPid'), {}))
        if [p.get('Name') for p in chain] != ['node', 'timeout', 'bash', 'opencode']:
            continue
        if not all(p.get('start_ticks') is not None
                   and p['start_ticks'] == old.get(p['Pid'], {}).get('start_ticks')
                   and p.get('PPid') == old.get(p['Pid'], {}).get('PPid') for p in chain):
            continue
        duration = chain[1].get('declared_timeout_s', 0)
        if duration > 0 and chain[1].get('elapsed_s', 0) > duration + 30:
            result.append({'chain': [{'pid': int(p['Pid']), 'start_ticks': p['start_ticks']} for p in chain],
                           'duration_s': duration})
    return result


def recovery_command(expected):
    """Send numeric identity only; model commands and credentials stay remote."""
    payload = {'chain': [{'pid': int(p['pid']), 'start_ticks': str(int(p['start_ticks']))}
                         for p in expected['chain']], 'duration_s': float(expected['duration_s'])}
    return ("python3 - <<'RECOVER'\nimport os, re, signal, json\nfrom pathlib import Path\n"
            + inspect.getsource(overdue_node) + '\n' + inspect.getsource(signal_overdue_node) + '\n'
            + f'print(json.dumps(signal_overdue_node(json.loads({json.dumps(payload)!r}))))\nRECOVER')
