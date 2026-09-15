"""Scoped recovery of a long-running grep of /proc/kcore, never ordinary files."""
import inspect
import json
import os
from pathlib import Path
import signal


def is_kcore_grep(expected, proc_root=Path('/proc')):
    """Validate grep -> bash -> OpenCode identity and the live kcore descriptor."""
    chain = expected['chain']
    if len(chain) != 3:
        return False
    for index, name in enumerate(('grep', 'bash', 'opencode')):
        item = chain[index]
        process = proc_root / str(item['pid'])
        stat = (process / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[19] != item['start_ticks'] or (process / 'comm').read_text().strip() != name:
            return False
        if index < 2 and int(stat[1]) != chain[index + 1]['pid']:
            return False
        if index == 0:
            elapsed = float((proc_root / 'uptime').read_text().split()[0]) - int(stat[19]) / os.sysconf('SC_CLK_TCK')
            if stat[0] not in ('R', 'S') or elapsed < 600:
                return False
    process = proc_root / str(chain[0]['pid'])
    for descriptor in (process / 'fd').iterdir():
        try:
            if os.readlink(descriptor) == '/proc/kcore':
                return True
        except FileNotFoundError:
            continue
    return False


def signal_kcore_grep(expected, proc_root=Path('/proc')):
    descriptor = None
    try:
        descriptor = os.pidfd_open(expected['chain'][0]['pid'])
        if not is_kcore_grep(expected, proc_root):
            return {'action': 'none', 'reason': 'evidence_changed'}
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        return {'action': 'SIGTERM', 'pid': expected['chain'][0]['pid'],
                'reason': 'confirmed_kcore_grep'}
    except (OSError, ValueError, KeyError, IndexError, AttributeError) as error:
        return {'action': 'none', 'reason': type(error).__name__}
    finally:
        if descriptor is not None:
            os.close(descriptor)


def recovery_candidates(previous, current):
    if (not previous or not current.get('sandbox_id')
            or previous.get('sandbox_id') != current['sandbox_id']
            or previous.get('phase') != current.get('phase')
            or current.get('phase') != 'agent_or_setup'):
        return []
    before, after = previous.get('remote', {}), current.get('remote', {})
    starts = {t.get('start_ms') for t in before.get('running_tools', []) if t.get('tool') == 'bash'}
    if not any(t.get('tool') == 'bash' and t.get('start_ms') is not None
               and t['start_ms'] in starts and (t.get('elapsed_s') or 0) >= 600
               for t in after.get('running_tools', [])):
        return []
    old = {p['Pid']: p for p in before.get('processes', [])}
    new = {p['Pid']: p for p in after.get('processes', [])}
    candidates = []
    for process in new.values():
        chain = [process]
        for _ in range(2):
            chain.append(new.get(chain[-1].get('PPid'), {}))
        if [p.get('Name') for p in chain] != ['grep', 'bash', 'opencode']:
            continue
        if all(p.get('start_ticks') is not None
               and p['start_ticks'] == old.get(p['Pid'], {}).get('start_ticks')
               and p.get('PPid') == old.get(p['Pid'], {}).get('PPid') for p in chain):
            candidates.append({'chain': [{'pid': int(p['Pid']), 'start_ticks': p['start_ticks']} for p in chain]})
    return candidates


def recovery_command(expected):
    payload = {'chain': [{'pid': int(p['pid']), 'start_ticks': str(int(p['start_ticks']))}
                         for p in expected['chain']]}
    return ("python3 - <<'RECOVER'\nimport os, signal, json\nfrom pathlib import Path\n"
            + inspect.getsource(is_kcore_grep) + '\n' + inspect.getsource(signal_kcore_grep)
            + f'\nprint(json.dumps(signal_kcore_grep(json.loads({json.dumps(payload)!r}))))\nRECOVER')
