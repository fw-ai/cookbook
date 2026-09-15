"""Opt-in recovery for an OpenCode root search blocked on kernel streams.

This is not a general timeout or slow-task policy. The caller must observe the
same grep/tool/PID twice without CPU progress, then revalidate inside the exact
trial sandbox. Only a confirmed child receives SIGTERM; never the agent, process
group, sandbox, or RL services. Fail closed if identity/evidence is unavailable.
"""
import inspect
import json
import os
from pathlib import Path
import re
import signal


def is_blocked_kernel_grep(expected, proc_root=Path('/proc')):
    """Recheck identity, no CPU progress, root cwd, stream FDs and wait channels."""
    process = proc_root / str(expected['pid'])
    stat = (process / 'stat').read_text().rsplit(')', 1)[1].split()
    parent = proc_root / str(expected['parent_pid'])
    parent_stat = (parent / 'stat').read_text().rsplit(')', 1)[1].split()
    cpu = (int(stat[11]) + int(stat[12])) / os.sysconf('SC_CLK_TCK')
    elapsed = float((proc_root / 'uptime').read_text().split()[0]) - int(stat[19]) / os.sysconf('SC_CLK_TCK')
    if not (
        stat[0] == 'S'
        and stat[19] == expected['start_ticks']
        and int(stat[1]) == expected['parent_pid']
        and parent_stat[19] == expected['parent_start_ticks']
        and cpu == expected['cpu_seconds']
        and elapsed >= 300
        and (process / 'comm').read_text().strip() == 'rg'
        and (parent / 'comm').read_text().strip() == 'opencode'
        and os.readlink(process / 'cwd') == '/'
    ):
        return False
    streams = []
    for descriptor in (process / 'fd').iterdir():
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            continue
        if target == '/proc/kmsg' or re.fullmatch(
            r'/sys/kernel/(?:debug/)?tracing/(?:per_cpu/cpu\d+/)?trace_pipe(?:_raw)?', target
        ):
            streams.append(target)
    waits = {(thread / 'wchan').read_text().strip() for thread in (process / 'task').iterdir()}
    return bool(streams) and bool(waits.intersection({'ring_buffer_wait', 'syslog_print'}))


def signal_blocked_kernel_grep(expected, proc_root=Path('/proc')):
    """Use a pidfd so a recycled numeric PID can never receive our signal."""
    descriptor = None
    try:
        descriptor = os.pidfd_open(expected['pid'])
        if not is_blocked_kernel_grep(expected, proc_root):
            return {'action': 'none', 'reason': 'evidence_changed'}
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        return {'action': 'SIGTERM', 'pid': expected['pid'], 'start_ticks': expected['start_ticks'],
                'reason': 'confirmed_kernel_stream_grep'}
    except (OSError, ValueError, KeyError, IndexError, AttributeError) as error:
        return {'action': 'none', 'reason': type(error).__name__}
    finally:
        if descriptor is not None:
            os.close(descriptor)


def recovery_candidates(previous, current):
    """Require two consistent sandbox/process/tool observations before probing."""
    if (not previous or not current.get('sandbox_id')
            or previous.get('sandbox_id') != current['sandbox_id']
            or current.get('phase') != 'agent_or_setup'
            or previous.get('phase') != current['phase']):
        return []
    before, after = previous.get('remote', {}), current.get('remote', {})
    prior_tools = {t.get('start_ms') for t in before.get('running_tools', []) if t.get('tool') == 'grep'}
    if not any(t.get('tool') == 'grep' and t.get('start_ms') is not None
               and t['start_ms'] in prior_tools and (t.get('elapsed_s') or 0) >= 300
               for t in after.get('running_tools', [])):
        return []
    old = {p['Pid']: p for p in before.get('processes', [])}
    new = {p['Pid']: p for p in after.get('processes', [])}
    candidates = []
    for process in new.values():
        prior = old.get(process['Pid'], {})
        parent = new.get(process.get('PPid'), {})
        if (process.get('Name') == 'rg' and parent.get('Name') == 'opencode'
                and process.get('start_ticks') is not None
                and process['start_ticks'] == prior.get('start_ticks')
                and parent.get('start_ticks') is not None
                and process.get('cpu_seconds') is not None
                and process['cpu_seconds'] == prior.get('cpu_seconds')
                and process.get('State', '').startswith('S ')):
            candidates.append({'pid': int(process['Pid']), 'start_ticks': process['start_ticks'],
                               'parent_pid': int(parent['Pid']), 'parent_start_ticks': parent['start_ticks'],
                               'cpu_seconds': process['cpu_seconds']})
    return candidates


def recovery_command(expected):
    """Serialize only the required numeric identity; never tool text or secrets."""
    payload = {'pid': int(expected['pid']), 'parent_pid': int(expected['parent_pid']),
               'start_ticks': str(int(expected['start_ticks'])),
               'parent_start_ticks': str(int(expected['parent_start_ticks'])),
               'cpu_seconds': float(expected['cpu_seconds'])}
    return ("python3 - <<'RECOVER'\nimport os, re, signal, json\nfrom pathlib import Path\n"
            + inspect.getsource(is_blocked_kernel_grep) + '\n'
            + inspect.getsource(signal_blocked_kernel_grep) + '\n'
            + f'print(json.dumps(signal_blocked_kernel_grep(json.loads({json.dumps(payload)!r}))))\nRECOVER')
