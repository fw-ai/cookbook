#!/usr/bin/env python3
"""E2B progress sampling every three minutes; read-only unless explicitly enabled.

Run with --pid LIVE_HARNESS_PID; add --once for a single observation. Reads
pending trials from health.jsonl and connects only to exact trial session IDs.
Records activity ages and process states, never tool inputs, model text or
credentials. By default it never signals processes. Opt-in kernel-stream grep
recovery terminates only a revalidated stuck search child, never the agent.
Opt-in kcore recovery terminates only a long-running grep reading /proc/kcore.
Opt-in overdue-Node recovery enforces only an existing model-authored GNU
timeout plus a 30-second termination grace; it never caps total sampling time.
Opt-in MIPS-probe recovery handles only revalidated unbounded Node VM probes
or frame polling left behind after a bounded VM exits inside
``make-mips-interpreter`` and signals only the exact stuck leaf.
It never retries samples or changes timeouts.
An old activity timestamp is a reason to inspect, not proof of a failed sample.
Assistant-message timing and log growth help distinguish a long model turn
from a tool wait; neither alone proves upstream request or GPU activity.
Exits if the original harness PID exits or is reused. Requires E2B_API_KEY.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from e2b import Sandbox, SandboxQuery

from training.examples.rl.harbor.recipes.terminal_bench import node_timeout_guard
from training.examples.rl.harbor.recipes.terminal_bench import kernel_core_guard
from training.examples.rl.harbor.recipes.terminal_bench import mips_probe_guard
from training.examples.rl.harbor.recipes.terminal_bench import overvalidation_guard
from training.examples.rl.harbor.recipes.terminal_bench import verifier_deadlock_guard

from training.examples.rl.harbor.recipes.terminal_bench.kernel_stream_guard import (
    recovery_candidates,
    recovery_command,
)


REMOTE = """python3 - <<'REMOTE'
import json, os, pathlib, re, sqlite3, time
p = pathlib.Path('/logs/agent/opencode/xdg-data/opencode/opencode.db')
out = {'observed_at_ms': time.time() * 1000}
try:
    memory = {}
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable'):
            memory[key] = int(value.split()[0]) * 1024
    out['guest_memory'] = memory
except (OSError, ValueError):
    pass
try:
    for line in pathlib.Path('/proc/vmstat').read_text().splitlines():
        key, value = line.split()
        if key == 'oom_kill':
            out['guest_oom_kills'] = int(value)
            break
except (OSError, ValueError):
    pass
root = pathlib.Path('/tmp/fireworks-tito-sidecar')
try:
    # Never read endpoint.json/spec.json: both can contain credentials.
    boot = {'endpoint_present': (root / 'endpoint.json').is_file(),
            'pid_file_present': (root / 'sidecar.pid').is_file()}
    if boot['pid_file_present']:
        raw_pid = (root / 'sidecar.pid').read_text().strip()
        if raw_pid.isdigit() and int(raw_pid) > 0:
            pid = int(raw_pid)
            boot['pid'] = pid
            try:
                fields = pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                boot['process_state'] = fields[0]
                boot['start_ticks'] = fields[19]
                boot['cpu_seconds'] = (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
            except FileNotFoundError:
                boot['process_state'] = 'missing'
    out['sidecar_readiness'] = boot
except (OSError, ValueError, IndexError):
    out['sidecar_readiness'] = {'observation_error': 'unavailable'}
if p.exists():
    with sqlite3.connect('file:' + str(p) + '?mode=ro', uri=True) as c:
        rows = c.execute('select time_updated,data from part order by time_updated desc limit 3').fetchall()
        active = c.execute("select id,data from part where json_extract(data, '$.state.status')='running'").fetchall()
        message = c.execute("select time_updated,data from message where json_extract(data, '$.role')='assistant' order by time_updated desc limit 1").fetchone()
    if message:
        updated, raw = message
        d = json.loads(raw)
        timing = d.get('time', {})
        created, completed = timing.get('created'), timing.get('completed')
        out['latest_assistant'] = {
            'age_s': round(time.time()-updated/1000, 1),
            'created_ms': created, 'completed_ms': completed,
            'duration_s': (completed-created)/1000 if created is not None and completed is not None else None,
            'finish': d.get('finish'),
        }
    out['running_tools'] = []
    for part_id, raw in active:
        d = json.loads(raw)
        state = d.get('state', {})
        start = state.get('time', {}).get('start')
        out['running_tools'].append({'part_id': part_id, 'tool': d.get('tool'), 'start_ms': start,
            'elapsed_s': round(time.time()-start/1000, 1) if start else None,
            'timeout_ms': state.get('input', {}).get('timeout')})
    out['activity'] = []
    for updated, raw in rows:
        d = json.loads(raw)
        out['activity'].append({'age_s': round(time.time()-updated/1000, 1),
            'type': d.get('type'), 'tool': d.get('tool'),
            'status': d.get('state', {}).get('status')})
p = pathlib.Path('/logs/agent/opencode.txt')
if p.exists():
    stat = p.stat()
    out['agent_log'] = {'bytes': stat.st_size, 'age_s': round(time.time()-stat.st_mtime, 1)}
p = pathlib.Path('/logs/verifier/test-stdout.txt')
if p.exists():
    stat = p.stat()
    out['verifier_log'] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
        'inode': stat.st_ino, 'age_s': round(time.time()-stat.st_mtime, 1)}
p = pathlib.Path('/tmp/fireworks-tito-opencode/agent-status')
if p.exists():
    value = p.read_text().strip()
    out['agent_exit'] = int(value) if value.lstrip('-').isdigit() else 'non-numeric'
out['processes'] = []
uptime = float(pathlib.Path('/proc/uptime').read_text().split()[0])
hz = os.sysconf('SC_CLK_TCK')
for p in pathlib.Path('/proc').glob('[0-9]*/status'):
    try:
        fields = dict(line.split(':', 1) for line in p.read_text().splitlines() if ':' in line)
        if int(fields['Pid']) > 1600 and fields['PPid'].strip() != '2':
            row = {k: fields[k].strip() for k in ('Name','Pid','PPid','State','TracerPid')}
            for key in ('VmRSS', 'VmPeak'):
                if key in fields:
                    row[key + '_bytes'] = int(fields[key].split()[0]) * 1024
            stat = (p.parent / 'stat').read_text().rsplit(')', 1)[1].split()
            row['cpu_seconds'] = (int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK')
            row['start_ticks'] = stat[19]
            row['elapsed_s'] = round(uptime-int(stat[19])/hz, 1)
            if row['Name'] == 'timeout':
                args = (p.parent / 'cmdline').read_bytes().split(bytes([0]))
                # Recognize plain timeout and the observed `-s INT` form.
                # Unknown option forms are omitted, never guessed. No command
                # text or child arguments are included in the observation.
                if args[1:3] == [b'-s', b'INT']:
                    args = args[:1] + args[3:]
                duration = re.fullmatch(rb'([0-9]+(?:[.][0-9]+)?)([smhd]?)', args[1]) if len(args) > 2 else None
                if duration:
                    row['declared_timeout_s'] = float(duration[1]) * {b'':1,b's':1,b'm':60,b'h':3600,b'd':86400}[duration[2]]
            out['processes'].append(row)
    except (OSError, KeyError, ValueError):
        pass
print(json.dumps(out))
REMOTE"""


def identity(pid):
    try:
        # Fields after the comm field begin at field 3; starttime is field 22.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def timeout_warnings(current):
    """An overdue inner timeout is an inspection trigger, not a kill policy."""
    warnings = []
    for process in current.get('remote', {}).get('processes', []):
        duration = process.get('declared_timeout_s', 0)
        elapsed = process.get('elapsed_s', 0)
        if process.get('Name') == 'timeout' and duration > 0 and elapsed > duration + 30:
            warnings.append({'code': 'inner_timeout_overrun', 'pid': process['Pid'],
                             'declared_timeout_s': duration, 'elapsed_s': elapsed,
                             'action': 'Inspect signal handling; SIGTERM may be ignored. Do not kill automatically'})
    return warnings


def search_wait_warnings(current):
    """Surface long grep waits early; elapsed time alone does not prove a hang."""
    return [
        {'code': 'long_grep_wait', 'elapsed_s': tool['elapsed_s'],
         'action': 'Inspect search cwd, worker wait channels and open descriptors for kernel streams; do not terminate automatically'}
        for tool in current.get('remote', {}).get('running_tools', [])
        if tool.get('tool') == 'grep' and (tool.get('elapsed_s') or 0) >= 300
    ]


def inspect_trial(root, trial):
    name = trial['trial']
    result = {'trial': name, 'phase': trial['phase'],
              'trial_age_s': trial.get('trial_age_s')}
    result_path = root / 'trials' / name / 'result.json'
    if result_path.exists():
        return {**result, 'observation': 'already_finalized'}
    try:
        session = name + '__env'
        candidates = Sandbox.list(query=SandboxQuery(metadata={'session_id': session}), limit=2).next_items()
        candidates = [s for s in candidates if s.metadata.get('session_id') == session]
        if len(candidates) != 1:
            if result_path.exists():
                return {**result, 'observation': 'already_finalized'}
            return {**result, 'matching_sandboxes': len(candidates)}
        sandbox = Sandbox.connect(candidates[0].sandbox_id)
        result['sandbox_id'] = candidates[0].sandbox_id
        command = sandbox.commands.run(REMOTE, timeout=15)
        result['remote'] = json.loads(command.stdout)
    except Exception as error:
        # Normal finalization deletes the ephemeral sandbox. Recheck after
        # remote I/O so that this race is not reported as a live failure.
        # A finalized result is not necessarily a successful/scored result;
        # scored_exception_inventory audits its contents separately.
        if result_path.exists():
            return {**result, 'observation': 'already_finalized'}
        result['observation_error'] = type(error).__name__
    return result


def retain_tool_start(previous, current):
    """Keep a lower-bound tool age when OpenCode metadata resets start_ms.

    Match the trial, sandbox and unique part ID, never the tool name. Raw
    timestamps remain intact. Missing history (including observer restart)
    can underestimate duration; this diagnostic never changes a deadline or
    the process-based recovery guards.
    """
    def valid(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    remote = current.get('remote', {})
    now = remote.get('observed_at_ms')
    if not valid(now):
        return
    previous = previous or {}
    same_trial = (current.get('trial') is not None
                  and current.get('trial') == previous.get('trial')
                  and current.get('sandbox_id') is not None
                  and current.get('sandbox_id') == previous.get('sandbox_id'))
    old = {tool.get('part_id'): tool for tool in
           previous.get('remote', {}).get('running_tools', [])} if same_trial else {}
    for tool in remote.get('running_tools', []):
        part_id, start = tool.get('part_id'), tool.get('start_ms')
        if not part_id or not valid(start) or start > now:
            continue
        prior = old.get(part_id, {}).get('earliest_recorded_start_ms')
        earliest = min(start, prior) if valid(prior) else start
        tool['earliest_recorded_start_ms'] = earliest
        tool['observed_elapsed_s_lower_bound'] = round((now - earliest) / 1000, 1)


def sampling_budget_warnings(current):
    """Inspection thresholds only; never change deadlines or discard samples."""
    if current.get('observation') == 'already_finalized':
        return []
    warnings = []
    if (current.get('trial_age_s') or 0) >= 1800:
        warnings.append({'code': 'sampling_target_exceeded',
                         'elapsed_s': current['trial_age_s'],
                         'action': 'Inspect current phase and critical path; 30 minutes is a target, not a termination deadline'})
    if current.get('phase') == 'agent_or_setup':
        for tool in current.get('remote', {}).get('running_tools', []):
            elapsed = max(tool.get('elapsed_s') or 0,
                          tool.get('observed_elapsed_s_lower_bound') or 0)
            if elapsed >= 600:
                warnings.append({'code': 'long_tool_call', 'tool': tool.get('tool'),
                                 'elapsed_s': elapsed,
                                 'action': 'Inspect process progress, explicit deadline and output handling; CPU activity or quiet output alone does not prove a hang'})
    return warnings


def memory_warnings(current):
    """Guest available-memory warning, not an OOM diagnosis or kill policy.

    Use MemAvailable rather than summing process RSS (shared pages overlap).
    This describes the E2B guest, not trainer GPU memory or a cgroup limit.
    """
    memory = current.get('remote', {}).get('guest_memory', {})
    total, available = memory.get('MemTotal'), memory.get('MemAvailable')
    if (isinstance(total, int) and isinstance(available, int)
            and total > 0 and 0 <= available <= total * 0.1):
        return [{'code': 'sandbox_memory_pressure',
                 'available_bytes': available, 'total_bytes': total,
                 'action': 'Inspect candidate-process RSS, progress and kernel OOM evidence; do not kill, retry or increase task resources automatically'}]
    return []


def sidecar_readiness_warnings(current):
    """Expose slow startup before the outer setup deadline; never signal/retry."""
    if (current.get('observation') == 'already_finalized'
            or current.get('phase') != 'agent_or_setup'
            or (current.get('trial_age_s') or 0) < 240):
        return []
    boot = current.get('remote', {}).get('sidecar_readiness', {})
    if boot.get('endpoint_present') is not False:
        return []
    return [{'code': 'sidecar_not_ready_after_four_minutes',
             'trial_age_s': current['trial_age_s'],
             'pid_file_present': boot.get('pid_file_present'),
             'process_state': boot.get('process_state'),
             'action': 'Inspect setup stage, sidecar process and sanitized startup logs; missing endpoint alone does not prove a hang. Do not terminate, retry or change deadlines automatically'}]


def guest_oom_warnings(current):
    """Guest-lifetime counter: not a trainer OOM or proof this trial caused it.

    Available memory recovers after an OOM kill, so current pressure alone
    misses repeated candidate failures. Inspect kernel logs for attribution.
    """
    count = current.get('remote', {}).get('guest_oom_kills')
    if type(count) is int and count > 0:
        return [{'code': 'sandbox_oom_kill_observed', 'guest_lifetime_kills': count,
                 'action': 'Inspect kernel victim records and verifier coverage; recovered free memory does not imply no OOM. Do not rewrite rewards or terminate/retry automatically'}]
    return []


def recover_trial_search(root, previous, current, *, node_deadlines=False, kcore=False,
                         mips_probes=False, overvalidation=False,
                         verifier_deadlock=False):
    """Attempt an exact recovery at most once per observed process identity.

    A command timeout or transport exception leaves the remote outcome unknown:
    the signal may already have been delivered.  Carry that uncertainty through
    subsequent snapshots and fail closed until the candidate identity changes.
    This prevents a monitor poll from repeatedly signaling the same process.
    """
    actions = []
    recovery_kind = 'kernel_stream_grep'
    candidates = node_timeout_guard.recovery_candidates if node_deadlines else recovery_candidates
    command = node_timeout_guard.recovery_command if node_deadlines else recovery_command
    if node_deadlines:
        recovery_kind = 'overdue_node'
    if kcore:
        candidates, command = kernel_core_guard.recovery_candidates, kernel_core_guard.recovery_command
        recovery_kind = 'kernel_core_grep'
    if mips_probes:
        candidates, command = mips_probe_guard.recovery_candidates, mips_probe_guard.recovery_command
        recovery_kind = 'mips_vm_probe'
    if overvalidation:
        candidates, command = overvalidation_guard.recovery_candidates, overvalidation_guard.recovery_command
        recovery_kind = 'known_overvalidation'
    if verifier_deadlock:
        candidates = verifier_deadlock_guard.recovery_candidates
        command = verifier_deadlock_guard.recovery_command
        recovery_kind = 'known_verifier_deadlock'

    held_keys = {
        action.get('candidate_key')
        for action in (previous or {}).get('recovery_actions', [])
        if (action.get('recovery_kind') == recovery_kind
            and action.get('candidate_key')
            and (action.get('action') in {'SIGINT', 'SIGTERM', 'SIGKILL', 'unknown'}
                 or action.get('reason') == 'prior_recovery_outcome_held'))
    }
    for expected in candidates(previous, current):
        candidate_key = json.dumps(expected, sort_keys=True, separators=(',', ':'))
        metadata = {'recovery_kind': recovery_kind, 'candidate_key': candidate_key}
        if candidate_key in held_keys:
            actions.append({
                'action': 'none',
                'reason': 'prior_recovery_outcome_held',
                'next': 'Reinspect this identity; retry only after the candidate identity changes',
                **metadata,
            })
            continue
        if (root / 'trials' / current['trial'] / 'result.json').exists():
            break
        session = current['trial'] + '__env'
        try:
            matches = Sandbox.list(query=SandboxQuery(metadata={'session_id': session}), limit=2).next_items()
            matches = [s for s in matches if s.metadata.get('session_id') == session]
            if len(matches) != 1 or matches[0].sandbox_id != current['sandbox_id']:
                actions.append({'action': 'none', 'reason': 'sandbox_identity_changed'})
                break
            sandbox = Sandbox.connect(current['sandbox_id'])
            result = sandbox.commands.run(command(expected), timeout=15)
            actions.append({**json.loads(result.stdout), **metadata})
        except Exception as error:
            actions.append({'action': 'unknown', 'reason': type(error).__name__,
                            'next': 'Reinspect the same process; do not assume success or resend blindly',
                            **metadata})
    return actions


def stall_warnings(previous, current):
    """Flag unchanged traced child/parent pairs; never classify task failure."""
    if not previous or previous.get('sandbox_id') != current.get('sandbox_id'):
        return []
    if (current.get('sandbox_id') and previous.get('observation_error')
            and current.get('observation_error')):
        return [{'code': 'repeated_sandbox_observation_error',
                 'action': 'Inspect sandbox control-plane state and metric freshness; observation errors do not prove failure. Do not restart automatically'}]
    before = previous.get('remote', {})
    after = current.get('remote', {})
    warnings = []
    old_log, new_log = before.get('verifier_log', {}), after.get('verifier_log', {})
    verifier_unchanged = (
        previous.get('phase') == current.get('phase') == 'verification_or_finalization'
        and all(new_log.get(k) is not None and new_log[k] == old_log.get(k)
                for k in ('bytes', 'mtime_ns', 'inode'))
    )
    if verifier_unchanged and new_log.get('age_s', 0) >= 300:
        warnings.append({'code': 'verifier_log_unchanged',
                         'action': 'Inspect verifier processes and deadline; quiet output alone is not failure. Never terminate automatically'})
    if verifier_unchanged and new_log.get('age_s', 0) >= 300:
        prior_processes = {p['Pid']: p for p in before.get('processes', [])}
        for process in after.get('processes', []):
            prior = prior_processes.get(process['Pid'], {})
            if (process.get('Name') == prior.get('Name') == 'apt-get'
                    and all(process.get(k) is not None and process[k] == prior.get(k)
                            for k in ('start_ticks', 'PPid', 'cpu_seconds'))):
                warnings.append({
                    'code': 'suspected_verifier_apt_wait', 'pid': process['Pid'],
                    'action': 'Inspect APT subcommand, child progress and repository access; unchanged parent CPU is not proof of failure. Never terminate automatically',
                })
    def active(record):
        return {(t.get('tool'), t['start_ms']) for t in record.get('running_tools', [])
                if t.get('start_ms') is not None}

    if not active(before).intersection(active(after)):
        return warnings
    old = {p['Pid']: p for p in before.get('processes', [])}
    new = {p['Pid']: p for p in after.get('processes', [])}

    def unchanged(p):
        prior = old.get(p['Pid'], {})
        return (p.get('start_ticks') is not None
                and p['start_ticks'] == prior.get('start_ticks')
                and p.get('cpu_seconds') is not None
                and p['cpu_seconds'] == prior.get('cpu_seconds'))

    for child in new.values():
        parent = new.get(child.get('TracerPid'))
        if (parent and child.get('PPid') == parent['Pid']
                and child.get('State', '').startswith('t ')
                and old.get(child['Pid'], {}).get('State', '').startswith('t ')
                and unchanged(child) and unchanged(parent)):
            warnings.append({'code': 'suspected_traced_child_stall',
                             'child_pid': child['Pid'], 'parent_pid': parent['Pid'],
                             'action': 'Inspect wait channels; do not infer failure or terminate automatically'})
    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval-seconds', type=int, default=180,
                        help='Observation cadence, 30–3600s; use 60s for short command-deadline recovery')
    parser.add_argument('--recover-kernel-stream-grep', action='store_true',
                        help='Opt in to SIGTERM only for revalidated OpenCode grep children blocked on kernel streams')
    parser.add_argument('--recover-overdue-node', action='store_true',
                        help='Opt in to killing only Node children exceeding their model-authored GNU timeout plus 30s grace')
    parser.add_argument('--recover-kcore-grep', action='store_true',
                        help='Opt in to SIGTERM only for revalidated grep children reading /proc/kcore after 10 minutes')
    parser.add_argument('--recover-mips-vm-probes', action='store_true',
                        help='Opt in to SIGKILL only for revalidated unbounded Node probes in make-mips-interpreter')
    parser.add_argument('--recover-known-overvalidation', action='store_true',
                        help='Opt in to SIGINT only for revalidated task-specific agent stress-test leaves')
    parser.add_argument('--recover-known-verifier-deadlock', action='store_true',
                        help='Opt in to SIGTERM only for the revalidated torch-tensor-parallelism spawned verifier leaf')
    args = parser.parse_args()
    if not 30 <= args.interval_seconds <= 3600:
        parser.error('--interval-seconds must be between 30 and 3600')
    root = args.run_dir.resolve()
    original = identity(args.pid)
    if original is None:
        raise SystemExit('Harness PID is not live; no remote inspection started')
    previous = {}
    while identity(args.pid) == original:
        cycle_started = time.monotonic()
        record = {'time': datetime.now(timezone.utc).isoformat()}
        record['scored_exception_audit'] = scored_exception_inventory(root / 'trials')
        try:
            with (root / 'health.jsonl').open() as source:
                health = json.loads(deque(source, maxlen=1)[0])
            record['health_time'] = health['time']
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(health['time'])).total_seconds()
            if age > 180:
                record['observation_error'] = 'stale_health_record'
            else:
                pending = [t for t in health['pending_trials'] if t['phase_age_s'] >= 180]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    record['trials'] = list(pool.map(lambda t: inspect_trial(root, t), pending))
                for trial in record['trials']:
                    retain_tool_start(previous.get(trial['trial']), trial)
                    trial['warnings'] = (stall_warnings(previous.get(trial['trial']), trial)
                                         + timeout_warnings(trial)
                                         + search_wait_warnings(trial)
                                         + memory_warnings(trial)
                                         + sidecar_readiness_warnings(trial)
                                         + guest_oom_warnings(trial)
                                         + sampling_budget_warnings(trial))
                    if args.recover_kernel_stream_grep:
                        trial['recovery_actions'] = recover_trial_search(root, previous.get(trial['trial']), trial)
                    if args.recover_overdue_node:
                        trial.setdefault('recovery_actions', []).extend(
                            recover_trial_search(root, previous.get(trial['trial']), trial, node_deadlines=True))
                    if args.recover_kcore_grep:
                        trial.setdefault('recovery_actions', []).extend(
                            recover_trial_search(root, previous.get(trial['trial']), trial, kcore=True))
                    if args.recover_mips_vm_probes:
                        trial.setdefault('recovery_actions', []).extend(
                            recover_trial_search(root, previous.get(trial['trial']), trial, mips_probes=True))
                    if args.recover_known_overvalidation:
                        trial.setdefault('recovery_actions', []).extend(
                            recover_trial_search(root, previous.get(trial['trial']), trial, overvalidation=True))
                    if args.recover_known_verifier_deadlock:
                        trial.setdefault('recovery_actions', []).extend(
                            recover_trial_search(root, previous.get(trial['trial']), trial,
                                                 verifier_deadlock=True))
        except Exception as error:
            record['observation_error'] = type(error).__name__
        print(json.dumps(record), flush=True)
        previous = {t['trial']: t for t in record.get('trials', [])}
        if args.once:
            return
        time.sleep(max(1, args.interval_seconds - (time.monotonic() - cycle_started)))


def scored_exception_inventory(trials_dir):
    """Expose scored agent failures; this is not an admission or retry policy.

    Only local, finalized results still retained on disk are counted. Never
    log exception messages or configuration: both can contain task text/secrets.
    """
    warnings = []
    unreadable = 0
    for path in sorted(trials_dir.glob('*/result.json')):
        try:
            result = json.loads(path.read_text())
            if not isinstance(result, dict) or not result.get('finished_at'):
                continue
            exception = result.get('exception_info')
            verifier = result.get('verifier_result')
            if not isinstance(exception, dict) or not isinstance(verifier, dict):
                continue
            rewards = verifier.get('rewards')
            if not isinstance(rewards, dict) or rewards.get('reward') is None:
                continue
            warnings.append({
                'code': 'scored_trial_with_exception',
                'trial': path.parent.name,
                'exception_type': exception.get('exception_type'),
                'finished_at': result['finished_at'],
                'action': 'Inspect exact terminal action and artifact; a score does not prove clean agent completion',
            })
        except FileNotFoundError:
            pass  # The checkpoint-aware archive may prune a completed trial.
        except (OSError, ValueError):
            unreadable += 1
    return {'warnings': warnings, 'unreadable_results': unreadable,
            'scope': 'finalized_local_results_not_pruned'}


if __name__ == '__main__':
    main()
