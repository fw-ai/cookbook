#!/usr/bin/env python3
"""Read-only E2B progress sampling for the live run, every three minutes.

Run with --pid LIVE_HARNESS_PID; add --once for a single observation. Reads
pending trials from health.jsonl and connects only to exact trial session IDs.
Records activity ages and process states, never tool inputs, model text or
credentials. It never signals processes, retries samples, or changes timeouts.
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
from pathlib import Path
import time

from e2b import Sandbox, SandboxQuery


REMOTE = """python3 - <<'REMOTE'
import json, os, pathlib, sqlite3, time
p = pathlib.Path('/logs/agent/opencode/xdg-data/opencode/opencode.db')
out = {}
if p.exists():
    with sqlite3.connect('file:' + str(p) + '?mode=ro', uri=True) as c:
        rows = c.execute('select time_updated,data from part order by time_updated desc limit 3').fetchall()
        active = c.execute("select data from part where json_extract(data, '$.state.status')='running'").fetchall()
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
    for (raw,) in active:
        d = json.loads(raw)
        state = d.get('state', {})
        start = state.get('time', {}).get('start')
        out['running_tools'].append({'tool': d.get('tool'), 'start_ms': start,
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
p = pathlib.Path('/tmp/fireworks-tito-opencode/agent-status')
if p.exists():
    value = p.read_text().strip()
    out['agent_exit'] = int(value) if value.lstrip('-').isdigit() else 'non-numeric'
out['processes'] = []
for p in pathlib.Path('/proc').glob('[0-9]*/status'):
    try:
        fields = dict(line.split(':', 1) for line in p.read_text().splitlines() if ':' in line)
        if int(fields['Pid']) > 1600 and fields['PPid'].strip() != '2':
            row = {k: fields[k].strip() for k in ('Name','Pid','PPid','State','TracerPid')}
            stat = (p.parent / 'stat').read_text().rsplit(')', 1)[1].split()
            row['cpu_seconds'] = (int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK')
            row['start_ticks'] = stat[19]
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


def inspect_trial(root, trial):
    name = trial['trial']
    result = {'trial': name, 'phase': trial['phase']}
    if (root / 'trials' / name / 'result.json').exists():
        return {**result, 'observation': 'already_finalized'}
    try:
        session = name + '__env'
        candidates = Sandbox.list(query=SandboxQuery(metadata={'session_id': session}), limit=2).next_items()
        candidates = [s for s in candidates if s.metadata.get('session_id') == session]
        if len(candidates) != 1:
            return {**result, 'matching_sandboxes': len(candidates)}
        sandbox = Sandbox.connect(candidates[0].sandbox_id)
        result['sandbox_id'] = candidates[0].sandbox_id
        command = sandbox.commands.run(REMOTE, timeout=15)
        result['remote'] = json.loads(command.stdout)
    except Exception as error:
        result['observation_error'] = type(error).__name__
    return result


def stall_warnings(previous, current):
    """Flag unchanged traced child/parent pairs; never classify task failure."""
    if not previous or previous.get('sandbox_id') != current.get('sandbox_id'):
        return []
    before = previous.get('remote', {})
    after = current.get('remote', {})
    active = lambda r: {(t.get('tool'), t['start_ms']) for t in r.get('running_tools', [])
                        if t.get('start_ms') is not None}
    if not active(before).intersection(active(after)):
        return []
    old = {p['Pid']: p for p in before.get('processes', [])}
    new = {p['Pid']: p for p in after.get('processes', [])}

    def unchanged(p):
        prior = old.get(p['Pid'], {})
        return (p.get('start_ticks') is not None
                and p['start_ticks'] == prior.get('start_ticks')
                and p.get('cpu_seconds') is not None
                and p['cpu_seconds'] == prior.get('cpu_seconds'))

    warnings = []
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
    args = parser.parse_args()
    root = args.run_dir.resolve()
    original = identity(args.pid)
    if original is None:
        raise SystemExit('Harness PID is not live; no remote inspection started')
    previous = {}
    while identity(args.pid) == original:
        record = {'time': datetime.now(timezone.utc).isoformat()}
        try:
            with (root / 'health.jsonl').open() as source:
                health = json.loads(deque(source, maxlen=1)[0])
            record['health_time'] = health['time']
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(health['time'])).total_seconds()
            if age > 180:
                record['observation_error'] = 'stale_health_record'
            else:
                pending = [t for t in health['pending_trials'] if t['phase_age_s'] >= 900]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    record['trials'] = list(pool.map(lambda t: inspect_trial(root, t), pending))
                for trial in record['trials']:
                    trial['warnings'] = stall_warnings(previous.get(trial['trial']), trial)
        except Exception as error:
            record['observation_error'] = type(error).__name__
        print(json.dumps(record), flush=True)
        previous = {t['trial']: t for t in record.get('trials', [])}
        if args.once:
            return
        time.sleep(180)


if __name__ == '__main__':
    main()
