#!/usr/bin/env python3
"""Read-only E2B progress sampling for the live run, every three minutes.

Run with --pid LIVE_HARNESS_PID; add --once for a single observation. Reads
pending trials from health.jsonl and connects only to exact trial session IDs.
Records activity ages and process states, never tool inputs, model text or
credentials. It never signals processes, retries samples, or changes timeouts.
An old activity timestamp is a reason to inspect, not proof of a failed sample.
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
        except Exception as error:
            record['observation_error'] = type(error).__name__
        print(json.dumps(record), flush=True)
        if args.once:
            return
        time.sleep(180)


if __name__ == '__main__':
    main()
