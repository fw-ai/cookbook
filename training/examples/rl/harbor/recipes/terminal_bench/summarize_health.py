"""Print a read-only, synchronization-checked table from monitor health.jsonl.

Usage: python -m training.examples.rl.harbor.recipes.terminal_bench.summarize_health \
    /path/to/run/health.jsonl --expected-peers 4

Includes a step only when train/step and rollout/step agree and every expected,
distinct hot-load peer is ready on that snapshot. Repeated monitor observations
are deduplicated, not averaged. This checks observed synchronization, not DCP
durability, successful restore, sample validity, or convergence. Trainer tok/s
is NOT dashboard generation throughput; step wall time is NOT sampling time.
The command does not contact services, modify files, or restart any process.
"""

import argparse
import json
from pathlib import Path
import re


def synchronized_step(record, expected_peers):
    metrics = record.get("metrics", {})
    step = metrics.get("train/step")
    if isinstance(step, bool) or not isinstance(step, (int, float)):
        return None
    if step < 1 or not float(step).is_integer() or metrics.get("rollout/step") != step:
        return None
    peers = record.get("hotload_peers", [])
    if len(peers) != expected_peers:
        return None
    identities = {peer.get("identity") for peer in peers}
    if None in identities or "" in identities or len(identities) != expected_peers:
        return None
    snapshots = {peer.get("current_snapshot_identity") for peer in peers}
    if len(snapshots) != 1 or not all(peer.get("readiness") is True for peer in peers):
        return None
    snapshot = next(iter(snapshots))
    match = re.fullmatch(r"step-(\d+)(?:-.+)?", snapshot or "")
    return int(step) if match and int(match[1]) == step else None


def collect_steps(records, expected_peers):
    if expected_peers < 1:
        raise ValueError("expected_peers must be positive")
    steps = {}
    for record in records:
        step = synchronized_step(record, expected_peers)
        if step is not None:
            steps[step] = record
    return steps


def read_records(path):
    with path.open() as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # The live writer may not have finished its final line yet.
                if line.endswith("\n"):
                    raise


def render(steps):
    columns = [
        ("Reward", "rollout/raw_reward", ".4f"),
        ("K3", "train/inference_k3", ".6f"),
        ("Grad norm", "train/grad_norm", ".4f"),
        ("Trainer tok/s", "perf/train_tokens_per_s", ".0f"),
        ("Sync s", "perf/weight_update_time", ".1f"),
        ("Step wall s", "perf/step_time", ".1f"),
    ]
    lines = ["| Step | " + " | ".join(c[0] for c in columns) + " | Observed UTC |",
             "|---:" + "|---:" * len(columns) + "|---|"]
    for step, record in sorted(steps.items()):
        metrics = record["metrics"]
        values = [format(metrics[key], spec) if isinstance(metrics.get(key), (int, float))
                  else "—" for _, key, spec in columns]
        lines.append(f"| {step} | " + " | ".join(values) + f" | {record.get('time', '—')} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("health", type=Path)
    parser.add_argument("--expected-peers", type=int, required=True)
    args = parser.parse_args()
    print(render(collect_steps(read_records(args.health), args.expected_peers)))
    print("\nObserved synchronized steps only; DCP and sample validity require separate checks.")


if __name__ == "__main__":
    main()
