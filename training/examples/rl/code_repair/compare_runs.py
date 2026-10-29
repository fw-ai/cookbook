#!/usr/bin/env python3
"""Did RL improve repair skill, or only stop the model running out of tokens?

Compares two per-row result files on the same eval rows (``sampled_rows.jsonl``
from ``eval_checkpoint`` or ``--sampling-only``). Older files without per-sample
``finished`` flags treat a sample of ``--legacy-cap`` tokens as truncated.

* overall: fully-fixed rate over all samples (what training optimized);
* finished only: fully-fixed rate over samples that did not hit the cap.
  Biased: a model that truncates on hard rows keeps mostly easy ones;
* matched rows: per row, A's rate over its finished samples vs B's rate, only
  on rows where A finished at least once, averaged over rows, with a paired
  bootstrap 95% interval on B - A;
* clean rows: rows where A never truncated, where length cannot explain a gap;
* rows where A always truncated: what B does on them.

    python -m training.examples.rl.code_repair.compare_runs \\
        runs/baseline_eval78 runs/eval_tuned_8k --names base tuned
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(run_dir: Path, legacy_cap: int) -> dict[str, dict]:
    rows = {}
    for line in (run_dir / "sampled_rows.jsonl").open():
        row = json.loads(line)
        if "finished" not in row:
            row["finished"] = [t < legacy_cap for t in row["completion_tokens"]]
        row["fixed"] = [r >= 1.0 for r in row["rewards"]]
        rows[row["id"]] = row
    return rows


def rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else float("nan")


def finished_fixed(row: dict) -> list[bool]:
    return [fx for fx, fin in zip(row["fixed"], row["finished"]) if fin]


def paired_interval(diffs: list[float], seed: int = 0, rounds: int = 5000) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def compare(a: dict[str, dict], b: dict[str, dict], names: tuple[str, str]) -> None:
    ids = sorted(set(a) & set(b))
    na, nb = names
    pool = lambda rows, fn: [x for i in ids for x in fn(rows[i])]  # noqa: E731
    print(f"rows compared: {len(ids)}\n")
    print(f"{'':34s}{na:>12s}{nb:>12s}")
    print(f"{'fully fixed, all samples':34s}{rate(pool(a, lambda r: r['fixed'])):12.1%}{rate(pool(b, lambda r: r['fixed'])):12.1%}")
    print(f"{'truncated':34s}{1 - rate(pool(a, lambda r: r['finished'])):12.1%}{1 - rate(pool(b, lambda r: r['finished'])):12.1%}")
    print(f"{'fully fixed, finished samples':34s}{rate(pool(a, finished_fixed)):12.1%}{rate(pool(b, finished_fixed)):12.1%}")

    def matched(subset: list[str], label: str) -> None:
        if not subset:
            print(f"\n{label}: no rows")
            return
        ra = [rate(finished_fixed(a[i])) for i in subset]
        rb = [rate(finished_fixed(b[i])) for i in subset]
        rb_all = [rate(b[i]["fixed"]) for i in subset]
        diffs = [y - x for x, y in zip(ra, rb)]
        lo, hi = paired_interval(diffs)
        better = sum(d > 1e-9 for d in diffs)
        worse = sum(d < -1e-9 for d in diffs)
        print(f"\n{label} ({len(subset)} rows)")
        print(f"  fully fixed, finished samples: {na} {sum(ra) / len(ra):.1%}  {nb} {sum(rb) / len(rb):.1%}")
        print(f"  {nb} - {na}: {sum(diffs) / len(diffs):+.1%}  (95% CI {lo:+.1%} to {hi:+.1%})")
        print(f"  rows {nb} better / worse / same: {better} / {worse} / {len(subset) - better - worse}")
        print(f"  {nb} fully fixed counting its truncations as failures: {sum(rb_all) / len(rb_all):.1%}")

    both = [i for i in ids if any(a[i]["finished"]) and any(b[i]["finished"])]
    matched(both, "matched rows (both finished at least once)")
    matched([i for i in ids if all(a[i]["finished"])], f"clean rows ({na} never truncated)")
    always = [i for i in ids if not any(a[i]["finished"])]
    if always:
        print(f"\nrows where {na} always truncated ({len(always)}): {nb} fully fixes {rate(pool_rows(b, always)):.1%}")


def pool_rows(rows: dict[str, dict], ids: list[str]) -> list[bool]:
    return [x for i in ids for x in rows[i]["fixed"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--names", nargs=2, default=("A", "B"))
    parser.add_argument("--legacy-cap", type=int, default=8192)
    args = parser.parse_args()
    compare(load(args.run_a, args.legacy_cap), load(args.run_b, args.legacy_cap), tuple(args.names))


if __name__ == "__main__":
    main()
