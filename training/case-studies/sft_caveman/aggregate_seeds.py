#!/usr/bin/env python3
"""Aggregate pack evaluations across seeds: pass rate per arm with a paired bootstrap CI vs base.

Each results directory is one evaluation run holding its own base (and skill)
arm next to one adapter arm. Greedy decoding is not bit-reproducible across
runs, so every adapter is paired with the base from ITS OWN run, never with
another run's base. The across-seed line is the mean of those per-run deltas.

Usage: python aggregate_seeds.py results/2026-10-08/pack-eval-s29 results/2026-10-08/pack-eval-s17 ...
"""
import json, random, statistics as st, sys
from pathlib import Path

rng = random.Random(0)


def load(d):
    rows = [json.loads(l) for l in (Path(d) / "samples.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("draw", 0) == 0]  # multi-draw evals: first draw only; pass^k lives in summary.json
    arms = {}
    for r in rows:
        arms.setdefault(r["arm"], {})[r["id"]] = r
    return arms


def boot_diff(a, b, ids, n=5000):
    diffs = []
    for _ in range(n):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        diffs.append(sum(a[i]["pass"] for i in s) / len(s) - sum(b[i]["pass"] for i in s) / len(s))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


runs = {d: load(d) for d in sys.argv[1:]}
ids = sorted(next(iter(runs.values()))["base"])
print(f"n = {len(ids)} held-out fixtures\n")
print("| run | arm | pass | median words | Δpass vs this run's base (95% paired bootstrap) |\n|---|---|---|---|---|")
deltas = []
for d, arms in runs.items():
    base = arms["base"]
    pb = sum(base[i]["pass"] for i in ids) / len(ids)
    for name, a in arms.items():
        p = sum(a[i]["pass"] for i in ids) / len(ids)
        w = st.median(a[i]["words"] for i in ids)
        if name == "base":
            print(f"| {Path(d).name} | base | {pb:.0%} | {w:.0f} | |")
            continue
        lo, hi = boot_diff(a, base, ids)
        print(f"| {Path(d).name} | {name} | {p:.0%} | {w:.0f} | {p - pb:+.0%} [{lo:+.0%}, {hi:+.0%}] |")
        if name.startswith("adapter"):
            deltas.append((name, p, p - pb, a))
if len(deltas) > 1:
    ps = [p for _, p, _, _ in deltas]
    print(f"\nadapter across {len(deltas)} seeds: mean pass {st.mean(ps):.0%}, range {min(ps):.0%}-{max(ps):.0%}; "
          f"mean Δ vs own base {st.mean(dl for _, _, dl, _ in deltas):+.0%}")
    agree = sum(all(a[i]["pass"] == deltas[0][3][i]["pass"] for _, _, _, a in deltas) for i in ids)
    print(f"fixtures where all seeds agree on pass/fail: {agree}/{len(ids)}")
