#!/usr/bin/env python3
"""Per-stage yield, cost per kept row, judge scores and failure modes for pipeline attempts.

    python stats.py runs/pilot/attempts.jsonl [--examples 10]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts")
    ap.add_argument("--examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rs = [json.loads(l) for l in Path(a.attempts).read_text().splitlines()]
    n = len(rs)
    drafted = [r for r in rs if r.get("original")]
    rewritten = [r for r in drafted if "e_free" in r]
    efree = [r for r in rewritten if r["e_free"]]
    long_ok = [r for r in efree if r.get("fail") != "too_short"]
    judged = [r for r in long_ok if "fidelity" in r and "answers" in r]
    acc = [r for r in rs if r.get("accepted")]
    first_try = sum(1 for r in efree if r["fix_history"][0]["violations"] == 0 and r["fix_history"][0]["finish"] == "stop")
    rounds = Counter(len(r["fix_history"]) - 1 for r in efree)
    cost = sum(r.get("cost", 0) for r in rs)
    print(f"seeds {n}")
    print(f"  drafted              {len(drafted):5}  ({len(drafted) / n:.1%})")
    print(f"  rewritten            {len(rewritten):5}")
    print(f"  e-free after fixes   {len(efree):5}  ({len(efree) / max(1, len(rewritten)):.1%} of rewritten; "
          f"{first_try} e-free on the first pass; fix rounds used {dict(sorted(rounds.items()))})")
    print(f"  length ok            {len(long_ok):5}")
    print(f"  judged               {len(judged):5}")
    print(f"  accepted             {len(acc):5}  ({len(acc) / n:.1%} of seeds, {len(acc) / max(1, len(judged)):.1%} of judged)")
    print(f"  cost total ${cost:.2f}, per seed ${cost / max(1, n):.4f}, per kept row ${cost / max(1, len(acc)):.4f}")
    for k in ("fidelity", "natural", "answers", "faithful"):
        v = [r[k] for r in judged if k in r]
        va = [r[k] for r in acc if k in r]
        if v:
            print(f"  {k:9} judged mean {statistics.mean(v):.3f}  accepted mean {statistics.mean(va):.3f}")
    ratio = [r["words_rewrite"] / r["words_original"] for r in acc if r.get("words_original")]
    if ratio:
        print(f"  words rewrite/original (accepted): median {statistics.median(ratio):.2f}; "
              f"accepted answer words median {statistics.median(r['words_rewrite'] for r in acc):.0f}")
    print("  failures:", dict(Counter(r.get("fail") for r in rs if r.get("fail")).most_common()))
    sec = [r["seconds"] for r in rs if r.get("seconds")]
    if sec:
        print(f"  seconds per seed: median {statistics.median(sec):.0f}, p90 {sorted(sec)[int(0.9 * (len(sec) - 1))]:.0f}")
    by = Counter(r["source"].split(":")[0] for r in acc)
    print("  accepted by source:", dict(by), " by length:", dict(Counter(r["length"] for r in acc)))
    if a.examples:
        for r in random.Random(a.seed).sample(acc, min(a.examples, len(acc))):
            print("\n---", r["id"], r["source"], r["type"], r["length"], f"fid {r['fidelity']} nat {r['natural']}")
            print("Q:", r["question"][:400])
            print("A:", r["rewrite"][:700])


if __name__ == "__main__":
    main()
