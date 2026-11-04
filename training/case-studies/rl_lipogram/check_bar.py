"""Pass/fail gate for a served evaluation summary, so every stage decision is the same check.

    python check_bar.py results/<run>/summary.json [--arm adapter]

The bar: on held-out prompts, with no instruction, at effort low, joint success (e-free AND
answered AND not truncated) >= 0.90 sampled and >= 0.95 greedy. Retention is checked separately
when a retention summary is present (adapter minus base within 2 points).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SAMPLED_MIN, GREEDY_MIN, RETENTION_MAX_LOSS = 0.90, 0.95, 0.02


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--arm", default="adapter", help="arm name in the summary (adapter = no instruction)")
    ap.add_argument("--diagnostic", action="store_true", help="report only; never prints BAR MET")
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    problems = []
    arm = s.get("arms", {}).get(a.arm)
    if arm is None:
        problems.append(f"arm {a.arm!r} missing from summary")
    if s.get("effort") != "low":
        problems.append(f"effort is {s.get('effort')!r}, the bar is defined at 'low'")
    if a.arm != "adapter":
        problems.append("the bar is defined on the unprompted adapter arm")
    sampled = (arm or {}).get("sampled", {}).get("success")
    greedy = (arm or {}).get("greedy", {}).get("success")
    if sampled is None or greedy is None:
        problems.append("joint success metric missing (an e-free rate is not a substitute)")
    ret = s.get("retention")
    if not ret or "base" not in ret or a.arm not in ret:
        problems.append("retention block missing")
    print(f"{a.summary}  arm={a.arm}  effort={s.get('effort')}")
    for msg in problems:
        print(f"  INVALID  {msg}")
    ok = not problems
    if sampled is not None and greedy is not None:
        for name, value, floor in (("sampled joint success", sampled, SAMPLED_MIN), ("greedy joint success", greedy, GREEDY_MIN)):
            passed = value >= floor
            ok &= passed
            print(f"  {'PASS' if passed else 'FAIL'}  {name}: {value:.1%} (need >= {floor:.0%})")
    if ret and "base" in ret and a.arm in ret:
        loss = ret["base"] - ret[a.arm]
        passed = loss <= RETENTION_MAX_LOSS
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  retention (constraint-compatible items): adapter {ret[a.arm]:.1%} vs base {ret['base']:.1%} (loss <= {RETENTION_MAX_LOSS:.0%})")
    if a.diagnostic:
        print("DIAGNOSTIC", "would pass" if ok else "would not pass")
        raise SystemExit(1)
    print("BAR", "MET" if ok else "NOT MET")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
