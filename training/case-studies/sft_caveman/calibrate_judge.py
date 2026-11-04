#!/usr/bin/env python3
"""Known-answer gate for the Caveman judge. Run before any paid evaluation.

Each known answer is a hand-written reply to one fixed task with a verdict we
already know. One passing reply, then one failing reply aimed at each judge
axis (supported, complete, on_request). If the judge disagrees with any
verdict, stop: the evaluation would be grading with a broken ruler.

Pattern borrowed from the cookbook's skill_signal harness (`judge.py`).

Usage:  python calibrate_judge.py          # exit 1 on any mismatch
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_caveman_pack import JUDGE_MIN, api_key, judge  # noqa: E402

TASK = {
    "question": "Our nightly job is scheduled with cron '0 3 * * 1' on a server that runs UTC. When does it run?",
    "facts": [
        "Cron format is minute hour day-of-month month day-of-week.",
        "'0 3 * * 1' means minute 0, hour 3, every day-of-month, every month, day-of-week 1.",
        "Day-of-week 1 is Monday.",
        "The server clock is UTC.",
    ],
    "expected": "It runs every Monday at 03:00 UTC.",
}

# name, reply, expect_pass, the axis a failing reply is aimed at
KNOWN_ANSWERS = [
    ("terse_pass", "Runs every Monday, 03:00 UTC.", True, None),
    ("invented_fact", "Runs every Monday, 03:00 UTC. Also retries at 04:00 if it fails.", False, "supported"),
    ("incomplete", "Runs at 03:00 UTC.", False, "complete"),
    ("extra_advice", "Runs every Monday, 03:00 UTC. You should move it to Sunday night and add alerting; "
                     "here is how to set up a Slack webhook: ...", False, "on_request"),
]


def main() -> int:
    api_key()
    bad = 0
    with httpx.Client() as c:
        for name, reply, expect_pass, axis in KNOWN_ANSWERS:
            j = judge(c, reply, TASK)
            if j is None:
                print(f"  {name}: judge returned nothing")
                bad += 1
                continue
            passed = all(v >= JUDGE_MIN for v in j.values())
            ok = passed == expect_pass and (axis is None or j[axis] < JUDGE_MIN)
            bad += not ok
            scores = " ".join(f"{k}={v:.2f}" for k, v in j.items())
            print(f"  {'ok ' if ok else 'BAD'} {name:<14} expect {'pass' if expect_pass else 'fail on ' + axis:<18} {scores}")
    if bad:
        print(f"\njudge calibration failed on {bad}/{len(KNOWN_ANSWERS)} known answers; fix the judge before spending")
        return 1
    print("\njudge calibration passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
