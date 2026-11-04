#!/usr/bin/env python3
"""Known-answer gate for the Humanizer judge. Run before any paid evaluation.

One source paragraph, one edit that should pass, and one failing edit aimed at
each judge axis (meaning_preserved, no_additions, natural, on_request). Any
mismatch stops the run before it spends on generations.

Pattern borrowed from the cookbook's skill_signal harness (`judge.py`).

Usage:  python calibrate_judge.py          # exit 1 on any mismatch
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_humanizer_pack import JUDGE_MIN, api_key, judge  # noqa: E402

ROW = {
    "audience": "existing customers",
    "source": (
        "Starting March 3, Standard-plan support tickets may take up to 48 business hours to get a first reply, "
        "up from the previous 24-hour target. Priority and Enterprise plans keep their 4-hour commitment. "
        "Phone hours and emergency escalation for outages are not changing."
    ),
}

KNOWN_ANSWERS = [
    ("light_edit_pass",
     "From March 3, a first reply to a Standard-plan ticket may take up to 48 business hours instead of the "
     "previous 24-hour target. Priority and Enterprise plans keep their 4-hour commitment. Phone hours and "
     "emergency escalation for outages are not changing.",
     True, None),
    ("hedge_dropped",
     "From March 3, Standard-plan tickets will get a first reply in 48 business hours, up from 24. "
     "Priority and Enterprise plans keep their 4-hour commitment. Phone hours and emergency escalation "
     "for outages are not changing.",
     False, "meaning_preserved"),
    ("added_fact",
     "From March 3, a first reply to a Standard-plan ticket may take up to 48 business hours instead of 24. "
     "Priority and Enterprise plans keep their 4-hour commitment, and Enterprise customers also get a named "
     "account manager. Phone hours and emergency escalation for outages are not changing.",
     False, "no_additions"),
    ("ai_tells",
     "This isn't just a change to response times, it's a commitment to doing support right. From March 3, "
     "Standard-plan tickets may take up to 48 business hours for a first reply, up from 24. Priority and "
     "Enterprise plans keep their 4-hour commitment. Phone hours and emergency escalation for outages are "
     "not changing. Because great support matters.",
     False, "natural"),
    ("preamble",
     "Here is a more natural version of your announcement:\n\nFrom March 3, a first reply to a Standard-plan "
     "ticket may take up to 48 business hours instead of the previous 24-hour target. Priority and Enterprise "
     "plans keep their 4-hour commitment. Phone hours and emergency escalation for outages are not changing.\n\n"
     "Changes made: simplified the opening sentence.",
     False, "on_request"),
]


def main() -> int:
    api_key()
    bad = 0
    with httpx.Client() as c:
        for name, reply, expect_pass, axis in KNOWN_ANSWERS:
            j = judge(c, reply, ROW)
            if j is None:
                print(f"  {name}: judge returned nothing")
                bad += 1
                continue
            passed = all(v >= JUDGE_MIN for v in j.values())
            ok = passed == expect_pass and (axis is None or j[axis] < JUDGE_MIN)
            bad += not ok
            scores = " ".join(f"{k}={v:.2f}" for k, v in j.items())
            print(f"  {'ok ' if ok else 'BAD'} {name:<16} expect {'pass' if expect_pass else 'fail on ' + axis:<26} {scores}")
    if bad:
        print(f"\njudge calibration failed on {bad}/{len(KNOWN_ANSWERS)} known answers; fix the judge before spending")
        return 1
    print("\njudge calibration passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
