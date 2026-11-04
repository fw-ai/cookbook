#!/usr/bin/env python3
"""Accepted pipeline attempts -> instruction-free SFT rows (user question -> e-free answer, no reasoning).

Rows are validated with Rob's train_sft.validate_rows in answer-only mode (no system message, ends in
a nonempty e-free assistant answer, no reasoning_content).

    python build_rows.py runs/pilot/attempts.jsonl --out data/sft_pilot.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rl_lipogram"))
from train_sft import validate_rows  # noqa: E402

sys.path.insert(0, str(HERE))
from checks import talks_about_constraint  # noqa: E402

# Answers that talk about the constraint itself ("a glyph I can't print", "this lipogram") teach the
# model to narrate the rule instead of following it silently.
META = re.compile(r"lipogram|can'?t print|cannot print|in this output|glyph I can|stand-in words|"
                  r"(can'?t|cannot) (writ|us)\w* (its|that|this|the)", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows, seen, leaked = [], set(), 0
    for p in a.attempts:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if not r.get("accepted") or r["id"] in seen:
                continue
            if META.search(r["rewrite"]) or talks_about_constraint(r["rewrite"]):
                leaked += 1
                continue
            seen.add(r["id"])
            rows.append({"id": r["id"], "source": r["source"], "type": r["type"], "length": r["length"],
                         "messages": [{"role": "user", "content": r["question"]},
                                      {"role": "assistant", "content": r["rewrite"]}]})
    random.Random(a.seed).shuffle(rows)
    validate_rows(rows, mode="answer-only")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"rows": len(rows), "dropped_meta_commentary": leaked,
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
