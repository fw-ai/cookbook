#!/usr/bin/env python3
"""Build managed-DPO preference JSONL from nvidia/HelpSteer3 (bucket #2: "write the way we write").

CLI mirror of the inlined logic in dpo_helpsteer3.ipynb.

HelpSteer3 `preference` rows carry a `context` (OpenAI-shaped message list), two
candidate responses, and an `overall_preference` score in [-3, 3]:

    -3..-1 => response1 is better,  +1..+3 => response2 is better,  0 => tie.

We emit the Fireworks **managed DPO** schema:

    {"input": {"messages": [...]},
     "preferred_output":     [{"role": "assistant", "content": <preferred>}],
     "non_preferred_output": [{"role": "assistant", "content": <other>}]}

Managed DPO is single-turn, so we keep only prompts with no prior assistant turns.

Usage:
    python make_helpsteer3_dpo.py                       # english, 2000 train / 200 holdout
    python make_helpsteer3_dpo.py --language "" --max-train 3000   # keep all languages
"""
from __future__ import annotations

import argparse
import json


def to_managed_row(ex: dict) -> dict | None:
    pref = ex.get("overall_preference")
    if pref is None or pref == 0:
        return None  # drop ties: no training signal
    context = ex.get("context") or []
    if not isinstance(context, list) or not context:
        return None
    # Managed DPO is single-turn: skip prompts that already contain assistant turns.
    if any(m.get("role") == "assistant" for m in context):
        return None
    if pref < 0:
        preferred, non_preferred = ex["response1"], ex["response2"]
    else:
        preferred, non_preferred = ex["response2"], ex["response1"]
    return {
        "input": {"messages": context},
        "preferred_output": [{"role": "assistant", "content": preferred}],
        "non_preferred_output": [{"role": "assistant", "content": non_preferred}],
    }


def build(language: str, max_train: int, max_holdout: int, out_dir: str) -> None:
    from datasets import load_dataset

    print("Loading nvidia/HelpSteer3 (preference) ...")
    ds = load_dataset("nvidia/HelpSteer3", "preference")

    def dump(split: str, path: str, limit: int) -> int:
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for ex in ds[split]:
                if language and ex.get("language", "english") != language:
                    continue
                row = to_managed_row(ex)
                if row is None:
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                if limit and n >= limit:
                    break
        print(f"[{split}] wrote {n} single-turn rows -> {path}")
        return n

    dump("train", f"{out_dir}/helpsteer3_dpo_train.jsonl", max_train)
    dump("validation", f"{out_dir}/helpsteer3_dpo_holdout.jsonl", max_holdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", default="english", help="Filter to one language (empty string keeps all).")
    ap.add_argument("--max-train", type=int, default=2000, help="0 = all rows.")
    ap.add_argument("--max-holdout", type=int, default=200, help="0 = all rows.")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    build(args.language, args.max_train, args.max_holdout, args.out_dir)


if __name__ == "__main__":
    main()
