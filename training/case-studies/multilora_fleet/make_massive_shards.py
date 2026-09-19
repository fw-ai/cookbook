#!/usr/bin/env python3
"""Shard MASSIVE into per-locale intent-classification SFT JSONL (bucket #3).

Uses the script-free parquet mirror `mteb/amazon_massive_intent` (config == locale
code, e.g. `en`, `de`, `es`). Each row has `text` (the utterance) and `label_text`
(the intent string, e.g. `alarm_set`). We emit one train + one holdout JSONL per
locale, each a chat row whose assistant turn is the gold intent label. Each shard
trains its own LoRA adapter; all adapters are then served on one base deployment.

Usage:
    python make_massive_shards.py                        # en, de, es
    python make_massive_shards.py --locales en fr ja --max-train 800
"""
from __future__ import annotations

import argparse
import json

SYSTEM = "Classify the user's utterance into exactly one intent label. Reply with only the label."


def build(locales, max_train: int, max_holdout: int, out_dir: str) -> None:
    from datasets import load_dataset

    for loc in locales:
        print(f"Loading mteb/amazon_massive_intent [{loc}] ...")
        ds = load_dataset("mteb/amazon_massive_intent", loc)
        for split_name, part, limit in [("train", "train", max_train), ("holdout", "test", max_holdout)]:
            rows = ds[part]
            path = f"{out_dir}/massive_{loc}_{split_name}.jsonl"
            n = 0
            with open(path, "w", encoding="utf-8") as f:
                for ex in rows:
                    f.write(json.dumps({"messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": ex["text"]},
                        {"role": "assistant", "content": ex["label_text"]},
                    ]}, ensure_ascii=False) + "\n")
                    n += 1
                    if limit and n >= limit:
                        break
            print(f"  [{loc}/{split_name}] wrote {n} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--locales", nargs="+", default=["en", "de", "es"])
    ap.add_argument("--max-train", type=int, default=800, help="0 = all rows.")
    ap.add_argument("--max-holdout", type=int, default=150, help="0 = all rows.")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    build(args.locales, args.max_train, args.max_holdout, args.out_dir)


if __name__ == "__main__":
    main()
