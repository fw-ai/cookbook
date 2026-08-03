#!/usr/bin/env python3
"""Prepare CORD-v2 receipts as a vision SFT dataset: receipt image -> structured JSON.

Each output row is an OpenAI multimodal chat example in the format the cookbook's
SFT recipe consumes (see examples/sft/food_reasoning.jsonl and utils/supervised.py):

    {"messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            {"type": "text", "text": <extraction instruction>}]},
        {"role": "assistant", "content": <gold JSON string>}]}

The gold JSON is a *flattened, readable* view of CORD's cryptic gt_parse:
  menu[].nm/cnt/price            -> items[].name/quantity/price
  sub_total.subtotal_price       -> subtotal
  sub_total.tax_price            -> tax
  sub_total.service_price        -> service
  total.total_price              -> total
Missing fields become null (this is deliberate -- it teaches the model to abstain,
which the later DPO step sharpens).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

from datasets import load_dataset

PROMPT = (
    "Extract this receipt into JSON with exactly this schema:\n"
    '{"items": [{"name": str, "quantity": str, "price": str}], '
    '"subtotal": str|null, "tax": str|null, "service": str|null, "total": str|null}\n'
    "Use null for any field that is not present on the receipt. "
    "Respond with only the JSON object, no prose."
)


def _as_list(v):
    """CORD stores a single menu item as a dict, multiple as a list."""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def flatten_gt(gt_parse: dict) -> dict:
    """CORD gt_parse -> clean, flat target schema. Missing -> None."""
    items = []
    for m in _as_list(gt_parse.get("menu")):
        if not isinstance(m, dict):
            continue
        items.append(
            {
                "name": m.get("nm"),
                "quantity": m.get("cnt"),
                "price": m.get("price"),
            }
        )
    sub = gt_parse.get("sub_total") or {}
    tot = gt_parse.get("total") or {}
    return {
        "items": items,
        "subtotal": sub.get("subtotal_price"),
        "tax": sub.get("tax_price"),
        "service": sub.get("service_price"),
        "total": tot.get("total_price"),
    }


def image_to_data_uri(img, max_side: int = 1024, quality: int = 85) -> str:
    """PIL image -> base64 JPEG data URI, downscaled to keep token/byte length sane."""
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_row(example) -> dict:
    gt = json.loads(example["ground_truth"])["gt_parse"]
    target = flatten_gt(gt)
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(example["image"])}},
                    {"type": "text", "text": PROMPT},
                ],
            },
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]
    }


def main():
    ap = argparse.ArgumentParser(description="Build CORD-v2 vision SFT JSONL")
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--max-examples", type=int, default=None, help="Cap rows (smoke tests)")
    ap.add_argument("--max-side", type=int, default=1024, help="Max image dimension (px)")
    ap.add_argument("--out", default=None, help="Output JSONL path")
    args = ap.parse_args()

    out = Path(args.out or Path(__file__).parent / f"cord_{args.split}.jsonl")
    ds = load_dataset("naver-clova-ix/cord-v2", split=args.split)
    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    n = 0
    with out.open("w") as f:
        for ex in ds:
            row = build_row(ex)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} rows -> {out}")


if __name__ == "__main__":
    main()
