"""Honest cost tracking for every paid API call in this study.

Each call appends one line to a ledger (jsonl): model, stage, prompt/completion tokens, time. `summary()`
prices it with the public serverless rates below (USD per 1M tokens, standard tier, 2026-10-17) and prints
a table by stage and model. Training tokens are added separately by the trainer script.

    python cost_ledger.py data-v2/costs.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

PRICES = {  # (input, output) USD per 1M tokens, serverless standard tier; verify against docs.fireworks.ai/serverless/pricing
    "accounts/fireworks/models/glm-5p3-flash": (0.15, 0.50),
    "accounts/fireworks/models/deepseek-v4p1-flash": (0.30, 1.20),
    "accounts/fireworks/models/kimi-k3": (3.00, 15.00),   # docs.fireworks.ai/serverless/pricing, 2026-10-20
}
# A LoRA adapter on serverless bills at its base model's rate. Adapter ids on this account are named by base.
ADAPTER_BASE = {"glm53-": "accounts/fireworks/models/glm-5p3-flash"}


def price_for(model: str):
    """(input, output) rate, or None when the model is not priced; never silently zero."""
    m = model.split("#", 1)[0]   # drop a deployment suffix
    if m in PRICES:
        return PRICES[m]          # may be None: known model, unknown price
    name = m.split("/")[-1]
    for prefix, base in ADAPTER_BASE.items():
        if name.startswith(prefix):
            return PRICES[base]
    return None
TRAIN_PRICE_PER_M = {"accounts/fireworks/models/glm-5p3-flash": 8.89}   # serverless Training API, train tokens
_lock = threading.Lock()


def record(path: Path, stage: str, model: str, usage: dict | None) -> None:
    u = usage or {}
    row = {"t": time.time(), "stage": stage, "model": model, "prompt_tokens": u.get("prompt_tokens", 0),
           "completion_tokens": u.get("completion_tokens", 0)}
    with _lock, path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def record_training(path: Path, model: str, train_tokens: int, note: str = "") -> None:
    with _lock, path.open("a") as f:
        f.write(json.dumps({"t": time.time(), "stage": "train", "model": model, "train_tokens": train_tokens, "note": note}) + "\n")


def summary(path: Path) -> dict:
    by: dict[tuple[str, str], dict] = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        r = json.loads(line)
        k = (r["stage"], r["model"])
        d = by.setdefault(k, {"calls": 0, "in": 0, "out": 0, "train": 0, "usd": 0.0})
        d["calls"] += 1
        d["in"] += r.get("prompt_tokens", 0)
        d["out"] += r.get("completion_tokens", 0)
        d["train"] += r.get("train_tokens", 0)
    unknown = []
    for (stage, model), d in by.items():
        rate = price_for(model)
        if rate is None and (d["in"] or d["out"]):
            unknown.append(model); d["usd"] = None
            continue
        pin, pout = rate or (0.0, 0.0)
        d["usd"] = d["in"] / 1e6 * pin + d["out"] / 1e6 * pout + d["train"] / 1e6 * TRAIN_PRICE_PER_M.get(model, 0.0)
    total = sum(d["usd"] for d in by.values() if d["usd"] is not None)
    return {"by_stage": {f"{s}|{m.split('/')[-1]}": d for (s, m), d in by.items()}, "total_usd": round(total, 2), "unpriced_models": sorted(set(unknown))}


def print_summary(path: Path) -> None:
    s = summary(path)
    print(f"{'stage|model':40s} {'calls':>7s} {'in tok':>12s} {'out tok':>12s} {'train tok':>12s} {'USD':>8s}")
    for k, d in sorted(s["by_stage"].items()):
        usd = f"{d['usd']:8.2f}" if d["usd"] is not None else f"{'UNKNOWN':>8s}"
        print(f"{k:40s} {d['calls']:7d} {d['in']:12,d} {d['out']:12,d} {d['train']:12,d} {usd}")
    print(f"{'TOTAL (priced rows only)':40s} {'':7s} {'':12s} {'':12s} {'':12s} {s['total_usd']:8.2f}")
    if s["unpriced_models"]:
        print("unpriced:", ", ".join(s["unpriced_models"]))


if __name__ == "__main__":
    print_summary(Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CAVEMAN_COSTS", "data/costs.jsonl")))
