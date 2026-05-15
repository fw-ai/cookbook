#!/usr/bin/env python3
"""DeepMath accuracy eval. Re-runnable harness for baseline + post-distill comparison.

Samples one completion per prompt at temperature=0 from the Fireworks
chat-completions API, scores with the same ``deepmath_reward`` used by the
RL recipe, and prints mean accuracy. Output JSONL captures per-row
predictions for later analysis.

Usage:
    # Baseline: base qwen3-8b
    python eval_deepmath.py \\
        --model accounts/fireworks/models/qwen3-8b \\
        --dataset ../rl/deepmath/dataset.jsonl \\
        --max-rows 100 \\
        --output ./eval_base_qwen3_8b.jsonl

    # Post-distillation: the distilled model produced by stage 2
    python eval_deepmath.py \\
        --model accounts/fireworks/models/<your-output-model-id> \\
        --dataset ../rl/deepmath/dataset.jsonl \\
        --max-rows 100 \\
        --output ./eval_distilled_qwen3_8b.jsonl

Note on the dataset split: this script reads the *same* dataset.jsonl used
by stage-1 RL and stage-2 distillation. The user-facing intent is "eval on
held-out". If you want a strict held-out split, pre-shuffle dataset.jsonl
and use --skip-rows / --max-rows to carve a slice the training jobs did
not see. For the cookbook default (max_rows=100 on each stage), passing
--skip-rows 100 --max-rows 100 here gives you a disjoint held-out 100.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

import httpx

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from training.examples.rl.deepmath.train_deepmath import deepmath_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Fireworks model resource name to evaluate.")
    p.add_argument("--dataset", required=True, help="DeepMath JSONL (one prompt per line).")
    p.add_argument("--max-rows", type=int, default=100)
    p.add_argument("--skip-rows", type=int, default=0,
                   help="Skip the first N rows (use this to carve a held-out slice).")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output", required=True, help="Write per-row predictions JSONL here.")
    p.add_argument("--base-url", default=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"))
    return p.parse_args()


def load_rows(path: str, skip: int, limit: int) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


async def score_one(
    client: httpx.AsyncClient,
    model: str,
    api_key: str,
    row: dict,
    max_tokens: int,
    temperature: float,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        payload = {
            "model": model,
            "messages": row["messages"],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            r = await client.post(
                "/inference/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=600,
            )
            r.raise_for_status()
            data = r.json()
            completion = data["choices"][0]["message"]["content"] or ""
        except Exception as e:
            logger.warning("sample failed: %s", e)
            return {"ground_truth": row.get("ground_truth"), "completion": None,
                    "reward": 0.0, "error": str(e)}
        reward = deepmath_reward(completion, row)
        return {
            "ground_truth": row.get("ground_truth"),
            "completion": completion,
            "reward": reward,
        }


async def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("FIREWORKS_API_KEY not set")

    rows = load_rows(args.dataset, args.skip_rows, args.max_rows)
    logger.info("Evaluating %d rows from %s against %s", len(rows), args.dataset, args.model)

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    async with httpx.AsyncClient(base_url=args.base_url) as client:
        results = await asyncio.gather(*[
            score_one(client, args.model, api_key, row, args.max_tokens, args.temperature, sem)
            for row in rows
        ])

    rewards = [r["reward"] for r in results]
    acc = sum(rewards) / max(1, len(rewards))
    elapsed = time.time() - t0
    logger.info("Accuracy: %.3f (%d/%d correct) in %.1fs",
                acc, int(sum(rewards)), len(rewards), elapsed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(json.dumps({
            "model": args.model, "n": len(rows), "accuracy": acc,
            "skip_rows": args.skip_rows, "max_rows": args.max_rows,
            "temperature": args.temperature, "max_tokens": args.max_tokens,
        }) + "\n")
        for row_in, row_out in zip(rows, results):
            f.write(json.dumps({**row_out, "prompt": row_in.get("messages")}) + "\n")
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
