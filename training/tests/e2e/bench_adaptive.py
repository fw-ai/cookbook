#!/usr/bin/env python3
"""Benchmark: adaptive concurrency (AIMD, initial_window=16).

32 batches × 256 requests/batch (32 prompts × 8 completions), 128K max tokens.
Uses AdaptiveConcurrencyController with batch-level AIMD.
"""

from __future__ import annotations

import os
import sys
import asyncio
import json
import logging
import time
import statistics

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import transformers
from fireworks.training.sdk.deployment import (
    AdaptiveConcurrencyController,
    DeploymentSampler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

API_KEY = os.environ["FIREWORKS_API_KEY"]
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
DEPLOYMENT_ID = os.environ["BENCH_DEPLOYMENT_ID"]
TOKENIZER_MODEL = "Qwen/Qwen3-30B-A3B"
DATASET_PATH = os.path.join(_SRC, "training", "examples", "deepmath_rl", "dataset.jsonl")

PROMPTS_PER_BATCH = 32
COMPLETIONS_PER_PROMPT = 8
NUM_BATCHES = 32
MAX_TOKENS = 131072
REPLICA_COUNT = 2
INITIAL_WINDOW = 8 * REPLICA_COUNT


def load_dataset():
    rows = []
    with open(DATASET_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


async def run():
    dataset = load_dataset()
    logger.info("Loaded %d rows", len(dataset))

    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    inference_model = f"accounts/pyroworks/deployments/{DEPLOYMENT_ID}"

    ctrl = AdaptiveConcurrencyController(
        initial_window=INITIAL_WINDOW,
        min_window=1,
        max_window=64,
        prefill_queue_target=0.5,
        ema_alpha=0.3,
    )

    sampler = DeploymentSampler(
        inference_url=BASE_URL,
        model=inference_model,
        api_key=API_KEY,
        tokenizer=tokenizer,
        concurrency_controller=ctrl,
    )

    # Warmup
    for attempt in range(10):
        try:
            r = await sampler.sample_with_tokens(
                messages=[{"role": "user", "content": "Say hi."}],
                n=1, max_tokens=8, temperature=0.0,
            )
            if r:
                logger.info("Warmup done")
                break
        except Exception:
            await asyncio.sleep(5)

    total_needed = NUM_BATCHES * PROMPTS_PER_BATCH
    all_rows = (dataset * ((total_needed // len(dataset)) + 1))[:total_needed]

    all_batch_results = []
    total_start = time.time()

    for batch in range(NUM_BATCHES):
        batch_start = time.time()
        batch_rows = all_rows[batch * PROMPTS_PER_BATCH : (batch + 1) * PROMPTS_PER_BATCH]

        async def _sample(row, idx):
            messages = row.get("messages", [])
            try:
                completions = await sampler.sample_with_tokens(
                    messages=messages,
                    n=COMPLETIONS_PER_PROMPT,
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                    logprobs=True,
                    http_timeout=600,
                )
                return completions
            except Exception as e:
                logger.warning("Batch %d prompt %d failed: %s", batch + 1, idx, e)
                return []

        tasks = [_sample(row, i) for i, row in enumerate(batch_rows)]
        results = await asyncio.gather(*tasks)

        batch_all = [c for r in results for c in r]
        batch_ok = len(batch_all)
        batch_err = sum(1 for r in results if not r)
        batch_elapsed = time.time() - batch_start

        # Step-level AIMD adjustment
        cc_summary = ctrl.step_completed()

        comp_lens = [c.completion_len for c in batch_all]
        avg_len = statistics.mean(comp_lens) if comp_lens else 0
        median_len = statistics.median(comp_lens) if comp_lens else 0
        finish = {}
        for c in batch_all:
            finish[c.finish_reason] = finish.get(c.finish_reason, 0) + 1

        first_preview = ""
        if batch_all:
            first = batch_all[0]
            first_preview = first.text[:300].replace("\n", " ")

        batch_result = {
            "batch": batch + 1,
            "completions": batch_ok,
            "expected": PROMPTS_PER_BATCH * COMPLETIONS_PER_PROMPT,
            "errors": batch_err,
            "time_s": round(batch_elapsed, 1),
            "avg_len": round(avg_len),
            "median_len": round(median_len),
            "finish": finish,
            "window": ctrl.window_size,
            "avg_pq": round(cc_summary.get("avg_pq", 0), 4),
            "cache_hit_rate": round(cc_summary.get("cache_hit_rate", 0), 3),
        }
        all_batch_results.append(batch_result)

        logger.info(
            "Batch %d/%d: %d/%d ok, %d err, %.1fs, avg_len=%d, median=%d, window=%d, avg_pq=%.3fs, cache=%.1f%%",
            batch + 1, NUM_BATCHES, batch_ok, batch_result["expected"],
            batch_err, batch_elapsed, avg_len, median_len,
            ctrl.window_size, cc_summary.get("avg_pq", 0), cc_summary.get("cache_hit_rate", 0) * 100,
        )
        logger.info("  First: (%d tok) %s%s", batch_all[0].completion_len if batch_all else 0,
                     first_preview[:200], "..." if len(first_preview) > 200 else "")

    total_elapsed = time.time() - total_start
    total_completions = sum(b["completions"] for b in all_batch_results)
    total_expected = NUM_BATCHES * PROMPTS_PER_BATCH * COMPLETIONS_PER_PROMPT
    total_errors = sum(b["errors"] for b in all_batch_results)
    all_times = [b["time_s"] for b in all_batch_results]
    all_lens = [b["avg_len"] for b in all_batch_results]
    windows = [b["window"] for b in all_batch_results]

    summary = {
        "mode": "adaptive",
        "initial_window": INITIAL_WINDOW,
        "final_window": ctrl.window_size,
        "window_range": f"{min(windows)}-{max(windows)}",
        "deployment": DEPLOYMENT_ID,
        "batches": NUM_BATCHES,
        "prompts_per_batch": PROMPTS_PER_BATCH,
        "completions_per_prompt": COMPLETIONS_PER_PROMPT,
        "max_tokens": MAX_TOKENS,
        "total_completions": total_completions,
        "total_expected": total_expected,
        "total_errors": total_errors,
        "success_rate": round(total_completions / total_expected * 100, 2),
        "total_time_s": round(total_elapsed, 1),
        "avg_batch_time_s": round(statistics.mean(all_times), 1),
        "median_batch_time_s": round(statistics.median(all_times), 1),
        "avg_response_len": round(statistics.mean(all_lens)),
        "throughput_completions_per_s": round(total_completions / total_elapsed, 2),
        "final_ema_pq": round(ctrl.ema_prefill_queue, 4) if ctrl.ema_prefill_queue else None,
    }

    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE: ADAPTIVE CONCURRENCY (initial=%d)", INITIAL_WINDOW)
    for k, v in summary.items():
        logger.info("  %-30s %s", k, v)
    logger.info("=" * 60)

    # Write results
    with open("bench_adaptive_results.json", "w") as f:
        json.dump({"summary": summary, "batches": all_batch_results}, f, indent=2)
    logger.info("Results written to bench_adaptive_results.json")

    sampler.close()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
