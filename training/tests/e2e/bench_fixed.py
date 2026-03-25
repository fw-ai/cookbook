#!/usr/bin/env python3
"""Benchmark: fixed concurrency (max_concurrency=32).

8 batches × 2048 requests (256 prompts × 8 completions), all fired at once.
SDK semaphore(32) throttles in-flight requests.
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
from fireworks.training.sdk.deployment import DeploymentSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

API_KEY = os.environ["FIREWORKS_API_KEY"]
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
DEPLOYMENT_ID = os.environ["BENCH_DEPLOYMENT_ID"]
TOKENIZER_MODEL = "Qwen/Qwen3-30B-A3B"
DATASET_PATH = os.path.join(_SRC, "training", "examples", "deepmath_rl", "dataset.jsonl")

PROMPTS_PER_BATCH = 256
COMPLETIONS_PER_PROMPT = 8
NUM_BATCHES = 8
MAX_TOKENS = 131072
MAX_CONCURRENCY = int(os.environ.get("BENCH_MAX_CONCURRENCY", "32"))


def load_dataset():
    rows = []
    with open(DATASET_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def summarize_metrics(metrics_list):
    """Aggregate ServerMetrics into a summary dict."""
    pqs = [m.prefill_queue_duration for m in metrics_list if m.prefill_queue_duration is not None]
    gqs = [m.generation_queue_duration for m in metrics_list if m.generation_queue_duration is not None]
    ttfts = [m.server_ttft for m in metrics_list if m.server_ttft is not None]
    client_ttfts = [m.client_ttft for m in metrics_list if m.client_ttft is not None]
    cached = sum(m.cached_prompt_tokens or 0 for m in metrics_list)
    total_prompt = sum(m.prompt_tokens or 0 for m in metrics_list)
    concurrent = [m.num_concurrent_requests for m in metrics_list if m.num_concurrent_requests is not None]

    d = {"n_metrics": len(metrics_list)}
    if pqs:
        d["avg_pq"] = round(statistics.mean(pqs), 4)
        d["p50_pq"] = round(statistics.median(pqs), 4)
        d["max_pq"] = round(max(pqs), 4)
    if gqs:
        d["avg_gq"] = round(statistics.mean(gqs), 4)
    if ttfts:
        d["avg_server_ttft"] = round(statistics.mean(ttfts), 3)
    if client_ttfts:
        d["avg_client_ttft"] = round(statistics.mean(client_ttfts), 3)
        d["p50_client_ttft"] = round(statistics.median(client_ttfts), 3)
    if total_prompt > 0:
        d["cache_hit_rate"] = round(cached / total_prompt, 3)
    if concurrent:
        d["avg_concurrent"] = round(statistics.mean(concurrent), 1)
        d["max_concurrent"] = max(concurrent)
    return d


async def run():
    dataset = load_dataset()
    logger.info("Loaded %d rows", len(dataset))

    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    inference_model = f"accounts/pyroworks/deployments/{DEPLOYMENT_ID}"

    sampler = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer,
        max_concurrency=MAX_CONCURRENCY,
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
    sampler.drain_metrics()  # discard warmup metrics

    total_needed = NUM_BATCHES * PROMPTS_PER_BATCH
    all_rows = (dataset * ((total_needed // len(dataset)) + 1))[:total_needed]

    all_batch_results = []
    total_start = time.time()

    async def _run_batch(batch_idx, batch_rows):
        batch_start = time.time()
        request_latencies = []

        async def _sample(row, idx):
            messages = row.get("messages", [])
            t0 = time.time()
            try:
                completions = await sampler.sample_with_tokens(
                    messages=messages, n=COMPLETIONS_PER_PROMPT,
                    max_tokens=MAX_TOKENS, temperature=0.7,
                    logprobs=True, http_timeout=600,
                )
                request_latencies.append(time.time() - t0)
                return completions
            except Exception as e:
                request_latencies.append(time.time() - t0)
                logger.warning("Batch %d prompt %d failed: %s", batch_idx + 1, idx, e)
                return []

        tasks = [_sample(row, i) for i, row in enumerate(batch_rows)]
        results = await asyncio.gather(*tasks)

        batch_all = [c for r in results for c in r]
        batch_ok = len(batch_all)
        batch_err = sum(1 for r in results if not r)
        batch_elapsed = time.time() - batch_start

        # Drain metrics collected during this batch
        batch_metrics = sampler.drain_metrics()
        metrics_summary = summarize_metrics(batch_metrics)

        comp_lens = [c.completion_len for c in batch_all]
        avg_len = statistics.mean(comp_lens) if comp_lens else 0
        median_len = statistics.median(comp_lens) if comp_lens else 0
        finish = {}
        for c in batch_all:
            finish[c.finish_reason] = finish.get(c.finish_reason, 0) + 1

        first_preview = batch_all[0].text[:200].replace("\n", " ") if batch_all else ""

        batch_result = {
            "batch": batch_idx + 1,
            "completions": batch_ok,
            "expected": PROMPTS_PER_BATCH * COMPLETIONS_PER_PROMPT,
            "errors": batch_err,
            "wall_time_s": round(batch_elapsed, 1),
            "avg_request_latency_s": round(statistics.mean(request_latencies), 1) if request_latencies else 0,
            "p50_request_latency_s": round(statistics.median(request_latencies), 1) if request_latencies else 0,
            "avg_len": round(avg_len),
            "median_len": round(median_len),
            "finish": finish,
            **metrics_summary,
        }

        logger.info(
            "Batch %d/%d: %d/%d ok, %d err | wall=%.0fs avg_lat=%.0fs | "
            "avg_len=%d median=%d | pq=%.3fs cache=%.1f%% concurrent=%s",
            batch_idx + 1, NUM_BATCHES, batch_ok, batch_result["expected"],
            batch_err, batch_elapsed,
            batch_result["avg_request_latency_s"],
            avg_len, median_len,
            metrics_summary.get("avg_pq", 0),
            metrics_summary.get("cache_hit_rate", 0) * 100,
            metrics_summary.get("avg_concurrent", "N/A"),
        )
        logger.info("  First: (%d tok) %s%s",
                     batch_all[0].completion_len if batch_all else 0,
                     first_preview, "..." if len(first_preview) >= 200 else "")
        return batch_result

    batch_tasks = []
    for batch in range(NUM_BATCHES):
        batch_rows = all_rows[batch * PROMPTS_PER_BATCH : (batch + 1) * PROMPTS_PER_BATCH]
        batch_tasks.append(_run_batch(batch, batch_rows))

    all_batch_results = await asyncio.gather(*batch_tasks)

    total_elapsed = time.time() - total_start
    total_completions = sum(b["completions"] for b in all_batch_results)
    total_expected = NUM_BATCHES * PROMPTS_PER_BATCH * COMPLETIONS_PER_PROMPT
    total_errors = sum(b["errors"] for b in all_batch_results)

    summary = {
        "mode": "fixed",
        "max_concurrency": MAX_CONCURRENCY,
        "total_completions": total_completions,
        "total_expected": total_expected,
        "total_errors": total_errors,
        "success_rate": round(total_completions / total_expected * 100, 2),
        "total_time_s": round(total_elapsed, 1),
        "avg_batch_wall_s": round(statistics.mean([b["wall_time_s"] for b in all_batch_results]), 1),
        "avg_request_latency_s": round(statistics.mean([b["avg_request_latency_s"] for b in all_batch_results]), 1),
        "avg_response_len": round(statistics.mean([b["avg_len"] for b in all_batch_results])),
        "throughput_completions_per_s": round(total_completions / total_elapsed, 2),
    }

    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE: FIXED (max_concurrency=%d)", MAX_CONCURRENCY)
    for k, v in summary.items():
        logger.info("  %-30s %s", k, v)
    logger.info("=" * 60)

    with open("bench_fixed_results.json", "w") as f:
        json.dump({"summary": summary, "batches": all_batch_results}, f, indent=2, default=str)

    sampler.close()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
