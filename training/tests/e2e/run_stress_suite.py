#!/usr/bin/env python3
"""Stress test suite: multiple configs against the same deployment.

Tests:
  1. 1 replica, adaptive (force AIMD decrease)
  2. 2 replicas, 64 prompts/step (high concurrency pressure)
  3. 2 replicas, fixed concurrency mode (backward compat)
  4. 2 replicas, adaptive with dynamic filtering

Usage:
    python training/tests/e2e/run_stress_suite.py
"""

from __future__ import annotations

import os
import sys
import asyncio
import json
import logging
import time
import traceback

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

API_KEY = os.environ.get("FIREWORKS_API_KEY", "fw_58efLjimG74e2zwAf69iqS")
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
DEPLOYMENT_ID = "e2e-pressure-chengxili-v2"
TOKENIZER_MODEL = "Qwen/Qwen3-30B-A3B"
DATASET_PATH = os.path.join(_SRC, "training", "examples", "deepmath_rl", "dataset.jsonl")

COMPLETIONS_PER_PROMPT = 8
MAX_TOKENS = 4096  # Short enough to finish fast, long enough to be meaningful


def load_dataset():
    rows = []
    with open(DATASET_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


async def run_test(
    name: str,
    sampler: DeploymentSampler,
    dataset: list,
    prompts_per_step: int,
    num_steps: int,
    ctrl: AdaptiveConcurrencyController | None = None,
    filter_fn=None,
):
    logger.info("=" * 60)
    logger.info("TEST: %s", name)
    logger.info("  prompts/step=%d, completions/prompt=%d, steps=%d, max_tokens=%d",
                prompts_per_step, COMPLETIONS_PER_PROMPT, num_steps, MAX_TOKENS)
    if ctrl:
        logger.info("  adaptive: initial=%d, range=%d-%d, target_pq=%.2fs",
                     ctrl.window_size, ctrl._min_window, ctrl._max_window, ctrl._prefill_queue_target)
    else:
        logger.info("  fixed concurrency (via sampler semaphore)")
    logger.info("=" * 60)

    total_needed = num_steps * prompts_per_step
    all_rows = (dataset * ((total_needed // len(dataset)) + 1))[:total_needed]
    total_completions = 0
    total_errors = 0
    total_filtered = 0
    t_start = time.time()

    for step in range(num_steps):
        step_start = time.time()
        step_rows = all_rows[step * prompts_per_step : (step + 1) * prompts_per_step]

        async def _sample_one(row: dict, idx: int):
            messages = row.get("messages", [])
            try:
                completions = await sampler.sample_with_tokens(
                    messages=messages,
                    n=COMPLETIONS_PER_PROMPT,
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                    logprobs=True,
                    http_timeout=300,
                )
                return completions
            except Exception as e:
                logger.warning("  Step %d prompt %d error: %s", step + 1, idx, e)
                return []

        tasks = [_sample_one(row, i) for i, row in enumerate(step_rows)]
        results = await asyncio.gather(*tasks)

        step_completions_list = [c for batch in results for c in batch]
        step_ok = len(step_completions_list)
        step_err = sum(1 for batch in results if not batch)

        # Apply filter if provided
        step_filtered = 0
        if filter_fn and step_completions_list:
            before = len(step_completions_list)
            step_completions_list = [c for c in step_completions_list if filter_fn(c)]
            step_filtered = before - len(step_completions_list)

        total_completions += step_ok
        total_errors += step_err
        total_filtered += step_filtered
        step_elapsed = time.time() - step_start

        comp_lens = [c.completion_len for c in step_completions_list] if step_completions_list else [0]
        avg_len = sum(comp_lens) / len(comp_lens) if comp_lens else 0

        summary_parts = [f"completions={step_ok}", f"errors={step_err}", f"filtered={step_filtered}",
                         f"avg_len={avg_len:.0f}", f"time={step_elapsed:.1f}s"]

        if ctrl:
            cc = ctrl.step_completed()
            summary_parts.append(f"window={cc.get('window_after', cc.get('window', '?'))}")
            if "avg_pq" in cc:
                summary_parts.append(f"avg_pq={cc['avg_pq']:.3f}s")
            if "cache_hit_rate" in cc:
                summary_parts.append(f"cache={cc['cache_hit_rate']:.1%}")

        logger.info("  Step %d/%d: %s", step + 1, num_steps, " | ".join(summary_parts))

    total_elapsed = time.time() - t_start
    expected = num_steps * prompts_per_step * COMPLETIONS_PER_PROMPT
    passed = total_completions == expected

    logger.info("  RESULT: %s -- %d/%d completions, %d errors, %d filtered, %.1fs",
                "PASS" if passed else "FAIL", total_completions, expected, total_errors, total_filtered, total_elapsed)
    if ctrl:
        logger.info("  Final window: %d, ema_pq: %s",
                     ctrl.window_size,
                     f"{ctrl.ema_prefill_queue:.4f}" if ctrl.ema_prefill_queue is not None else "N/A")
    return passed


def main():
    dataset = load_dataset()
    logger.info("Loaded %d rows from %s", len(dataset), DATASET_PATH)

    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    inference_model = f"accounts/pyroworks/deployments/{DEPLOYMENT_ID}"

    results = {}

    # ---- Test 1: 1 replica, adaptive (should trigger AIMD decrease) ----
    ctrl1 = AdaptiveConcurrencyController(
        initial_window=8,  # 8 * 1 replica
        min_window=1, max_window=32,
        prefill_queue_target=0.3,  # Lower target to make decrease more likely
        ema_alpha=0.5,
    )
    sampler1 = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer, concurrency_controller=ctrl1,
    )
    results["1_replica_adaptive"] = asyncio.run(run_test(
        "1 replica, adaptive (target_pq=0.3s)",
        sampler1, dataset, prompts_per_step=16, num_steps=4, ctrl=ctrl1,
    ))
    sampler1.close()

    # ---- Scale back to 2 replicas and warmup before remaining tests ----
    import httpx
    logger.info("Scaling back to 2 replicas...")
    resp = httpx.patch(
        f"{BASE_URL}/v1/accounts/pyroworks/deployments/{DEPLOYMENT_ID}",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"minReplicaCount": 2, "maxReplicaCount": 2},
        timeout=30,
    )
    logger.info("Scale response: %d", resp.status_code)
    # Wait for 2 replicas to be ready.
    for _ in range(30):
        r = httpx.get(
            f"{BASE_URL}/v1/accounts/pyroworks/deployments/{DEPLOYMENT_ID}",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=15,
        )
        data = r.json()
        ready = data.get("replicaStats", {}).get("readyReplicaCount", 0)
        if ready >= 2 and data.get("state") == "READY":
            logger.info("2 replicas ready")
            break
        time.sleep(10)
    # Warmup: send a single small request to verify the deployment is
    # actually serving (not just READY but still hotloading).
    logger.info("Warming up after scale...")
    warmup_sampler = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer,
    )
    for attempt in range(20):
        try:
            warmup_result = asyncio.run(warmup_sampler.sample_with_tokens(
                messages=[{"role": "user", "content": "Say hi."}],
                n=1, max_tokens=8, temperature=0.0, logprobs=False,
            ))
            if warmup_result:
                logger.info("Warmup succeeded after %d attempt(s)", attempt + 1)
                break
        except Exception as e:
            logger.info("Warmup attempt %d: %s", attempt + 1, e)
            time.sleep(5)
    warmup_sampler.close()

    # ---- Test 2: 2 replicas, 64 prompts/step (high concurrency) ----
    ctrl3 = AdaptiveConcurrencyController(
        initial_window=16,  # 8 * 2
        min_window=1, max_window=64,
        prefill_queue_target=0.5, ema_alpha=0.3,
    )
    sampler3 = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer, concurrency_controller=ctrl3,
    )
    results["high_concurrency_64"] = asyncio.run(run_test(
        "2 replicas, 64 prompts/step (512 concurrent requests)",
        sampler3, dataset, prompts_per_step=64, num_steps=3, ctrl=ctrl3,
    ))
    sampler3.close()

    # ---- Test 3: Fixed concurrency mode ----
    sampler4 = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer,
        max_concurrency=16,
    )
    results["fixed_concurrency"] = asyncio.run(run_test(
        "2 replicas, fixed concurrency=16",
        sampler4, dataset, prompts_per_step=32, num_steps=3,
    ))
    sampler4.close()

    # ---- Test 4: Adaptive with filtering (drop completions < 200 tokens) ----
    ctrl5 = AdaptiveConcurrencyController(
        initial_window=16, min_window=1, max_window=64,
        prefill_queue_target=0.5, ema_alpha=0.3,
    )
    sampler5 = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer, concurrency_controller=ctrl5,
    )
    results["adaptive_with_filter"] = asyncio.run(run_test(
        "2 replicas, adaptive + filter (drop < 200 tokens)",
        sampler5, dataset, prompts_per_step=32, num_steps=3, ctrl=ctrl5,
        filter_fn=lambda c: c.completion_len >= 200,
    ))
    sampler5.close()

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("STRESS SUITE RESULTS")
    for name, passed in results.items():
        logger.info("  %-30s %s", name, "PASS" if passed else "FAIL")
    all_passed = all(results.values())
    logger.info("OVERALL: %s", "ALL PASSED" if all_passed else "SOME FAILED")
    logger.info("=" * 60)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
