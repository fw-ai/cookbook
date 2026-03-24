#!/usr/bin/env python3
"""Thorough E2E test: fresh deployment, no scaling, proper warmup.

Tests on a single fresh deployment (no prior hotloads):
  1. Adaptive concurrency, 32 prompts/step × 8 completions, 4 steps
  2. Adaptive concurrency, 64 prompts/step × 8 completions, 2 steps (high concurrency)
  3. Fixed concurrency mode, 32 prompts/step × 8 completions, 2 steps
  4. Adaptive with filtering (drop < 200 tokens), 32 prompts/step, 2 steps

All tests use the SAME deployment with 2 replicas, no scaling between tests.
"""

from __future__ import annotations

import os
import sys
import asyncio
import json
import logging
import time

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import transformers

from fireworks.training.sdk.deployment import (
    AdaptiveConcurrencyController,
    DeploymentConfig,
    DeploymentManager,
    DeploymentSampler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("FIREWORKS_API_KEY", "fw_58efLjimG74e2zwAf69iqS")
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
DEPLOYMENT_ID = "e2e-thorough-chengxili-v1"
DEPLOYMENT_SHAPE = "accounts/fireworks/deploymentShapes/rft-qwen3-30b-a3b-throughput/versions/ai5r0aoy"
BASE_MODEL = "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507"
TOKENIZER_MODEL = "Qwen/Qwen3-30B-A3B"
REPLICA_COUNT = 2
COMPLETIONS_PER_PROMPT = 8
MAX_TOKENS = 4096
DATASET_PATH = os.path.join(_SRC, "training", "examples", "deepmath_rl", "dataset.jsonl")


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
    logger.info("  prompts/step=%d, completions/prompt=%d, steps=%d",
                prompts_per_step, COMPLETIONS_PER_PROMPT, num_steps)
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

        step_all = [c for batch in results for c in batch]
        step_ok = len(step_all)
        step_err = sum(1 for batch in results if not batch)

        step_filtered = 0
        if filter_fn and step_all:
            before = len(step_all)
            step_all = [c for c in step_all if filter_fn(c)]
            step_filtered = before - len(step_all)

        total_completions += step_ok
        total_errors += step_err
        total_filtered += step_filtered
        step_elapsed = time.time() - step_start

        comp_lens = [c.completion_len for c in step_all] if step_all else [0]
        avg_len = sum(comp_lens) / len(comp_lens)
        finish = {}
        for c in step_all:
            finish[c.finish_reason] = finish.get(c.finish_reason, 0) + 1

        parts = [f"ok={step_ok}", f"err={step_err}", f"filtered={step_filtered}",
                 f"avg_len={avg_len:.0f}", f"finish={finish}", f"time={step_elapsed:.1f}s"]

        if ctrl:
            cc = ctrl.step_completed()
            parts.append(f"window={cc.get('window_after', cc.get('window', '?'))}")
            if "avg_pq" in cc:
                parts.append(f"avg_pq={cc['avg_pq']:.3f}s")
            if "cache_hit_rate" in cc:
                parts.append(f"cache={cc['cache_hit_rate']:.1%}")

        logger.info("  Step %d/%d: %s", step + 1, num_steps, " | ".join(parts))

    total_elapsed = time.time() - t_start
    expected = num_steps * prompts_per_step * COMPLETIONS_PER_PROMPT
    passed = total_completions == expected

    logger.info("  RESULT: %s -- %d/%d completions, %d errors, %d filtered, %.1fs",
                "PASS" if passed else "FAIL", total_completions, expected, total_errors, total_filtered, total_elapsed)
    return passed


def main():
    mgr = DeploymentManager(api_key=API_KEY, base_url=BASE_URL)
    dataset = load_dataset()
    logger.info("Loaded %d rows", len(dataset))

    # --- Create fresh deployment ---
    logger.info("Creating fresh deployment %s (2 replicas)...", DEPLOYMENT_ID)
    config = DeploymentConfig(
        deployment_id=DEPLOYMENT_ID,
        base_model=BASE_MODEL,
        deployment_shape=DEPLOYMENT_SHAPE,
        min_replica_count=REPLICA_COUNT,
        max_replica_count=REPLICA_COUNT,
        hot_load_bucket_type="FW_HOSTED",
    )
    mgr.create_or_get(config)
    logger.info("Waiting for READY...")
    mgr.wait_for_ready(DEPLOYMENT_ID, timeout_s=600)
    logger.info("Deployment READY")

    inference_model = f"accounts/{mgr.account_id}/deployments/{DEPLOYMENT_ID}"
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)

    # --- Warmup: send a few requests to make sure it's serving ---
    logger.info("Warming up...")
    warmup_sampler = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer,
    )
    for attempt in range(10):
        try:
            result = asyncio.run(warmup_sampler.sample_with_tokens(
                messages=[{"role": "user", "content": "Say hi."}],
                n=1, max_tokens=8, temperature=0.0,
            ))
            if result:
                logger.info("Warmup done (attempt %d)", attempt + 1)
                break
        except Exception as e:
            logger.info("Warmup attempt %d: %s", attempt + 1, e)
            time.sleep(5)
    warmup_sampler.close()

    results = {}

    # --- Test 1: Adaptive, 32 prompts/step, 4 steps ---
    ctrl1 = AdaptiveConcurrencyController(
        initial_window=8 * REPLICA_COUNT,
        min_window=1, max_window=64,
        prefill_queue_target=0.5, ema_alpha=0.3,
    )
    s1 = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer, concurrency_controller=ctrl1,
    )
    results["adaptive_32x4"] = asyncio.run(run_test(
        "Adaptive 32 prompts/step × 4 steps", s1, dataset,
        prompts_per_step=32, num_steps=4, ctrl=ctrl1,
    ))
    s1.close()

    # --- Test 2: Adaptive, 64 prompts/step, 2 steps (high concurrency) ---
    ctrl2 = AdaptiveConcurrencyController(
        initial_window=8 * REPLICA_COUNT,
        min_window=1, max_window=64,
        prefill_queue_target=0.5, ema_alpha=0.3,
    )
    s2 = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer, concurrency_controller=ctrl2,
    )
    results["adaptive_64x2"] = asyncio.run(run_test(
        "Adaptive 64 prompts/step × 2 steps (512 concurrent)", s2, dataset,
        prompts_per_step=64, num_steps=2, ctrl=ctrl2,
    ))
    s2.close()

    # --- Test 3: Fixed concurrency, 32 prompts/step, 2 steps ---
    s3 = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer,
        max_concurrency=16,
    )
    results["fixed_32x2"] = asyncio.run(run_test(
        "Fixed concurrency=16, 32 prompts/step × 2 steps", s3, dataset,
        prompts_per_step=32, num_steps=2,
    ))
    s3.close()

    # --- Test 4: Adaptive with filter ---
    ctrl4 = AdaptiveConcurrencyController(
        initial_window=8 * REPLICA_COUNT,
        min_window=1, max_window=64,
        prefill_queue_target=0.5, ema_alpha=0.3,
    )
    s4 = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer, concurrency_controller=ctrl4,
    )
    results["adaptive_filter"] = asyncio.run(run_test(
        "Adaptive + filter (drop < 200 tokens), 32 prompts/step × 2 steps",
        s4, dataset, prompts_per_step=32, num_steps=2, ctrl=ctrl4,
        filter_fn=lambda c: c.completion_len >= 200,
    ))
    s4.close()

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("E2E THOROUGH TEST RESULTS")
    for name, passed in results.items():
        logger.info("  %-25s %s", name, "PASS" if passed else "FAIL")
    all_passed = all(results.values())
    logger.info("OVERALL: %s", "ALL PASSED" if all_passed else "SOME FAILED")
    logger.info("=" * 60)

    # Cleanup
    logger.info("Deleting deployment %s...", DEPLOYMENT_ID)
    try:
        mgr.scale_to_zero(DEPLOYMENT_ID)
        mgr._delete(
            f"/v1/accounts/{mgr.account_id}/deployments/{DEPLOYMENT_ID}?ignoreChecks=true&hard=true"
        )
        logger.info("Deleted")
    except Exception as e:
        logger.warning("Cleanup: %s", e)
    mgr.close()

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
