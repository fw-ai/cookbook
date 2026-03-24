#!/usr/bin/env python3
"""Create deployment, run pressure test, clean up.

Usage:
    python training/tests/e2e/run_pressure_test.py
"""

from __future__ import annotations

import os
import sys
import time
import asyncio
import logging

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
DEPLOYMENT_ID = "e2e-pressure-chengxili-v2"
DEPLOYMENT_SHAPE = "accounts/fireworks/deploymentShapes/rft-qwen3-30b-a3b-throughput/versions/ai5r0aoy"
BASE_MODEL = "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507"
TOKENIZER_MODEL = "Qwen/Qwen3-30B-A3B"
REPLICA_COUNT = 2

DATASET_PATH = os.environ.get(
    "FIREWORKS_E2E_DATASET",
    os.path.join(os.path.dirname(__file__), "../../examples/deepmath_rl/dataset.jsonl"),
)
PROMPTS_PER_STEP = 32
COMPLETIONS_PER_PROMPT = 8
NUM_STEPS = 8
MAX_TOKENS = 1024
INITIAL_WINDOW = 8 * REPLICA_COUNT


def load_dataset():
    import json
    rows = []
    with open(DATASET_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def create_deployment(mgr: DeploymentManager) -> str:
    config = DeploymentConfig(
        deployment_id=DEPLOYMENT_ID,
        base_model=BASE_MODEL,
        deployment_shape=DEPLOYMENT_SHAPE,
        min_replica_count=REPLICA_COUNT,
        max_replica_count=REPLICA_COUNT,
        hot_load_bucket_type="FW_HOSTED",
    )
    logger.info("Creating deployment %s (%d replicas)...", DEPLOYMENT_ID, REPLICA_COUNT)
    mgr.create_or_get(config)
    logger.info("Waiting for READY...")
    mgr.wait_for_ready(DEPLOYMENT_ID, timeout_s=600)
    logger.info("Deployment READY")
    return f"accounts/{mgr.account_id}/deployments/{DEPLOYMENT_ID}"




async def run_pressure_test(sampler: DeploymentSampler, ctrl: AdaptiveConcurrencyController):
    dataset = load_dataset()
    logger.info("Loaded %d rows from %s", len(dataset), DATASET_PATH)
    total_needed = NUM_STEPS * PROMPTS_PER_STEP
    # Cycle through dataset if we need more prompts than available.
    all_rows = (dataset * ((total_needed // len(dataset)) + 1))[:total_needed]

    total_completions = 0
    total_errors = 0
    total_start = time.time()

    for step in range(NUM_STEPS):
        step_start = time.time()
        step_rows = all_rows[step * PROMPTS_PER_STEP : (step + 1) * PROMPTS_PER_STEP]

        async def _sample_one(row: dict, prompt_idx: int):
            messages = row.get("messages", [])
            try:
                completions = await sampler.sample_with_tokens(
                    messages=messages,
                    n=COMPLETIONS_PER_PROMPT,
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                    logprobs=True,
                )
                return len(completions), 0
            except Exception as e:
                logger.warning("Step %d prompt %d failed: %s", step + 1, prompt_idx, e)
                return 0, 1

        tasks = [_sample_one(row, i) for i, row in enumerate(step_rows)]
        results = await asyncio.gather(*tasks)

        step_completions = sum(r[0] for r in results)
        step_errors = sum(r[1] for r in results)
        total_completions += step_completions
        total_errors += step_errors
        step_elapsed = time.time() - step_start

        summary = ctrl.step_completed()

        logger.info(
            "Step %d/%d: %d completions, %d errors, %.1fs | %s",
            step + 1, NUM_STEPS, step_completions, step_errors, step_elapsed,
            " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()),
        )

    total_elapsed = time.time() - total_start
    expected = NUM_STEPS * PROMPTS_PER_STEP * COMPLETIONS_PER_PROMPT

    logger.info("=" * 60)
    logger.info("PRESSURE TEST COMPLETE")
    logger.info("  Total completions: %d/%d (errors: %d)", total_completions, expected, total_errors)
    logger.info("  Total time: %.1fs", total_elapsed)
    logger.info("  Throughput: %.1f completions/s", total_completions / total_elapsed if total_elapsed > 0 else 0)
    logger.info("  Final window: %d", ctrl.window_size)
    logger.info("  Final ema_pq: %s", f"{ctrl.ema_prefill_queue:.4f}" if ctrl.ema_prefill_queue is not None else "N/A")
    logger.info("=" * 60)

    return total_completions == expected


def main():
    mgr = DeploymentManager(api_key=API_KEY, base_url=BASE_URL)
    inference_model = None
    inference_model = create_deployment(mgr)

    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
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

    success = asyncio.run(run_pressure_test(sampler, ctrl))
    sampler.close()
    mgr.close()

    if not success:
        logger.error("PRESSURE TEST FAILED")
        sys.exit(1)
    logger.info("PRESSURE TEST PASSED")
    logger.info("Deployment %s kept alive for further testing", DEPLOYMENT_ID)


if __name__ == "__main__":
    main()
