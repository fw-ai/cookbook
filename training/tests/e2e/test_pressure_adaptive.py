#!/usr/bin/env python3
"""Pressure test: adaptive concurrency + deprecated max_concurrency compat.

Two runs on the same deployment:
  1. AdaptiveConcurrencyController (primary path)
  2. max_concurrency=16 (deprecated compat — must emit DeprecationWarning)

Each run: 8 steps × 32 prompts × 8 completions = 2048 requests, 128 max tokens.

Usage:
    FIREWORKS_API_KEY=... FIREWORKS_E2E_DEPLOYMENT_ID=... \
    python training/tests/e2e/test_pressure_adaptive.py
"""

from __future__ import annotations

import os
import asyncio
import logging
import time
import warnings

import transformers

from fireworks.training.sdk.deployment import (
    AdaptiveConcurrencyController,
    DeploymentSampler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

API_KEY = os.environ["FIREWORKS_API_KEY"]
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
DEPLOYMENT_ID = os.environ["FIREWORKS_E2E_DEPLOYMENT_ID"]
TOKENIZER_MODEL = os.environ.get("FIREWORKS_E2E_TOKENIZER_MODEL", "Qwen/Qwen3-30B-A3B")
ACCOUNT_ID = os.environ.get("FIREWORKS_ACCOUNT_ID", "pyroworks")

PROMPTS_PER_STEP = 32
COMPLETIONS_PER_PROMPT = 8
NUM_STEPS = 8
MAX_TOKENS = 128

SAMPLE_PROMPTS = [
    "What is 2 + 2?",
    "Explain photosynthesis in one sentence.",
    "Write a haiku about coding.",
    "What is the capital of France?",
    "Explain what a neural network is.",
    "What is the speed of light?",
    "Name three programming languages.",
    "What year did World War 2 end?",
    "Describe the color blue.",
    "What is machine learning?",
    "How does a compiler work?",
    "What is recursion?",
    "Explain quantum computing simply.",
    "What is the Pythagorean theorem?",
    "Name the planets in our solar system.",
    "What is an API?",
    "What is a database index?",
    "Explain the difference between TCP and UDP.",
    "What is a hash function?",
    "How does HTTPS work?",
    "What is a binary tree?",
    "Explain Big O notation.",
    "What is a REST API?",
    "What is containerization?",
    "Explain MapReduce.",
    "What is gradient descent?",
    "Define entropy in information theory.",
    "What is a Markov chain?",
    "Explain backpropagation.",
    "What is a transformer model?",
    "What is attention in deep learning?",
    "Explain the softmax function.",
]


async def run_pressure(sampler: DeploymentSampler, label: str, ctrl: AdaptiveConcurrencyController | None = None):
    """Run NUM_STEPS batches and return (total_completions, total_errors)."""
    total_completions = 0
    total_errors = 0
    total_start = time.time()

    for step in range(NUM_STEPS):
        step_start = time.time()

        async def _sample_one(prompt_text: str, prompt_idx: int):
            messages = [{"role": "user", "content": prompt_text}]
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
                logger.warning("[%s] Step %d prompt %d failed: %s", label, step + 1, prompt_idx, e)
                return 0, 1

        tasks = [_sample_one(p, i) for i, p in enumerate(SAMPLE_PROMPTS[:PROMPTS_PER_STEP])]
        results = await asyncio.gather(*tasks)

        step_completions = sum(r[0] for r in results)
        step_errors = sum(r[1] for r in results)
        total_completions += step_completions
        total_errors += step_errors
        step_elapsed = time.time() - step_start

        cc_info = ""
        if ctrl is not None:
            summary = ctrl.step_completed()
            cc_info = " | " + " ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()
            )

        logger.info(
            "[%s] Step %d/%d: %d completions, %d errors, %.1fs%s",
            label, step + 1, NUM_STEPS, step_completions, step_errors, step_elapsed, cc_info,
        )

    total_elapsed = time.time() - total_start
    expected = NUM_STEPS * PROMPTS_PER_STEP * COMPLETIONS_PER_PROMPT

    logger.info("=" * 60)
    logger.info("[%s] COMPLETE: %d/%d (errors: %d) in %.1fs", label, total_completions, expected, total_errors, total_elapsed)
    if ctrl is not None:
        logger.info("[%s] Final window: %d, ema_pq: %s", label, ctrl.window_size,
                     f"{ctrl.ema_prefill_queue:.4f}" if ctrl.ema_prefill_queue is not None else "N/A")
    logger.info("=" * 60)

    return total_completions, total_errors, expected


async def main():
    inference_model = f"accounts/{ACCOUNT_ID}/deployments/{DEPLOYMENT_ID}"
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)

    # -- Test 1: Adaptive concurrency controller (primary path) ----------------
    logger.info(">>> TEST 1: AdaptiveConcurrencyController")
    ctrl = AdaptiveConcurrencyController(
        initial_window=16, min_window=1, max_window=64,
        prefill_queue_target=0.5, ema_alpha=0.3,
    )
    sampler = DeploymentSampler(
        inference_url=BASE_URL, model=inference_model,
        api_key=API_KEY, tokenizer=tokenizer,
        concurrency_controller=ctrl,
    )
    ok1, err1, expected1 = await run_pressure(sampler, "adaptive", ctrl)
    sampler.close()

    # -- Test 2: Deprecated max_concurrency (backward compat) ------------------
    logger.info(">>> TEST 2: max_concurrency (deprecated compat)")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampler2 = DeploymentSampler(
            inference_url=BASE_URL, model=inference_model,
            api_key=API_KEY, tokenizer=tokenizer,
            max_concurrency=16,
        )
        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecation_warnings, "Expected DeprecationWarning for max_concurrency but got none"
        logger.info("DeprecationWarning correctly emitted: %s", deprecation_warnings[0].message)

    ok2, err2, expected2 = await run_pressure(sampler2, "fixed-deprecated")
    sampler2.close()

    # -- Assertions ------------------------------------------------------------
    assert ok1 == expected1, f"[adaptive] Expected {expected1}, got {ok1} (errors: {err1})"
    assert ok2 == expected2, f"[fixed-deprecated] Expected {expected2}, got {ok2} (errors: {err2})"

    logger.info("ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
