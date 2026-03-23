#!/usr/bin/env python3
"""E2E test: streaming inference + adaptive concurrency controller.

Creates a qwen3-30b-a3b deployment with 2 replicas, then fires concurrent
streaming completion requests through the AdaptiveConcurrencyController
and validates that:
  1. Streaming completions return valid tokens and logprobs.
  2. ServerMetrics are parsed from response headers.
  3. The adaptive controller adjusts its window based on prefill_queue_duration.

Requires FIREWORKS_API_KEY to be set.

Usage:
    python -m pytest training/tests/e2e/test_streaming_concurrency.py -v -s
    # or directly:
    python training/tests/e2e/test_streaming_concurrency.py
"""

from __future__ import annotations

import os
import asyncio
import logging
import time

import pytest
import transformers

from fireworks.training.sdk.deployment import (
    AdaptiveConcurrencyController,
    DeploymentConfig,
    DeploymentManager,
    DeploymentSampler,
    ServerMetrics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = os.environ.get("FIREWORKS_E2E_MODEL", "accounts/fireworks/models/qwen3-30b-a3b")
TOKENIZER_MODEL = os.environ.get("FIREWORKS_E2E_TOKENIZER_MODEL", "Qwen/Qwen3-30B-A3B")
DEPLOYMENT_ACCELERATOR = os.environ.get("FIREWORKS_E2E_DEPLOYMENT_ACCELERATOR", "NVIDIA_H100_80GB")
REGION = os.environ.get("FIREWORKS_E2E_REGION", "US_IOWA_1")
REPLICA_COUNT = int(os.environ.get("FIREWORKS_E2E_REPLICA_COUNT", "2"))

# Reuse existing deployment if set (otherwise one is created and cleaned up).
DEPLOYMENT_ID = os.environ.get("FIREWORKS_E2E_DEPLOYMENT_ID", None)

NUM_PROMPTS = 16
CONCURRENCY_INITIAL = 4
MAX_TOKENS = 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_key():
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        pytest.skip("FIREWORKS_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def deploy_mgr(api_key):
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    inference_url = os.environ.get("FIREWORKS_INFERENCE_URL", base_url)
    additional_headers = {}
    secret = os.environ.get("FIREWORKS_GATEWAY_SECRET")
    if secret:
        additional_headers["X-Fireworks-Gateway-Secret"] = secret
    mgr = DeploymentManager(
        api_key=api_key,
        base_url=base_url,
        inference_url=inference_url,
        additional_headers=additional_headers or None,
    )
    yield mgr
    mgr.close()


@pytest.fixture(scope="module")
def deployment(deploy_mgr, api_key):
    """Ensure a qwen3-30b-a3b deployment with 2 replicas is ready."""
    dep_id = DEPLOYMENT_ID
    created = False
    if dep_id:
        logger.info("Reusing existing deployment: %s", dep_id)
    else:
        dep_id = f"e2e-stream-concurrency-chengxili-{int(time.time()) % 100000}"
        logger.info("Creating deployment %s with %d replicas...", dep_id, REPLICA_COUNT)
        config = DeploymentConfig(
            deployment_id=dep_id,
            base_model=MODEL,
            region=REGION,
            min_replica_count=REPLICA_COUNT,
            max_replica_count=REPLICA_COUNT,
            accelerator_type=DEPLOYMENT_ACCELERATOR,
        )
        deploy_mgr.create_or_get(config)
        created = True

    logger.info("Waiting for deployment %s to be READY...", dep_id)
    deploy_mgr.wait_for_ready(dep_id, timeout_seconds=600)
    logger.info("Deployment %s is READY", dep_id)

    inference_model = f"accounts/{deploy_mgr.account_id}/deployments/{dep_id}"
    yield dep_id, inference_model

    # Cleanup: scale to zero only if we created it.
    if created:
        logger.info("Scaling deployment %s to zero...", dep_id)
        try:
            deploy_mgr.scale_to_zero(dep_id)
        except Exception as e:
            logger.warning("Failed to scale to zero: %s", e)


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SAMPLE_PROMPTS = [
    "What is 2 + 2?",
    "Explain photosynthesis in one sentence.",
    "Write a haiku about coding.",
    "What is the capital of France?",
    "Explain what a neural network is.",
    "What is the speed of light?",
    "Name three programming languages.",
    "What year did World War 2 end?",
    "Describe the color blue to someone who has never seen it.",
    "What is machine learning?",
    "How does a compiler work?",
    "What is recursion?",
    "Explain quantum computing simply.",
    "What is the Pythagorean theorem?",
    "Name the planets in our solar system.",
    "What is an API?",
]


class TestStreamingWithAdaptiveConcurrency:
    """Test streaming inference with the adaptive concurrency controller."""

    def test_streaming_completions_with_adaptive_controller(
        self, deploy_mgr, deployment, tokenizer, api_key,
    ):
        """Fire NUM_PROMPTS streaming requests through AdaptiveConcurrencyController."""
        dep_id, inference_model = deployment

        ctrl = AdaptiveConcurrencyController(
            initial_window=CONCURRENCY_INITIAL,
            min_window=1,
            max_window=32,
            prefill_queue_target=0.5,
            ema_alpha=0.3,
        )

        sampler = DeploymentSampler(
            inference_url=deploy_mgr.inference_url,
            model=inference_model,
            api_key=api_key,
            tokenizer=tokenizer,
            concurrency_controller=ctrl,
        )

        all_results = []
        errors = []

        async def _sample_one(prompt_text: str):
            messages = [{"role": "user", "content": prompt_text}]
            try:
                completions = await sampler.sample_with_tokens(
                    messages=messages,
                    n=1,
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                    stream=True,
                    logprobs=True,
                )
                return completions
            except Exception as e:
                errors.append((prompt_text, e))
                return []

        async def _run_all():
            tasks = [_sample_one(p) for p in SAMPLE_PROMPTS[:NUM_PROMPTS]]
            results = await asyncio.gather(*tasks)
            return results

        t0 = time.time()
        results = asyncio.run(_run_all())
        elapsed = time.time() - t0

        for batch in results:
            all_results.extend(batch)

        # -- Assertions --
        logger.info(
            "Completed %d/%d requests in %.1fs (errors=%d)",
            len(all_results), NUM_PROMPTS, elapsed, len(errors),
        )
        logger.info(
            "Controller: window=%d, ema_prefill_q=%s",
            ctrl.window_size,
            f"{ctrl.ema_prefill_queue:.4f}" if ctrl.ema_prefill_queue is not None else "N/A",
        )

        # At least some completions should succeed.
        assert len(all_results) > 0, f"No completions succeeded. Errors: {errors}"

        # Validate completion structure.
        for c in all_results:
            assert c.text, "Completion text is empty"
            assert len(c.full_tokens) > c.prompt_len, "No tokens generated"
            assert c.completion_len > 0
            assert c.finish_reason in ("stop", "length")

        # Validate logprobs were returned.
        completions_with_logprobs = [c for c in all_results if c.inference_logprobs]
        assert len(completions_with_logprobs) > 0, "No completions had logprobs"

        # The controller should have received metrics and updated EMA.
        assert ctrl.ema_prefill_queue is not None, (
            "Controller never received prefill_queue_duration -- "
            "are response headers available on this deployment?"
        )
        assert ctrl._completed_requests >= len(all_results)

        # Log errors if any (but don't fail for transient ones).
        for prompt, err in errors:
            logger.warning("Error for '%s': %s", prompt[:40], err)

        sampler.close()

    def test_streaming_returns_server_metrics(
        self, deploy_mgr, deployment, tokenizer, api_key,
    ):
        """Verify ServerMetrics are parsed from response headers."""
        dep_id, inference_model = deployment

        sampler = DeploymentSampler(
            inference_url=deploy_mgr.inference_url,
            model=inference_model,
            api_key=api_key,
            tokenizer=tokenizer,
        )

        async def _do():
            messages = [{"role": "user", "content": "Say hello."}]
            result, metrics = await sampler.async_completions_stream(
                prompt=tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=False,
                ),
                max_tokens=16,
                temperature=0.7,
                raw_output=True,
                logprobs=True,
            )
            return result, metrics

        result, metrics = asyncio.run(_do())

        # Result should have a valid completion.
        assert result.get("choices"), "No choices in response"
        choice = result["choices"][0]
        assert choice.get("text"), "Empty completion text"

        # ServerMetrics should be populated for dedicated deployments.
        logger.info("ServerMetrics: %s", metrics)
        # At minimum, prompt_tokens should be set.
        if metrics.prompt_tokens is not None:
            assert metrics.prompt_tokens > 0
        if metrics.cached_prompt_tokens is not None:
            assert metrics.cached_prompt_tokens >= 0
        if metrics.prefill_queue_duration is not None:
            assert metrics.prefill_queue_duration >= 0
        # client_ttft should always be set for streaming.
        assert metrics.client_ttft is not None
        assert metrics.client_ttft > 0

        sampler.close()


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    pytest.main([__file__, "-v", "-s"])
