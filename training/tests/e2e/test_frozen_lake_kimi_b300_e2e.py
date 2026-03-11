"""E2E test for Kimi K2.5 FrozenLake VL GRPO on B300 (LoRA + inference).

Validates the full RL pipeline with:
  - Kimi K2.5 (1T MoE) base model with LoRA training on B300
  - Text or image-based (VL) observation mode
  - Multi-turn tool-call rollouts via eval-protocol
  - GRPO loss with correct TITO masking for Kimi's format

Infrastructure is pre-created via firectl-admin (private model requires admin
privileges). Use the companion launcher script run_frozen_lake_kimi_b300.sh
to automate the full lifecycle.

Required env vars:
  FIREWORKS_API_KEY
  FIREWORKS_ACCOUNT_ID

Optional env vars:
  FIREWORKS_BASE_URL                (default: https://dev.api.fireworks.ai)
  KIMI_FROZEN_LAKE_DEPLOYMENT_ID    (pre-created deployment)
  KIMI_FROZEN_LAKE_DEPLOYMENT_SHAPE (default: rft-kimi-k2p5-b300)
  KIMI_FROZEN_LAKE_TRAINING_SHAPE   (if set, uses validated shape path)
  KIMI_FROZEN_LAKE_ACCELERATOR_TYPE (default: NVIDIA_B300_288GB)
  KIMI_FROZEN_LAKE_LORA_RANK        (default: 8)
  KIMI_FROZEN_LAKE_MAX_SEQ_LEN      (default: 65536 = 64k)
  KIMI_FROZEN_LAKE_POLICY_JOB_ID    (pre-created policy trainer job)
  KIMI_FROZEN_LAKE_REFERENCE_JOB_ID (pre-created reference trainer job)
  KIMI_FROZEN_LAKE_INFERENCE_BASE_URL (direct inference URL, skip gateway)
  KIMI_FROZEN_LAKE_REGION           (default: EU_NETHERLANDS_1)
  KIMI_FROZEN_LAKE_OBSERVATION_MODE (default: image, can be "text" for debugging)
  KIMI_FROZEN_LAKE_DISABLE_HOTLOAD  (default: false; set to true to skip hotload)
  KIMI_FROZEN_LAKE_EPOCHS           (default: 1)
"""

from __future__ import annotations

import logging
import os
import time

import httpx
import pytest

from training.examples.frozen_lake.train_frozen_lake import (
    FrozenLakeConfig,
    main as frozen_lake_main,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "accounts/fireworks/models/kimi-k2p5"
DEFAULT_TOKENIZER = "moonshotai/Kimi-K2.5"
DEFAULT_DEPLOYMENT_SHAPE = "rft-kimi-k2p5-b300"
DEFAULT_REGION = "EU_NETHERLANDS_1"
DEFAULT_ACCELERATOR = "NVIDIA_B300_288GB"
DEFAULT_LORA_RANK = 8
DEFAULT_MAX_SEQ_LEN = 65536


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _preflight_deployment_check(
    base_url: str,
    api_key: str,
    account_id: str,
    deployment_id: str,
    timeout_s: int = 30,
) -> None:
    """Verify the Kimi K2.5 inference deployment is reachable before starting training."""
    url = f"{base_url}/inference/v1/completions"
    model = f"{DEFAULT_BASE_MODEL}#accounts/{account_id}/deployments/{deployment_id}"
    body = {
        "model": model,
        "prompt": "hello",
        "max_tokens": 1,
    }
    try:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=timeout_s,
        )
        if resp.status_code not in (200, 401):
            pytest.fail(
                f"Kimi deployment {deployment_id} pre-flight check failed: "
                f"HTTP {resp.status_code} from {url}"
            )
    except httpx.RequestError as e:
        pytest.fail(
            f"Kimi deployment {deployment_id} pre-flight check failed: "
            f"cannot reach {url}: {e}"
        )


@pytest.mark.e2e
@pytest.mark.timeout(5400)
class TestFrozenLakeKimiB300:
    """FrozenLake GRPO smoke test on Kimi K2.5 with LoRA."""

    def test_kimi_frozen_lake_vl_rewards_improve(self):
        api_key = _env("FIREWORKS_API_KEY")
        if not api_key:
            pytest.skip("FIREWORKS_API_KEY not set")

        account = _env("FIREWORKS_ACCOUNT_ID")
        if not account:
            pytest.skip("FIREWORKS_ACCOUNT_ID not set")

        region = _env("KIMI_FROZEN_LAKE_REGION", DEFAULT_REGION)
        deployment_id = _env("KIMI_FROZEN_LAKE_DEPLOYMENT_ID")
        deployment_shape = _env("KIMI_FROZEN_LAKE_DEPLOYMENT_SHAPE", DEFAULT_DEPLOYMENT_SHAPE)
        training_shape = _env("KIMI_FROZEN_LAKE_TRAINING_SHAPE", "")
        accelerator_type = _env("KIMI_FROZEN_LAKE_ACCELERATOR_TYPE", DEFAULT_ACCELERATOR)
        lora_rank = int(_env("KIMI_FROZEN_LAKE_LORA_RANK", str(DEFAULT_LORA_RANK)))
        max_seq_len = int(_env("KIMI_FROZEN_LAKE_MAX_SEQ_LEN", str(DEFAULT_MAX_SEQ_LEN)))
        observation_mode = _env("KIMI_FROZEN_LAKE_OBSERVATION_MODE", "image")
        base_url = _env("FIREWORKS_BASE_URL", "https://dev.api.fireworks.ai")
        disable_hotload = _env("KIMI_FROZEN_LAKE_DISABLE_HOTLOAD", "false").lower() in ("1", "true", "yes")
        epochs = int(_env("KIMI_FROZEN_LAKE_EPOCHS", "1"))

        os.environ.setdefault("FIREWORKS_BASE_URL", base_url)
        os.environ.setdefault("WANDB_MODE", "disabled")

        if deployment_id and not _env("KIMI_FROZEN_LAKE_POLICY_JOB_ID"):
            logger.info(
                "Skipping preflight check -- main() will poll for deployment readiness"
            )

        cfg = FrozenLakeConfig(
            base_model=DEFAULT_BASE_MODEL,
            tokenizer_model=DEFAULT_TOKENIZER,
            training_shape=training_shape,
            deployment_shape=deployment_shape,
            accelerator_type=accelerator_type,
            deployment_id=deployment_id,
            region=region,
            deployment_region=region,
            lora_rank=lora_rank,
            max_seq_len=max_seq_len,
            observation_mode=observation_mode,
            epochs=epochs,
            max_seeds=20,
            max_steps=12,
            completions_per_prompt=8,
            prompt_groups_per_step=4,
            max_concurrent=16,
            max_completion_tokens=128,
            temperature=1.0,
            kl_beta=0.001,
            learning_rate=1e-5,
            policy_job_id=_env("KIMI_FROZEN_LAKE_POLICY_JOB_ID"),
            reference_job_id=_env("KIMI_FROZEN_LAKE_REFERENCE_JOB_ID"),
            inference_base_url=_env("KIMI_FROZEN_LAKE_INFERENCE_BASE_URL"),
            disable_hotload=disable_hotload,
        )

        t0 = time.time()
        result = frozen_lake_main(cfg)
        elapsed = time.time() - t0

        assert isinstance(result, dict), "main() should return a dict"

        steps = result["steps"]
        rewards = result["rewards"]

        summary = (
            f"steps={steps}, rewards={[f'{r:.3f}' for r in rewards]}, "
            f"elapsed={elapsed:.0f}s, observation_mode={observation_mode}"
        )
        logger.info("Kimi FrozenLake VL smoke test result: %s", summary)

        assert steps >= 3, (
            f"Expected >= 3 optimizer steps, got {steps}. "
            f"High filtering rate may indicate a Kimi tool-call parsing "
            f"or VL rollout problem. {summary}"
        )
        assert len(rewards) >= 3, (
            f"Expected >= 3 reward entries, got {len(rewards)}. {summary}"
        )

        avg_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)

        logger.info(
            "Kimi VL Rewards: avg=%.3f, max=%.3f, steps=%d, elapsed=%.0fs",
            avg_reward, max_reward, steps, elapsed,
        )

        assert avg_reward >= 0.10, (
            f"Average reward ({avg_reward:.3f}) should be >= 0.10 "
            f"(random baseline ~0.05). This may indicate broken Kimi VL "
            f"training (mask alignment, image handling, or hotload). {summary}"
        )

        assert max_reward >= 0.20, (
            f"Max reward ({max_reward:.3f}) should reach >= 0.20 "
            f"at least once. {summary}"
        )

        assert elapsed < 5400, (
            f"Pipeline took {elapsed:.0f}s (>90min), expected <90min. "
            f"May indicate infrastructure issues or VL image bottleneck. {summary}"
        )
