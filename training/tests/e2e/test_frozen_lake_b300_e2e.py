"""E2E smoke test for FrozenLake GRPO on B300 GPUs.

Validates the full Firetitan RL pipeline end-to-end:
  - RLOR trainer job lifecycle (policy + reference)
  - Multi-turn tool-call rollouts via eval-protocol
  - GRPO loss with KL divergence
  - Hotloading (weight sync to inference deployment)
  - Reward-based learning signal

Required env vars:
  FIREWORKS_API_KEY
  FIREWORKS_ACCOUNT_ID
  FROZEN_LAKE_DEPLOYMENT_ID    (pre-created deployment with hotload)

Optional env vars (CI script sets these):
  FIREWORKS_BASE_URL           (default: https://dev.api.fireworks.ai)
  FROZEN_LAKE_POLICY_JOB_ID    (pre-created policy trainer job)
  FROZEN_LAKE_REFERENCE_JOB_ID (pre-created reference trainer job)
  FROZEN_LAKE_DEPLOYMENT_ID    (pre-created deployment)
  FROZEN_LAKE_TRAINING_SHAPE   (default: qwen3-4b-b300)
  FROZEN_LAKE_REGION           (default: EU_NETHERLANDS_1)
  FROZEN_LAKE_LORA_RANK        (default: 0, must match training shape)
"""

from __future__ import annotations

import os
import time
import logging

import httpx
import pytest

from training.examples.frozen_lake.train_frozen_lake import (
    FrozenLakeConfig,
    main as frozen_lake_main,
)

logger = logging.getLogger(__name__)


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _preflight_deployment_check(
    base_url: str,
    api_key: str,
    account_id: str,
    deployment_id: str,
    timeout_s: int = 30,
) -> None:
    """Verify the inference deployment is reachable before starting training.

    Raises pytest.fail with a clear message if the deployment is not ready,
    avoiding a 50-minute hang in the training loop's readiness polling.
    """
    url = f"{base_url}/inference/v1/completions"
    body = {
        "model": f"accounts/fireworks/models/qwen3-4b#accounts/{account_id}/deployments/{deployment_id}",
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
                f"Deployment {deployment_id} pre-flight check failed: "
                f"HTTP {resp.status_code} from {url}"
            )
    except httpx.RequestError as e:
        pytest.fail(
            f"Deployment {deployment_id} pre-flight check failed: "
            f"cannot reach {url}: {e}"
        )


@pytest.mark.e2e
@pytest.mark.timeout(3600)
class TestFrozenLakeB300:
    """FrozenLake GRPO smoke test on Qwen3-4B with B300 GPUs in EU_NETHERLANDS_1."""

    def test_frozen_lake_rewards_improve(self):
        api_key = _env("FIREWORKS_API_KEY")
        if not api_key:
            pytest.skip("FIREWORKS_API_KEY not set")

        account = _env("FIREWORKS_ACCOUNT_ID")
        if not account:
            pytest.skip("FIREWORKS_ACCOUNT_ID not set")

        deployment_id = _env("FROZEN_LAKE_DEPLOYMENT_ID")
        if not deployment_id:
            pytest.skip("FROZEN_LAKE_DEPLOYMENT_ID not set — need pre-created deployment")

        region = _env("FROZEN_LAKE_REGION", "EU_NETHERLANDS_1")
        training_shape = _env("FROZEN_LAKE_TRAINING_SHAPE", "qwen3-4b-b300")
        # TODO(bennychen): LoRA hotload has a loading perf bug that needs
        # fixing before re-enabling. Use full-parameter (rank=0) for now.
        lora_rank = int(_env("FROZEN_LAKE_LORA_RANK", "0"))
        base_url = _env("FIREWORKS_BASE_URL", "https://dev.api.fireworks.ai")

        os.environ.setdefault("FIREWORKS_BASE_URL", base_url)
        os.environ.setdefault("WANDB_MODE", "disabled")

        # Pre-flight: fail fast if deployment is unreachable.
        # Skip when jobs are pre-created (CI script already verified via firectl).
        if not _env("FROZEN_LAKE_POLICY_JOB_ID"):
            _preflight_deployment_check(base_url, api_key, account, deployment_id)

        cfg = FrozenLakeConfig(
            base_model="accounts/fireworks/models/qwen3-4b",
            tokenizer_model="Qwen/Qwen3-4B",
            training_shape=training_shape,
            deployment_id=deployment_id,
            region=region,
            deployment_region=region,
            lora_rank=lora_rank,
            epochs=3,
            max_seeds=20,
            max_steps=12,
            completions_per_prompt=16,
            prompt_groups_per_step=4,
            max_concurrent=16,
            policy_job_id=_env("FROZEN_LAKE_POLICY_JOB_ID"),
            reference_job_id=_env("FROZEN_LAKE_REFERENCE_JOB_ID"),
            inference_base_url=_env("FROZEN_LAKE_INFERENCE_BASE_URL"),
        )

        t0 = time.time()
        result = frozen_lake_main(cfg)
        elapsed = time.time() - t0

        # -- Assertions -------------------------------------------------------
        assert isinstance(result, dict), "main() should return a dict"

        steps = result["steps"]
        rewards = result["rewards"]

        summary = (
            f"steps={steps}, rewards={[f'{r:.3f}' for r in rewards]}, "
            f"elapsed={elapsed:.0f}s"
        )
        logger.info("Smoke test result: %s", summary)

        # At least 5 completed optimizer steps (branch primary intent)
        assert steps >= 5, (
            f"Expected >= 5 optimizer steps, got {steps}. "
            f"High filtering rate may indicate a rollout or env problem. {summary}"
        )
        assert len(rewards) >= 5, (
            f"Expected >= 5 reward entries, got {len(rewards)}. {summary}"
        )

        avg_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)

        logger.info(
            "Rewards: avg=%.3f, max=%.3f, steps=%d, elapsed=%.0fs, all=%s",
            avg_reward, max_reward, steps, elapsed, [f"{r:.3f}" for r in rewards],
        )

        # Random baseline for 4x4 FrozenLake is ~0.15.
        # A learning model should average well above that (branch primary intent).
        assert avg_reward >= 0.3, (
            f"Average reward ({avg_reward:.3f}) should be >= 0.30 "
            f"(random baseline ~0.15). {summary}"
        )
        assert max_reward >= 0.5, (
            f"Max reward ({max_reward:.3f}) should reach >= 0.50 "
            f"at least once. {summary}"
        )

        # Sanity: pipeline didn't take absurdly long (upstream addition).
        assert elapsed < 4200, (
            f"Pipeline took {elapsed:.0f}s (>70min), expected <70min. "
            f"May indicate infrastructure issues. {summary}"
        )
