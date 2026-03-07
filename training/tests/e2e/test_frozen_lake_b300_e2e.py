"""E2E test for FrozenLake GRPO on Qwen3-4B with B300 GPUs.

Verifies the full multi-turn RL pipeline: policy + reference trainers,
hotloading, eval-protocol rollouts, and reward improvement.

Required env vars:
  FIREWORKS_API_KEY
  FIREWORKS_ACCOUNT_ID

Optional env vars (CI script sets these):
  FIREWORKS_BASE_URL           (default: https://dev.api.fireworks.ai)
  FROZEN_LAKE_POLICY_JOB_ID    (pre-created policy trainer job)
  FROZEN_LAKE_REFERENCE_JOB_ID (pre-created reference trainer job)
  FROZEN_LAKE_DEPLOYMENT_ID    (pre-created deployment, required)
  FROZEN_LAKE_TRAINING_SHAPE   (default: qwen3-4b-b300)
  FROZEN_LAKE_REGION           (default: EU_NETHERLANDS_1)
  FROZEN_LAKE_LORA_RANK        (default: 8, must match training shape)
"""

from __future__ import annotations

import os
import logging

import pytest

from training.examples.frozen_lake.train_frozen_lake import (
    FrozenLakeConfig,
    main as frozen_lake_main,
)

logger = logging.getLogger(__name__)


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@pytest.mark.e2e
@pytest.mark.timeout(3600)
class TestFrozenLakeB300:
    """FrozenLake GRPO on Qwen3-4B with B300 GPUs in EU_NETHERLANDS_1."""

    def test_frozen_lake_rewards_improve(self):
        api_key = _env("FIREWORKS_API_KEY")
        if not api_key:
            pytest.skip("FIREWORKS_API_KEY not set")

        account = _env("FIREWORKS_ACCOUNT_ID")
        if not account:
            pytest.skip("FIREWORKS_ACCOUNT_ID not set")

        deployment_id = _env("FROZEN_LAKE_DEPLOYMENT_ID")
        if not deployment_id:
            pytest.skip("FROZEN_LAKE_DEPLOYMENT_ID not set")

        region = _env("FROZEN_LAKE_REGION", "EU_NETHERLANDS_1")
        training_shape = _env("FROZEN_LAKE_TRAINING_SHAPE", "qwen3-4b-b300")
        lora_rank = int(_env("FROZEN_LAKE_LORA_RANK", "8"))

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
            completions_per_prompt=4,
            prompt_groups_per_step=4,
            max_concurrent=16,
            policy_job_id=_env("FROZEN_LAKE_POLICY_JOB_ID"),
            reference_job_id=_env("FROZEN_LAKE_REFERENCE_JOB_ID"),
        )

        os.environ.setdefault("FIREWORKS_BASE_URL", "https://dev.api.fireworks.ai")
        os.environ.setdefault("WANDB_MODE", "disabled")

        result = frozen_lake_main(cfg)

        assert isinstance(result, dict), "main() should return a dict"
        assert result["steps"] >= 5, f"Expected >=5 steps, got {result['steps']}"

        rewards = result["rewards"]
        assert len(rewards) >= 5, f"Expected >=5 reward entries, got {len(rewards)}"

        avg_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)

        logger.info(
            "Rewards: avg=%.3f, max=%.3f, all=%s",
            avg_reward, max_reward, [f"{r:.3f}" for r in rewards],
        )

        assert avg_reward >= 0.3, (
            f"Average reward ({avg_reward:.3f}) should be >= 0.30 "
            f"(random baseline ~0.15). Full history: {rewards}"
        )
        assert max_reward >= 0.5, (
            f"Max reward ({max_reward:.3f}) should reach >= 0.50 "
            f"at least once. Full history: {rewards}"
        )
