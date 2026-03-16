"""E2E test for GRPO training on qwen3-30b-a3b (MoE).

Full pipeline: policy + reference trainers, deployment with weight sync,
Router Replay (R3), and Truncated Importance Sampling (TIS).

Requires:
  FIREWORKS_API_KEY     -- API key with training/deployment access
  FIREWORKS_ACCOUNT_ID  -- target account ID
  FIREWORKS_BASE_URL    -- optional (defaults to "https://api.fireworks.ai")
  FIREWORKS_E2E_DEPLOYMENT_SHAPE -- required for this MoE GRPO test
"""

from __future__ import annotations

import re
import time

import pytest

from training.utils import InfraConfig, DeployConfig, WeightSyncConfig
from training.utils.rl import TISConfig
from training.tests.e2e.conftest import GSM8K_SAMPLE_URL
from training.recipes.rl_loop import Config, main


def _gsm8k_reward(completion: str, row: dict) -> float:
    """Extract a numeric answer from the completion and compare to ground truth."""
    gt = row.get("ground_truth", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", completion)
    if numbers and gt:
        gt_numbers = re.findall(r"-?\d+(?:\.\d+)?", gt)
        if gt_numbers and numbers[-1] == gt_numbers[-1]:
            return 1.0
    return 0.0


@pytest.mark.e2e
@pytest.mark.timeout(3600)
class TestGRPOE2E:
    """GRPO on qwen3-30b-a3b with R3, TIS, and weight sync."""

    def test_grpo_full_pipeline(
        self,
        sdk_managers,
        e2e_region,
        e2e_model,
        e2e_tokenizer_model,
        e2e_training_accelerator,
        e2e_deployment_accelerator,
        e2e_deployment_shape,
        custom_image_tag,
    ):
        rlor_mgr, deploy_mgr = sdk_managers
        ts = int(time.time())
        if not e2e_deployment_shape:
            pytest.skip("Set FIREWORKS_E2E_DEPLOYMENT_SHAPE for GRPO E2E runs")

        # Inject the test reward function into the module.
        import training.recipes.rl_loop as grpo_mod

        grpo_mod.reward_fn = _gsm8k_reward

        config = Config(
            base_model=e2e_model,
            dataset=GSM8K_SAMPLE_URL,
            completions_per_prompt=4,
            max_rows=10,
            epochs=1,
            router_replay=True,
            tis=TISConfig(cap=10.0),
            infra=InfraConfig(
                region=e2e_region,
                accelerator_type=e2e_training_accelerator,
                custom_image_tag=custom_image_tag,
            ),
            deployment=DeployConfig(
                deployment_id=f"grpo-e2e-{ts}",
                deployment_shape=e2e_deployment_shape,
                deployment_region=e2e_region,
                tokenizer_model=e2e_tokenizer_model,
            ),
            weight_sync=WeightSyncConfig(
                weight_sync_interval=1,
                first_checkpoint_type="base",
                weight_sync_before_training=True,
                weight_sync_timeout=600,
            ),
        )

        metrics = main(config, rlor_mgr=rlor_mgr, deploy_mgr=deploy_mgr)

        assert isinstance(metrics, dict)
        assert "steps" in metrics
        assert metrics["steps"] >= 2, "Expected at least 2 optimizer steps"
