"""E2E test for DPO training on qwen3-30b-a3b.

Creates a real RLOR trainer job, trains DPO on a small synthetic preference
dataset, and verifies metrics. No deployment or hotloading.

Requires:
  FIREWORKS_API_KEY     -- API key with training access
  FIREWORKS_ACCOUNT_ID  -- target account ID
  FIREWORKS_BASE_URL    -- optional (defaults to "https://api.fireworks.ai")
"""

from __future__ import annotations

import os
import json
import tempfile

import pytest

from fireworks.training.cookbook.utils import InfraConfig, DeployConfig, HotloadConfig
from fireworks.training.cookbook.recipes.dpo_loop import Config, main


def _make_preference_dataset(path: str, num_pairs: int = 10) -> None:
    """Generate a synthetic preference dataset for testing."""
    with open(path, "w") as f:
        for i in range(num_pairs):
            row = {
                "chosen": {
                    "messages": [
                        {"role": "user", "content": f"What is {i} + {i}?"},
                        {"role": "assistant", "content": f"The answer is {i + i}."},
                    ]
                },
                "rejected": {
                    "messages": [
                        {"role": "user", "content": f"What is {i} + {i}?"},
                        {"role": "assistant", "content": f"I think it's {i * 3}."},
                    ]
                },
            }
            f.write(json.dumps(row) + "\n")


@pytest.mark.e2e
@pytest.mark.timeout(3600)
class TestDPOE2E:
    """DPO training on qwen3-30b-a3b -- trainer only, no deployment."""

    def test_dpo_trains_several_steps(
        self,
        sdk_managers,
        e2e_region,
        e2e_model,
        e2e_training_accelerator,
        custom_image_tag,
    ):
        rlor_mgr, deploy_mgr = sdk_managers

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            dataset_path = f.name

        try:
            _make_preference_dataset(dataset_path, num_pairs=10)

            config = Config(
                base_model=e2e_model,
                dataset=dataset_path,
                beta=0.1,
                learning_rate=1e-5,
                epochs=2,
                grad_accum=2,
                max_seq_len=4096,
                max_pairs=10,
                infra=InfraConfig(
                    region=e2e_region,
                    skip_validations=True,
                    accelerator_type=e2e_training_accelerator,
                    custom_image_tag=custom_image_tag,
                ),
                deployment=DeployConfig(create_deployment=False),
                hotload=HotloadConfig(hot_load_interval=0),
            )

            metrics = main(config, rlor_mgr=rlor_mgr, deploy_mgr=deploy_mgr)

            assert isinstance(metrics, dict)
            assert "steps" in metrics
            assert metrics["steps"] >= 2, "Expected at least 2 optimizer steps"
        finally:
            os.unlink(dataset_path)
