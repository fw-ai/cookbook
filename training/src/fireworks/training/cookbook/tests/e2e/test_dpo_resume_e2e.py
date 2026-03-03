"""E2E test: DPO training -> DCP checkpoint -> resume -> verify continuation.

Two-phase test on qwen3-30b-a3b.

Requires:
  FIREWORKS_API_KEY     -- API key with training access
  FIREWORKS_ACCOUNT_ID  -- target account ID
  FIREWORKS_BASE_URL    -- optional (defaults to "https://api.fireworks.ai")
"""

from __future__ import annotations

import os
import json
import logging
import tempfile

import pytest

from fireworks.training.cookbook.utils import InfraConfig, DeployConfig, ResumeConfig, HotloadConfig
from fireworks.training.cookbook.recipes.dpo_loop import Config, main

logger = logging.getLogger(__name__)


def _make_preference_dataset(path: str, num_pairs: int = 8) -> None:
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
@pytest.mark.timeout(5400)
class TestDPOResumeE2E:
    """DPO checkpoint-resume on qwen3-30b-a3b."""

    def test_dpo_resume_from_checkpoint(
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
            _make_preference_dataset(dataset_path, num_pairs=8)

            shared_infra = InfraConfig(
                region=e2e_region,
                skip_validations=True,
                accelerator_type=e2e_training_accelerator,
                custom_image_tag=custom_image_tag,
            )

            # Phase 1: train, save DCP
            logger.info("PHASE 1: initial DPO training")

            phase1_config = Config(
                base_model=e2e_model,
                dataset=dataset_path,
                beta=0.1,
                learning_rate=1e-5,
                epochs=1,
                grad_accum=2,
                max_seq_len=4096,
                max_pairs=8,
                infra=shared_infra,
                deployment=DeployConfig(create_deployment=False),
                hotload=HotloadConfig(hot_load_interval=0, dcp_save_interval=2),
            )

            phase1_metrics = main(phase1_config, rlor_mgr=rlor_mgr, deploy_mgr=deploy_mgr)

            assert isinstance(phase1_metrics, dict)
            assert "steps" in phase1_metrics
            phase1_steps = phase1_metrics["steps"]
            assert phase1_steps >= 2, f"Expected >= 2 steps in phase 1, got {phase1_steps}"

            phase1_job_id = phase1_metrics["policy_job_id"]
            dcp_name = f"step-{phase1_steps}"
            logger.info("Phase 1 done: %d steps, job=%s", phase1_steps, phase1_job_id)

            # Phase 2: resume from checkpoint
            logger.info("PHASE 2: resume from '%s' (source job: %s)", dcp_name, phase1_job_id)

            phase2_config = Config(
                base_model=e2e_model,
                dataset=dataset_path,
                beta=0.1,
                learning_rate=1e-5,
                epochs=1,
                grad_accum=2,
                max_seq_len=4096,
                max_pairs=8,
                infra=shared_infra,
                deployment=DeployConfig(create_deployment=False),
                hotload=HotloadConfig(hot_load_interval=0),
                resume=ResumeConfig(resume_from=dcp_name, resume_job_id=phase1_job_id),
            )

            phase2_metrics = main(phase2_config, rlor_mgr=rlor_mgr, deploy_mgr=deploy_mgr)

            assert isinstance(phase2_metrics, dict)
            assert "steps" in phase2_metrics
            phase2_steps = phase2_metrics["steps"]
            assert phase2_steps >= 2, f"Expected >= 2 steps in phase 2, got {phase2_steps}"

            logger.info("Resume verified: phase1=%d, phase2=%d", phase1_steps, phase2_steps)
        finally:
            os.unlink(dataset_path)
