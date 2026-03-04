"""E2E test: GRPO training -> DCP checkpoint -> resume.

Two-phase test on qwen3-30b-a3b (MoE) with Router Replay, TIS, and
hotloading:

  Phase 1: Train ~2 steps with hotloading and dcp_save_interval=2.
  Phase 2: Create new RLOR jobs, reuse deployment, resume from checkpoint.

Requires:
  FIREWORKS_API_KEY     -- API key with training/deployment access
  FIREWORKS_ACCOUNT_ID  -- target account ID
  FIREWORKS_BASE_URL    -- optional (defaults to "https://api.fireworks.ai")
  FIREWORKS_E2E_DEPLOYMENT_SHAPE -- required for this MoE GRPO test
"""

from __future__ import annotations

import os
import re
import logging

import pytest

from fireworks.training.cookbook.utils import InfraConfig, DeployConfig, ResumeConfig, HotloadConfig
from fireworks.training.cookbook.utils.rl import ISConfig
from fireworks.training.cookbook.tests.e2e.conftest import GSM8K_SAMPLE_URL
from fireworks.training.cookbook.recipes.rl_loop import Config, main

logger = logging.getLogger(__name__)


def _gsm8k_reward(completion: str, row: dict) -> float:
    gt = row.get("ground_truth", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", completion)
    if numbers and gt:
        gt_numbers = re.findall(r"-?\d+(?:\.\d+)?", gt)
        if gt_numbers and numbers[-1] == gt_numbers[-1]:
            return 1.0
    return 0.0


@pytest.mark.e2e
@pytest.mark.timeout(5400)
class TestGRPOResumeE2E:
    """GRPO checkpoint-resume on qwen3-30b-a3b with R3, TIS, and hotloading."""

    def test_grpo_resume_from_checkpoint(
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
        if not e2e_deployment_shape:
            pytest.skip("Set FIREWORKS_E2E_DEPLOYMENT_SHAPE for GRPO E2E runs")

        import fireworks.training.cookbook.recipes.rl_loop as grpo_mod

        grpo_mod.reward_fn = _gsm8k_reward

        deployment_id = os.environ.get("GRPO_RESUME_DEPLOYMENT_ID")

        shared_infra = InfraConfig(
            region=e2e_region,
            skip_validations=True,
            accelerator_type=e2e_training_accelerator,
            custom_image_tag=custom_image_tag,
        )

        # Phase 1: train ~2 steps, save DCP
        logger.info("PHASE 1: initial training")

        phase1_config = Config(
            base_model=e2e_model,
            dataset=GSM8K_SAMPLE_URL,
            completions_per_prompt=4,
            max_rows=8,
            epochs=1,
            grad_accum=2,
            router_replay=True,
            tis_enabled=True,
            tis=ISConfig(clip_high=10.0),
            infra=shared_infra,
            deployment=DeployConfig(
                deployment_id=deployment_id,
                deployment_shape=e2e_deployment_shape,
                deployment_region=e2e_region,
                tokenizer_model=e2e_tokenizer_model,
            ),
            hotload=HotloadConfig(
                hot_load_interval=1,
                dcp_save_interval=2,
                first_checkpoint_type="base",
                hot_load_before_training=True,
                hot_load_timeout=600,
            ),
        )

        phase1_metrics = main(phase1_config, rlor_mgr=rlor_mgr, deploy_mgr=deploy_mgr)

        assert isinstance(phase1_metrics, dict)
        assert "steps" in phase1_metrics
        phase1_steps = phase1_metrics["steps"]
        assert phase1_steps >= 2, f"Expected >= 2 steps in phase 1, got {phase1_steps}"

        phase1_policy_job_id = phase1_metrics["policy_job_id"]
        dcp_name = f"step-{phase1_steps}"
        logger.info("Phase 1 done: %d steps, job=%s", phase1_steps, phase1_policy_job_id)

        # Phase 2: resume from checkpoint
        logger.info("PHASE 2: resume from '%s' (source job: %s)", dcp_name, phase1_policy_job_id)

        phase2_config = Config(
            base_model=e2e_model,
            dataset=GSM8K_SAMPLE_URL,
            completions_per_prompt=4,
            max_rows=6,
            epochs=1,
            grad_accum=2,
            router_replay=True,
            tis_enabled=True,
            tis=ISConfig(clip_high=10.0),
            infra=shared_infra,
            deployment=DeployConfig(
                deployment_id=deployment_id,
                tokenizer_model=e2e_tokenizer_model,
            ),
            hotload=HotloadConfig(
                hot_load_interval=1,
                first_checkpoint_type="base",
                hot_load_before_training=True,
                hot_load_timeout=600,
            ),
            resume=ResumeConfig(
                resume_from=dcp_name,
                resume_job_id=phase1_policy_job_id,
            ),
        )

        phase2_metrics = main(phase2_config, rlor_mgr=rlor_mgr, deploy_mgr=deploy_mgr)

        assert isinstance(phase2_metrics, dict)
        assert "steps" in phase2_metrics
        phase2_steps = phase2_metrics["steps"]
        assert phase2_steps > phase1_steps, f"Expected global_step > {phase1_steps} after resume, got {phase2_steps}"
        logger.info("Resume verified: phase1=%d, phase2=%d", phase1_steps, phase2_steps)
