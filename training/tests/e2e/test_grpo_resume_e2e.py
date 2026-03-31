"""E2E test: GRPO training -> DCP checkpoint -> resume from dataloader position.

Two-phase test on qwen3-30b-a3b (MoE) with Router Replay and TIS:

  Phase 1: Train a few steps, save DCP checkpoints to checkpoints.jsonl.
  Phase 2: New RLOR jobs, same log_dir -- resolve_resume picks up from
           checkpoints.jsonl, loads cross-job checkpoint, resumes at the
           saved step (continues dataloader, not from beginning).

Requires:
  FIREWORKS_API_KEY     -- API key with training/deployment access
  FIREWORKS_BASE_URL    -- optional (defaults to "https://api.fireworks.ai")
  FIREWORKS_E2E_DEPLOYMENT_SHAPE -- required for this MoE GRPO test
"""

from __future__ import annotations

import os
import re
import logging
import tempfile

import pytest

from training.utils.checkpoint_utils import get_last_checkpoint
from training.utils import InfraConfig, DeployConfig, WeightSyncConfig
from training.tests.e2e.conftest import GSM8K_SAMPLE_URL
from training.recipes.rl_loop import Config, main

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
    """GRPO checkpoint-resume on qwen3-30b-a3b with R3, TIS, and weight sync."""

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

        import training.recipes.rl_loop as grpo_mod

        grpo_mod.reward_fn = _gsm8k_reward

        deployment_id = os.environ.get("GRPO_RESUME_DEPLOYMENT_ID")
        log_dir = tempfile.mkdtemp(prefix="grpo_resume_")

        shared_infra = InfraConfig(
            region=e2e_region,
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
            kl_beta=0,
            log_path=log_dir,
            infra=shared_infra,
            deployment=DeployConfig(
                deployment_id=deployment_id,
                deployment_shape=e2e_deployment_shape,
                deployment_region=e2e_region,
                tokenizer_model=e2e_tokenizer_model,
            ),
            weight_sync=WeightSyncConfig(
                weight_sync_interval=1,
                dcp_save_interval=2,
                first_checkpoint_type="base",
                weight_sync_before_training=True,
                weight_sync_timeout=600,
            ),
        )

        phase1_metrics = main(phase1_config, rlor_mgr=rlor_mgr, deploy_mgr=deploy_mgr)

        assert isinstance(phase1_metrics, dict)
        assert "steps" in phase1_metrics
        phase1_steps = phase1_metrics["steps"]
        assert phase1_steps >= 2, f"Expected >= 2 steps in phase 1, got {phase1_steps}"

        phase1_deployment_id = phase1_config.deployment.deployment_id
        logger.info("Phase 1 done: %d steps, job=%s, deployment=%s",
                     phase1_steps, phase1_metrics["policy_job_id"], phase1_deployment_id)

        last_ckpt = get_last_checkpoint(log_dir)
        assert last_ckpt is not None, "Expected at least one checkpoint in checkpoints.jsonl"
        assert "state_path" in last_ckpt
        saved_step = last_ckpt["step"]
        saved_data_consumed = last_ckpt["data_consumed"]
        logger.info("Phase 1 checkpoint: step=%d data_consumed=%d",
                     saved_step, saved_data_consumed)

        # Phase 2: new jobs, same log_dir -- resume from checkpoints.jsonl
        logger.info("PHASE 2: resume from checkpoints.jsonl (step=%d)", saved_step)

        phase2_config = Config(
            base_model=e2e_model,
            dataset=GSM8K_SAMPLE_URL,
            completions_per_prompt=4,
            max_rows=8,
            epochs=1,
            kl_beta=0,
            log_path=log_dir,
            infra=shared_infra,
            deployment=DeployConfig(
                deployment_id=phase1_deployment_id,
                tokenizer_model=e2e_tokenizer_model,
            ),
            weight_sync=WeightSyncConfig(
                weight_sync_interval=1,
                first_checkpoint_type="base",
                weight_sync_before_training=True,
                weight_sync_timeout=600,
            ),
        )

        phase2_metrics = main(phase2_config, rlor_mgr=rlor_mgr, deploy_mgr=deploy_mgr)

        assert isinstance(phase2_metrics, dict)
        assert "steps" in phase2_metrics
        phase2_steps = phase2_metrics["steps"]

        assert phase2_steps > saved_step, (
            f"Phase 2 should continue beyond phase 1's saved step {saved_step}, "
            f"but got {phase2_steps}"
        )

        final_ckpt = get_last_checkpoint(log_dir)
        assert final_ckpt is not None
        assert final_ckpt["data_consumed"] > saved_data_consumed, (
            f"Phase 2 data_consumed ({final_ckpt['data_consumed']}) should exceed "
            f"phase 1's ({saved_data_consumed}) -- dataloader should continue, not restart"
        )

        logger.info(
            "Resume verified: phase1=%d steps (data_consumed=%d), "
            "phase2=%d steps (data_consumed=%d)",
            saved_step, saved_data_consumed,
            phase2_steps, final_ckpt["data_consumed"],
        )
