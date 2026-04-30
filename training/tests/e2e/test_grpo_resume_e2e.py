"""E2E test: GRPO training -> DCP checkpoint -> resume from dataloader position.

Two-phase test on qwen3-30b-a3b (MoE) with Router Replay and TIS:

  Phase 1: Train a few steps, save DCP checkpoints. Capture the policy
           and reference job IDs.
  Phase 2: Reattach to the same trainers (``policy_job_id``,
           ``reference_job_id``) with the same ``log_path``.
           ``TrainingCheckpoints.resume()`` lists the policy trainer's
           checkpoints on the control plane, picks the newest resumable
           row, and restores the rollout cursor from ``dataloader.json``.

Requires:
  FIREWORKS_API_KEY     -- API key with training/deployment access
  FIREWORKS_BASE_URL    -- optional (defaults to "https://api.fireworks.ai")
  FIREWORKS_E2E_DEPLOYMENT_SHAPE -- required for this MoE GRPO test
"""

from __future__ import annotations

import json
import os
import re
import logging
import tempfile

import pytest

from fireworks.training.sdk import FireworksClient
from training.utils import InfraConfig, DeployConfig, WeightSyncConfig
from training.utils.checkpoints import DATALOADER_BASE_NAME
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
        phase1_steps = phase1_metrics["steps"]
        phase1_policy_job_id = phase1_metrics["policy_job_id"]
        phase1_reference_job_id = phase1_metrics.get("reference_job_id")
        phase1_deployment_id = phase1_config.deployment.deployment_id
        assert phase1_steps >= 2, f"Expected >= 2 steps in phase 1, got {phase1_steps}"
        logger.info(
            "Phase 1 done: %d steps, policy=%s, reference=%s, deployment=%s",
            phase1_steps, phase1_policy_job_id, phase1_reference_job_id,
            phase1_deployment_id,
        )

        # Read the persisted rollout cursor from dataloader.json — the only
        # cookbook-side state in the new model (one int per checkpoint name).
        dataloader_path = os.path.join(log_dir, DATALOADER_BASE_NAME)
        assert os.path.exists(dataloader_path), (
            f"Phase 1 should have written {DATALOADER_BASE_NAME} under {log_dir}"
        )
        with open(dataloader_path) as f:
            phase1_dataloader = json.load(f)
        assert phase1_dataloader, "dataloader.json should be non-empty after phase 1"
        phase1_data_consumed = max(int(v) for v in phase1_dataloader.values())

        # Verify the control plane has at least one resumable row for the
        # phase-1 policy trainer — phase 2's resume reads from there.
        api_key = os.environ["FIREWORKS_API_KEY"]
        base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
        fw_client = FireworksClient(api_key=api_key, base_url=base_url)
        phase1_rows = fw_client.list_checkpoints(phase1_policy_job_id)
        phase1_resumable = [
            r for r in phase1_rows
            if (r.get("checkpointType") or "").endswith(("TRAINING", "TRAINING_LORA"))
        ]
        assert phase1_resumable, (
            f"Expected phase 1 to leave a resumable row on the control plane "
            f"for policy job {phase1_policy_job_id}; got rows: {phase1_rows}"
        )
        logger.info(
            "Phase 1 CP rows: %d resumable (data_consumed=%d)",
            len(phase1_resumable), phase1_data_consumed,
        )

        # Phase 2: reattach to the same policy + reference trainers so resume
        # can find the phase-1 CP rows. Auto-resume across separate trainer
        # jobs is not supported in the new model.
        logger.info(
            "PHASE 2: reattach to policy=%s, reference=%s",
            phase1_policy_job_id, phase1_reference_job_id,
        )

        phase2_config = Config(
            base_model=e2e_model,
            dataset=GSM8K_SAMPLE_URL,
            completions_per_prompt=4,
            max_rows=8,
            epochs=1,
            kl_beta=0,
            log_path=log_dir,
            policy_job_id=phase1_policy_job_id,
            reference_job_id=phase1_reference_job_id,
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
        phase2_steps = phase2_metrics["steps"]
        assert phase2_steps > phase1_steps, (
            f"Phase 2 step count ({phase2_steps}) should exceed phase 1's "
            f"({phase1_steps}); resume probably did not pick up phase 1's CP rows."
        )

        with open(dataloader_path) as f:
            phase2_dataloader = json.load(f)
        phase2_data_consumed = max(int(v) for v in phase2_dataloader.values())
        assert phase2_data_consumed > phase1_data_consumed, (
            f"Phase 2 data_consumed ({phase2_data_consumed}) should exceed "
            f"phase 1's ({phase1_data_consumed}); rollout cursor should advance."
        )
        logger.info(
            "Resume verified: phase1=%d steps (data_consumed=%d), "
            "phase2=%d steps (data_consumed=%d)",
            phase1_steps, phase1_data_consumed, phase2_steps, phase2_data_consumed,
        )
