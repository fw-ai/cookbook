"""E2E test: async GRPO training -> DCP checkpoint -> resume.

Two-phase test on the SDK-managed single-shape trainer:

  Phase 1: Create an RL trainer/deployment, train a few steps, and save DCP
           checkpoints. Capture the policy, reference, and deployment IDs.
  Phase 2: Reattach to the same trainer and deployment with the same ``log_path``.
           ``TrainingCheckpoints.resume()`` lists the policy trainer's
           checkpoints on the control plane, picks the newest resumable
           row, and restores the rollout cursor from ``dataloader.json``.

Requires:
  FIREWORKS_API_KEY     -- API key with training/deployment access
  FIREWORKS_BASE_URL    -- optional (defaults to "https://api.fireworks.ai")
  FIREWORKS_E2E_DEPLOYMENT_SHAPE -- optional; defaults to the qwen3p5-9b B200 shape
"""

from __future__ import annotations

import json
import os
import logging
import tempfile
import time
from dataclasses import replace

import httpx
import pytest

from training.utils import DeployConfig, TrainerConfig
from training.utils.checkpoints import DATALOADER_BASE_NAME
from training.tests.e2e.conftest import GSM8K_SAMPLE_URL
from training.recipes.async_rl_loop import Config, main
from training.tests.async_grpo_helpers import (
    MAX_REALISTIC_COMPLETION_TOKENS,
    gsm8k_numeric_reward,
    make_message_rollout_fn_factory,
)

logger = logging.getLogger(__name__)

_TRAINER_CLEANUP_STATES = {
    "JOB_STATE_ARCHIVED",
    "JOB_STATE_DELETING",
    "JOB_STATE_DELETING_CLEANING_UP",
    "JOB_STATE_DELETED",
    "JOB_STATE_CANCELLED",
}


def _delete_deployment_and_assert_cleanup(deploy_mgr, deployment_id: str) -> None:
    try:
        deploy_mgr.delete(deployment_id)
        deploy_mgr._wait_for_deletion(deployment_id, timeout_s=120)
        deployment = deploy_mgr.get(deployment_id)
        if deployment is not None and deployment.state not in {"DELETING", "DELETED"}:
            raise AssertionError(
                f"Deployment {deployment_id} should be deleting or gone after "
                f"cleanup; got {deployment.state}"
            )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise


def _delete_trainer_and_assert_cleanup(rlor_mgr, job_id: str) -> None:
    try:
        rlor_mgr.delete(job_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise
    deadline = time.time() + 120
    last_state = "UNKNOWN"
    while time.time() < deadline:
        try:
            job = rlor_mgr.get(job_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return
            raise
        last_state = job.get("state", "UNKNOWN")
        if last_state in _TRAINER_CLEANUP_STATES:
            return
        time.sleep(3)
    raise AssertionError(
        f"Trainer {job_id} should be deleting or gone after cleanup; got {last_state}"
    )


@pytest.mark.e2e
@pytest.mark.timeout(5400)
class TestGRPOResumeE2E:
    """Async GRPO checkpoint-resume on the SDK-managed single-shape trainer."""

    def test_grpo_resume_from_checkpoint(
        self,
        sdk_managers,
        e2e_region,
        e2e_model,
        e2e_tokenizer_model,
        e2e_training_shape,
        e2e_reference_training_shape,
        e2e_training_profile,
        e2e_reference_training_profile,
        e2e_deployment_accelerator,
        custom_image_tag,
        port_lora_rank,
    ):
        rlor_mgr, deploy_mgr = sdk_managers
        _ = e2e_training_profile
        _ = e2e_reference_training_profile
        _ = e2e_deployment_accelerator

        phase1_policy_job_id = None
        phase1_reference_job_id = None
        phase1_deployment_id = None
        log_dir = tempfile.mkdtemp(prefix="grpo_resume_")
        verified = False

        shared_trainer = TrainerConfig(
            training_shape_id=e2e_training_shape,
            reference_training_shape_id=e2e_reference_training_shape,
            cleanup_reference_on_close=False,
            region=e2e_region,
            custom_image_tag=custom_image_tag,
        )
        rollout_fn_factory = make_message_rollout_fn_factory(gsm8k_numeric_reward)

        # Phase 1: train ~2 steps, save DCP
        logger.info("PHASE 1: initial training")

        phase1_config = Config(
            base_model=e2e_model,
            dataset=GSM8K_SAMPLE_URL,
            completions_per_prompt=4,
            max_completion_tokens=MAX_REALISTIC_COMPLETION_TOKENS,
            max_rows=4,
            epochs=1,
            shuffle=False,
            kl_beta=0 if port_lora_rank else 0.001,
            log_path=log_dir,
            lora_rank=port_lora_rank,
            trainer=shared_trainer,
            deployment=DeployConfig(
                deployment_region=e2e_region,
                tokenizer_model=e2e_tokenizer_model,
            ),
            weight_sync_before_training=True,
            weight_sync_timeout=600,
            dcp_save_interval=2,
        )

        try:
            phase1_metrics = main(
                phase1_config,
                rollout_fn_factory=rollout_fn_factory,
            )

            assert isinstance(phase1_metrics, dict)
            phase1_steps = phase1_metrics["steps"]
            phase1_policy_job_id = phase1_metrics["policy_job_id"]
            phase1_reference_job_id = phase1_metrics.get("reference_job_id")
            phase1_deployment_id = phase1_metrics["deployment_id"]
            if port_lora_rank:
                assert phase1_reference_job_id is None, (
                    f"LoRA GRPO should use the shared-reference path: {phase1_metrics}"
                )
            else:
                assert phase1_reference_job_id, (
                    f"Full-param GRPO should create a separate reference trainer: {phase1_metrics}"
                )
            assert phase1_deployment_id, (
                f"Expected managed deployment id in metrics: {phase1_metrics}"
            )
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
            phase1_rows = rlor_mgr.list_checkpoints(phase1_policy_job_id)
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
                max_completion_tokens=MAX_REALISTIC_COMPLETION_TOKENS,
                max_rows=8,
                epochs=1,
                shuffle=False,
                kl_beta=0 if port_lora_rank else 0.001,
                log_path=log_dir,
                lora_rank=port_lora_rank,
                trainer=replace(
                    shared_trainer,
                    job_id=phase1_policy_job_id,
                    reference_job_id=phase1_reference_job_id,
                ),
                deployment=DeployConfig(
                    deployment_id=phase1_deployment_id,
                    deployment_region=e2e_region,
                    tokenizer_model=e2e_tokenizer_model,
                ),
                weight_sync_before_training=True,
                weight_sync_timeout=600,
            )

            phase2_metrics = main(
                phase2_config,
                rollout_fn_factory=rollout_fn_factory,
            )
            assert phase2_metrics["policy_job_id"] == phase1_policy_job_id, (
                f"Phase 2 should reuse policy trainer {phase1_policy_job_id}: {phase2_metrics}"
            )
            if phase1_reference_job_id:
                assert phase2_metrics.get("reference_job_id") == phase1_reference_job_id, (
                    f"Phase 2 should reuse reference trainer {phase1_reference_job_id}: "
                    f"{phase2_metrics}"
                )
            phase2_steps = phase2_metrics["steps"]
            assert phase2_steps > phase1_steps, (
                f"Phase 2 step count ({phase2_steps}) should exceed phase 1's "
                f"({phase1_steps}); resume probably did not pick up phase 1's CP rows."
            )

            with open(dataloader_path) as f:
                phase2_dataloader = json.load(f)
            phase2_data_consumed = max(int(v) for v in phase2_dataloader.values())
            assert phase2_data_consumed >= phase1_data_consumed, (
                f"Phase 2 data_consumed ({phase2_data_consumed}) should not regress "
                f"from phase 1's ({phase1_data_consumed}); rollout cursor should remain aligned."
            )
            logger.info(
                "Resume verified: phase1=%d steps (data_consumed=%d), "
                "phase2=%d steps (data_consumed=%d)",
                phase1_steps, phase1_data_consumed, phase2_steps, phase2_data_consumed,
            )
            verified = True
        finally:
            if verified and phase1_deployment_id:
                _delete_deployment_and_assert_cleanup(deploy_mgr, phase1_deployment_id)
            if verified and phase1_policy_job_id:
                _delete_trainer_and_assert_cleanup(rlor_mgr, phase1_policy_job_id)
            if (
                verified
                and phase1_reference_job_id
                and phase1_reference_job_id != phase1_policy_job_id
            ):
                _delete_trainer_and_assert_cleanup(rlor_mgr, phase1_reference_job_id)
