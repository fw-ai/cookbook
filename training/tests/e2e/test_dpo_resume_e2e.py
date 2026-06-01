"""E2E test: DPO training -> DCP checkpoint -> resume -> verify continuation.

Two-phase test on the SDK-managed single-shape trainer.

Requires:
  FIREWORKS_API_KEY     -- API key with training access
  FIREWORKS_BASE_URL    -- optional (defaults to "https://api.fireworks.ai")
"""

from __future__ import annotations

import os
import json
import logging
import tempfile

import pytest

from dataclasses import replace

from training.utils import DeployConfig, TrainerConfig
from training.utils.checkpoints import DATALOADER_BASE_NAME
from training.recipes.dpo_loop import Config, main

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
    """DPO checkpoint-resume on the SDK-managed single-shape trainer."""

    def test_dpo_resume_from_checkpoint(
        self,
        sdk_managers,
        e2e_region,
        e2e_model,
        e2e_tokenizer_model,
        e2e_training_shape,
        e2e_reference_training_shape,
        e2e_training_profile,
        e2e_reference_training_profile,
        custom_image_tag,
        port_lora_rank,
        port_track_state,
    ):
        rlor_mgr, _deploy_mgr = sdk_managers
        _ = e2e_training_profile
        _ = e2e_reference_training_profile
        phase1_job_id = None

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            dataset_path = f.name

        log_dir = tempfile.mkdtemp(prefix="dpo_resume_")

        try:
            _make_preference_dataset(dataset_path, num_pairs=8)

            shared_trainer = TrainerConfig(
                training_shape_id=e2e_training_shape,
                reference_training_shape_id=e2e_reference_training_shape,
                region=e2e_region,
                custom_image_tag=custom_image_tag,
            )
            track = port_track_state.load()
            phase1_job_id = track.get("policy_job_id") or track.get("trainer_job_id")
            phase1_reference_job_id = track.get("reference_job_id")
            if not phase1_job_id:
                pytest.skip(
                    "Run test_dpo_smoke.py first to provision the shared "
                    "precision-track trainer"
                )
            if port_lora_rank == 0 and not phase1_reference_job_id:
                pytest.skip(
                    "Run test_dpo_smoke.py first to provision the full-param "
                    "reference trainer"
                )

            # Phase 1: train, save DCP
            logger.info("PHASE 1: initial DPO training")

            phase1_config = Config(
                log_path=log_dir,
                base_model=e2e_model,
                dataset=dataset_path,
                tokenizer_model=e2e_tokenizer_model,
                beta=0.1,
                learning_rate=1e-5,
                epochs=1,
                max_pairs=8,
                lora_rank=port_lora_rank,
                dcp_save_interval=2,
                init_from_checkpoint=f"{phase1_job_id}:step-4",
                trainer=replace(
                    shared_trainer,
                    job_id=phase1_job_id,
                    reference_job_id=phase1_reference_job_id,
                ),
                deployment=DeployConfig(),
                release_reference_after_cache=port_lora_rank > 0,
            )

            phase1_metrics = main(phase1_config)

            assert isinstance(phase1_metrics, dict)
            assert "steps" in phase1_metrics
            phase1_steps = phase1_metrics["steps"]
            assert phase1_steps >= 2, f"Expected >= 2 steps in phase 1, got {phase1_steps}"
            phase1_job_id = phase1_metrics["policy_job_id"]
            phase1_reference_job_id = phase1_metrics.get("reference_job_id")
            if port_lora_rank == 0:
                assert phase1_reference_job_id == track.get("reference_job_id"), (
                    f"DPO resume should reuse the stored reference trainer: {phase1_metrics}"
                )
            else:
                assert phase1_reference_job_id is None, (
                    f"LoRA DPO resume should use the shared-reference path: {phase1_metrics}"
                )
            port_track_state.update(
                policy_job_id=phase1_job_id,
                reference_job_id=phase1_reference_job_id,
                log_path=log_dir,
                dpo_log_path=log_dir,
                lora_rank=port_lora_rank,
            )
            logger.info("Phase 1 done: %d steps, job=%s", phase1_steps, phase1_job_id)

            dataloader_path = os.path.join(log_dir, DATALOADER_BASE_NAME)
            assert os.path.exists(dataloader_path), (
                f"Phase 1 should have written {DATALOADER_BASE_NAME} under {log_dir}"
            )
            with open(dataloader_path) as f:
                phase1_dataloader = json.load(f)
            assert phase1_dataloader, "dataloader.json should be non-empty after phase 1"
            phase1_data_consumed = max(int(v) for v in phase1_dataloader.values())

            phase1_rows = rlor_mgr.list_checkpoints(phase1_job_id)
            phase1_resumable = [
                r for r in phase1_rows
                if (r.get("checkpointType") or "").endswith(("TRAINING", "TRAINING_LORA"))
            ]
            assert phase1_resumable, (
                f"Expected phase 1 to leave a resumable row on the control plane "
                f"for policy job {phase1_job_id}; got rows: {phase1_rows}"
            )

            # Phase 2: reattach to the same trainer and auto-resume from CP rows.
            logger.info("PHASE 2: reattach to job=%s, expect resume", phase1_job_id)

            phase2_config = Config(
                log_path=log_dir,
                base_model=e2e_model,
                dataset=dataset_path,
                tokenizer_model=e2e_tokenizer_model,
                beta=0.1,
                learning_rate=1e-5,
                epochs=1,
                max_pairs=8,
                lora_rank=port_lora_rank,
                trainer=replace(
                    shared_trainer,
                    job_id=phase1_job_id,
                    reference_job_id=phase1_reference_job_id,
                ),
                deployment=DeployConfig(),
                release_reference_after_cache=port_lora_rank > 0,
            )

            phase2_metrics = main(phase2_config)

            assert isinstance(phase2_metrics, dict)
            assert "steps" in phase2_metrics
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
                f"from phase 1's ({phase1_data_consumed}); dataloader cursor should remain aligned."
            )

            logger.info("Resume verified: phase1=%d, phase2=%d", phase1_steps, phase2_steps)
        finally:
            os.unlink(dataset_path)
