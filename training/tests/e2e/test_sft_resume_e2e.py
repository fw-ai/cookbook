"""E2E test: SFT training -> DCP checkpoint -> resume from dataloader position.

Two-phase test on qwen3-30b-a3b:
  Phase 1: Train on first portion of data, save DCP checkpoints. Capture
           the trainer job ID.
  Phase 2: Reattach to the same trainer (``trainer_job_id=phase1_job_id``)
           with the same ``log_path``. ``TrainingCheckpoints.resume()``
           lists the trainer's checkpoints on the control plane, picks
           the newest resumable row, and restores ``raw_rows_consumed``
           from ``dataloader.json``.

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

from fireworks.training.sdk import FireworksClient
from training.utils import InfraConfig
from training.utils.checkpoints import DATALOADER_BASE_NAME
from training.recipes.sft_loop import Config, main

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "Qwen/Qwen3-30B-A3B"


def _get_tokenizer_model(e2e_model: str) -> str:
    if "qwen3-1p7b" in e2e_model:
        return "Qwen/Qwen3-1.7B"
    if "qwen3-30b-a3b" in e2e_model.lower() or "qwen3-30b" in e2e_model.lower():
        return "Qwen/Qwen3-30B-A3B"
    return os.environ.get("FIREWORKS_E2E_TOKENIZER", DEFAULT_TOKENIZER)


def _make_chat_dataset(path: str, num_examples: int = 10) -> None:
    with open(path, "w") as f:
        for i in range(num_examples):
            row = {
                "messages": [
                    {"role": "user", "content": f"What is {i} times 2?"},
                    {"role": "assistant", "content": f"The answer is {i * 2}."},
                ]
            }
            f.write(json.dumps(row) + "\n")


@pytest.mark.e2e
@pytest.mark.timeout(5400)
class TestSFTResumeE2E:
    """SFT checkpoint-resume on qwen3-30b-a3b."""

    def test_sft_resume_from_checkpoint(
        self,
        sdk_managers,
        e2e_region,
        e2e_model,
        e2e_training_accelerator,
        custom_image_tag,
    ):
        rlor_mgr, deploy_mgr = sdk_managers
        tokenizer_model = _get_tokenizer_model(e2e_model)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            dataset_path = f.name

        try:
            _make_chat_dataset(dataset_path, num_examples=20)

            shared_infra = InfraConfig(
                region=e2e_region,
                custom_image_tag=custom_image_tag or "0.33.0",
            )

            log_dir = tempfile.mkdtemp(prefix="sft_resume_")

            # Phase 1: train, save DCP checkpoints to checkpoints.jsonl
            logger.info("PHASE 1: initial SFT training")

            phase1_config = Config(
                base_model=e2e_model,
                dataset=dataset_path,
                tokenizer_model=tokenizer_model,
                learning_rate=1e-4,
                epochs=2,
                batch_size=4,
                max_seq_len=4096,
                max_examples=20,
                dcp_save_interval=2,
                log_path=log_dir,
                infra=shared_infra,
            )

            phase1_metrics = main(phase1_config, rlor_mgr=rlor_mgr)

            assert isinstance(phase1_metrics, dict)
            phase1_steps = phase1_metrics["steps"]
            phase1_job_id = phase1_metrics["job_id"]
            assert phase1_steps >= 2, f"Expected >= 2 steps in phase 1, got {phase1_steps}"
            logger.info("Phase 1 done: %d steps, job=%s", phase1_steps, phase1_job_id)

            # Read the persisted raw_rows_consumed from dataloader.json.
            # In the new model, dataloader.json holds the only cookbook-side
            # state — one int per checkpoint name.
            dataloader_path = os.path.join(log_dir, DATALOADER_BASE_NAME)
            assert os.path.exists(dataloader_path), (
                f"Phase 1 should have written {DATALOADER_BASE_NAME} under {log_dir}"
            )
            with open(dataloader_path) as f:
                phase1_dataloader = json.load(f)
            assert phase1_dataloader, "dataloader.json should be non-empty after phase 1"
            phase1_raw_rows = max(int(v) for v in phase1_dataloader.values())

            # Verify the control plane has at least one resumable row for the
            # phase-1 trainer — that is what phase 2's resume will read.
            api_key = os.environ["FIREWORKS_API_KEY"]
            base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
            fw_client = FireworksClient(api_key=api_key, base_url=base_url)
            phase1_rows = fw_client.list_checkpoints(phase1_job_id)
            phase1_resumable = [
                r for r in phase1_rows
                if (r.get("checkpointType") or "").endswith(("TRAINING", "TRAINING_LORA"))
            ]
            assert phase1_resumable, (
                f"Expected phase 1 to leave a resumable row on the control plane "
                f"for job {phase1_job_id}; got rows: {phase1_rows}"
            )
            logger.info(
                "Phase 1 CP rows: %d resumable (raw_rows_consumed=%d)",
                len(phase1_resumable), phase1_raw_rows,
            )

            # Phase 2: reattach to the same trainer so resume can find the
            # phase-1 CP rows. (Auto-resume across separate trainer jobs is
            # not supported in the new model — explicit init_from_checkpoint
            # would be the cross-job equivalent and would reset the step.)
            logger.info("PHASE 2: reattach to job=%s, expect resume", phase1_job_id)

            phase2_config = Config(
                base_model=e2e_model,
                dataset=dataset_path,
                tokenizer_model=tokenizer_model,
                learning_rate=1e-4,
                epochs=2,
                batch_size=4,
                max_seq_len=4096,
                max_examples=20,
                log_path=log_dir,
                trainer_job_id=phase1_job_id,
                infra=shared_infra,
            )

            phase2_metrics = main(phase2_config, rlor_mgr=rlor_mgr)
            phase2_steps = phase2_metrics["steps"]
            assert phase2_steps > phase1_steps, (
                f"Phase 2 step count ({phase2_steps}) should exceed phase 1's "
                f"({phase1_steps}); resume probably did not pick up phase 1's CP rows."
            )

            with open(dataloader_path) as f:
                phase2_dataloader = json.load(f)
            phase2_raw_rows = max(int(v) for v in phase2_dataloader.values())
            assert phase2_raw_rows > phase1_raw_rows, (
                f"Phase 2 raw_rows_consumed ({phase2_raw_rows}) should exceed "
                f"phase 1's ({phase1_raw_rows}); dataloader cursor should advance."
            )
            logger.info(
                "Resume verified: phase1=%d steps (raw_rows=%d), "
                "phase2=%d steps (raw_rows=%d) -- dataloader continued",
                phase1_steps, phase1_raw_rows, phase2_steps, phase2_raw_rows,
            )
        finally:
            os.unlink(dataset_path)
