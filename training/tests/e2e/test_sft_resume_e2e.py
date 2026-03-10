"""E2E test: SFT training -> DCP checkpoint -> resume -> verify continuation.

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

from training.utils import InfraConfig
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

            # Phase 1: train, save DCP
            logger.info("PHASE 1: initial SFT training")

            import tempfile as _tf
            log_dir = _tf.mkdtemp(prefix="sft_resume_")

            phase1_config = Config(
                base_model=e2e_model,
                dataset=dataset_path,
                tokenizer_model=tokenizer_model,
                learning_rate=1e-4,
                epochs=2,
                batch_size=4,
                grad_accum=2,
                max_seq_len=4096,
                max_examples=20,
                dcp_save_interval=2,
                log_path=log_dir,
                infra=shared_infra,
            )

            phase1_metrics = main(phase1_config, rlor_mgr=rlor_mgr)

            assert isinstance(phase1_metrics, dict)
            assert "steps" in phase1_metrics
            phase1_steps = phase1_metrics["steps"]
            assert phase1_steps >= 2, f"Expected >= 2 steps in phase 1, got {phase1_steps}"

            phase1_job_id = phase1_metrics["job_id"]
            dcp_name = f"step-{phase1_steps}"
            logger.info("Phase 1 done: %d steps, job=%s", phase1_steps, phase1_job_id)

            # Phase 2: resume from checkpoint
            logger.info("PHASE 2: resume from '%s' (source job: %s)", dcp_name, phase1_job_id)

            phase2_config = Config(
                base_model=e2e_model,
                dataset=dataset_path,
                tokenizer_model=tokenizer_model,
                learning_rate=1e-4,
                epochs=2,
                batch_size=4,
                grad_accum=2,
                max_seq_len=4096,
                max_examples=20,
                log_path=log_dir,
                infra=shared_infra,
                init_from_checkpoint=f"{phase1_job_id}:{dcp_name}",
            )

            phase2_metrics = main(phase2_config, rlor_mgr=rlor_mgr)

            assert isinstance(phase2_metrics, dict)
            assert "steps" in phase2_metrics
            phase2_steps = phase2_metrics["steps"]
            assert phase2_steps >= 2, (
                f"Expected >= 2 steps in phase 2 (init_from_checkpoint), got {phase2_steps}"
            )

            logger.info(
                "Resume verified: phase1=%d steps, phase2=%d steps (from init_from_checkpoint)",
                phase1_steps,
                phase2_steps,
            )
        finally:
            os.unlink(dataset_path)
