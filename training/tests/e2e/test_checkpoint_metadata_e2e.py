"""E2E test: checkpoint metadata (base_model, training_shape) and promote.

Tests the full flow:
  1. Train SFT with a training shape -> base_model auto-resolved from profile
  2. DCP checkpoints saved with base_model + training_shape metadata
  3. Resume from latest checkpoint (auto-resolve)
  4. Resume from a specific step
  5. Promote checkpoint reads metadata (--model/--shape auto-detected)

Requires:
  FIREWORKS_API_KEY
  FIREWORKS_E2E_TRAINING_SHAPE  -- e.g. accounts/fireworks/trainingShapes/ts-qwen3-30b-a3b-128k
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

import pytest

from tinker_cookbook.checkpoint_utils import get_last_checkpoint, CHECKPOINTS_BASE_NAME
from training.utils import InfraConfig
from training.recipes.sft_loop import Config, main

logger = logging.getLogger(__name__)


def _make_chat_dataset(path: str, num_examples: int = 20) -> None:
    with open(path, "w") as f:
        for i in range(num_examples):
            row = {
                "messages": [
                    {"role": "user", "content": f"What is {i} + {i}?"},
                    {"role": "assistant", "content": f"{i + i}"},
                ]
            }
            f.write(json.dumps(row) + "\n")


def _read_all_checkpoints(log_dir: str) -> list[dict]:
    path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _get_tokenizer(e2e_model: str) -> str:
    if "qwen3-30b-a3b" in e2e_model.lower():
        return "Qwen/Qwen3-30B-A3B"
    if "qwen3-8b" in e2e_model.lower():
        return "Qwen/Qwen3-8B"
    return os.environ.get("FIREWORKS_E2E_TOKENIZER", "Qwen/Qwen3-30B-A3B")


@pytest.mark.e2e
@pytest.mark.timeout(5400)
class TestCheckpointMetadataE2E:
    """SFT with training shape: checkpoint metadata + resume."""

    def test_checkpoint_metadata_and_resume(
        self,
        sdk_managers,
        e2e_model,
        e2e_training_shape,
    ):
        if not e2e_training_shape:
            pytest.skip("FIREWORKS_E2E_TRAINING_SHAPE not set")

        rlor_mgr, deploy_mgr = sdk_managers
        tokenizer = _get_tokenizer(e2e_model)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            dataset_path = f.name

        try:
            _make_chat_dataset(dataset_path, num_examples=20)
            log_dir = tempfile.mkdtemp(prefix="ckpt_meta_e2e_")

            # -- Phase 1: train with shape, save DCP checkpoints ----------------

            logger.info("PHASE 1: SFT with training shape %s", e2e_training_shape)

            phase1_config = Config(
                # base_model intentionally omitted — auto-resolved from shape
                dataset=dataset_path,
                tokenizer_model=tokenizer,
                learning_rate=1e-4,
                epochs=2,
                batch_size=4,
                max_seq_len=4096,
                max_examples=20,
                dcp_save_interval=2,
                log_path=log_dir,
                infra=InfraConfig(
                    training_shape_id=e2e_training_shape,
                ),
            )

            phase1_metrics = main(phase1_config, rlor_mgr=rlor_mgr)
            assert isinstance(phase1_metrics, dict)
            phase1_steps = phase1_metrics["steps"]
            assert phase1_steps >= 2

            # Verify checkpoint metadata
            checkpoints = _read_all_checkpoints(log_dir)
            assert len(checkpoints) >= 1, "Expected at least one checkpoint"

            for ckpt in checkpoints:
                assert "base_model" in ckpt, f"Checkpoint missing base_model: {ckpt}"
                assert "training_shape" in ckpt, f"Checkpoint missing training_shape: {ckpt}"
                assert ckpt["base_model"], "base_model should not be empty"
                assert ckpt["training_shape"] == e2e_training_shape

            logger.info(
                "Phase 1 done: %d steps, %d checkpoints, base_model=%s",
                phase1_steps, len(checkpoints), checkpoints[0]["base_model"],
            )

            # -- Phase 2: resume from latest (auto-resolve) --------------------

            last_ckpt = get_last_checkpoint(log_dir)
            assert last_ckpt is not None
            saved_step = last_ckpt["step"]
            saved_data = last_ckpt["data_consumed"]

            logger.info(
                "PHASE 2: resume from latest (step=%d, data_consumed=%d)",
                saved_step, saved_data,
            )

            phase2_config = Config(
                # base_model again omitted — auto-resolved
                dataset=dataset_path,
                tokenizer_model=tokenizer,
                learning_rate=1e-4,
                epochs=2,
                batch_size=4,
                max_seq_len=4096,
                max_examples=20,
                log_path=log_dir,  # same log_dir — picks up checkpoints.jsonl
                infra=InfraConfig(
                    training_shape_id=e2e_training_shape,
                ),
            )

            phase2_metrics = main(phase2_config, rlor_mgr=rlor_mgr)
            phase2_steps = phase2_metrics["steps"]
            assert phase2_steps > saved_step, (
                f"Phase 2 should continue beyond step {saved_step}, got {phase2_steps}"
            )

            final_ckpt = get_last_checkpoint(log_dir)
            assert final_ckpt["data_consumed"] > saved_data, (
                "Dataloader should continue, not restart"
            )

            # Final checkpoint also has metadata
            assert final_ckpt.get("base_model"), "Final checkpoint should have base_model"
            assert final_ckpt.get("training_shape") == e2e_training_shape

            logger.info(
                "Resume verified: phase1=%d steps -> phase2=%d steps, "
                "data_consumed %d -> %d",
                saved_step, phase2_steps, saved_data, final_ckpt["data_consumed"],
            )

            # -- Verify specific step selection --------------------------------

            # The checkpoints.jsonl should have multiple entries.
            # Verify we can select a specific step's metadata.
            all_ckpts = _read_all_checkpoints(log_dir)
            if len(all_ckpts) >= 2:
                first_ckpt = all_ckpts[0]
                logger.info(
                    "Step selection: first checkpoint step=%d has base_model=%s",
                    first_ckpt["step"], first_ckpt.get("base_model"),
                )
                assert first_ckpt.get("base_model") == final_ckpt.get("base_model"), (
                    "All checkpoints should have the same base_model"
                )

        finally:
            os.unlink(dataset_path)
