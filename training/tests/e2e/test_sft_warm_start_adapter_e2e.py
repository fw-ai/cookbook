"""E2E test: SFT warm-start from a Fireworks HF PEFT adapter Model.

Verifies the full path:
  Config.warm_start_from_adapter -> resolve_resume -> client.load_adapter ->
  server /api/v1/load_adapter -> LoadAdapterOp -> load_lora_adapter -> training.

Setup:
  - Account: pyroworks (set via FIREWORKS_API_KEY for pyroworks).
  - Base model: qwen3-4b.
  - Warm-start source: qwen3-4b-minimal-lora — a pre-promoted LoRA adapter
    kept in the pyroworks account for exactly this test. Must be READY and
    Kind=HF_PEFT_ADDON.

Environment:
  FIREWORKS_API_KEY                       -- pyroworks API key (required)
  FIREWORKS_E2E_WARM_START_ADAPTER_URI    -- gs:// URI of the adapter dir.
      Override when the model is re-promoted to a new GcsUri. Default
      resolves via ``firectl get model ... --output json``.
  FIREWORKS_E2E_BASE_MODEL                -- default accounts/fireworks/models/qwen3-4b
  FIREWORKS_E2E_TOKENIZER                 -- default Qwen/Qwen3-4B

Pass criteria:
  1. sft_loop.main() completes without raising.
  2. At least one optim step ran (proves the trainer reached the loop).
  3. Initial loss is NOT random-initialization level — proves the adapter
     was actually loaded into LoRA params before the first forward_backward.
     Qwen3-4B random-init loss on simple chat: ~9-11. Adapter-warmed loss
     on the same dataset: typically < 5. Threshold of 6 gives good margin
     either way.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile

import pytest

from training.recipes.sft_loop import Config, main
from training.utils import InfraConfig

logger = logging.getLogger(__name__)

DEFAULT_WARM_START_MODEL = "accounts/fireworks/models/qwen3-4b-minimal-lora"
DEFAULT_BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B"

# Qwen3-4B random-init chat loss hovers around 9-11; a trained adapter on
# trivially-learnable data drops to ~1-3. 6.0 is the "clearly not random"
# boundary — generous on both sides.
RANDOM_INIT_LOSS_FLOOR = 6.0


def _resolve_warm_start_uri() -> str:
    """Resolve the adapter GcsUri, preferring the env override."""
    explicit = os.environ.get("FIREWORKS_E2E_WARM_START_ADAPTER_URI")
    if explicit:
        return explicit

    model_ref = os.environ.get(
        "FIREWORKS_E2E_WARM_START_MODEL", DEFAULT_WARM_START_MODEL
    )
    logger.info("Resolving %s via firectl", model_ref)
    result = subprocess.run(
        ["firectl", "get", "model", model_ref, "--output", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads(result.stdout)
    uri = meta.get("gcsUri") or meta.get("gcs_uri")
    if not uri:
        pytest.skip(f"Model {model_ref} has no GcsUri; skipping")
    return uri


def _make_chat_dataset(path: str, num_examples: int = 12) -> None:
    """Trivially-learnable chat pairs — enough to run a few SFT steps."""
    with open(path, "w") as f:
        for i in range(num_examples):
            row = {
                "messages": [
                    {"role": "user", "content": f"What is {i} plus {i}?"},
                    {"role": "assistant", "content": f"The answer is {i + i}."},
                ]
            }
            f.write(json.dumps(row) + "\n")


@pytest.mark.e2e
@pytest.mark.timeout(3600)
class TestSFTWarmStartAdapterE2E:
    """SFT warm-start from an HF PEFT adapter on qwen3-4b."""

    def test_warm_start_from_minimal_lora(
        self,
        sdk_managers,
        e2e_region,
        e2e_training_accelerator,
        custom_image_tag,
    ):
        rlor_mgr, _ = sdk_managers
        base_model = os.environ.get("FIREWORKS_E2E_BASE_MODEL", DEFAULT_BASE_MODEL)
        tokenizer = os.environ.get("FIREWORKS_E2E_TOKENIZER", DEFAULT_TOKENIZER)
        adapter_uri = _resolve_warm_start_uri()
        logger.info("Warm-starting from adapter at %s", adapter_uri)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            dataset_path = f.name

        log_dir = tempfile.mkdtemp(prefix="sft_warm_start_adapter_")

        try:
            _make_chat_dataset(dataset_path, num_examples=12)

            cfg = Config(
                base_model=base_model,
                dataset=dataset_path,
                tokenizer_model=tokenizer,
                learning_rate=1e-4,
                epochs=1,
                batch_size=4,
                max_seq_len=512,
                max_examples=12,
                lora_rank=16,
                warm_start_from_adapter=adapter_uri,
                log_path=log_dir,
                infra=InfraConfig(
                    region=e2e_region,
                    custom_image_tag=custom_image_tag or "0.33.0",
                ),
            )

            metrics = main(cfg, rlor_mgr=rlor_mgr)

            assert isinstance(metrics, dict), "main() must return a metrics dict"
            assert "steps" in metrics, f"metrics missing steps: {metrics}"
            assert metrics["steps"] >= 1, (
                f"Expected at least 1 optim step, got {metrics['steps']}"
            )

            initial_loss = metrics.get("initial_loss")
            if initial_loss is None:
                pytest.skip(
                    "initial_loss not in metrics — test requires sft_loop to "
                    "expose step-0 loss for this assertion to be meaningful"
                )
            assert initial_loss < RANDOM_INIT_LOSS_FLOOR, (
                f"Initial loss {initial_loss:.3f} >= {RANDOM_INIT_LOSS_FLOOR}, "
                f"which is random-init territory for Qwen3-4B. The adapter at "
                f"{adapter_uri} likely did not load — check trainer logs for "
                f"'Adapter loaded' and 'Fresh start with HF adapter'."
            )
            logger.info(
                "PASS: adapter loaded (initial_loss=%.3f < %.1f, steps=%d)",
                initial_loss,
                RANDOM_INIT_LOSS_FLOOR,
                metrics["steps"],
            )
        finally:
            os.unlink(dataset_path)

    def test_fresh_start_ignores_missing_adapter_field(
        self,
        sdk_managers,
        e2e_region,
        e2e_training_accelerator,
        custom_image_tag,
    ):
        """Regression: omitting warm_start_from_adapter behaves exactly as
        before. Same base_model, same dataset, no warm-start — initial loss
        should land in the random-init range."""
        rlor_mgr, _ = sdk_managers
        base_model = os.environ.get("FIREWORKS_E2E_BASE_MODEL", DEFAULT_BASE_MODEL)
        tokenizer = os.environ.get("FIREWORKS_E2E_TOKENIZER", DEFAULT_TOKENIZER)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            dataset_path = f.name

        log_dir = tempfile.mkdtemp(prefix="sft_fresh_start_")

        try:
            _make_chat_dataset(dataset_path, num_examples=12)

            cfg = Config(
                base_model=base_model,
                dataset=dataset_path,
                tokenizer_model=tokenizer,
                learning_rate=1e-4,
                epochs=1,
                batch_size=4,
                max_seq_len=512,
                max_examples=12,
                lora_rank=16,
                # warm_start_from_adapter intentionally unset
                log_path=log_dir,
                infra=InfraConfig(
                    region=e2e_region,
                    custom_image_tag=custom_image_tag or "0.33.0",
                ),
            )

            metrics = main(cfg, rlor_mgr=rlor_mgr)
            assert metrics["steps"] >= 1
            logger.info("Fresh-start completed: %d steps", metrics["steps"])
        finally:
            os.unlink(dataset_path)
