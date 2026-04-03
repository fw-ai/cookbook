"""E2E test for visual FrozenLake RFT on Qwen3-VL-8B.

Exercises the image-observation FrozenLake path with the shared validated
Qwen3-VL training shape. This is a policy-only RL run: Qwen3-VL currently
does not publish a forward-only reference shape, so we run with ``kl_beta=0``.

Requires:
  FIREWORKS_API_KEY

Optional env vars:
  FIREWORKS_BASE_URL                  (default: https://api.fireworks.ai)
  FROZEN_LAKE_VL_BASE_MODEL          (default: accounts/fireworks/models/qwen3-vl-8b-instruct)
  FROZEN_LAKE_VL_TOKENIZER_MODEL     (default: Qwen/Qwen3-VL-8B-Instruct)
  FROZEN_LAKE_VL_TRAINING_SHAPE      (default: accounts/fireworks/trainingShapes/qwen3-vl-8b-65k)
  FROZEN_LAKE_VL_DEPLOYMENT_SHAPE    (optional explicit override)
  FROZEN_LAKE_VL_EPOCHS              (default: 1)
  FROZEN_LAKE_VL_MAX_SEEDS           (default: 8)
  FROZEN_LAKE_VL_MAX_STEPS           (default: 8)
  FROZEN_LAKE_VL_COMPLETIONS         (default: 4)
  FROZEN_LAKE_VL_PROMPT_GROUPS       (default: 1)
  FROZEN_LAKE_VL_MAX_CONCURRENT      (default: 4)
"""

from __future__ import annotations

import os
import tempfile

import pytest

from training.examples.rl.frozen_lake.train_frozen_lake import (
    FrozenLakeConfig,
    main as frozen_lake_main,
)


DEFAULT_VL_BASE_MODEL = "accounts/fireworks/models/qwen3-vl-8b-instruct"
DEFAULT_VL_TOKENIZER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_VL_TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3-vl-8b-65k"


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@pytest.mark.e2e
@pytest.mark.timeout(5400)
class TestFrozenLakeVLE2E:
    """Visual FrozenLake GRPO on Qwen3-VL-8B with weight sync."""

    def test_frozen_lake_vl_pipeline(self):
        api_key = _env("FIREWORKS_API_KEY")
        if not api_key:
            pytest.skip("FIREWORKS_API_KEY not set")

        base_url = _env("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
        os.environ.setdefault("FIREWORKS_BASE_URL", base_url)
        os.environ.setdefault("WANDB_MODE", "disabled")

        config = FrozenLakeConfig(
            log_path=tempfile.mkdtemp(prefix="frozen_lake_vl_e2e_"),
            base_model=_env("FROZEN_LAKE_VL_BASE_MODEL", DEFAULT_VL_BASE_MODEL),
            tokenizer_model=_env(
                "FROZEN_LAKE_VL_TOKENIZER_MODEL",
                DEFAULT_VL_TOKENIZER_MODEL,
            ),
            training_shape=_env(
                "FROZEN_LAKE_VL_TRAINING_SHAPE",
                DEFAULT_VL_TRAINING_SHAPE,
            )
            or "",
            deployment_shape=_env("FROZEN_LAKE_VL_DEPLOYMENT_SHAPE", "") or "",
            learning_rate=1e-5,
            kl_beta=0.0,
            completions_per_prompt=int(_env("FROZEN_LAKE_VL_COMPLETIONS", "4") or "4"),
            prompt_groups_per_step=int(_env("FROZEN_LAKE_VL_PROMPT_GROUPS", "1") or "1"),
            max_concurrent=int(_env("FROZEN_LAKE_VL_MAX_CONCURRENT", "4") or "4"),
            epochs=int(_env("FROZEN_LAKE_VL_EPOCHS", "1") or "1"),
            max_seeds=int(_env("FROZEN_LAKE_VL_MAX_SEEDS", "8") or "8"),
            max_steps=int(_env("FROZEN_LAKE_VL_MAX_STEPS", "8") or "8"),
            observation_mode="image",
            allow_plaintext_action_fallback=True,
        )

        result = frozen_lake_main(config)

        assert isinstance(result, dict), "main() should return a dict"
        assert result["steps"] >= 2, f"Expected >= 2 optimizer steps, got {result['steps']}"
        assert result["rewards"], "Expected non-empty reward history"
        assert max(result["rewards"]) > 0.0, (
            f"Expected the visual FrozenLake run to achieve a non-zero reward, "
            f"got {result['rewards']}"
        )
