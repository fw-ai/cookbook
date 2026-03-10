"""Minimal GRPO e2e smoke test on Qwen3-4B.

Runs 2 optimizer steps with hotloading on a single GPU:
  Step 1: fwd/bkwd + optim_step -> hotload to deployment
  Step 2: fwd/bkwd + optim_step (sampling from updated weights)
  Cleanup: delete trainer job, scale deployment to zero

Requires env vars (skipped if not set):
    FIREWORKS_API_KEY
    FIREWORKS_ACCOUNT_ID
    FIREWORKS_BASE_URL      (optional, defaults to https://dev.api.fireworks.ai)

Usage:
    pytest training/tests/smoke_test/test_grpo_smoke.py -v -s
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)


def _make_prompt_dataset(path: str, n: int = 4) -> None:
    with open(path, "w") as f:
        for i in range(n):
            row = {
                "messages": [
                    {"role": "user", "content": f"What is {i + 1} + {i + 1}?"},
                ],
                "ground_truth": str((i + 1) * 2),
            }
            f.write(json.dumps(row) + "\n")


def _constant_reward(_completion: str, _row: dict) -> float:
    return 1.0


@pytest.mark.e2e
@pytest.mark.timeout(3600)
def test_grpo_smoke(
    smoke_sdk_managers,
    smoke_base_model,
    smoke_tokenizer_model,
    smoke_training_shape,
    smoke_custom_image_tag,
):
    """2-step GRPO on Qwen3-4B: train, hotload, train again, cleanup."""
    from training.utils import InfraConfig, DeployConfig, HotloadConfig, WandBConfig
    from training.recipes.rl_loop import Config, main
    import training.recipes.rl_loop as rl_mod

    rlor_mgr, deploy_mgr = smoke_sdk_managers

    rl_mod.reward_fn = _constant_reward

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        dataset_path = f.name

    try:
        _make_prompt_dataset(dataset_path, n=4)

        config = Config(
            log_path=tempfile.mkdtemp(prefix="grpo_smoke_"),
            base_model=smoke_base_model,
            dataset=dataset_path,
            learning_rate=1e-4,
            kl_beta=0,
            completions_per_prompt=4,
            prompt_groups_per_step=1,
            max_completion_tokens=32,
            max_rows=2,
            epochs=1,
            infra=InfraConfig(
                training_shape_id=smoke_training_shape,
                skip_validations=True,
                custom_image_tag=smoke_custom_image_tag,
            ),
            deployment=DeployConfig(
                tokenizer_model=smoke_tokenizer_model,
            ),
            hotload=HotloadConfig(
                hot_load_interval=1,
                first_checkpoint_type="base",
                hot_load_before_training=True,
                hot_load_timeout=600,
            ),
            wandb=WandBConfig(),
        )

        metrics = main(
            config,
            rlor_mgr=rlor_mgr,
            deploy_mgr=deploy_mgr,
            cleanup_on_exit=True,
        )

        assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"
        assert "steps" in metrics, f"Missing 'steps' in metrics: {metrics}"
        assert metrics["steps"] >= 2, f"Expected >= 2 steps, got {metrics['steps']}"
    finally:
        os.unlink(dataset_path)
