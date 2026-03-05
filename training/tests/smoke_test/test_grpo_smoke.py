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
    pytest training/tests/e2e/smoke_test/test_grpo_smoke.py -v -s
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

import pytest

from fireworks.training.sdk.trainer import TrainerJobManager
from fireworks.training.sdk.deployment import DeploymentManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
TOKENIZER_MODEL = "Qwen/Qwen3-4B"
TRAINING_SHAPE = "ts-qwen3-4b-smoke-v1"
DEFAULT_BASE_URL = "https://dev.api.fireworks.ai"


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


def _get_sdk_managers():
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        pytest.skip("FIREWORKS_API_KEY not set")

    account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if not account_id:
        pytest.skip("FIREWORKS_ACCOUNT_ID not set")

    base_url = os.environ.get("FIREWORKS_BASE_URL", DEFAULT_BASE_URL)
    inference_url = os.environ.get("FIREWORKS_INFERENCE_URL", base_url)
    hotload_api_url = os.environ.get("FIREWORKS_HOTLOAD_API_URL", base_url)

    additional_headers = {}
    gateway_secret = os.environ.get("FIREWORKS_GATEWAY_SECRET")
    if gateway_secret:
        additional_headers["X-Fireworks-Gateway-Secret"] = gateway_secret

    rlor_mgr = TrainerJobManager(
        api_key=api_key,
        account_id=account_id,
        base_url=base_url,
        additional_headers=additional_headers or None,
    )
    deploy_mgr = DeploymentManager(
        api_key=api_key,
        account_id=account_id,
        base_url=base_url,
        inference_url=inference_url,
        hotload_api_url=hotload_api_url,
        additional_headers=additional_headers or None,
    )
    return rlor_mgr, deploy_mgr


@pytest.mark.e2e
@pytest.mark.timeout(3600)
def test_grpo_smoke():
    """2-step GRPO on Qwen3-4B: train, hotload, train again, cleanup."""
    from training.utils import InfraConfig, DeployConfig, HotloadConfig, WandBConfig
    from training.recipes.rl_loop import Config, main
    import training.recipes.rl_loop as rl_mod

    rlor_mgr, deploy_mgr = _get_sdk_managers()

    rl_mod.reward_fn = _constant_reward

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        dataset_path = f.name

    try:
        _make_prompt_dataset(dataset_path, n=4)

        config = Config(
            base_model=BASE_MODEL,
            dataset=dataset_path,
            learning_rate=1e-4,
            kl_beta=0,
            completions_per_prompt=4,
            prompt_groups_per_step=1,
            max_completion_tokens=32,
            max_rows=2,
            epochs=1,
            infra=InfraConfig(
                training_shape_id=TRAINING_SHAPE,
                skip_validations=True,
            ),
            deployment=DeployConfig(
                tokenizer_model=TOKENIZER_MODEL,
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
