"""Minimal DPO smoke test on Qwen3-4B."""

from __future__ import annotations

import json
import os
import tempfile

import pytest


def _make_preference_dataset(path: str, num_pairs: int = 4) -> None:
    with open(path, "w") as f:
        for i in range(num_pairs):
            row = {
                "chosen": {
                    "messages": [
                        {"role": "user", "content": f"What is {i} plus {i}?"},
                        {"role": "assistant", "content": f"The answer is {i + i}."},
                    ]
                },
                "rejected": {
                    "messages": [
                        {"role": "user", "content": f"What is {i} plus {i}?"},
                        {"role": "assistant", "content": f"I think it is {i * 3}."},
                    ]
                },
            }
            f.write(json.dumps(row) + "\n")


@pytest.mark.e2e
@pytest.mark.timeout(3600)
def test_dpo_smoke(
    smoke_sdk_managers,
    smoke_base_model,
    smoke_tokenizer_model,
    smoke_infra,
):
    from training.recipes.dpo_loop import Config, main
    from training.utils import DeployConfig, HotloadConfig, WandBConfig

    rlor_mgr, deploy_mgr = smoke_sdk_managers

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        dataset_path = f.name

    try:
        _make_preference_dataset(dataset_path, num_pairs=4)
        config = Config(
            log_path=tempfile.mkdtemp(prefix="dpo_smoke_"),
            base_model=smoke_base_model,
            dataset=dataset_path,
            tokenizer_model=smoke_tokenizer_model,
            beta=0.1,
            learning_rate=1e-5,
            epochs=1,
            batch_size=1,
            grad_accum=1,
            max_pairs=4,
            infra=smoke_infra,
            deployment=DeployConfig(),
            hotload=HotloadConfig(hot_load_interval=0, dcp_save_interval=0),
            wandb=WandBConfig(),
        )

        metrics = main(config, rlor_mgr=rlor_mgr, deploy_mgr=deploy_mgr)
        assert isinstance(metrics, dict)
        assert metrics["steps"] >= 2, f"Expected >= 2 steps, got {metrics['steps']}"
    finally:
        os.unlink(dataset_path)
