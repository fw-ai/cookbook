"""Minimal SFT smoke test on Qwen3-4B."""

from __future__ import annotations

import json
import os
import tempfile

import httpx
import pytest


_DELETING_STATES = {
    "JOB_STATE_DELETING",
    "JOB_STATE_DELETING_CLEANING_UP",
    "JOB_STATE_DELETED",
}


def _delete_trainer_if_present(rlor_mgr, job_id: str) -> None:
    try:
        rlor_mgr.delete(job_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise


def _make_chat_dataset(path: str, num_examples: int = 4) -> None:
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
def test_sft_smoke(
    smoke_sdk_managers,
    smoke_base_model,
    smoke_tokenizer_model,
    smoke_trainer_config,
    port_lora_rank,
):
    from training.recipes.sft_loop import Config, main
    from training.utils import WandBConfig

    rlor_mgr, _deploy_mgr = smoke_sdk_managers
    job_id = None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        dataset_path = f.name

    try:
        _make_chat_dataset(dataset_path, num_examples=4)
        config = Config(
            log_path=tempfile.mkdtemp(prefix="sft_smoke_"),
            base_model=smoke_base_model,
            dataset=dataset_path,
            tokenizer_model=smoke_tokenizer_model,
            learning_rate=1e-4,
            epochs=1,
            batch_size=2,
            max_examples=4,
            lora_rank=port_lora_rank,
            trainer=smoke_trainer_config,
            wandb=WandBConfig(),
        )

        metrics = main(config)
        assert isinstance(metrics, dict)
        assert metrics["steps"] >= 2, f"Expected >= 2 steps, got {metrics['steps']}"
        job_id = metrics.get("job_id")

        import time
        time.sleep(3)
        if job_id:
            _delete_trainer_if_present(rlor_mgr, job_id)
            try:
                job = rlor_mgr.get(job_id)
                state = job.get("state", "")
                assert state in _DELETING_STATES, (
                    f"ResourceCleanup failed: job {job_id} still {state}"
                )
            except httpx.HTTPStatusError as e:
                assert e.response.status_code == 404
    finally:
        if job_id:
            _delete_trainer_if_present(rlor_mgr, job_id)
        os.unlink(dataset_path)
