"""Minimal DPO smoke test on Qwen3-4B."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace

import pytest

from training.utils import DeployConfig, WandBConfig


_DELETING_STATES = {
    "JOB_STATE_DELETING",
    "JOB_STATE_DELETING_CLEANING_UP",
    "JOB_STATE_DELETED",
}


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
    smoke_trainer_config,
    port_lora_rank,
    port_track_state,
):
    from training.recipes.dpo_loop import Config, main

    rlor_mgr, _deploy_mgr = smoke_sdk_managers
    job_ids: list[str] = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        dataset_path = f.name

    try:
        port_track_state.clear()
        _make_preference_dataset(dataset_path, num_pairs=4)
        trainer = smoke_trainer_config
        if port_lora_rank == 0:
            trainer = replace(trainer, cleanup_reference_on_close=False)
        config = Config(
            log_path=tempfile.mkdtemp(prefix="dpo_smoke_"),
            base_model=smoke_base_model,
            dataset=dataset_path,
            tokenizer_model=smoke_tokenizer_model,
            beta=0.1,
            learning_rate=1e-5,
            epochs=1,
            batch_size=1,
            max_pairs=4,
            lora_rank=port_lora_rank,
            dcp_save_interval=2,
            trainer=trainer,
            deployment=DeployConfig(),
            release_reference_after_cache=port_lora_rank > 0,
            wandb=WandBConfig(),
        )

        metrics = main(config)
        assert isinstance(metrics, dict)
        assert metrics["steps"] >= 2, f"Expected >= 2 steps, got {metrics['steps']}"
        policy_job_id = metrics.get("policy_job_id")
        reference_job_id = metrics.get("reference_job_id")
        assert policy_job_id, f"DPO should return a policy trainer id: {metrics}"
        if port_lora_rank == 0:
            assert reference_job_id, (
                f"Full-param DPO should keep a reusable reference trainer id: {metrics}"
            )
        else:
            assert reference_job_id is None, (
                f"LoRA DPO should use the shared-reference path by default: {metrics}"
            )
        job_ids = [j for j in (policy_job_id, reference_job_id) if j]
        if policy_job_id:
            port_track_state.update(
                policy_job_id=policy_job_id,
                reference_job_id=reference_job_id,
                log_path=config.log_path,
                dpo_log_path=config.log_path,
                lora_rank=port_lora_rank,
            )

        import httpx, time
        time.sleep(3)
        for job_id in job_ids:
            try:
                job = rlor_mgr.get(job_id)
                state = job.get("state", "")
                assert state not in _DELETING_STATES, (
                    f"Trainer {job_id} should remain live for the next track phase; got {state}"
                )
            except httpx.HTTPStatusError as e:
                raise AssertionError(f"Trainer {job_id} should remain visible for reuse") from e
    finally:
        os.unlink(dataset_path)
