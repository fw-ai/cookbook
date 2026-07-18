"""Minimal GRPO e2e smoke test on Qwen3-4B.

Runs 2 optimizer steps with weight syncing on a single GPU:
  Step 1: fwd/bkwd + optim_step -> weight sync to deployment
  Step 2: fwd/bkwd + optim_step (sampling from updated weights)
  Cleanup: cancel trainer job, scale deployment to zero

Requires env vars (skipped if not set):
    FIREWORKS_API_KEY
    FIREWORKS_BASE_URL      (optional, defaults to https://dev.api.fireworks.ai)

Usage:
    pytest training/tests/smoke_test/test_grpo_smoke.py -v -s
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import replace

import httpx
import pytest

from training.tests.async_grpo_helpers import (
    MAX_REALISTIC_COMPLETION_TOKENS,
    completion_hash_reward,
    make_message_rollout_fn_factory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

_DELETING_STATES = {
    "JOB_STATE_DELETING",
    "JOB_STATE_DELETING_CLEANING_UP",
    "JOB_STATE_DELETED",
}


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


@pytest.mark.e2e
@pytest.mark.timeout(3600)
def test_grpo_smoke(
    smoke_sdk_managers,
    smoke_base_model,
    smoke_tokenizer_model,
    smoke_trainer_config,
    port_lora_rank,
):
    """2-step async GRPO: train, weight sync, train again, cleanup."""
    from training.utils import DeployConfig, WandBConfig
    from training.recipes.async_rl_loop import Config, main

    rlor_mgr, deploy_mgr = smoke_sdk_managers
    policy_job_id = None
    reference_job_id = None
    deployment_id = None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        dataset_path = f.name

    try:
        _make_prompt_dataset(dataset_path, n=4)
        trainer = replace(
            smoke_trainer_config,
            cleanup_reference_on_close=False,
        )
        config = Config(
            log_path=tempfile.mkdtemp(prefix="grpo_smoke_"),
            base_model=smoke_base_model,
            dataset=dataset_path,
            learning_rate=1e-4,
            kl_beta=0 if port_lora_rank else 0.001,
            completions_per_prompt=4,
            prompt_groups_per_step=1,
            max_completion_tokens=MAX_REALISTIC_COMPLETION_TOKENS,
            # GRPO may filter prompt groups when sampled completions are not
            # usable. Keep enough rows that the smoke test still performs at
            # least two optimizer steps under normal stochastic sampling.
            max_rows=4,
            epochs=1,
            lora_rank=port_lora_rank,
            trainer=trainer,
            deployment=DeployConfig(
                tokenizer_model=smoke_tokenizer_model,
            ),
            weight_sync_before_training=True,
            weight_sync_timeout=600,
            wandb=WandBConfig(),
        )

        metrics = main(
            config,
            rollout_fn_factory=make_message_rollout_fn_factory(completion_hash_reward),
        )

        assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"
        assert "steps" in metrics, f"Missing 'steps' in metrics: {metrics}"
        assert metrics["steps"] >= 2, f"Expected >= 2 steps, got {metrics['steps']}"
        returned_policy_job_id = metrics.get("policy_job_id")
        returned_reference_job_id = metrics.get("reference_job_id")
        deployment_id = metrics.get("deployment_id")
        assert returned_policy_job_id, (
            f"GRPO should return policy trainer id: {metrics}"
        )
        policy_job_id = returned_policy_job_id
        reference_job_id = returned_reference_job_id
        if port_lora_rank:
            assert returned_reference_job_id is None, (
                f"LoRA GRPO should not create a separate reference trainer: {metrics}"
            )
        else:
            assert returned_reference_job_id, (
                f"Full-param GRPO should return reference trainer id: {metrics}"
            )
        assert deployment_id, f"Async GRPO should report deployment id: {metrics}"

        import time

        time.sleep(3)
        try:
            job = rlor_mgr.get(returned_policy_job_id)
            state = job.get("state", "")
            assert state not in _DELETING_STATES, (
                f"Trainer {returned_policy_job_id} should remain visible before cleanup; got {state}"
            )
        except httpx.HTTPStatusError as e:
            raise AssertionError(
                f"Trainer {returned_policy_job_id} should remain visible before cleanup"
            ) from e
    finally:
        os.unlink(dataset_path)
        if deployment_id:
            _delete_deployment_if_present(deploy_mgr, deployment_id)
        if policy_job_id:
            _delete_trainer_if_present(rlor_mgr, policy_job_id)
        if reference_job_id:
            _delete_trainer_if_present(rlor_mgr, reference_job_id)


def _delete_deployment_if_present(deploy_mgr, deployment_id: str) -> None:
    try:
        deploy_mgr.delete(deployment_id)
        deploy_mgr._wait_for_deletion(deployment_id, timeout_s=120)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise


def _delete_trainer_if_present(rlor_mgr, job_id: str) -> None:
    try:
        rlor_mgr.delete(job_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise
