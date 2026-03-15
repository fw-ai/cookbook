"""Minimal GRPO e2e smoke test on Qwen3-4B.

Runs 2 optimizer steps with weight syncing on a single GPU:
  Step 1: fwd/bkwd + optim_step -> weight sync to deployment
  Step 2: fwd/bkwd + optim_step (sampling from updated weights)
  Cleanup: delete trainer job, scale deployment to zero

Requires env vars (skipped if not set):
    FIREWORKS_API_KEY
    FIREWORKS_ACCOUNT_ID
    FIREWORKS_BASE_URL      (optional, defaults to https://dev.api.fireworks.ai)

Usage:
    pytest training/tests/smoke_test/test_grpo_smoke.py -v -s

Optional env vars to exercise LoRA reference reuse:
    FIREWORKS_SMOKE_GRPO_LORA_RANK
    FIREWORKS_SMOKE_GRPO_KL_BETA
    FIREWORKS_SMOKE_GRPO_POLICY_JOB_ID
    FIREWORKS_SMOKE_GRPO_POLICY_BASE_URL
    FIREWORKS_SMOKE_GRPO_DEPLOYMENT_ID
    FIREWORKS_SMOKE_GRPO_MAX_SEQ_LEN
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


def _get_float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


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
    """2-step GRPO on Qwen3-4B: train, weight sync, train again, cleanup."""
    from training.utils import DeployConfig, InfraConfig, WeightSyncConfig, WandBConfig
    from training.recipes.rl_loop import Config, main
    import training.recipes.rl_loop as rl_mod

    rlor_mgr, deploy_mgr = smoke_sdk_managers

    original_reward_fn = rl_mod.reward_fn
    original_should_accept = rl_mod.should_accept
    rl_mod.reward_fn = _constant_reward
    rl_mod.should_accept = lambda _pg: True

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        dataset_path = f.name

    try:
        _make_prompt_dataset(dataset_path, n=4)

        smoke_lora_rank = int(os.environ.get("FIREWORKS_SMOKE_GRPO_LORA_RANK", "0"))
        default_kl_beta = 0.001 if smoke_lora_rank > 0 else 0.0
        smoke_kl_beta = _get_float_env("FIREWORKS_SMOKE_GRPO_KL_BETA", default_kl_beta)
        smoke_skip_validations = _get_bool_env(
            "FIREWORKS_SMOKE_SKIP_VALIDATIONS",
            default=smoke_custom_image_tag is not None,
        )
        smoke_policy_job_id = os.environ.get("FIREWORKS_SMOKE_GRPO_POLICY_JOB_ID")
        smoke_policy_base_url = os.environ.get("FIREWORKS_SMOKE_GRPO_POLICY_BASE_URL")
        smoke_deployment_id = os.environ.get("FIREWORKS_SMOKE_GRPO_DEPLOYMENT_ID")
        smoke_max_seq_len = (
            int(os.environ.get("FIREWORKS_SMOKE_GRPO_MAX_SEQ_LEN", "4096"))
            if smoke_policy_job_id
            else None
        )
        smoke_training_shape_id = None if smoke_policy_job_id else smoke_training_shape

        config = Config(
            log_path=tempfile.mkdtemp(prefix="grpo_smoke_"),
            base_model=smoke_base_model,
            dataset=dataset_path,
            learning_rate=1e-4,
            kl_beta=smoke_kl_beta,
            lora_rank=smoke_lora_rank,
            completions_per_prompt=4,
            prompt_groups_per_step=1,
            max_completion_tokens=32,
            max_rows=2,
            epochs=1,
            max_seq_len=smoke_max_seq_len,
            policy_job_id=smoke_policy_job_id,
            policy_base_url=smoke_policy_base_url,
            infra=InfraConfig(
                training_shape_id=smoke_training_shape_id,
                skip_validations=smoke_skip_validations,
                custom_image_tag=smoke_custom_image_tag,
            ),
            deployment=DeployConfig(
                deployment_id=smoke_deployment_id,
                tokenizer_model=smoke_tokenizer_model,
            ),
            weight_sync=WeightSyncConfig(
                weight_sync_interval=1,
                first_checkpoint_type="base",
                weight_sync_before_training=True,
                weight_sync_timeout=600,
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
        if smoke_lora_rank > 0 and smoke_kl_beta != 0:
            assert metrics["reference_job_id"] is None, (
                "LoRA GRPO should reuse the policy trainer for reference logprobs "
                "instead of launching a separate reference trainer."
            )

        if not smoke_policy_job_id:
            import httpx
            import time

            time.sleep(3)
            policy_job_id = metrics.get("policy_job_id")
            if policy_job_id:
                try:
                    job = rlor_mgr.get(policy_job_id)
                    state = job.get("state", "")
                    assert state in ("JOB_STATE_DELETING", "JOB_STATE_DELETED"), (
                        f"ResourceCleanup failed: policy job {policy_job_id} still {state}"
                    )
                except httpx.HTTPStatusError as e:
                    assert e.response.status_code == 404
    finally:
        rl_mod.reward_fn = original_reward_fn
        rl_mod.should_accept = original_should_accept
        os.unlink(dataset_path)
