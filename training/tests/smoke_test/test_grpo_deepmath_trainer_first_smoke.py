"""GRPO deepmath smoke test on minimal 1xGPU qwen3-4b shapes (trainer-first).

40 rows, 4 completions/prompt, 2 prompt groups/step, kl_beta>0. Asserts the
trainer-first invariant (deployment created with hot_load_trainer_job, not
hot_load_deployment_id).
"""

from __future__ import annotations

import logging
import os
import tempfile
import time

import httpx
import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

_DEEPMATH_DATASET = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "examples",
        "rl",
        "deepmath",
        "dataset.jsonl",
    )
)


@pytest.mark.e2e
@pytest.mark.timeout(3600)
def test_grpo_deepmath_trainer_first(
    smoke_sdk_managers,
    smoke_base_model,
    smoke_tokenizer_model,
    smoke_minimal_grpo_infra,
    monkeypatch,
):
    # Late imports: module collection must not require FIREWORKS_API_KEY.
    from training.utils import DeployConfig, WeightSyncConfig, WandBConfig
    from training.recipes.rl_loop import Config, main
    import training.recipes.rl_loop as rl_mod
    from training.examples.rl.deepmath.train_deepmath import deepmath_reward

    if not os.path.exists(_DEEPMATH_DATASET):
        pytest.skip(f"deepmath dataset not found at {_DEEPMATH_DATASET}")

    rlor_mgr, deploy_mgr = smoke_sdk_managers

    # monkeypatch (not direct rl_mod.x = ...) so the override is auto-undone
    # at test exit; otherwise running multiple smoke tests in one pytest
    # process leaks the patched globals into later tests.
    monkeypatch.setattr(rl_mod, "reward_fn", deepmath_reward)
    # Disable zero-variance filter so a 40-row run still produces steps.
    monkeypatch.setattr(rl_mod, "should_accept", lambda _: True)

    config = Config(
        log_path=tempfile.mkdtemp(prefix="grpo_deepmath_smoke_"),
        base_model=smoke_base_model,
        dataset=_DEEPMATH_DATASET,
        learning_rate=1e-5,
        kl_beta=0.001,
        completions_per_prompt=4,
        prompt_groups_per_step=2,
        max_completion_tokens=256,
        max_rows=40,
        epochs=1,
        infra=smoke_minimal_grpo_infra,
        deployment=DeployConfig(
            tokenizer_model=smoke_tokenizer_model,
            sample_timeout=600,
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
        cancel_on_exit=True,
    )

    # No seed pinning: this is an API contract smoke (steps complete, cleanup
    # runs, trainer-first invariant holds), not a numerical fidelity check.
    assert isinstance(metrics, dict)
    assert metrics.get("steps", 0) >= 1, f"no training steps: {metrics}"
    assert metrics.get("reference_job_id"), f"reference trainer not created: {metrics}"
    assert config.deployment.hot_load_trainer_job, (
        "trainer-first regression: deployment was not attached via hot_load_trainer_job"
    )

    time.sleep(3)
    for key in ("policy_job_id", "reference_job_id"):
        job_id = metrics.get(key)
        if not job_id:
            continue
        try:
            job = rlor_mgr.get(job_id)
            state = job.get("state", "")
            assert state in ("JOB_STATE_DELETING", "JOB_STATE_DELETED"), (
                f"ResourceCleanup failed: {key} {job_id} still {state}"
            )
        except httpx.HTTPStatusError as e:
            assert e.response.status_code == 404
