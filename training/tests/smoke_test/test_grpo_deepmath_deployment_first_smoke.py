"""DEPRECATED deployment-first creation smoke test on minimal qwen3-4b shapes.

Validates the legacy server contract: create the deployment first, then
create the trainer with hot_load_deployment_id pointing at it. The
production cookbook recipe (training/recipes/rl_loop.py) is trainer-first;
this test exists only to keep the deprecated path from silently breaking.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import httpx
import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)


@pytest.mark.e2e
@pytest.mark.timeout(3600)
def test_grpo_deepmath_deployment_first_smoke(
    smoke_sdk_managers,
    smoke_base_model,
    smoke_minimal_training_shape,
):
    from fireworks.training.sdk.deployment import DeploymentConfig
    from fireworks.training.sdk.trainer import TrainerJobConfig

    rlor_mgr, deploy_mgr = smoke_sdk_managers

    suffix = uuid.uuid4().hex[:8]
    deployment_id = f"smoke-depfirst-{suffix}"

    # Resolve the policy training shape so we can pin the trainer to a
    # validated shape version (the trainer create will reject the manual
    # path without superuser).
    profile = rlor_mgr.resolve_training_profile(smoke_minimal_training_shape)
    deployment_shape = profile.deployment_shape
    if not deployment_shape:
        pytest.skip(
            f"training shape {smoke_minimal_training_shape} has no linked "
            "deployment_shape; deployment-first smoke needs one"
        )

    deployment_created = False
    trainer_job_id: str | None = None
    try:
        # Step 1: deployment first (the deprecated order).
        deploy_cfg = DeploymentConfig(
            deployment_id=deployment_id,
            base_model=smoke_base_model,
            deployment_shape=deployment_shape,
            min_replica_count=1,
            max_replica_count=1,
        )
        deploy_mgr.create_or_get(deploy_cfg)
        deployment_created = True
        deploy_mgr.wait_for_ready(deployment_id, timeout_s=1500)

        # Step 2: trainer with hot_load_deployment_id pointing at the
        # already-existing deployment. This is the contract under test:
        # the gateway must accept it and the trainer must reach RUNNING.
        trainer_cfg = TrainerJobConfig(
            base_model=smoke_base_model,
            display_name=f"smoke-depfirst-{suffix}",
            hot_load_deployment_id=deployment_id,
            training_shape_ref=profile.training_shape_version,
        )
        endpoint = rlor_mgr.create_and_wait(trainer_cfg, timeout_s=1500)
        trainer_job_id = endpoint.job_id
        assert endpoint.job_id, "trainer create returned empty job_id"
    finally:
        if trainer_job_id:
            try:
                rlor_mgr.delete(trainer_job_id)
            except Exception as e:
                logging.warning("trainer cleanup failed: %s", e)
        if deployment_created:
            try:
                deploy_mgr.delete(deployment_id)
            except Exception as e:
                logging.warning("deployment cleanup failed: %s", e)

    time.sleep(3)
    if trainer_job_id:
        try:
            job = rlor_mgr.get(trainer_job_id)
            state = job.get("state", "")
            assert state in ("JOB_STATE_DELETING", "JOB_STATE_DELETED")
        except httpx.HTTPStatusError as e:
            assert e.response.status_code == 404
