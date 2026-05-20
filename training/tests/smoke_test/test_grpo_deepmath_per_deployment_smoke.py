"""Smoke test for the ``WeightSyncScope.PER_DEPLOYMENT`` creation path on minimal qwen3-4b shapes.

Validates the server contract for the deployment-owned-bucket scope:
create the deployment first (so it owns the bucket URL), then create
the trainer with ``hot_load_deployment_id`` pointing at it. The
production cookbook recipe (``training/recipes/rl_loop.py``) defaults
to ``PER_TRAINER``; this test exercises the alternate ``PER_DEPLOYMENT``
scope end-to-end so it doesn't silently break.
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid

import httpx
import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

# qwen3-4b-minimum cold-start on a fresh dev pod (image pull + checkpoint
# download + replica become-ready) is ~10-15 min in observed dev runs;
# 25 min budget gives ~2x slack to absorb pull cache misses without making
# the smoke timeout a hair trigger.
_PROVISION_TIMEOUT_S = 1500

# Network errors that are legitimate during best-effort cleanup against
# a remote dev gateway. We catch these narrowly so genuine programming
# errors (AttributeError, TypeError, ...) still surface.
_CLEANUP_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    ConnectionError,
    TimeoutError,
)


def _delete_deployment_safe(deploy_mgr, deployment_id: str) -> None:
    try:
        deploy_mgr.delete(deployment_id)
    except _CLEANUP_NETWORK_ERRORS as e:
        logging.warning("deployment %s cleanup failed: %s", deployment_id, e)


def _delete_trainer_safe(rlor_mgr, job_id: str) -> None:
    try:
        rlor_mgr.delete(job_id)
    except _CLEANUP_NETWORK_ERRORS as e:
        logging.warning("trainer %s cleanup failed: %s", job_id, e)


@pytest.mark.e2e
@pytest.mark.timeout(3600)
def test_grpo_deepmath_per_deployment_smoke(
    smoke_sdk_managers,
    smoke_base_model,
    smoke_minimal_training_shape,
):
    from fireworks.training.sdk.deployment import DeploymentConfig
    from fireworks.training.sdk.trainer import TrainerJobConfig

    rlor_mgr, deploy_mgr = smoke_sdk_managers

    suffix = uuid.uuid4().hex[:8]
    deployment_id = f"smoke-perdeploy-{suffix}"

    # Resolve the policy training shape so we can pin the trainer to a
    # validated shape version (the trainer create will reject the manual
    # path without superuser).
    profile = rlor_mgr.resolve_training_profile(smoke_minimal_training_shape)
    deployment_shape = profile.deployment_shape
    if not deployment_shape:
        pytest.skip(
            f"training shape {smoke_minimal_training_shape} has no linked "
            "deployment_shape; PER_DEPLOYMENT smoke needs one"
        )

    trainer_job_id: str | None = None
    with contextlib.ExitStack() as stack:
        # Step 1: deployment first — it owns the bucket URL under
        # PER_DEPLOYMENT. Register cleanup immediately after
        # create_or_get returns so a failure in wait_for_ready does not
        # leak the deployment.
        deploy_cfg = DeploymentConfig(
            deployment_id=deployment_id,
            base_model=smoke_base_model,
            deployment_shape=deployment_shape,
            min_replica_count=1,
            max_replica_count=1,
        )
        deploy_mgr.create_or_get(deploy_cfg)
        stack.callback(_delete_deployment_safe, deploy_mgr, deployment_id)
        deploy_mgr.wait_for_ready(deployment_id, timeout_s=_PROVISION_TIMEOUT_S)

        # Step 2: trainer with hot_load_deployment_id pointing at the
        # already-existing deployment. This is the contract under test:
        # the gateway must accept it and the trainer must reach RUNNING.
        # Split create() and wait_for_ready() so cleanup is registered as
        # soon as the trainer ID exists, not after wait_for_ready returns.
        trainer_cfg = TrainerJobConfig(
            base_model=smoke_base_model,
            display_name=f"smoke-perdeploy-{suffix}",
            hot_load_deployment_id=deployment_id,
            training_shape_ref=profile.training_shape_version,
        )
        created = rlor_mgr.create(trainer_cfg)
        trainer_job_id = created.job_id
        assert trainer_job_id, "trainer create returned empty job_id"
        stack.callback(_delete_trainer_safe, rlor_mgr, trainer_job_id)
        rlor_mgr.wait_for_ready(
            trainer_job_id,
            job_name=created.job_name,
            timeout_s=_PROVISION_TIMEOUT_S,
        )
    # ExitStack fired here in LIFO order: trainer deleted first, then deployment.

    # Verify cleanup actually landed (best-effort: control plane may take a
    # few seconds to flip state, and the resource may already be 404).
    time.sleep(3)
    if trainer_job_id:
        try:
            job = rlor_mgr.get(trainer_job_id)
            state = job.get("state", "")
            assert state in ("JOB_STATE_DELETING", "JOB_STATE_DELETED"), (
                f"trainer {trainer_job_id} cleanup failed: state={state}"
            )
        except httpx.HTTPStatusError as e:
            assert e.response.status_code == 404, (
                f"unexpected error fetching trainer {trainer_job_id}: {e}"
            )
