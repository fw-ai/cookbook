"""Smoke test: ResourceCleanup deletes trainer jobs on scope exit.

Creates a real trainer job, registers it with ResourceCleanup, exits
scope via exception, then verifies the job no longer exists on the
control plane.
"""

from __future__ import annotations

import time

import pytest

from fireworks.training.sdk.trainer import TrainerJobConfig
from training.utils.infra import ResourceCleanup


@pytest.mark.e2e
@pytest.mark.timeout(120)
def test_resource_cleanup_deletes_trainer_on_exception(
    smoke_sdk_managers,
    smoke_infra,
    smoke_base_model,
):
    """Create a job, crash, verify ResourceCleanup actually deleted it."""
    rlor_mgr, _ = smoke_sdk_managers

    profile = None
    if smoke_infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(smoke_infra.training_shape_id)

    config = TrainerJobConfig(
        base_model=smoke_base_model,
        training_shape_ref=profile.training_shape_version if profile else None,
        display_name="cleanup-smoke-test",
    )

    raw = rlor_mgr._create(config)
    job_name = raw.get("name", "")
    job_id = job_name.split("/")[-1] if "/" in job_name else job_name
    assert job_id

    job_before = rlor_mgr.get(job_id)
    assert job_before.get("state") != "", f"Job {job_id} should exist"

    try:
        with ResourceCleanup(rlor_mgr) as cleanup:
            cleanup.trainer(job_id)
            raise RuntimeError("simulate crash")
    except RuntimeError:
        pass

    time.sleep(3)
    import httpx
    try:
        job_after = rlor_mgr.get(job_id)
        state = job_after.get("state", "")
        assert state in ("JOB_STATE_DELETING", "JOB_STATE_DELETED"), (
            f"ResourceCleanup failed to delete job {job_id}: state={state}"
        )
    except httpx.HTTPStatusError as e:
        assert e.response.status_code == 404, (
            f"Expected 404 (deleted) but got {e.response.status_code}"
        )
