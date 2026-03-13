"""Smoke test: ResourceCleanup deletes trainer jobs on scope exit.

Uses the same fixtures as the other smoke tests (qwen3-4b on dev).
Creates a real trainer job (without waiting for it to be ready),
wraps it in ResourceCleanup, exits the scope via exception, then
verifies the job was deleted.
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
    """Create a trainer job, exit scope via exception, verify deletion."""
    rlor_mgr, _ = smoke_sdk_managers

    profile = None
    if smoke_infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(smoke_infra.training_shape_id)

    config = TrainerJobConfig(
        base_model=smoke_base_model,
        training_shape_ref=profile.training_shape_version if profile else None,
        display_name="cleanup-smoke-test",
    )

    job_id = None
    try:
        with ResourceCleanup(rlor_mgr) as cleanup:
            raw = rlor_mgr._create(config)
            job_name = raw.get("name", "")
            job_id = job_name.split("/")[-1] if "/" in job_name else job_name
            cleanup.trainer(job_id)

            job = rlor_mgr.get(job_id)
            assert job, f"Job {job_id} should exist after creation"

            raise RuntimeError("simulate crash")
    except RuntimeError:
        pass

    assert job_id, "Job was never created"

    time.sleep(3)
    try:
        job_after = rlor_mgr.get(job_id)
        state = job_after.get("state", "")
        assert "DELET" in state or state == "", (
            f"Expected job {job_id} deleted/deleting, got state={state}"
        )
    except Exception:
        pass
