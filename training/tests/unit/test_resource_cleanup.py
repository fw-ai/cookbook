"""Tests for ResourceCleanup context manager."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from training.utils.config import DeployConfig, InfraConfig
from training.utils.infra import ResourceCleanup, create_trainer_job, setup_deployment


class TestResourceCleanup:
    def test_cleans_registered_resources_in_reverse_order(self):
        rlor_mgr = MagicMock()
        deploy_mgr = MagicMock()

        with ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup:
            cleanup.trainer("job-1")
            cleanup.trainer("job-2")
            cleanup.deployment("dep-1")
            cleanup.deployment("dep-2", action="scale_to_zero")

        assert rlor_mgr.delete.call_args_list == [
            (("job-2",),),
            (("job-1",),),
        ]
        deploy_mgr.scale_to_zero.assert_called_once_with("dep-2")
        deploy_mgr.delete.assert_called_once_with("dep-1")

    def test_cleans_on_exception_and_skips_unregistered(self):
        rlor_mgr = MagicMock()

        try:
            with ResourceCleanup(rlor_mgr) as cleanup:
                cleanup.trainer("created-job")
                raise RuntimeError("training crashed")
        except RuntimeError:
            pass

        rlor_mgr.delete.assert_called_once_with("created-job")

    def test_skips_duplicate_registrations(self):
        rlor_mgr = MagicMock()
        deploy_mgr = MagicMock()

        with ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup:
            cleanup.trainer("job-1")
            cleanup.trainer("job-1")
            cleanup.deployment("dep-1")
            cleanup.deployment("dep-1")

        rlor_mgr.delete.assert_called_once_with("job-1")
        deploy_mgr.delete.assert_called_once_with("dep-1")


def test_create_trainer_job_registers_before_wait():
    events = {"deleted_jobs": []}

    class FakeRlorMgr:
        account_id = "acct"

        def create(self, config):
            events["config"] = config
            return SimpleNamespace(
                job_id="job-created",
                job_name="accounts/acct/rlorTrainerJobs/job-created",
            )

        def wait_for_ready(self, job_id, job_name=None, poll_interval_s=5.0, timeout_s=0):
            events["wait_call"] = {
                "job_id": job_id,
                "job_name": job_name,
                "poll_interval_s": poll_interval_s,
                "timeout_s": timeout_s,
            }
            raise TimeoutError("trainer wait failed")

        def delete(self, job_id):
            events["deleted_jobs"].append(job_id)

    mgr = FakeRlorMgr()
    with pytest.raises(TimeoutError, match="trainer wait failed"):
        with ResourceCleanup(mgr) as cleanup:
            create_trainer_job(
                mgr,
                base_model="accounts/test/models/qwen3-4b",
                infra=InfraConfig(),
                cleanup=cleanup,
            )

    assert events["wait_call"]["job_id"] == "job-created"
    assert events["deleted_jobs"] == ["job-created"]


def test_setup_deployment_registers_before_wait():
    events = {"deleted_deployments": []}

    class FakeDeployMgr:
        def get(self, deployment_id):
            events["get_id"] = deployment_id
            return None

        def create_or_get(self, config):
            events["created_id"] = config.deployment_id
            return SimpleNamespace(state="CREATING")

        def wait_for_ready(self, deployment_id, timeout_s=0, poll_interval_s=15):
            events["wait_call"] = {
                "deployment_id": deployment_id,
                "timeout_s": timeout_s,
                "poll_interval_s": poll_interval_s,
            }
            raise TimeoutError("deployment wait failed")

        def delete(self, deployment_id):
            events["deleted_deployments"].append(deployment_id)

    deploy_mgr = FakeDeployMgr()
    with pytest.raises(TimeoutError, match="deployment wait failed"):
        with ResourceCleanup(MagicMock(), deploy_mgr) as cleanup:
            setup_deployment(
                deploy_mgr,
                DeployConfig(deployment_id="dep-created"),
                "accounts/test/models/qwen3-4b",
                InfraConfig(),
                cleanup=cleanup,
            )

    assert events["wait_call"]["deployment_id"] == "dep-created"
    assert events["deleted_deployments"] == ["dep-created"]
