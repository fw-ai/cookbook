from __future__ import annotations

from types import SimpleNamespace

import training.utils.infra as infra_module


def test_reuse_or_resume_job_reuses_existing_job():
    expected_endpoint = SimpleNamespace(job_id="job-123", base_url="https://trainer.unit.test")
    events: dict[str, object] = {}

    class FakeTrainerJobManager:
        account_id = "acct"

        def get(self, job_id):
            events["get_job_id"] = job_id
            return {"state": "JOB_STATE_RUNNING"}

        def wait_for_existing(self, job_id):
            events["wait_job_id"] = job_id
            return expected_endpoint

    endpoint = infra_module._reuse_or_resume_job(FakeTrainerJobManager(), "job-123")

    assert endpoint is expected_endpoint
    assert events == {
        "get_job_id": "job-123",
        "wait_job_id": "job-123",
    }


def test_reuse_or_resume_job_falls_back_to_gateway_when_lookup_fails(monkeypatch):
    events: dict[str, object] = {
        "healthz_urls": [],
        "sleeps": [],
    }

    class FakeTrainerJobManager:
        account_id = "acct"

        def get(self, _job_id):
            raise RuntimeError("403 forbidden")

        def _get_trainer_gateway_url(self, job_id):
            return f"https://gateway.unit.test/training/v1/rlorTrainerJobs/acct/{job_id}"

        def _check_healthz(self, base_url):
            events["healthz_urls"].append(base_url)
            return len(events["healthz_urls"]) >= 2

    monkeypatch.setattr(infra_module.time, "sleep", lambda seconds: events["sleeps"].append(seconds))

    endpoint = infra_module._reuse_or_resume_job(FakeTrainerJobManager(), "job-123")

    assert endpoint.job_id == "job-123"
    assert endpoint.job_name == "accounts/acct/rlorTrainerJobs/job-123"
    assert endpoint.base_url == "https://gateway.unit.test/training/v1/rlorTrainerJobs/acct/job-123"
    assert events["healthz_urls"] == [
        "https://gateway.unit.test/training/v1/rlorTrainerJobs/acct/job-123",
        "https://gateway.unit.test/training/v1/rlorTrainerJobs/acct/job-123",
    ]
    assert events["sleeps"] == [5]
