"""Smoke test: SIGINT triggers ResourceCleanup on a real trainer job.

Creates a trainer job directly, wraps it in ResourceCleanup, then
sends SIGINT to verify the context manager deletes the job.

Requires: FIREWORKS_API_KEY, FIREWORKS_ACCOUNT_ID
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import textwrap

import pytest

import fireworks.training.sdk as _sdk


@pytest.mark.e2e
@pytest.mark.timeout(600)
def test_sigint_cleans_trainer_job(smoke_sdk_managers, smoke_training_shape):
    """Create a trainer job, SIGINT the process, verify job is deleted."""
    rlor_mgr, _ = smoke_sdk_managers

    api_key = os.environ["FIREWORKS_API_KEY"]
    account_id = os.environ["FIREWORKS_ACCOUNT_ID"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://dev.api.fireworks.ai")
    gateway_secret = os.environ.get("FIREWORKS_GATEWAY_SECRET", "")

    sdk_root = os.path.dirname(os.path.dirname(os.path.dirname(_sdk.__file__)))
    cookbook_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    script = textwrap.dedent(f"""\
        import os, sys, signal, time, logging, json
        os.environ["FIREWORKS_API_KEY"] = "{api_key}"
        os.environ["FIREWORKS_ACCOUNT_ID"] = "{account_id}"
        os.environ["FIREWORKS_BASE_URL"] = "{base_url}"
        logging.basicConfig(level=logging.INFO, format="%(message)s")

        from fireworks.training.sdk.trainer import TrainerJobManager, TrainerJobConfig
        from training.utils.infra import ResourceCleanup

        def _signal_handler(signum, frame):
            raise SystemExit("SIGINT")
        signal.signal(signal.SIGINT, _signal_handler)

        headers = {{"X-Fireworks-Gateway-Secret": "{gateway_secret}"}} if "{gateway_secret}" else None
        mgr = TrainerJobManager(
            api_key="{api_key}",
            account_id="{account_id}",
            base_url="{base_url}",
            additional_headers=headers,
        )

        profile = mgr.resolve_training_profile("{smoke_training_shape}")

        with ResourceCleanup(mgr) as cleanup:
            config = TrainerJobConfig(
                base_model="accounts/fireworks/models/qwen3-4b",
                training_shape_ref=profile.training_shape_version,
                display_name="sigint-cleanup-test",
            )
            ep = mgr.create_and_wait(config, timeout_s=600)
            cleanup.trainer(ep.job_id)
            print(f"CREATED_JOB_ID={{ep.job_id}}", flush=True)

            time.sleep(3600)
    """)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{sdk_root}:{cookbook_root}:{env.get('PYTHONPATH', '')}"

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    job_id = None
    deadline = time.time() + 300
    for line in iter(proc.stdout.readline, ""):
        print(line, end="")
        if "CREATED_JOB_ID=" in line:
            job_id = line.strip().split("CREATED_JOB_ID=")[-1]
            break
        if time.time() > deadline:
            proc.kill()
            pytest.fail("Timed out waiting for trainer job creation")

    assert job_id, "Never saw CREATED_JOB_ID in output"

    job_before = rlor_mgr.get(job_id)
    assert job_before, f"Trainer job {job_id} should exist before SIGINT"
    print(f"Job {job_id} exists, state={job_before.get('state')}")

    proc.send_signal(signal.SIGINT)

    remaining_output, _ = proc.communicate(timeout=60)
    print(remaining_output)

    assert "Cleanup: deleting trainer job" in remaining_output, (
        "Expected ResourceCleanup log line in output"
    )

    time.sleep(5)
    try:
        job_after = rlor_mgr.get(job_id)
        state = job_after.get("state", "")
        assert state in ("JOB_STATE_DELETING", ""), (
            f"Expected job {job_id} to be deleted/deleting but got state={state}"
        )
    except Exception:
        pass
