from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from training.utils.config import DeployConfig, InfraConfig
from training.utils.infra import create_trainer_job, setup_deployment, _fetch_job_failure_reason


def test_deploy_config_omits_region_for_shape_backed_deployments():
    config = DeployConfig(
        deployment_id="dep-123",
        deployment_shape="accounts/fireworks/deploymentShapes/rft-kimi-k2p5-v2",
    ).to_deployment_config(
        "accounts/fireworks/models/kimi-k2p5",
        InfraConfig(region="US_VIRGINIA_1"),
    )

    assert config.deployment_shape == "accounts/fireworks/deploymentShapes/rft-kimi-k2p5-v2"
    assert config.region is None


def test_setup_deployment_omits_placement_for_shape_backed_create():
    captured = {}

    class FakeResponse:
        def __init__(self, payload=None):
            self._payload = payload or {"name": "accounts/acct/deployments/dep-123", "state": "CREATING"}

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeMgr:
        account_id = "acct"

        def get(self, _deployment_id):
            return None

        def _get(self, path, timeout):
            captured["shape_path"] = path
            captured["shape_timeout"] = timeout
            return FakeResponse(
                {
                    "deploymentShapeVersions": [
                        {
                            "snapshot": {},
                        }
                    ]
                }
            )

        def _post(self, path, json, timeout):
            captured["path"] = path
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResponse()

        def _parse_deployment_info(self, deployment_id, data):
            return SimpleNamespace(deployment_id=deployment_id, state=data["state"])

        def wait_for_ready(self, deployment_id, timeout_s):
            return SimpleNamespace(deployment_id=deployment_id, state="READY")

    info = setup_deployment(
        FakeMgr(),
        DeployConfig(
            deployment_id="dep-123",
            deployment_shape="accounts/fireworks/deploymentShapes/rft-kimi-k2p5-v2",
        ),
        "accounts/fireworks/models/kimi-k2p5",
        InfraConfig(region="US_VIRGINIA_1"),
    )

    assert info.state == "READY"
    assert captured["shape_timeout"] == 30
    assert captured["timeout"] == 60
    assert captured["json"]["deploymentShape"] == "accounts/fireworks/deploymentShapes/rft-kimi-k2p5-v2"
    assert "placement" not in captured["json"]


def test_setup_deployment_infers_ohio_for_b200_shape():
    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeMgr:
        account_id = "acct"

        def get(self, _deployment_id):
            return None

        def _get(self, path, timeout):
            captured["shape_path"] = path
            captured["shape_timeout"] = timeout
            return FakeResponse(
                {
                    "deploymentShapeVersions": [
                        {
                            "snapshot": {"acceleratorType": "NVIDIA_B200_180GB"},
                        }
                    ]
                }
            )

        def create_or_get(self, config):
            captured["config"] = config
            return SimpleNamespace(deployment_id=config.deployment_id, state="READY")

    info = setup_deployment(
        FakeMgr(),
        DeployConfig(
            deployment_id="dep-123",
            deployment_shape="accounts/fireworks/deploymentShapes/rft-kimi-k2p5-v2",
        ),
        "accounts/fireworks/models/kimi-k2p5",
        InfraConfig(region="US_VIRGINIA_1"),
    )

    assert info.state == "READY"
    assert captured["shape_timeout"] == 30
    assert captured["shape_path"].startswith(
        "/v1/accounts/fireworks/deploymentShapes/rft-kimi-k2p5-v2/versions?"
    )
    assert "pageSize=1" in captured["shape_path"]
    assert captured["config"].region == "US_OHIO_1"


def test_setup_deployment_infers_virginia_for_versioned_h200_shape():
    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeMgr:
        account_id = "acct"

        def get(self, _deployment_id):
            return None

        def _get(self, path, timeout):
            captured["shape_path"] = path
            captured["shape_timeout"] = timeout
            return FakeResponse(
                {
                    "snapshot": {"acceleratorType": "NVIDIA_H200_141GB"},
                }
            )

        def create_or_get(self, config):
            captured["config"] = config
            return SimpleNamespace(deployment_id=config.deployment_id, state="READY")

    info = setup_deployment(
        FakeMgr(),
        DeployConfig(
            deployment_id="dep-456",
            deployment_shape="accounts/fireworks/deploymentShapes/qwen-h200/versions/rv1",
        ),
        "accounts/fireworks/models/qwen3-4b",
        InfraConfig(),
    )

    assert info.state == "READY"
    assert captured["shape_path"] == "/v1/accounts/fireworks/deploymentShapes/qwen-h200/versions/rv1"
    assert captured["shape_timeout"] == 30
    assert captured["config"].region == "US_VIRGINIA_1"


def test_setup_deployment_retries_fireworks_create_with_description_workaround():
    captured = {}

    class FakeResponse:
        def __init__(self, payload=None, status_code=200, text=""):
            self._payload = payload or {"name": "accounts/fireworks/deployments/dep-789", "state": "CREATING"}
            self.status_code = status_code
            self.text = text
            self.request = httpx.Request("POST", "https://unit.test")
            self.headers = {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"Client error '{self.status_code}' for url '{self.request.url}'",
                    request=self.request,
                    response=self,
                )
            return None

        def json(self):
            return self._payload

    class FakeMgr:
        account_id = "fireworks"

        def get(self, _deployment_id):
            return None

        def _get(self, path, timeout):
            captured["shape_path"] = path
            return FakeResponse(
                {
                    "snapshot": {"acceleratorType": "NVIDIA_B200_180GB"},
                }
            )

        def create_or_get(self, _config):
            response = FakeResponse(
                {"error": {"message": "description is required for deployments under fireworks account"}},
                status_code=400,
                text="description is required for deployments under fireworks account",
            )
            raise httpx.HTTPStatusError(
                "Client error '400 Bad Request'",
                request=response.request,
                response=response,
            )

        def _post(self, path, json, timeout):
            captured["path"] = path
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResponse()

        def _parse_deployment_info(self, deployment_id, data):
            return SimpleNamespace(deployment_id=deployment_id, state=data["state"])

        def wait_for_ready(self, deployment_id, timeout_s):
            return SimpleNamespace(deployment_id=deployment_id, state="READY")

    info = setup_deployment(
        FakeMgr(),
        DeployConfig(
            deployment_id="dep-789",
            deployment_shape="accounts/fireworks/deploymentShapes/rft-qwen3-4b/versions/shape-1",
        ),
        "accounts/fireworks/models/qwen3-4b",
        InfraConfig(),
    )

    assert info.state == "READY"
    assert captured["timeout"] == 60
    assert captured["json"]["description"] == "Cookbook deployment for qwen3-4b"
    assert captured["json"]["placement"] == {"region": "US_OHIO_1"}
    assert captured["json"]["deploymentShape"] == (
        "accounts/fireworks/deploymentShapes/rft-qwen3-4b/versions/shape-1"
    )


# ---------------------------------------------------------------------------
# create_trainer_job: error messages expose failure reason
# ---------------------------------------------------------------------------


class _FakeRlorMgr:
    """Minimal fake for TrainerJobManager used by create_trainer_job tests."""

    account_id = "test-account"

    def __init__(
        self,
        *,
        create_result=None,
        wait_for_ready_result=None,
        wait_error=None,
        create_error=None,
        get_result=None,
    ):
        self._create_result = create_result
        self._wait_for_ready_result = wait_for_ready_result
        self._wait_error = wait_error
        self._create_error = create_error
        self._get_result = get_result or {}

    def create(self, config):
        if self._create_error:
            raise self._create_error
        return self._create_result or SimpleNamespace(
            job_id="job-123", job_name="accounts/test-account/rlorTrainerJobs/job-123"
        )

    def wait_for_ready(self, job_id, job_name=None, timeout_s=3600):
        if self._wait_error:
            raise self._wait_error
        return self._wait_for_ready_result or SimpleNamespace(
            job_id=job_id, job_name=job_name, base_url="http://localhost:8080"
        )

    def get(self, job_id):
        return self._get_result


class TestCreateTrainerJobErrorMessages:
    def test_runtime_error_includes_original_exception_message(self):
        mgr = _FakeRlorMgr(wait_error=RuntimeError("NCCL timeout after 300s"))
        with pytest.raises(RuntimeError, match="NCCL timeout after 300s"):
            create_trainer_job(
                mgr, base_model="model", infra=InfraConfig(), display_name="test"
            )

    def test_runtime_error_includes_server_reason_when_available(self):
        mgr = _FakeRlorMgr(
            wait_error=TimeoutError("timed out waiting for ready"),
            get_result={
                "state": "JOB_STATE_FAILED",
                "error": {"message": "Insufficient H100 capacity in US_VIRGINIA_1"},
            },
        )
        with pytest.raises(RuntimeError) as exc_info:
            create_trainer_job(
                mgr, base_model="model", infra=InfraConfig(), display_name="test"
            )
        msg = str(exc_info.value)
        assert "timed out waiting for ready" in msg
        assert "Insufficient H100 capacity" in msg
        assert "JOB_STATE_FAILED" in msg

    def test_runtime_error_includes_error_message_field(self):
        mgr = _FakeRlorMgr(
            wait_error=RuntimeError("wait failed"),
            get_result={
                "state": "JOB_STATE_FAILED",
                "errorMessage": "Pod scheduling timeout",
            },
        )
        with pytest.raises(RuntimeError, match="Pod scheduling timeout"):
            create_trainer_job(
                mgr, base_model="model", infra=InfraConfig(), display_name="test"
            )

    def test_runtime_error_includes_status_message_field(self):
        mgr = _FakeRlorMgr(
            wait_error=RuntimeError("wait failed"),
            get_result={
                "state": "JOB_STATE_FAILED",
                "statusMessage": "OOM killed",
            },
        )
        with pytest.raises(RuntimeError, match="OOM killed"):
            create_trainer_job(
                mgr, base_model="model", infra=InfraConfig(), display_name="test"
            )

    def test_create_error_surfaces_directly(self):
        mgr = _FakeRlorMgr(create_error=ValueError("invalid accelerator type"))
        with pytest.raises(RuntimeError, match="invalid accelerator type"):
            create_trainer_job(
                mgr, base_model="model", infra=InfraConfig(), display_name="test"
            )

    def test_error_still_raised_when_get_fails(self):
        class BadGetMgr(_FakeRlorMgr):
            def get(self, job_id):
                raise ConnectionError("server unreachable")

        mgr = BadGetMgr(wait_error=RuntimeError("connection lost"))
        with pytest.raises(RuntimeError, match="connection lost"):
            create_trainer_job(
                mgr, base_model="model", infra=InfraConfig(), display_name="test"
            )


# ---------------------------------------------------------------------------
# create_trainer_job: on_status callback
# ---------------------------------------------------------------------------


class TestCreateTrainerJobStatusCallback:
    def test_on_status_called_on_success(self):
        messages = []
        mgr = _FakeRlorMgr()
        create_trainer_job(
            mgr,
            base_model="model",
            infra=InfraConfig(),
            display_name="grpo-policy",
            on_status=messages.append,
        )
        assert any("creating" in m for m in messages)
        assert any("waiting" in m for m in messages)
        assert any("ready" in m for m in messages)

    def test_on_status_called_on_failure(self):
        messages = []
        mgr = _FakeRlorMgr(wait_error=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            create_trainer_job(
                mgr,
                base_model="model",
                infra=InfraConfig(),
                display_name="grpo-policy",
                on_status=messages.append,
            )
        assert any("creating" in m for m in messages)
        assert any("Failed" in m for m in messages)

    def test_on_status_for_precreated_job(self):
        messages = []
        mgr = _FakeRlorMgr()
        create_trainer_job(
            mgr,
            base_model="model",
            infra=InfraConfig(),
            display_name="test",
            job_id="pre-created-123",
            base_url_override="http://preexisting:8080",
            on_status=messages.append,
        )
        assert any("pre-created" in m for m in messages)

    def test_broken_callback_does_not_crash_creation(self):
        def bad_callback(msg):
            raise ValueError("callback broke")

        mgr = _FakeRlorMgr()
        endpoint = create_trainer_job(
            mgr,
            base_model="model",
            infra=InfraConfig(),
            display_name="test",
            on_status=bad_callback,
        )
        assert endpoint.job_id == "job-123"


# ---------------------------------------------------------------------------
# _fetch_job_failure_reason
# ---------------------------------------------------------------------------


class TestFetchJobFailureReason:
    def test_returns_state_and_error_message(self):
        mgr = _FakeRlorMgr(get_result={
            "state": "JOB_STATE_FAILED",
            "error": {"message": "GPU OOM"},
        })
        reason = _fetch_job_failure_reason(mgr, "job-1")
        assert "JOB_STATE_FAILED" in reason
        assert "GPU OOM" in reason

    def test_returns_none_when_no_details(self):
        mgr = _FakeRlorMgr(get_result={})
        reason = _fetch_job_failure_reason(mgr, "job-1")
        assert reason is None

    def test_returns_none_when_get_raises(self):
        class ErrorMgr:
            def get(self, job_id):
                raise ConnectionError("server down")

        reason = _fetch_job_failure_reason(ErrorMgr(), "job-1")
        assert reason is None

    def test_returns_error_message_field(self):
        mgr = _FakeRlorMgr(get_result={
            "state": "JOB_STATE_FAILED",
            "errorMessage": "NCCL init failed",
        })
        reason = _fetch_job_failure_reason(mgr, "job-1")
        assert "NCCL init failed" in reason

    def test_returns_status_message_field(self):
        mgr = _FakeRlorMgr(get_result={
            "statusMessage": "Pod evicted",
        })
        reason = _fetch_job_failure_reason(mgr, "job-1")
        assert "Pod evicted" in reason
