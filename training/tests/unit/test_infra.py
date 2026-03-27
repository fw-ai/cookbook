from __future__ import annotations

from types import SimpleNamespace

from training.utils.config import DeployConfig, InfraConfig
from training.utils.infra import setup_deployment


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
        def __init__(self, payload=None):
            self._payload = payload or {"name": "accounts/acct/deployments/dep-123", "state": "READY"}

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

        def _post(self, path, json, timeout):
            captured["json"] = json
            return FakeResponse()

        def _parse_deployment_info(self, deployment_id, data):
            return SimpleNamespace(deployment_id=deployment_id, state=data["state"])

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
    assert captured["json"]["placement"] == {"region": "US_OHIO_1"}


def test_setup_deployment_infers_virginia_for_versioned_h200_shape():
    captured = {}

    class FakeResponse:
        def __init__(self, payload=None):
            self._payload = payload or {"name": "accounts/acct/deployments/dep-456", "state": "READY"}

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

        def _post(self, path, json, timeout):
            captured["json"] = json
            return FakeResponse()

        def _parse_deployment_info(self, deployment_id, data):
            return SimpleNamespace(deployment_id=deployment_id, state=data["state"])

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
    assert captured["json"]["placement"] == {"region": "US_VIRGINIA_1"}


def test_setup_deployment_injects_purpose_annotation():
    captured = {}

    class FakeResponse:
        def __init__(self, payload=None):
            self._payload = payload or {"name": "accounts/acct/deployments/dep-p", "state": "READY"}
        def raise_for_status(self): return None
        def json(self): return self._payload

    class FakeMgr:
        account_id = "acct"
        def get(self, _): return None
        def _get(self, path, timeout):
            return FakeResponse({"deploymentShapeVersions": [{"snapshot": {}}]})
        def _post(self, path, json, timeout):
            captured["json"] = json
            return FakeResponse()
        def _parse_deployment_info(self, did, data):
            return SimpleNamespace(deployment_id=did, state=data["state"])

    setup_deployment(
        FakeMgr(),
        DeployConfig(
            deployment_id="dep-p",
            deployment_shape="accounts/fireworks/deploymentShapes/test-shape",
        ),
        "accounts/fireworks/models/test-model",
        InfraConfig(purpose="PURPOSE_PILOT"),
    )

    assert captured["json"]["annotations"] == {"internal/purpose": "pilot"}
