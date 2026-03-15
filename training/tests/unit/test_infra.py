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
        def raise_for_status(self):
            return None

        def json(self):
            return {"name": "accounts/acct/deployments/dep-123", "state": "CREATING"}

    class FakeMgr:
        account_id = "acct"

        def get(self, _deployment_id):
            return None

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
    assert captured["timeout"] == 60
    assert captured["json"]["deploymentShape"] == "accounts/fireworks/deploymentShapes/rft-kimi-k2p5-v2"
    assert "placement" not in captured["json"]
