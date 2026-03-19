from __future__ import annotations

import training.utils.config as config_module
from fireworks.training.sdk.deployment import DeploymentConfig


def test_to_deployment_config_includes_extra_values():
    deploy_cfg = config_module.DeployConfig(
        deployment_id="dep-123",
        deployment_region="US_OHIO_1",
        extra_values={"priorityClass": "deployment"},
    )

    deployment_config = deploy_cfg.to_deployment_config(
        "accounts/test/models/qwen3-4b",
        config_module.InfraConfig(),
    )

    assert isinstance(deployment_config, DeploymentConfig)
    assert deployment_config.deployment_id == "dep-123"
    assert deployment_config.region == "US_OHIO_1"
    assert deployment_config.replica_count == 4
    assert deployment_config.disable_speculative_decoding is True
    assert deployment_config.extra_values == {"priorityClass": "deployment"}


def test_to_deployment_config_does_not_infer_region_from_trainer():
    deploy_cfg = config_module.DeployConfig(deployment_id="dep-123")

    deployment_config = deploy_cfg.to_deployment_config(
        "accounts/test/models/qwen3-4b",
        config_module.InfraConfig(region="US_VIRGINIA_1"),
    )

    assert isinstance(deployment_config, DeploymentConfig)
    assert deployment_config.region is None


def test_to_deployment_config_propagates_replica_count():
    deploy_cfg = config_module.DeployConfig(
        deployment_id="dep-123",
        replica_count=6,
    )

    deployment_config = deploy_cfg.to_deployment_config(
        "accounts/test/models/qwen3-4b",
        config_module.InfraConfig(),
    )

    assert isinstance(deployment_config, DeploymentConfig)
    assert deployment_config.replica_count == 6
