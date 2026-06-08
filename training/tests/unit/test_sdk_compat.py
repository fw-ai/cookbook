from __future__ import annotations

import training.utils.config as config_module
from fireworks.training.sdk.deployment import DeploymentConfig


def test_to_deployment_config_includes_extra_values():
    deploy_cfg = config_module.DeployConfig(
        deployment_id="dep-123",
        extra_values={"priorityClass": "deployment"},
    )

    deployment_config = deploy_cfg.to_deployment_config(
        "accounts/test/models/qwen3-4b",
        config_module.InfraConfig(),
    )

    assert isinstance(deployment_config, DeploymentConfig)
    assert deployment_config.deployment_id == "dep-123"
    assert deployment_config.region is None
    assert deployment_config.disable_speculative_decoding is False
    assert deployment_config.extra_values == {"priorityClass": "deployment"}


def test_to_deployment_config_omits_region():
    deploy_cfg = config_module.DeployConfig(deployment_id="dep-123")

    deployment_config = deploy_cfg.to_deployment_config(
        "accounts/test/models/qwen3-4b",
        config_module.InfraConfig(),
    )

    assert isinstance(deployment_config, DeploymentConfig)
    assert deployment_config.region is None


def test_to_deployment_config_uses_explicit_trainer_region():
    deploy_cfg = config_module.DeployConfig(deployment_id="dep-123")

    deployment_config = deploy_cfg.to_deployment_config(
        "accounts/test/models/qwen3-4b",
        config_module.InfraConfig(region="US_OHIO_1"),
    )

    assert isinstance(deployment_config, DeploymentConfig)
    assert deployment_config.region == "US_OHIO_1"


def test_to_deployment_config_sets_fixed_replica_count():
    deploy_cfg = config_module.DeployConfig(
        deployment_id="dep-123",
        replica_count=3,
    )

    deployment_config = deploy_cfg.to_deployment_config(
        "accounts/test/models/qwen3-4b",
        config_module.InfraConfig(),
    )

    assert isinstance(deployment_config, DeploymentConfig)
    assert deployment_config.min_replica_count == 3
    assert deployment_config.max_replica_count == 3


def test_to_deployment_config_drops_hotload_fields_when_disabled():
    deploy_cfg = config_module.DeployConfig(
        deployment_id="dep-123",
        enable_hot_load=False,
        hot_load_bucket_type="FW_HOSTED",
        hot_load_trainer_job="accounts/test/rlorTrainerJobs/job-123",
    )

    deployment_config = deploy_cfg.to_deployment_config(
        "accounts/test/models/qwen3-4b",
        config_module.InfraConfig(),
    )

    assert isinstance(deployment_config, DeploymentConfig)
    assert deployment_config.enable_hot_load is False
    assert deployment_config.hot_load_bucket_type is None
    assert deployment_config.hot_load_trainer_job is None
