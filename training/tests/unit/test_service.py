"""Tests for cookbook mapping into the SDK-managed service boundary."""

from __future__ import annotations

import pytest

from training.utils import service
from training.utils.config import DeployConfig, TrainerConfig
from training.utils.service import build_service_client


def _trainer_config(**overrides) -> TrainerConfig:
    fields = dict(
        training_shape_id="ts-x",
        reference_training_shape_id="ref-ts-x",
        job_id="job-1",
        reference_job_id="ref-job-1",
        cleanup_reference_on_close=False,
        region="US_OHIO_1",
        node_count=2,
        accelerator_type="NVIDIA_B200",
        accelerator_count=8,
        custom_image_tag="0.0.0-dev",
        extra_args=["--foo"],
        replica_count=4,
        timeout_s=1800,
        pending_timeout_s=172800,
        inactivity_timeout="7200s",
        disable_inactivity_cleanup=True,
        purpose="PURPOSE_UNSPECIFIED",
        managed_by="parent-job",
        skip_validations=True,
    )
    fields.update(overrides)
    return TrainerConfig(**fields)


def _deployment_config(**overrides) -> DeployConfig:
    fields = dict(
        deployment_shape="ds-x",
        deployment_id="dep-1",
        deployment_extra_args=["--enable-moe-stats"],
        extra_values={"devShmSize": "200Gi"},
        deployment_timeout_s=5400,
        replica_count=3,
        disable_speculative_decoding=True,
    )
    fields.update(overrides)
    return DeployConfig(**fields)


def test_build_service_client_maps_cookbook_config_to_sdk_kwargs(monkeypatch):
    calls: list[dict] = []

    def fake_from_firetitan_config(**kwargs):
        calls.append(kwargs)
        return "service-sentinel"

    class FakeServiceClient:
        from_firetitan_config = staticmethod(fake_from_firetitan_config)

    monkeypatch.setattr(service, "FiretitanServiceClient", FakeServiceClient)

    result = build_service_client(
        api_key="k",
        base_url="https://api",
        additional_headers={"X-Fireworks-Test": "1"},
        base_model="accounts/acct/models/base",
        tokenizer_model="Qwen/Qwen3-1.7B",
        lora_rank=16,
        max_context_length=4096,
        learning_rate=1e-5,
        trainer=_trainer_config(),
        deployment=_deployment_config(),
        hotload_timeout_s=600,
        cleanup_trainer_on_close=True,
    )

    assert result == "service-sentinel"
    assert calls == [
        {
            "api_key": "k",
            "base_url": "https://api",
            "inference_url": None,
            "additional_headers": {"X-Fireworks-Test": "1"},
            "base_model": "accounts/acct/models/base",
            "tokenizer_model": "Qwen/Qwen3-1.7B",
            "lora_rank": 16,
            "training_shape_id": "ts-x",
            "reference_training_shape_id": "ref-ts-x",
            "trainer_job_id": "job-1",
            "reference_trainer_job_id": "ref-job-1",
            "cleanup_reference_trainer_on_close": False,
            "reference_required": False,
            "region": "US_OHIO_1",
            "max_context_length": 4096,
            "learning_rate": 1e-5,
            "gradient_accumulation_steps": None,
            "node_count": 2,
            "accelerator_type": "NVIDIA_B200",
            "accelerator_count": 8,
            "custom_image_tag": "0.0.0-dev",
            "extra_args": ["--foo"],
            "trainer_replica_count": 4,
            "trainer_timeout_s": 1800,
            "trainer_pending_timeout_s": 172800,
            "inactivity_timeout": "7200s",
            "disable_inactivity_cleanup": True,
            "purpose": "PURPOSE_UNSPECIFIED",
            "managed_by": "parent-job",
            "skip_validations": True,
            "cleanup_trainer_on_close": True,
            "cleanup_deployment_on_close": None,
            "create_deployment": True,
            "hotload_timeout_s": 600,
            "deployment_shape": "ds-x",
            "deployment_id": "dep-1",
            "deployment_extra_args": ["--enable-moe-stats"],
            "deployment_extra_values": {"devShmSize": "200Gi"},
            "deployment_timeout_s": 5400,
            "replica_count": 3,
            "disable_speculative_decoding": True,
        }
    ]


def test_build_service_client_forwards_lora_alpha(monkeypatch):
    calls: list[dict] = []

    def fake_from_firetitan_config(**kwargs):
        calls.append(kwargs)
        return "service-sentinel"

    class FakeServiceClient:
        from_firetitan_config = staticmethod(fake_from_firetitan_config)

    monkeypatch.setattr(service, "FiretitanServiceClient", FakeServiceClient)

    build_service_client(
        api_key="k",
        base_url="https://api",
        additional_headers=None,
        base_model="accounts/acct/models/base",
        tokenizer_model=None,
        lora_rank=32,
        lora_alpha=128,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-x"),
        deployment=DeployConfig(deployment_id="dep-1"),
    )

    assert calls[0]["lora_alpha"] == 128


def test_build_service_client_defaults_speculative_decoding_enabled(monkeypatch):
    calls: list[dict] = []

    def fake_from_firetitan_config(**kwargs):
        calls.append(kwargs)
        return "service-sentinel"

    class FakeServiceClient:
        from_firetitan_config = staticmethod(fake_from_firetitan_config)

    monkeypatch.setattr(service, "FiretitanServiceClient", FakeServiceClient)

    result = build_service_client(
        api_key="k",
        base_url="https://api",
        additional_headers=None,
        base_model="accounts/acct/models/base",
        tokenizer_model=None,
        lora_rank=None,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-x"),
        deployment=DeployConfig(deployment_id="dep-1"),
    )

    assert result == "service-sentinel"
    assert calls[0]["disable_speculative_decoding"] is False


def test_train_only_config_disables_deployment(monkeypatch):
    calls: list[dict] = []

    def fake_from_firetitan_config(**kwargs):
        calls.append(kwargs)
        return "service-sentinel"

    class FakeServiceClient:
        from_firetitan_config = staticmethod(fake_from_firetitan_config)

    monkeypatch.setattr(service, "FiretitanServiceClient", FakeServiceClient)

    result = build_service_client(
        api_key="k",
        base_url="https://api",
        additional_headers=None,
        base_model="accounts/acct/models/base",
        tokenizer_model=None,
        lora_rank=None,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-x"),
        deployment=None,
    )

    assert result == "service-sentinel"
    assert calls[0]["lora_rank"] == 0
    assert calls[0]["create_deployment"] is False
    assert calls[0]["replica_count"] == 1
    assert "deployment_shape" not in calls[0]


def test_build_service_client_forwards_inference_url(monkeypatch):
    calls: list[dict] = []

    def fake_from_firetitan_config(**kwargs):
        calls.append(kwargs)
        return "service-sentinel"

    class FakeServiceClient:
        from_firetitan_config = staticmethod(fake_from_firetitan_config)

    monkeypatch.setattr(service, "FiretitanServiceClient", FakeServiceClient)

    result = build_service_client(
        api_key="k",
        base_url="https://api",
        inference_url="https://gateway",
        additional_headers=None,
        base_model="accounts/acct/models/base",
        tokenizer_model=None,
        lora_rank=0,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-x"),
        deployment=_deployment_config(),
    )

    assert result == "service-sentinel"
    assert calls[0]["inference_url"] == "https://gateway"


def test_trainer_region_becomes_sdk_region():
    client = build_service_client(
        api_key="k",
        base_url="https://api",
        additional_headers=None,
        base_model="accounts/acct/models/base",
        tokenizer_model=None,
        lora_rank=0,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=_trainer_config(
            region="US_OHIO_1",
            accelerator_type=None,
            accelerator_count=None,
        ),
        deployment=_deployment_config(),
    )

    assert client._managed_config.region == "US_OHIO_1"


def test_deprecated_trainer_accelerator_fields_warn_and_are_ignored():
    with pytest.warns(DeprecationWarning):
        client = build_service_client(
            api_key="k",
            base_url="https://api",
            additional_headers=None,
            base_model="accounts/acct/models/base",
            tokenizer_model=None,
            lora_rank=0,
            max_context_length=None,
            learning_rate=1e-5,
            trainer=_trainer_config(),
            deployment=_deployment_config(),
        )

    assert client._managed_config.accelerator_type is None
    assert client._managed_config.accelerator_count is None
