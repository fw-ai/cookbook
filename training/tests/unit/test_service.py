"""Tests for cookbook mapping into the SDK-managed service boundary."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from unittest.mock import Mock

import pytest

from types import SimpleNamespace

from training.utils import service
from training.utils.config import DeployConfig, TrainerConfig, WeightSyncScope
from training.utils.service import build_service_client, resolve_router_replay_enabled


@pytest.mark.parametrize("capability", [None, False, True])
@pytest.mark.parametrize("configured_rdma", [False, True])
def test_weight_sync_uses_negotiated_capability(capability, configured_rdma, caplog):
    from unittest.mock import Mock

    policy = SimpleNamespace(
        weight_sync=Mock(return_value=SimpleNamespace(optimizer_version=0)),
        save_weights_for_sampler=Mock(return_value=SimpleNamespace(path="saved")),
    )
    if capability is not None:
        policy.supports_rdma_weight_sync = capability
    managed_service = SimpleNamespace(hotload_sampler_snapshot=Mock())
    deployment = DeployConfig(weight_sync_transport="RDMA" if configured_rdma else None)
    sync = service.make_weight_sync(policy, managed_service, deployment)
    with caplog.at_level("INFO", logger=service.__name__):
        sync("step-1", checkpoint_type="base")
    assert f"Weight sync completed: {'RDMA' if capability is True else 'FILE'}" in caplog.text
    if capability is True:
        policy.weight_sync.assert_called_once_with()
        policy.save_weights_for_sampler.assert_not_called()
        managed_service.hotload_sampler_snapshot.assert_not_called()
    else:
        policy.save_weights_for_sampler.assert_called_once_with("step-1", checkpoint_type="base")
        managed_service.hotload_sampler_snapshot.assert_called_once_with("saved")
        policy.weight_sync.assert_not_called()


def test_weight_sync_logs_file_when_sdk_falls_back(caplog):
    from unittest.mock import Mock

    result = SimpleNamespace(optimizer_version=None)
    policy = SimpleNamespace(supports_rdma_weight_sync=True, weight_sync=Mock(return_value=result))
    sync = service.make_weight_sync(policy, None, DeployConfig())
    with caplog.at_level("INFO", logger=service.__name__):
        assert sync("step-1") is result
    assert "Weight sync completed: FILE" in caplog.text


def test_resumed_full_parameter_base_preserves_published_delta(tmp_path: Path) -> None:
    def save(name: str, *, checkpoint_type: str = "delta") -> SimpleNamespace:
        path = tmp_path / name
        path.write_text(checkpoint_type)
        return SimpleNamespace(path=str(path))

    cached_formats: dict[str, str] = {}

    def hotload(path: str) -> None:
        current = Path(path).read_text()
        assert cached_formats.setdefault(path, current) == current

    policy = SimpleNamespace(
        supports_rdma_weight_sync=False, lora_rank=0, save_weights_for_sampler=save
    )
    publish = service.make_weight_sync(
        policy,
        SimpleNamespace(hotload_sampler_snapshot=hotload),
        DeployConfig(),
    )
    publish("step-19")
    publish("step-19", checkpoint_type="base")

    assert (tmp_path / "step-19").read_text() == "delta"
    assert (tmp_path / "step-19-base").read_text() == "base"
    assert len(cached_formats) == 2


@pytest.mark.parametrize("lora_rank", [None, 32])
def test_initial_base_preserves_legacy_and_lora_names(lora_rank: int | None) -> None:
    policy = SimpleNamespace(
        lora_rank=lora_rank,
        supports_rdma_weight_sync=False,
        save_weights_for_sampler=Mock(return_value=SimpleNamespace(path="saved")),
    )
    publish = service.make_weight_sync(
        policy,
        SimpleNamespace(hotload_sampler_snapshot=Mock()),
        DeployConfig(),
    )
    publish("step-19", checkpoint_type="base")
    policy.save_weights_for_sampler.assert_called_once_with(
        "step-19", checkpoint_type="base"
    )


def test_full_parameter_base_keeps_sdk_rdma_publication() -> None:
    result = SimpleNamespace(optimizer_version=19)
    policy = SimpleNamespace(
        supports_rdma_weight_sync=True, lora_rank=0, weight_sync=Mock(return_value=result)
    )
    publish = service.make_weight_sync(policy, None, DeployConfig())
    assert publish("step-19", checkpoint_type="base") is result
    policy.weight_sync.assert_called_once_with()


def _trainer_config(**overrides) -> TrainerConfig:
    fields = dict(
        training_shape_id="ts-x",
        reference_training_shape_id="ref-ts-x",
        job_id="job-1",
        reference_job_id="ref-job-1",
        cleanup_reference_on_close=False,
        use_reservation=True,
        region="US_OHIO_1",
        node_count=2,
        custom_image_tag="0.0.0-dev",
        extra_args=["--test-flag"],
        replica_count=4,
        timeout_s=1800,
        pending_timeout_s=172800,
        inactivity_timeout="7200s",
        disable_inactivity_cleanup=True,
        purpose="PURPOSE_UNSPECIFIED",
        preemptible=True,
        managed_by="parent-job",
        skip_validations=True,
    )
    fields.update(overrides)
    return TrainerConfig(**fields)


@pytest.mark.parametrize("max_lora_rank", [0, 8])
@pytest.mark.parametrize("transport", [None, "RDMA"])
def test_weight_sync_config_does_not_inject_admin_overrides(max_lora_rank, transport):
    config = dict(
        base_model="accounts/acct/models/base",
        tokenizer_model=None,
        max_lora_rank=max_lora_rank,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-policy"),
        deployment=DeployConfig(deployment_shape="ds-policy", weight_sync_transport=transport),
    )
    if transport is not None and "weight_sync_transport" not in inspect.signature(
        service.FiretitanProvisioningConfig
    ).parameters:
        # The minimum supported SDK predates the explicit RDMA hint. Automatic
        # mode must still work, while an unsupported explicit hint stays loud.
        with pytest.raises(RuntimeError, match="has no 'weight_sync_transport' field"):
            service._firetitan_service_kwargs(**config)
        return
    kwargs = service._firetitan_service_kwargs(**config)

    for key in ("extra_args", "extra_values", "deployment_extra_args", "deployment_extra_values"):
        assert kwargs.get(key) is None


def _deployment_config(**overrides) -> DeployConfig:
    fields = dict(
        deployment_shape="ds-x",
        deployment_id="dep-1",
        deployment_extra_args=["--test-flag"],
        extra_values={"testValue": "test-value"},
        deployment_timeout_s=5400,
        replica_count=3,
        disable_speculative_decoding=True,
        hot_load_transition_type="SYNC",
    )
    fields.update(overrides)
    return DeployConfig(**fields)


def test_router_replay_skips_model_lookup_when_not_requested(monkeypatch):
    monkeypatch.setattr(
        service,
        "FireworksClient",
        lambda **_kwargs: pytest.fail("model lookup should not run"),
    )

    assert (
        resolve_router_replay_enabled(
            requested=False,
            api_key="k",
            base_url="https://api",
            additional_headers=None,
            base_model="accounts/acct/models/base",
        )
        is False
    )


@pytest.mark.parametrize("is_moe", [False, True])
def test_router_replay_follows_model_architecture(monkeypatch, is_moe):
    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def model_is_moe(self, model):
            assert model == "accounts/acct/models/base"
            return is_moe

    monkeypatch.setattr(service, "FireworksClient", FakeClient)

    assert (
        resolve_router_replay_enabled(
            requested=True,
            api_key="k",
            base_url="https://api",
            additional_headers=None,
            base_model="accounts/acct/models/base",
        )
        is is_moe
    )


@pytest.mark.parametrize("advertised", [False, True])
def test_router_replay_uses_trainer_capability_without_model_lookup(
    monkeypatch, advertised
):
    # Trainer-authoritative: no GET model at all, so a private early-access
    # base model (GET model -> 403) no longer breaks the R3 decision.
    monkeypatch.setattr(
        service,
        "FireworksClient",
        lambda **_kwargs: pytest.fail("model lookup should not run"),
    )
    training_client = SimpleNamespace(supports_router_replay=advertised)

    assert (
        resolve_router_replay_enabled(
            requested=True,
            api_key="k",
            base_url="https://api",
            additional_headers=None,
            base_model="accounts/fireworks/models/private",
            training_client=training_client,
        )
        is advertised
    )


def test_router_replay_not_requested_ignores_trainer_capability(monkeypatch):
    monkeypatch.setattr(
        service,
        "FireworksClient",
        lambda **_kwargs: pytest.fail("model lookup should not run"),
    )

    assert (
        resolve_router_replay_enabled(
            requested=False,
            api_key="k",
            base_url="https://api",
            additional_headers=None,
            base_model="accounts/acct/models/base",
            training_client=SimpleNamespace(supports_router_replay=True),
        )
        is False
    )


@pytest.mark.parametrize(
    "training_client",
    [
        None,  # recipe did not pass a client (pre-change callers)
        SimpleNamespace(),  # older SDK: attribute absent
        SimpleNamespace(supports_router_replay=None),  # older trainer: unknown
    ],
)
def test_router_replay_falls_back_to_model_probe_when_capability_unknown(
    monkeypatch, training_client
):
    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def model_is_moe(self, model):
            assert model == "accounts/acct/models/base"
            return True

    monkeypatch.setattr(service, "FireworksClient", FakeClient)

    assert (
        resolve_router_replay_enabled(
            requested=True,
            api_key="k",
            base_url="https://api",
            additional_headers=None,
            base_model="accounts/acct/models/base",
            training_client=training_client,
        )
        is True
    )


# The last-resort path needs the SDK's typed ModelDetailsUnavailableError; on
# an older SDK the placeholder is never raised and a 403 propagates as before.
_TYPED_PROBE_ERROR = hasattr(
    __import__("fireworks.training.sdk", fromlist=["x"]),
    "ModelDetailsUnavailableError",
)
requires_typed_probe_error = pytest.mark.skipif(
    not _TYPED_PROBE_ERROR,
    reason="SDK predates ModelDetailsUnavailableError (last resort disabled)",
)


def _probe_raising(exc):
    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def model_is_moe(self, model):
            raise exc

    return FakeClient


@requires_typed_probe_error
@pytest.mark.parametrize("status", [403, 404])
def test_router_replay_last_resort_honors_user_flag_when_model_not_visible(
    monkeypatch, caplog, status
):
    # Last resort: trainer did not report and the model record is not visible
    # to this caller (private early-access base model). Honor the user's
    # router_replay=True, with a warning, instead of crashing the recipe.
    monkeypatch.setattr(
        service,
        "FireworksClient",
        _probe_raising(
            service.ModelDetailsUnavailableError(
                f"Failed to fetch model details (HTTP {status})", status_code=status
            )
        ),
    )

    with caplog.at_level("WARNING"):
        enabled = resolve_router_replay_enabled(
            requested=True,
            api_key="k",
            base_url="https://api",
            additional_headers=None,
            base_model="accounts/fireworks/models/private",
            training_client=SimpleNamespace(supports_router_replay=None),
        )

    assert enabled is True
    assert "requesting Router Replay" in caplog.text


def test_router_replay_last_resort_still_off_when_user_disabled(monkeypatch):
    # The user flag can only keep R3 on; it never turns it on by itself.
    monkeypatch.setattr(
        service,
        "FireworksClient",
        lambda **_kwargs: pytest.fail("model lookup should not run"),
    )

    assert (
        resolve_router_replay_enabled(
            requested=False,
            api_key="k",
            base_url="https://api",
            additional_headers=None,
            base_model="accounts/fireworks/models/private",
        )
        is False
    )


@requires_typed_probe_error
@pytest.mark.parametrize(
    "exc",
    [
        # A real control-plane failure is not "not visible": keep raising.
        pytest.param(
            "server-error",
            id="http-500",
        ),
        # A readable record with no MoE flag: capability is genuinely unknown.
        pytest.param("missing-flag", id="missing-moe-flag"),
    ],
)
def test_router_replay_real_probe_failures_still_raise(monkeypatch, exc):
    err = (
        service.ModelDetailsUnavailableError(
            "Failed to fetch model details (HTTP 500)", status_code=500
        )
        if exc == "server-error"
        else ValueError("Base model is missing baseModelDetails.moe")
    )
    monkeypatch.setattr(service, "FireworksClient", _probe_raising(err))

    with pytest.raises(type(err)):
        resolve_router_replay_enabled(
            requested=True,
            api_key="k",
            base_url="https://api",
            additional_headers=None,
            base_model="accounts/acct/models/base",
        )


def test_router_replay_clear_dense_answer_overrides_user_flag(monkeypatch):
    # A clear "dense" answer wins over router_replay=True from either source.
    class DenseProbe:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def model_is_moe(self, model):
            return False

    monkeypatch.setattr(service, "FireworksClient", DenseProbe)
    for training_client in (None, SimpleNamespace(supports_router_replay=False)):
        assert (
            resolve_router_replay_enabled(
                requested=True,
                api_key="k",
                base_url="https://api",
                additional_headers=None,
                base_model="accounts/acct/models/dense",
                training_client=training_client,
            )
            is False
        )


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
        max_lora_rank=16,
        projection_head_dim=3,
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
            "max_lora_rank": 16,
            "projection_head_dim": 3,
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
            "custom_image_tag": "0.0.0-dev",
            "extra_args": ["--test-flag"],
            "trainer_replica_count": 4,
            "trainer_timeout_s": 1800,
            "trainer_pending_timeout_s": 172800,
            "inactivity_timeout": "7200s",
            "disable_inactivity_cleanup": True,
            "purpose": "PURPOSE_UNSPECIFIED",
            "preemptible": True,
            "managed_by": "parent-job",
            "skip_validations": True,
            "use_reservation": True,
            "cleanup_trainer_on_close": True,
            "cleanup_deployment_on_close": None,
            "create_deployment": True,
            "hotload_timeout_s": 600,
            "deployment_shape": "ds-x",
            "deployment_id": "dep-1",
            "deployment_extra_args": ["--test-flag"],
            "deployment_extra_values": {"testValue": "test-value"},
            "deployment_timeout_s": 5400,
            "replica_count": 3,
            "disable_speculative_decoding": True,
            "hot_load_transition_type": "SYNC",
        }
    ]


def test_build_service_client_forwards_max_lora_rank(monkeypatch):
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
        max_lora_rank=32,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-x"),
        deployment=DeployConfig(deployment_id="dep-1"),
    )

    assert calls[0]["max_lora_rank"] == 32
    assert "projection_head_dim" not in calls[0]


def test_build_service_client_treats_zero_projection_head_dim_as_unset(monkeypatch):
    calls: list[dict] = []

    class FakeServiceClient:
        from_firetitan_config = staticmethod(
            lambda **kwargs: calls.append(kwargs) or "service-sentinel"
        )

    monkeypatch.setattr(service, "FiretitanServiceClient", FakeServiceClient)

    build_service_client(
        api_key="k",
        base_url="https://api",
        additional_headers=None,
        base_model="accounts/acct/models/base",
        tokenizer_model=None,
        max_lora_rank=32,
        projection_head_dim=0,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-x"),
    )

    assert "projection_head_dim" not in calls[0]


def test_build_service_client_forwards_service_projection_head_dim(monkeypatch):
    calls: list[dict] = []

    class FakeServiceClient:
        from_firetitan_config = staticmethod(
            lambda **kwargs: calls.append(kwargs) or "service-sentinel"
        )

    monkeypatch.setattr(service, "FiretitanServiceClient", FakeServiceClient)

    build_service_client(
        api_key="k",
        base_url="https://api",
        additional_headers=None,
        base_model="accounts/acct/models/base",
        tokenizer_model=None,
        max_lora_rank=32,
        projection_head_dim=2,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-x"),
    )

    assert calls[0]["projection_head_dim"] == 2


def test_build_service_client_defaults_use_reservation_true(monkeypatch):
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
        max_lora_rank=0,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-x"),
    )

    assert calls[0]["use_reservation"] is True


def test_build_service_client_forwards_explicit_use_reservation_false(monkeypatch):
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
        max_lora_rank=0,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-x", use_reservation=False),
    )

    assert calls[0]["use_reservation"] is False


def test_build_service_client_forwards_reservation_target(monkeypatch):
    calls: list[dict] = []

    def fake_from_firetitan_config(**kwargs):
        calls.append(kwargs)
        return "service-sentinel"

    class FakeServiceClient:
        from_firetitan_config = staticmethod(fake_from_firetitan_config)

    monkeypatch.setattr(service, "FiretitanServiceClient", FakeServiceClient)

    target = "accounts/acct/reservations/team-training"
    build_service_client(
        api_key="k",
        base_url="https://api",
        additional_headers=None,
        base_model="accounts/acct/models/base",
        tokenizer_model=None,
        max_lora_rank=0,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(
            training_shape_id="ts-x",
            reservation_target=target,
        ),
    )

    assert calls[0]["reservation_target"] == target


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
        max_lora_rank=None,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-x"),
        deployment=DeployConfig(deployment_id="dep-1"),
    )

    assert result == "service-sentinel"
    assert calls[0]["disable_speculative_decoding"] is False


def test_build_service_client_leaves_hot_load_transition_type_unset(monkeypatch):
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
        max_lora_rank=None,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-x"),
        deployment=DeployConfig(deployment_id="dep-1"),
    )

    assert calls[0]["hot_load_transition_type"] is None


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
        max_lora_rank=None,
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


def test_build_service_client_rejects_negative_max_lora_rank() -> None:
    with pytest.raises(ValueError, match="max_lora_rank must be non-negative"):
        build_service_client(
            api_key="k",
            base_url="https://api",
            additional_headers=None,
            base_model="accounts/acct/models/base",
            tokenizer_model=None,
            max_lora_rank=-1,
            max_context_length=None,
            learning_rate=1e-5,
            trainer=TrainerConfig(training_shape_id="ts-x"),
        )


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
        base_url="https://api.example.com",
        inference_url="https://gateway.example.com",
        additional_headers=None,
        base_model="accounts/acct/models/base",
        tokenizer_model=None,
        max_lora_rank=0,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-x"),
        deployment=_deployment_config(),
    )

    assert result == "service-sentinel"
    assert calls[0]["inference_url"] == "https://gateway.example.com"


def test_trainer_region_becomes_sdk_region():
    client = build_service_client(
        api_key="k",
        base_url="https://api",
        additional_headers=None,
        base_model="accounts/acct/models/base",
        tokenizer_model=None,
        max_lora_rank=0,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=_trainer_config(region="US_OHIO_1", use_reservation=False),
        deployment=_deployment_config(),
    )

    assert client._managed_config.region == "US_OHIO_1"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("accelerator_type", "NVIDIA_B200"),
        ("accelerator_count", 8),
    ],
)
def test_trainer_config_rejects_removed_accelerator_fields(field_name, value):
    with pytest.raises(
        TypeError,
        match=(
            rf"TrainerConfig no longer supports `{field_name}`.*"
            r"select a training shape with `training_shape_id`"
        ),
    ):
        TrainerConfig(**{field_name: value})


def test_build_service_client_rejects_trainer_wait_without_per_trainer():
    with pytest.raises(ValueError, match="wait_for_trainer_before_deployment"):
        build_service_client(
            api_key="k",
            base_url="https://api",
            additional_headers=None,
            base_model="accounts/acct/models/base",
            tokenizer_model=None,
            max_lora_rank=None,
            max_context_length=None,
            learning_rate=1e-5,
            trainer=TrainerConfig(training_shape_id="ts-x"),
            deployment=DeployConfig(
                deployment_id="dep-1",
                wait_for_trainer_before_deployment=True,
                weight_sync_scope=WeightSyncScope.PER_DEPLOYMENT,
            ),
        )


def test_build_service_client_forwards_trainer_wait_when_sdk_declares_it(monkeypatch):
    calls: list[dict] = []

    def fake_from_firetitan_config(**kwargs):
        calls.append(kwargs)
        return "service-sentinel"

    class FakeServiceClient:
        from_firetitan_config = staticmethod(fake_from_firetitan_config)

    monkeypatch.setattr(service, "FiretitanServiceClient", FakeServiceClient)
    monkeypatch.setattr(
        service,
        "fields",
        lambda _cls: (SimpleNamespace(name="wait_for_trainer_before_deployment"),),
    )

    build_service_client(
        api_key="k",
        base_url="https://api",
        additional_headers=None,
        base_model="accounts/acct/models/base",
        tokenizer_model=None,
        max_lora_rank=None,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=TrainerConfig(training_shape_id="ts-x"),
        deployment=DeployConfig(
            deployment_id="dep-1",
            wait_for_trainer_before_deployment=True,
        ),
    )

    assert calls[0]["wait_for_trainer_before_deployment"] is True


def test_build_service_client_errors_when_trainer_wait_requested_on_old_sdk(monkeypatch):
    monkeypatch.setattr(service, "fields", lambda _cls: ())
    with pytest.raises(RuntimeError, match="wait_for_trainer_before_deployment"):
        build_service_client(
            api_key="k",
            base_url="https://api",
            additional_headers=None,
            base_model="accounts/acct/models/base",
            tokenizer_model=None,
            max_lora_rank=None,
            max_context_length=None,
            learning_rate=1e-5,
            trainer=TrainerConfig(training_shape_id="ts-x"),
            deployment=DeployConfig(
                deployment_id="dep-1",
                wait_for_trainer_before_deployment=True,
            ),
        )


def test_trainer_config_does_not_advertise_removed_accelerator_fields():
    field_names = {field.name for field in dataclasses.fields(TrainerConfig)}
    parameters = inspect.signature(TrainerConfig).parameters

    assert "accelerator_type" not in field_names
    assert "accelerator_count" not in field_names
    assert "accelerator_type" not in parameters
    assert "accelerator_count" not in parameters


class _FakeTrainerJobManager:
    existing: set[str] = set()
    deleted: list[str] = []

    def __init__(self, **_kwargs):
        pass

    def try_get(self, job_id):
        return {"name": job_id} if job_id in self.existing else None

    def delete(self, job_id):
        self.deleted.append(job_id)
        self.existing.discard(job_id)


def _provisioning_kwargs(trainer: TrainerConfig, cleanup: bool = True) -> dict:
    return dict(
        api_key="k",
        base_url="https://api",
        additional_headers=None,
        base_model="accounts/acct/models/base",
        tokenizer_model=None,
        max_lora_rank=None,
        max_context_length=None,
        learning_rate=1e-5,
        trainer=trainer,
        cleanup_trainer_on_close=cleanup,
    )


class _FakeProvisioningService:
    """Stands in for an SDK service whose trainer finishes provisioning in create_training_client."""

    def __init__(self, job_id, on_create):
        self.job_id = job_id
        self.on_create = on_create

    def create_training_client(self, *args, **kwargs):
        return self.on_create(self.job_id)


@pytest.fixture
def provisioning(monkeypatch):
    import signal as signal_module

    _FakeTrainerJobManager.existing = set()
    _FakeTrainerJobManager.deleted = []
    monkeypatch.setattr(service, "TrainerJobManager", _FakeTrainerJobManager)
    monkeypatch.setattr(service, "_PROVISIONING", {})
    monkeypatch.setattr(service, "_DELETE_RETRY_S", 0.0)
    handlers = {s: signal_module.getsignal(s) for s in (signal_module.SIGINT, signal_module.SIGTERM)}

    def patch(from_config):
        class FakeServiceClient:
            from_firetitan_config = staticmethod(from_config)

        monkeypatch.setattr(service, "FiretitanServiceClient", FakeServiceClient)

    yield patch
    for signum, handler in handlers.items():
        signal_module.signal(signum, handler)


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit, TimeoutError])
def test_failed_provisioning_deletes_the_new_trainer(provisioning, error):
    def from_config(**kwargs):
        _FakeTrainerJobManager.existing.add(kwargs["trainer_job_id"])
        raise error()

    provisioning(from_config)
    with pytest.raises(error):
        build_service_client(**_provisioning_kwargs(TrainerConfig()))

    assert len(_FakeTrainerJobManager.deleted) == 1
    assert _FakeTrainerJobManager.deleted[0].startswith("training-api-service-")


def test_interrupted_create_training_client_deletes_the_new_trainer(provisioning):
    def interrupted(job_id):
        _FakeTrainerJobManager.existing.add(job_id)
        raise KeyboardInterrupt

    provisioning(lambda **kwargs: _FakeProvisioningService(kwargs["trainer_job_id"], interrupted))
    svc = build_service_client(**_provisioning_kwargs(TrainerConfig()))
    with pytest.raises(KeyboardInterrupt):
        svc.create_training_client("accounts/acct/models/base")

    assert _FakeTrainerJobManager.deleted == [svc.job_id]


def test_ready_trainer_is_left_to_the_service(provisioning):
    import signal as signal_module

    def ready(job_id):
        _FakeTrainerJobManager.existing.add(job_id)
        return "training-client"

    provisioning(lambda **kwargs: _FakeProvisioningService(kwargs["trainer_job_id"], ready))
    signal_module.signal(signal_module.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("stopped")))
    svc = build_service_client(**_provisioning_kwargs(TrainerConfig()))
    assert svc.create_training_client("accounts/acct/models/base") == "training-client"
    with pytest.raises(SystemExit, match="stopped"):
        signal_module.raise_signal(signal_module.SIGTERM)

    assert _FakeTrainerJobManager.deleted == []


def test_provisioning_never_deletes_a_reattached_trainer(provisioning):
    def from_config(**kwargs):
        _FakeTrainerJobManager.existing.add(kwargs["trainer_job_id"])
        raise KeyboardInterrupt

    provisioning(from_config)
    with pytest.raises(KeyboardInterrupt):
        build_service_client(**_provisioning_kwargs(TrainerConfig(job_id="existing-job")))
    with pytest.raises(KeyboardInterrupt):
        build_service_client(**_provisioning_kwargs(TrainerConfig(), cleanup=False))

    assert _FakeTrainerJobManager.deleted == []


def test_signal_while_provisioning_deletes_trainer_then_runs_previous_handler(provisioning):
    import signal as signal_module

    seen = []

    def previous_handler(signum, _frame):
        seen.append(signum)
        raise SystemExit("stopped")

    def signalled(job_id):
        _FakeTrainerJobManager.existing.add(job_id)
        signal_module.raise_signal(signal_module.SIGTERM)

    provisioning(lambda **kwargs: _FakeProvisioningService(kwargs["trainer_job_id"], signalled))
    signal_module.signal(signal_module.SIGTERM, previous_handler)
    svc = build_service_client(**_provisioning_kwargs(TrainerConfig()))
    with pytest.raises(SystemExit, match="stopped"):
        svc.create_training_client("accounts/acct/models/base")

    assert seen == [signal_module.SIGTERM]
    assert _FakeTrainerJobManager.deleted == [svc.job_id]


def test_trainer_created_after_the_interrupt_is_still_deleted(provisioning, monkeypatch):
    import threading
    import time

    monkeypatch.setattr(service, "_CREATE_POLL_S", 0.01)
    job_ids = []

    def interrupted_before_create_lands(job_id):
        job_ids.append(job_id)
        # The create request lands after cleanup first looks for the trainer.
        threading.Timer(0.05, _FakeTrainerJobManager.existing.add, args=(job_id,)).start()
        raise KeyboardInterrupt

    provisioning(lambda **kwargs: _FakeProvisioningService(kwargs["trainer_job_id"], interrupted_before_create_lands))
    svc = build_service_client(**_provisioning_kwargs(TrainerConfig()))
    with pytest.raises(KeyboardInterrupt):
        svc.create_training_client("accounts/acct/models/base")

    deadline = time.monotonic() + 2
    while not _FakeTrainerJobManager.deleted and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _FakeTrainerJobManager.deleted == job_ids


def test_late_trainer_watch_retries_failed_lookups(monkeypatch):
    monkeypatch.setattr(service, "_CREATE_POLL_S", 0.0)
    deleted = []

    class FlakyManager:
        lookups = 0

        def try_get(self, job_id):
            self.lookups += 1
            if self.lookups == 1:
                raise RuntimeError("503 Service Unavailable")
            return None if deleted else {"name": job_id}

        def delete(self, job_id):
            deleted.append(job_id)

    service._delete_when_created(FlakyManager(), "training-api-service-12345678")

    assert deleted == ["training-api-service-12345678"]


def test_failed_first_lookup_still_watches_for_the_trainer(provisioning, monkeypatch):
    import time

    monkeypatch.setattr(service, "_CREATE_POLL_S", 0.01)
    lookups = []
    real_try_get = _FakeTrainerJobManager.try_get

    def flaky_try_get(self, job_id):
        lookups.append(job_id)
        if len(lookups) == 1:
            raise RuntimeError("503 Service Unavailable")
        return real_try_get(self, job_id)

    monkeypatch.setattr(_FakeTrainerJobManager, "try_get", flaky_try_get)

    def interrupted(job_id):
        _FakeTrainerJobManager.existing.add(job_id)
        raise KeyboardInterrupt

    provisioning(lambda **kwargs: _FakeProvisioningService(kwargs["trainer_job_id"], interrupted))
    svc = build_service_client(**_provisioning_kwargs(TrainerConfig()))
    with pytest.raises(KeyboardInterrupt):
        svc.create_training_client("accounts/acct/models/base")

    deadline = time.monotonic() + 2
    while not _FakeTrainerJobManager.deleted and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _FakeTrainerJobManager.deleted == [svc.job_id]


def test_late_trainer_watch_warns_when_lookups_keep_failing(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(service, "_CREATE_POLL_S", 0.0)
    monkeypatch.setattr(service, "_CREATE_SETTLE_S", 0.05)

    class DownManager:
        def try_get(self, job_id):
            raise RuntimeError("503 Service Unavailable")

        def delete(self, job_id):
            pytest.fail("nothing was found to delete")

    with caplog.at_level(logging.WARNING, logger=service.logger.name):
        service._delete_when_created(DownManager(), "training-api-service-12345678")

    assert "Could not confirm whether trainer training-api-service-12345678 was created" in caplog.text


def test_unconfirmed_delete_is_retried_then_reported(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(service, "_DELETE_RETRY_S", 0.0)
    attempts = []

    class SwallowingManager:
        """Like TrainerJobManager.delete: a failed request is logged, not raised."""

        def delete(self, job_id):
            attempts.append(job_id)

        def try_get(self, job_id):
            if len(attempts) < 2:
                return {"name": job_id, "state": "JOB_STATE_PENDING"}
            return {"name": job_id, "state": "JOB_STATE_DELETING"}

    assert service._delete_confirmed(SwallowingManager(), "training-api-service-12345678") is True
    assert len(attempts) == 2

    class StuckManager(SwallowingManager):
        def try_get(self, job_id):
            return {"name": job_id, "state": "JOB_STATE_PENDING"}

    attempts.clear()
    with caplog.at_level(logging.WARNING, logger=service.logger.name):
        assert service._delete_confirmed(StuckManager(), "training-api-service-12345678") is False
    assert len(attempts) == service._DELETE_ATTEMPTS
    assert "Could not confirm trainer training-api-service-12345678 was deleted" in caplog.text


def test_process_exit_waits_for_an_in_flight_trainer_create(tmp_path):
    """A create that lands just after the interrupted process exits must still be deleted."""
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    deleted_file = tmp_path / "deleted.txt"
    script = textwrap.dedent(
        f"""
        import sys, time
        from training.utils import service

        service._CREATE_POLL_S = 0.05
        started = time.monotonic()

        class Manager:
            def try_get(self, job_id):
                # The in-flight create lands 0.3s after the process starts exiting.
                if time.monotonic() - started < 0.3 or open({str(deleted_file)!r}).read():
                    return None
                return {{"name": job_id, "state": "JOB_STATE_PENDING"}}

            def delete(self, job_id):
                open({str(deleted_file)!r}, "a").write(job_id)

        open({str(deleted_file)!r}, "w").close()
        service._start_late_trainer_watch(Manager(), "training-api-service-12345678")
        sys.exit(0)
        """
    )
    repo_root = Path(__file__).resolve().parents[3]
    subprocess.run([sys.executable, "-c", script], check=True, cwd=repo_root, timeout=60)

    assert deleted_file.read_text() == "training-api-service-12345678"
