from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from training.utils import (
    InfraConfig,
    ShapeSelectionRequest,
    canonical_base_model,
    materialize_profile_infra,
    prepare_training_shape_launch,
    select_validated_launch_shapes,
)


def _training_shape_version(
    name: str,
    *,
    trainer_mode: str,
    max_supported_context_length: int,
    deployment_shape_version: str = "",
    accelerator_type: str = "NVIDIA_H200_141GB",
    accelerator_count: int = 8,
) -> dict:
    return {
        "name": name,
        "snapshot": {
            "baseModel": "accounts/fireworks/models/qwen3-8b",
            "modelType": "qwen3",
            "parameterCount": 8_400_000_000,
            "trainerMode": trainer_mode,
            "trainerImageTag": "trainer:1",
            "maxSupportedContextLength": max_supported_context_length,
            "nodeCount": 2,
            "deploymentShapeVersion": deployment_shape_version,
            "deploymentImageTag": "deployment:1",
            "acceleratorType": accelerator_type,
            "acceleratorCount": accelerator_count,
            "baseModelWeightPrecision": "WEIGHT_PRECISION_BFLOAT16",
            "trainerShardingScheme": {
                "pipelineParallelism": 1,
            },
        },
    }


def _deployment_shape_version(name: str, *, accelerator_type: str = "NVIDIA_H200_141GB") -> dict:
    return {
        "name": name,
        "snapshot": {
            "baseModel": "accounts/fireworks/models/qwen3-8b",
            "modelType": "qwen3",
            "parameterCount": 8_400_000_000,
            "acceleratorType": accelerator_type,
            "acceleratorCount": 8,
            "engine": "VLLM",
        },
    }


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300

    def json(self) -> dict:
        return self._payload


class _FakeSelectorMgr:
    def __init__(self, handler, profile=None):
        self._handler = handler
        self._profile = profile
        self.paths: list[str] = []
        self.resolved_shapes: list[str] = []

    def _get(self, path: str, timeout: int = 30):
        self.paths.append(path)
        return _FakeResponse(self._handler(path))

    def resolve_training_profile(self, shape_id: str):
        self.resolved_shapes.append(shape_id)
        if self._profile is None:
            raise AssertionError("resolve_training_profile should not have been called")
        return self._profile


def _filter_for(path: str) -> str:
    return parse_qs(urlsplit(path).query).get("filter", [""])[0]


def test_canonical_base_model_is_identity_for_runtime_selection():
    model = "accounts/fireworks/models/qwen3-30b-a3b"
    assert canonical_base_model(model) == model


def test_select_validated_launch_shapes_prefers_smallest_exact_match():
    def handler(path: str) -> dict:
        filt = _filter_for(path)
        if "/trainingShapes/-/versions" in path:
            assert 'snapshot.base_model="accounts/fireworks/models/qwen3-8b"' in filt
            assert 'snapshot.trainer_mode="POLICY_TRAINER"' in filt
            return {
                "trainingShapeVersions": [
                    _training_shape_version(
                        "accounts/fireworks/trainingShapes/qwen3-8b-128k/versions/v2",
                        trainer_mode="POLICY_TRAINER",
                        max_supported_context_length=131072,
                        deployment_shape_version=(
                            "accounts/fireworks/deploymentShapes/qwen3-8b-128k/versions/d2"
                        ),
                    ),
                    _training_shape_version(
                        "accounts/fireworks/trainingShapes/qwen3-8b-32k/versions/v1",
                        trainer_mode="POLICY_TRAINER",
                        max_supported_context_length=32768,
                        deployment_shape_version=(
                            "accounts/fireworks/deploymentShapes/qwen3-8b-32k/versions/d1"
                        ),
                    ),
                ]
            }
        raise AssertionError(f"unexpected path: {path}")

    mgr = _FakeSelectorMgr(handler)
    selection = select_validated_launch_shapes(
        mgr,
        request=ShapeSelectionRequest(
            base_model="accounts/fireworks/models/qwen3-8b",
            max_seq_len=32000,
            trainer_role="policy",
            needs_deployment=True,
        ),
    )

    assert selection.training_shape_id == "accounts/fireworks/trainingShapes/qwen3-8b-32k"
    assert (
        selection.training_profile.training_shape_version
        == "accounts/fireworks/trainingShapes/qwen3-8b-32k/versions/v1"
    )
    assert (
        selection.deployment_shape
        == "accounts/fireworks/deploymentShapes/qwen3-8b-32k/versions/d1"
    )
    assert selection.inferred_training_shape is True
    assert selection.inferred_deployment_shape is True
    assert not any("/v1/accounts/fireworks/models/qwen3-8b" in path for path in mgr.paths)


def test_select_validated_launch_shapes_falls_back_to_compat_buckets():
    def handler(path: str) -> dict:
        filt = _filter_for(path)
        if path == "/v1/accounts/fireworks/models/qwen3-8b":
            return {
                "baseModelDetails": {
                    "modelType": "qwen3",
                    "parameterCount": 8_400_000_000,
                    "supportsFireattention": False,
                }
            }
        if "/trainingShapes/-/versions" in path:
            if 'snapshot.base_model="accounts/fireworks/models/qwen3-8b"' in filt:
                return {"trainingShapeVersions": []}
            assert 'snapshot.model_type="qwen3"' in filt
            assert "snapshot.parameter_count>=8000000000" in filt
            assert "snapshot.parameter_count<=9000000000" in filt
            return {
                "trainingShapeVersions": [
                    _training_shape_version(
                        "accounts/fireworks/trainingShapes/qwen3-compat/versions/v3",
                        trainer_mode="POLICY_TRAINER",
                        max_supported_context_length=65536,
                    )
                ]
            }
        if "/deploymentShapes/-/versions" in path:
            if 'snapshot.base_model="accounts/fireworks/models/qwen3-8b"' in filt:
                return {"deploymentShapeVersions": []}
            assert 'snapshot.model_type="qwen3"' in filt
            assert 'snapshot.engine!="FIREATTENTION"' in filt
            return {
                "deploymentShapeVersions": [
                    _deployment_shape_version(
                        "accounts/fireworks/deploymentShapes/qwen3-compat/versions/d5"
                    )
                ]
            }
        raise AssertionError(f"unexpected path: {path}")

    mgr = _FakeSelectorMgr(handler)
    selection = select_validated_launch_shapes(
        mgr,
        request=ShapeSelectionRequest(
            base_model="accounts/fireworks/models/qwen3-8b",
            max_seq_len=4096,
            trainer_role="policy",
            needs_deployment=True,
        ),
    )

    assert selection.training_shape_id == "accounts/fireworks/trainingShapes/qwen3-compat"
    assert (
        selection.deployment_shape
        == "accounts/fireworks/deploymentShapes/qwen3-compat/versions/d5"
    )


def test_select_validated_launch_shapes_filters_lora_and_reference_modes():
    def handler(path: str) -> dict:
        filt = _filter_for(path)
        if "/trainingShapes/-/versions" in path and 'snapshot.trainer_mode="LORA_TRAINER"' in filt:
            return {
                "trainingShapeVersions": [
                    _training_shape_version(
                        "accounts/fireworks/trainingShapes/qwen3-8b-lora/versions/v7",
                        trainer_mode="LORA_TRAINER",
                        max_supported_context_length=8192,
                    )
                ]
            }
        if "/trainingShapes/-/versions" in path and 'snapshot.trainer_mode="FORWARD_ONLY"' in filt:
            return {
                "trainingShapeVersions": [
                    _training_shape_version(
                        "accounts/fireworks/trainingShapes/qwen3-8b-ref/versions/v2",
                        trainer_mode="FORWARD_ONLY",
                        max_supported_context_length=8192,
                    )
                ]
            }
        raise AssertionError(f"unexpected path: {path}")

    mgr = _FakeSelectorMgr(handler)
    lora_selection = select_validated_launch_shapes(
        mgr,
        request=ShapeSelectionRequest(
            base_model="accounts/fireworks/models/qwen3-8b",
            max_seq_len=4096,
            trainer_role="policy",
            lora_rank=16,
        ),
    )
    ref_selection = select_validated_launch_shapes(
        mgr,
        request=ShapeSelectionRequest(
            base_model="accounts/fireworks/models/qwen3-8b",
            max_seq_len=4096,
            trainer_role="reference",
        ),
    )

    assert lora_selection.training_shape_id == "accounts/fireworks/trainingShapes/qwen3-8b-lora"
    assert ref_selection.training_shape_id == "accounts/fireworks/trainingShapes/qwen3-8b-ref"


def test_select_validated_launch_shapes_preserves_explicit_overrides():
    profile = SimpleNamespace(
        training_shape_version="accounts/custom/trainingShapes/policy/versions/v1",
        trainer_image_tag="trainer:1",
        max_supported_context_length=16384,
        node_count=2,
        deployment_shape_version="accounts/custom/deploymentShapes/policy/versions/d1",
        deployment_image_tag="deployment:1",
        accelerator_type="NVIDIA_H200_141GB",
        accelerator_count=8,
        base_model_weight_precision="WEIGHT_PRECISION_BFLOAT16",
        pipeline_parallelism=1,
    )
    mgr = _FakeSelectorMgr(lambda path: (_ for _ in ()).throw(AssertionError(path)), profile=profile)

    selection = select_validated_launch_shapes(
        mgr,
        request=ShapeSelectionRequest(
            base_model="accounts/fireworks/models/qwen3-8b",
            max_seq_len=4096,
            trainer_role="policy",
            needs_deployment=True,
            explicit_training_shape_id="accounts/custom/trainingShapes/policy",
            explicit_deployment_shape="accounts/custom/deploymentShapes/policy/versions/d9",
        ),
    )

    assert mgr.resolved_shapes == ["accounts/custom/trainingShapes/policy"]
    assert selection.training_shape_id == "accounts/custom/trainingShapes/policy"
    assert selection.deployment_shape == "accounts/custom/deploymentShapes/policy/versions/d9"
    assert selection.inferred_training_shape is False
    assert selection.inferred_deployment_shape is False


def test_select_validated_launch_shapes_raises_when_no_validated_shape_matches():
    def handler(path: str) -> dict:
        if path == "/v1/accounts/fireworks/models/qwen3-8b":
            return {
                "baseModelDetails": {
                    "modelType": "qwen3",
                    "parameterCount": 8_400_000_000,
                    "supportsFireattention": True,
                }
            }
        if "/versions" in path:
            if "trainingShapes" in path:
                return {"trainingShapeVersions": []}
            return {"deploymentShapeVersions": []}
        raise AssertionError(f"unexpected path: {path}")

    mgr = _FakeSelectorMgr(handler)
    with pytest.raises(ValueError, match="No training configuration is available"):
        select_validated_launch_shapes(
            mgr,
            request=ShapeSelectionRequest(
                base_model="accounts/fireworks/models/qwen3-8b",
                max_seq_len=16000,
                trainer_role="policy",
                needs_deployment=False,
            ),
        )


def test_select_validated_launch_shapes_filters_by_accelerator_type():
    """When accelerator_type is set, only shapes with that accelerator are returned."""
    h200_shape = _training_shape_version(
        "accounts/fireworks/trainingShapes/qwen3-8b-h200/versions/v1",
        trainer_mode="POLICY_TRAINER",
        max_supported_context_length=131072,
        accelerator_type="NVIDIA_H200_141GB",
    )
    b200_shape = _training_shape_version(
        "accounts/fireworks/trainingShapes/qwen3-8b-b200/versions/v2",
        trainer_mode="POLICY_TRAINER",
        max_supported_context_length=131072,
        accelerator_type="NVIDIA_B200_180GB",
    )

    def handler(path: str) -> dict:
        filt = _filter_for(path)
        if "/trainingShapes/-/versions" in path:
            assert 'snapshot.accelerator_type="NVIDIA_B200_180GB"' in filt
            return {"trainingShapeVersions": [b200_shape]}
        raise AssertionError(f"unexpected path: {path}")

    mgr = _FakeSelectorMgr(handler)
    selection = select_validated_launch_shapes(
        mgr,
        request=ShapeSelectionRequest(
            base_model="accounts/fireworks/models/qwen3-8b",
            trainer_role="policy",
            accelerator_type="NVIDIA_B200_180GB",
        ),
    )

    assert selection.training_shape_id == "accounts/fireworks/trainingShapes/qwen3-8b-b200"
    assert selection.training_profile.accelerator_type == "NVIDIA_B200_180GB"


def test_select_validated_launch_shapes_accelerator_mismatch_raises_generic_error():
    """When no shapes match the required accelerator, the user sees a generic
    error while the detailed constraint info is logged."""
    def handler(path: str) -> dict:
        if path == "/v1/accounts/fireworks/models/qwen3-8b":
            return {
                "baseModelDetails": {
                    "modelType": "qwen3",
                    "parameterCount": 8_400_000_000,
                    "supportsFireattention": False,
                }
            }
        if "/versions" in path:
            if "trainingShapes" in path:
                return {"trainingShapeVersions": []}
            return {"deploymentShapeVersions": []}
        raise AssertionError(f"unexpected path: {path}")

    mgr = _FakeSelectorMgr(handler)
    with pytest.raises(ValueError, match="No training configuration is available"):
        select_validated_launch_shapes(
            mgr,
            request=ShapeSelectionRequest(
                base_model="accounts/fireworks/models/qwen3-8b",
                trainer_role="policy",
                accelerator_type="NVIDIA_B200_180GB",
            ),
        )


def test_select_validated_launch_shapes_client_side_accelerator_filter():
    """Even if the server returns shapes with mixed accelerators, the client
    filters to only the requested type."""
    h200_shape = _training_shape_version(
        "accounts/fireworks/trainingShapes/qwen3-8b-h200/versions/v1",
        trainer_mode="POLICY_TRAINER",
        max_supported_context_length=131072,
        accelerator_type="NVIDIA_H200_141GB",
    )
    b200_shape = _training_shape_version(
        "accounts/fireworks/trainingShapes/qwen3-8b-b200/versions/v2",
        trainer_mode="POLICY_TRAINER",
        max_supported_context_length=131072,
        accelerator_type="NVIDIA_B200_180GB",
    )

    def handler(path: str) -> dict:
        if "/trainingShapes/-/versions" in path:
            return {"trainingShapeVersions": [h200_shape, b200_shape]}
        raise AssertionError(f"unexpected path: {path}")

    mgr = _FakeSelectorMgr(handler)
    selection = select_validated_launch_shapes(
        mgr,
        request=ShapeSelectionRequest(
            base_model="accounts/fireworks/models/qwen3-8b",
            trainer_role="policy",
            accelerator_type="NVIDIA_B200_180GB",
        ),
    )

    assert selection.training_shape_id == "accounts/fireworks/trainingShapes/qwen3-8b-b200"


def test_select_validated_launch_shapes_no_accelerator_filter_returns_any():
    """When accelerator_type is None, all shapes are eligible (backward compat)."""
    h200_shape = _training_shape_version(
        "accounts/fireworks/trainingShapes/qwen3-8b-h200/versions/v1",
        trainer_mode="POLICY_TRAINER",
        max_supported_context_length=131072,
        accelerator_type="NVIDIA_H200_141GB",
    )

    def handler(path: str) -> dict:
        filt = _filter_for(path)
        if "/trainingShapes/-/versions" in path:
            assert "snapshot.accelerator_type" not in filt
            return {"trainingShapeVersions": [h200_shape]}
        raise AssertionError(f"unexpected path: {path}")

    mgr = _FakeSelectorMgr(handler)
    selection = select_validated_launch_shapes(
        mgr,
        request=ShapeSelectionRequest(
            base_model="accounts/fireworks/models/qwen3-8b",
            trainer_role="policy",
        ),
    )

    assert selection.training_shape_id == "accounts/fireworks/trainingShapes/qwen3-8b-h200"


def test_materialize_profile_infra_copies_shape_owned_fields():
    infra = InfraConfig(training_shape_id="shape-a", region="US_VIRGINIA_1")
    profile = SimpleNamespace(
        trainer_image_tag="trainer:1",
        accelerator_type="NVIDIA_H200_141GB",
        accelerator_count=8,
        node_count=2,
    )

    resolved = materialize_profile_infra(infra, profile)

    assert resolved is not infra
    assert resolved.training_shape_id == "shape-a"
    assert resolved.region == "US_VIRGINIA_1"
    assert resolved.custom_image_tag == "trainer:1"
    assert resolved.accelerator_type == "NVIDIA_H200_141GB"
    assert resolved.accelerator_count == 8
    assert resolved.node_count == 2


def test_prepare_training_shape_launch_uses_manual_path_for_client_managed_shape():
    infra = InfraConfig(training_shape_id="shape-a")
    profile = SimpleNamespace(
        trainer_image_tag="trainer:1",
        accelerator_type="NVIDIA_H200_141GB",
        accelerator_count=8,
        node_count=2,
    )

    launch_infra, launch_profile = prepare_training_shape_launch(
        infra,
        profile,
        client_managed=True,
    )

    assert launch_profile is None
    assert launch_infra.custom_image_tag == "trainer:1"
    assert launch_infra.accelerator_type == "NVIDIA_H200_141GB"
    assert launch_infra.accelerator_count == 8
    assert launch_infra.node_count == 2
