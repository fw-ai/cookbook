from __future__ import annotations

from types import SimpleNamespace

from training.utils import (
    InfraConfig,
    apply_recommended_training_shapes,
    canonical_base_model,
    get_recommended_training_shapes,
    materialize_profile_infra,
    prepare_training_shape_launch,
    recommend_training_shape,
)


def test_recommend_training_shape_returns_documented_public_paths():
    assert recommend_training_shape(
        "accounts/fireworks/models/qwen3-8b"
    ) == "accounts/fireworks/trainingShapes/qwen3-8b-128k-h200"
    assert recommend_training_shape(
        "accounts/fireworks/models/qwen3-8b",
        forward_only=True,
    ) == "accounts/fireworks/trainingShapes/qwen3-8b-128k-h200-forward"
    assert recommend_training_shape(
        "accounts/fireworks/models/qwen3-vl-8b-instruct",
        forward_only=True,
    ) is None


def test_recommend_training_shape_supports_canonical_aliases():
    assert canonical_base_model(
        "accounts/fireworks/models/qwen3-30b-a3b"
    ) == "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507"
    assert recommend_training_shape(
        "accounts/fireworks/models/qwen3-30b-a3b"
    ) == "accounts/fireworks/trainingShapes/qwen3-30b-a3b-instruct-2507-128k-b200"


def test_recommend_training_shape_uses_kimi_lora_shape_when_requested():
    assert get_recommended_training_shapes("accounts/fireworks/models/kimi-k2p5").lora == (
        "accounts/fireworks/trainingShapes/kimi-k2p5-80k-lora"
    )
    assert recommend_training_shape(
        "accounts/fireworks/models/kimi-k2p5",
        lora_rank=16,
    ) == "accounts/fireworks/trainingShapes/kimi-k2p5-80k-lora"


def test_apply_recommended_training_shapes_preserves_explicit_overrides():
    infra = InfraConfig(
        training_shape_id="custom-policy-shape",
        ref_training_shape_id="custom-reference-shape",
    )

    selected = apply_recommended_training_shapes(
        infra,
        base_model="accounts/fireworks/models/qwen3-8b",
        prefer_reference=True,
    )

    assert selected.policy == "custom-policy-shape"
    assert selected.reference == "custom-reference-shape"
    assert selected.inferred_policy is False
    assert selected.inferred_reference is False


def test_apply_recommended_training_shapes_fills_policy_and_reference():
    infra = InfraConfig()

    selected = apply_recommended_training_shapes(
        infra,
        base_model="accounts/fireworks/models/qwen3-4b",
        prefer_reference=True,
    )

    assert selected.policy == "accounts/fireworks/trainingShapes/qwen3-4b-minimum-h200"
    assert selected.reference == "accounts/fireworks/trainingShapes/qwen3-4b-minimum-h200-forward"
    assert infra.training_shape_id == selected.policy
    assert infra.ref_training_shape_id == selected.reference
    assert selected.inferred_policy is True
    assert selected.inferred_reference is True


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
