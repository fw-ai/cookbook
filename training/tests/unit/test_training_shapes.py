from __future__ import annotations

from training.utils import (
    InfraConfig,
    apply_recommended_training_shapes,
    canonical_base_model,
    get_recommended_training_shapes,
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
