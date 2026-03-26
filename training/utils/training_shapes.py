"""Recommended training-shape mappings for supported Fireworks base models.

The mapping is intentionally hard-coded from the public Fireworks training
shapes catalog:
https://docs.fireworks.ai/fine-tuning/training-sdk/training-shapes
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from fireworks.training.sdk.trainer import TrainingShapeProfile
from training.utils.config import InfraConfig

TRAINING_SHAPES_DOCS_URL = (
    "https://docs.fireworks.ai/fine-tuning/training-sdk/training-shapes"
)

_SHARED_SHAPE_PREFIX = "accounts/fireworks/trainingShapes"


def _shape_id(name: str) -> str:
    return f"{_SHARED_SHAPE_PREFIX}/{name}"


@dataclass(frozen=True)
class RecommendedTrainingShapes:
    policy: str
    reference: str | None = None
    lora: str | None = None


@dataclass(frozen=True)
class SelectedTrainingShapes:
    policy: str | None
    reference: str | None
    inferred_policy: bool
    inferred_reference: bool


def materialize_profile_infra(
    infra: InfraConfig,
    profile: TrainingShapeProfile,
) -> InfraConfig:
    """Return an InfraConfig copy populated from a resolved training shape."""
    return replace(
        infra,
        custom_image_tag=getattr(profile, "trainer_image_tag", None) or infra.custom_image_tag,
        accelerator_type=getattr(profile, "accelerator_type", None) or infra.accelerator_type,
        accelerator_count=getattr(profile, "accelerator_count", None) or infra.accelerator_count,
        node_count=getattr(profile, "node_count", None) or infra.node_count,
    )


def prepare_training_shape_launch(
    infra: InfraConfig,
    profile: TrainingShapeProfile | None,
    *,
    client_managed: bool,
) -> tuple[InfraConfig, TrainingShapeProfile | None]:
    """Choose manual-vs-shape launch config for a resolved profile."""
    if not client_managed or profile is None:
        return infra, profile
    return materialize_profile_infra(infra, profile), None


_MODEL_ALIASES: dict[str, str] = {
    "accounts/fireworks/models/qwen3-30b-a3b": (
        "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507"
    ),
    "accounts/fireworks/models/qwen3-235b-a22b": (
        "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"
    ),
}

_MODEL_TRAINING_SHAPES: dict[str, RecommendedTrainingShapes] = {
    "accounts/fireworks/models/qwen3-4b": RecommendedTrainingShapes(
        policy=_shape_id("qwen3-4b-minimum-h200"),
        reference=_shape_id("qwen3-4b-minimum-h200-forward"),
    ),
    "accounts/fireworks/models/qwen3-8b": RecommendedTrainingShapes(
        policy=_shape_id("qwen3-8b-128k-h200"),
        reference=_shape_id("qwen3-8b-128k-h200-forward"),
    ),
    "accounts/fireworks/models/qwen3-32b": RecommendedTrainingShapes(
        policy=_shape_id("qwen3-32b-65k-b200"),
        reference=_shape_id("qwen3-32b-65k-b200-forward-only"),
    ),
    "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507": RecommendedTrainingShapes(
        policy=_shape_id("qwen3-30b-a3b-instruct-2507-128k-b200"),
        reference=_shape_id("qwen3-30b-a3b-instruct-2507-128k-b200-forward-only"),
    ),
    "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507": RecommendedTrainingShapes(
        policy=_shape_id("qwen3-235b-2507-instruct-128k-b200"),
        reference=_shape_id("qwen3-235b-2507-instruct-128k-b200-forward-only"),
    ),
    "accounts/fireworks/models/qwen3-vl-8b-instruct": RecommendedTrainingShapes(
        policy=_shape_id("qwen3-vl-8b-65k-h200"),
    ),
    "accounts/fireworks/models/llama-v3p3-70b-instruct": RecommendedTrainingShapes(
        policy=_shape_id("ts-llama70b-b200-policy"),
        reference=_shape_id("ts-llama70b-b200-ref"),
    ),
    "accounts/fireworks/models/kimi-k2p5": RecommendedTrainingShapes(
        policy=_shape_id("kimi-2p5-text-only-256k-4b200"),
        reference=_shape_id("kimi-2p5-text-only-256k-4b200-forward"),
        lora=_shape_id("kimi-k2p5-80k-lora"),
    ),
}


def canonical_base_model(base_model: str) -> str:
    """Return the canonical public model ID for lookup."""
    return _MODEL_ALIASES.get(base_model, base_model)


def get_recommended_training_shapes(
    base_model: str,
) -> RecommendedTrainingShapes | None:
    """Return the documented shape bundle for *base_model*, if known."""
    return _MODEL_TRAINING_SHAPES.get(canonical_base_model(base_model))


def recommend_training_shape(
    base_model: str,
    *,
    forward_only: bool = False,
    lora_rank: int = 0,
) -> str | None:
    """Return the documented training shape ID for *base_model*, if known."""
    shapes = get_recommended_training_shapes(base_model)
    if shapes is None:
        return None
    if forward_only:
        return shapes.reference
    if lora_rank > 0 and shapes.lora is not None:
        return shapes.lora
    return shapes.policy


def apply_recommended_training_shapes(
    infra: InfraConfig,
    *,
    base_model: str,
    lora_rank: int = 0,
    prefer_reference: bool = False,
) -> SelectedTrainingShapes:
    """Populate missing training-shape IDs from the documented model mapping."""
    policy = infra.training_shape_id
    reference = infra.ref_training_shape_id
    inferred_policy = False
    inferred_reference = False

    if not policy:
        policy = recommend_training_shape(base_model, lora_rank=lora_rank)
        if policy:
            infra.training_shape_id = policy
            inferred_policy = True

    if not reference and prefer_reference:
        reference = recommend_training_shape(base_model, forward_only=True)
        if reference:
            infra.ref_training_shape_id = reference
            inferred_reference = True

    return SelectedTrainingShapes(
        policy=policy,
        reference=reference,
        inferred_policy=inferred_policy,
        inferred_reference=inferred_reference,
    )
