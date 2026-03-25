"""Tests for TrainingShapeProfile.base_model and TrainerJobConfig validation.

Requires the SDK with base_model field on TrainingShapeProfile
(stainless-sdks/fireworks-ai-python#82). Skipped when running
against an older SDK release.
"""

import pytest

from fireworks.training.sdk.trainer import TrainerJobConfig, TrainingShapeProfile

_HAS_BASE_MODEL = "base_model" in {f.name for f in TrainingShapeProfile.__dataclass_fields__.values()}
pytestmark = pytest.mark.skipif(not _HAS_BASE_MODEL, reason="SDK missing TrainingShapeProfile.base_model")


def _make_profile(**overrides) -> TrainingShapeProfile:
    defaults = dict(
        training_shape_version="accounts/fw/trainingShapes/ts-qwen3-8b/versions/1",
        trainer_image_tag="0.0.0-test",
        max_supported_context_length=4096,
        node_count=1,
        deployment_shape_version="accounts/fw/deploymentShapes/ds-x/versions/1",
        deployment_image_tag="4.24.22",
        accelerator_type="NVIDIA_B200",
        accelerator_count=8,
        base_model_weight_precision="BF16",
        base_model="accounts/fireworks/models/qwen3-8b",
        pipeline_parallelism=1,
    )
    defaults.update(overrides)
    return TrainingShapeProfile(**defaults)


class TestProfileBaseModel:
    def test_profile_exposes_base_model(self):
        profile = _make_profile(base_model="accounts/fw/models/qwen3-8b")
        assert profile.base_model == "accounts/fw/models/qwen3-8b"

    def test_config_uses_profile_base_model(self):
        profile = _make_profile()
        cfg = TrainerJobConfig(
            base_model=profile.base_model,
            training_shape_ref=profile.training_shape_version,
        )
        cfg.validate()
        assert cfg.base_model == "accounts/fireworks/models/qwen3-8b"

    def test_rejects_accelerator_count_with_shape(self):
        profile = _make_profile()
        cfg = TrainerJobConfig(
            base_model=profile.base_model,
            training_shape_ref=profile.training_shape_version,
            accelerator_count=8,
        )
        with pytest.raises(ValueError, match="accelerator_count.*cannot be set"):
            cfg.validate()

    def test_requires_base_model(self):
        cfg = TrainerJobConfig(base_model="")
        with pytest.raises(ValueError, match="base_model is required"):
            cfg.validate()

    def test_manual_path_accepts_all_fields(self):
        cfg = TrainerJobConfig(
            base_model="accounts/fw/models/qwen3-8b",
            accelerator_count=8,
            accelerator_type="NVIDIA_B200",
            custom_image_tag="my-tag",
            node_count=1,
        )
        cfg.validate()  # should not raise
