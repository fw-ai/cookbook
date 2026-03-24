"""Tests for TrainerJobConfig profile auto-fill."""

import logging
import pytest

from fireworks.training.sdk.trainer import TrainerJobConfig, TrainingShapeProfile


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


class TestTrainerJobConfigValidateWithProfile:
    def test_autofills_base_model_from_profile(self):
        profile = _make_profile(base_model="accounts/fw/models/qwen3-8b")
        cfg = TrainerJobConfig(profile=profile)
        cfg.validate()
        assert cfg.base_model == "accounts/fw/models/qwen3-8b"

    def test_autofills_training_shape_ref_from_profile(self):
        profile = _make_profile()
        cfg = TrainerJobConfig(profile=profile)
        cfg.validate()
        assert cfg.training_shape_ref == profile.training_shape_version

    def test_profile_overrides_explicit_base_model_with_warning(self, caplog):
        profile = _make_profile(base_model="accounts/fw/models/qwen3-8b")
        with caplog.at_level(logging.WARNING):
            cfg = TrainerJobConfig(profile=profile, base_model="accounts/fw/models/other")
            cfg.validate()
        assert cfg.base_model == "accounts/fw/models/qwen3-8b"
        assert "shape-owned" in caplog.text
        assert "will become an error" in caplog.text

    def test_rejects_accelerator_count_with_profile(self):
        profile = _make_profile()
        cfg = TrainerJobConfig(profile=profile, accelerator_count=8)
        with pytest.raises(ValueError, match="accelerator_count.*cannot be set"):
            cfg.validate()

    def test_rejects_custom_image_tag_with_profile(self):
        profile = _make_profile()
        cfg = TrainerJobConfig(profile=profile, custom_image_tag="my-tag")
        with pytest.raises(ValueError, match="custom_image_tag.*cannot be set"):
            cfg.validate()

    def test_requires_base_model_without_profile(self):
        cfg = TrainerJobConfig()
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
