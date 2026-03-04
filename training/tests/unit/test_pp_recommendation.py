"""Tests for PP batch recommendation computation."""

from __future__ import annotations

from fireworks.training.sdk.trainer import TrainingShapeProfile
from training.utils.rl.pp import compute_pp_recommendation


def _make_profile(**overrides) -> TrainingShapeProfile:
    defaults = dict(
        training_shape_version="v1",
        trainer_image_tag="0.33.0",
        max_supported_context_length=4096,
        node_count=1,
        deployment_shape_version="dsv",
        deployment_image_tag="4.24.22",
        accelerator_type="NVIDIA_B200_180GB",
        accelerator_count=8,
        base_model_weight_precision="bfloat16",
        pipeline_parallelism=1,
    )
    defaults.update(overrides)
    return TrainingShapeProfile(**defaults)


class TestComputePPRecommendation:
    def test_no_pp(self):
        rec = compute_pp_recommendation(_make_profile(pipeline_parallelism=1), group_size=4)
        assert rec.pp_degree == 1
        assert rec.bubble_ratio == 0.0
        assert rec.recommended_prompts_per_step == 1

    def test_pp4_group4_ctx4096(self):
        rec = compute_pp_recommendation(
            _make_profile(pipeline_parallelism=4, max_supported_context_length=4096),
            group_size=4,
        )
        assert rec.local_batch_size == 16  # max(4, 65536//4096)
        assert rec.pp_degree == 4
        assert rec.recommended_group_size == 16
        assert rec.recommended_prompts_per_step == 4  # 16 // 4
        assert 0.15 < rec.bubble_ratio < 0.17  # 3/19 ≈ 0.158

    def test_pp4_group16(self):
        rec = compute_pp_recommendation(
            _make_profile(pipeline_parallelism=4, max_supported_context_length=4096),
            group_size=16,
        )
        assert rec.recommended_prompts_per_step == 1

    def test_pp8_large_context(self):
        rec = compute_pp_recommendation(
            _make_profile(pipeline_parallelism=8, max_supported_context_length=16384),
            group_size=4,
        )
        assert rec.local_batch_size == 8  # max(8, 65536//16384=4) → 8
        assert rec.recommended_prompts_per_step == 2  # 8 // 4

    def test_custom_max_batch_tokens(self):
        rec = compute_pp_recommendation(
            _make_profile(pipeline_parallelism=4, max_supported_context_length=4096),
            group_size=4,
            max_batch_size_tokens=16384,
        )
        assert rec.local_batch_size == 4  # max(4, 16384//4096=4)
        assert rec.recommended_prompts_per_step == 1  # 4 // 4
