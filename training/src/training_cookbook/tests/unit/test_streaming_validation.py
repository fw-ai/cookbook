"""Tests for streaming config validation helpers."""

from __future__ import annotations

import pytest

from training_cookbook.utils.validation import validate_streaming_config


class TestValidateStreamingConfig:
    def test_valid_config(self):
        validate_streaming_config(
            prompt_groups_per_step=16,
            completions_per_prompt=8,
            min_samples_per_fwd_bwd=32,
            max_samples_per_fwd_bwd=256,
        )

    def test_min_batch_divisibility_error(self):
        with pytest.raises(
            ValueError,
            match="min_samples_per_fwd_bwd \\(30\\) should be divisible by completions_per_prompt \\(8\\)",
        ):
            validate_streaming_config(
                prompt_groups_per_step=16,
                completions_per_prompt=8,
                min_samples_per_fwd_bwd=30,
                max_samples_per_fwd_bwd=256,
            )
