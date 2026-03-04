"""Tests for streaming config validation helpers."""

from __future__ import annotations

import pytest

from training.utils.validation import validate_streaming_config


class TestValidateStreamingConfig:
    def test_valid_config(self):
        validate_streaming_config(
            prompt_groups_per_step=16,
            completions_per_prompt=8,
            min_samples_per_fwd_bwd=32,
        )

    def test_valid_without_min(self):
        validate_streaming_config(
            prompt_groups_per_step=4,
            completions_per_prompt=8,
        )

    def test_invalid_completions_per_prompt(self):
        with pytest.raises(ValueError, match="completions_per_prompt must be >= 1"):
            validate_streaming_config(
                prompt_groups_per_step=1,
                completions_per_prompt=0,
            )
