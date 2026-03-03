"""Tests for GRPO streaming config defaults."""

from __future__ import annotations

from fireworks.training.cookbook.recipes.rl_loop import Config


class TestConfigDefaults:
    def test_defaults(self):
        cfg = Config()
        assert cfg.completions_per_prompt == 4
        assert cfg.prompt_groups_per_step == 1
        assert cfg.max_samples_per_fwd_bwd == 256
        assert cfg.min_samples_per_fwd_bwd is None

    def test_custom_values(self):
        cfg = Config(
            completions_per_prompt=8,
            prompt_groups_per_step=16,
            min_samples_per_fwd_bwd=32,
            max_samples_per_fwd_bwd=256,
        )
        assert cfg.completions_per_prompt == 8
        assert cfg.prompt_groups_per_step == 16
        assert cfg.min_samples_per_fwd_bwd == 32
        assert cfg.max_samples_per_fwd_bwd == 256
