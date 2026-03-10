"""Tests for GRPO config defaults."""

from __future__ import annotations

from training.recipes.rl_loop import Config


class TestConfigDefaults:
    def test_defaults(self):
        cfg = Config(log_path="/tmp/test")
        assert cfg.completions_per_prompt == 4
        assert cfg.prompt_groups_per_step == 1

    def test_custom_values(self):
        cfg = Config(
            log_path="/tmp/test",
            completions_per_prompt=8,
            prompt_groups_per_step=16,
        )
        assert cfg.completions_per_prompt == 8
        assert cfg.prompt_groups_per_step == 16
