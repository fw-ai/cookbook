"""Tests for cookbook and SDK default values."""

from __future__ import annotations


def test_grpo_temperature():
    from fireworks.training.cookbook.recipes.rl_loop import Config

    assert Config().temperature == 1.0


def test_grpo_max_completion_tokens():
    from fireworks.training.cookbook.recipes.rl_loop import Config

    assert Config().max_completion_tokens == 1024


def test_tis_clip_high():
    from fireworks.training.cookbook.utils.importance_sampling import ISConfig

    assert ISConfig().clip_high == 2.0


def test_cispo_config_defaults():
    from fireworks.training.cookbook.recipes.rl_loop import Config

    cfg = Config()
    assert cfg.cispo.eps_low == 0.2
    assert cfg.cispo.eps_high == 0.28


def test_cispo_is_valid_policy_loss():
    from fireworks.training.cookbook.recipes.rl_loop import Config

    cfg = Config(policy_loss="cispo")
    assert cfg.policy_loss == "cispo"


def test_dpo_has_tokenizer_model():
    from fireworks.training.cookbook.recipes.dpo_loop import Config

    assert hasattr(Config(), "tokenizer_model")
