from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.examples.serverless_rl.countdown_rl import (
    Config,
    _group_relative_advantages,
    _mean_policy_sample_logprob_gap,
    _serverless_base_url,
    _validate_config,
    _validate_length,
)


def test_serverless_defaults_use_supported_model_and_matching_tokenizer():
    cfg = Config()
    assert cfg.base_model == "accounts/fireworks/models/kimi-k3"
    assert cfg.tokenizer_model == "moonshotai/Kimi-K3"
    # renderer_name defaults to "" and is auto-resolved from the tokenizer.
    assert cfg.renderer_name == ""
    assert cfg.max_seq_len > 0
    assert cfg.lora_rank > 0


def test_renderer_name_auto_resolves_from_tokenizer():
    from training.examples.serverless_rl.countdown_rl import resolve_renderer_name

    assert resolve_renderer_name("moonshotai/Kimi-K3", "") == "kimi_k3"
    assert resolve_renderer_name("deepseek-ai/DeepSeek-V4-Flash-0731", "") == "deepseek_v4"
    # An explicit override is honored.
    assert resolve_renderer_name("moonshotai/Kimi-K3", "kimi_k3_disable_thinking") == "kimi_k3_disable_thinking"


@pytest.mark.parametrize("max_seq_len", [0, -1])
def test_serverless_rejects_invalid_sequence_bound(max_seq_len):
    cfg = Config(max_seq_len=max_seq_len)
    with pytest.raises(ValueError, match="max_seq_len > 0"):
        _validate_config(cfg)


def test_serverless_rejects_non_lora_config():
    cfg = Config(lora_rank=0)
    with pytest.raises(ValueError, match="lora_rank > 0"):
        _validate_config(cfg)


def test_serverless_length_accepts_exact_bound():
    _validate_length("prompt", 1024, 1024)


def test_serverless_length_rejects_overflow():
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        _validate_length("prompt", 1025, 1024)


def test_serverless_base_url_appends_suffix():
    assert _serverless_base_url("https://api.fireworks.ai") == "https://api.fireworks.ai/training/v1/serverless"
    assert _serverless_base_url("https://api.fireworks.ai/training/v1") == "https://api.fireworks.ai/training/v1/serverless"
    assert _serverless_base_url("https://api.fireworks.ai/training/v1/serverless") == "https://api.fireworks.ai/training/v1/serverless"


def test_group_relative_advantages_standardizes():
    advs = _group_relative_advantages([0.0, 1.0])
    assert advs[0] < 0 < advs[1]
    assert abs(advs[0] + advs[1]) < 1e-6


def test_group_relative_advantages_single_sample_is_zero():
    assert _group_relative_advantages([0.7]) == [0.0]


def test_group_relative_advantages_uses_true_std_below_one():
    """Countdown rewards live in [0, 1], so group std is < 1; GRPO must divide by
    the true std, not a 1.0 floor (which would reduce it to mean-centering)."""
    # std of [0.0, 0.2] is 0.1414 (< 1). If floored to 1.0 the spread would be ~0.2.
    advs = _group_relative_advantages([0.0, 0.2])
    assert advs[1] - advs[0] > 1.0  # standardized spread, not the raw 0.2


def test_group_relative_advantages_constant_group_is_zero_not_nan():
    advs = _group_relative_advantages([0.5, 0.5, 0.5])
    assert all(a == 0.0 for a in advs)


class _TensorData:
    def __init__(self, data: list[float]) -> None:
        self.data = data


def test_k1_reads_dict_and_object_outputs_plus_tensordata_inputs():
    """forward_backward may return dict or object rows; Datum stores TensorData."""
    datum = SimpleNamespace(
        loss_fn_inputs={
            "logprobs": _TensorData([-0.1, -0.2, -2.0]),
            "target_tokens": _TensorData([0.0, 0.0, 7.0]),
        }
    )
    dict_fb = SimpleNamespace(loss_fn_outputs=[{"logprobs": [0.0, 0.0, -1.0]}])
    object_fb = SimpleNamespace(
        loss_fn_outputs=[SimpleNamespace(logprobs=_TensorData([0.0, 0.0, -1.0]))]
    )
    assert _mean_policy_sample_logprob_gap([datum], dict_fb) == pytest.approx(1.0)
    assert _mean_policy_sample_logprob_gap([datum], object_fb) == pytest.approx(1.0)


def test_k1_returns_none_when_logprobs_are_absent():
    datum = SimpleNamespace(loss_fn_inputs={"logprobs": [0.0], "target_tokens": [1.0]})
    assert _mean_policy_sample_logprob_gap([datum], SimpleNamespace(loss_fn_outputs=[{}])) is None
    assert _mean_policy_sample_logprob_gap([datum], SimpleNamespace(loss_fn_outputs=[])) is None
