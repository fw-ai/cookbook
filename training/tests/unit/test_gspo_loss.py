"""Unit tests for the GSPO sequence-level policy ratio."""

from __future__ import annotations

import math

import pytest
import torch

from training.utils.rl.gspo import GSPOConfig, make_gspo_loss_fn


def test_gspo_uses_one_geometric_mean_ratio_for_the_sequence() -> None:
    old_policy = [0.0, 0.0]
    pi_values = [math.log(1.1), math.log(0.9)]
    expected_sequence_ratio = math.sqrt(1.1 * 0.9)
    pi = torch.tensor(pi_values, requires_grad=True)
    loss_fn = make_gspo_loss_fn(
        advantages=[1.0],
        ref_logprobs=[],
        prompt_len=1,
        inf_logprobs=[old_policy],
        old_policy_logprobs=[old_policy],
        gspo_config=GSPOConfig(clip_ratio_low=0.2, clip_ratio_high=0.2),
    )

    loss, metrics = loss_fn([], [pi])
    loss.backward()

    # GSPO uses exp(mean(log(pi / pi_old))) for every token in the sequence.
    # Per-token GRPO would instead use the distinct ratios 1.1 and 0.9.
    assert metrics["ppo_ratio_mean"] == pytest.approx(expected_sequence_ratio)
    assert loss.item() == pytest.approx(-2.0 * expected_sequence_ratio)
    assert pi.grad is not None
    assert pi.grad.tolist() == pytest.approx(
        [-expected_sequence_ratio, -expected_sequence_ratio]
    )
