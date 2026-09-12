"""Unit tests for the GSPO sequence-level policy ratio."""

from __future__ import annotations

import math

import pytest
import torch
import tinker

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
    assert metrics["gspo_sequence_ratio_mean"] == pytest.approx(expected_sequence_ratio)
    assert metrics["ppo_clip_frac"] == 0.0
    assert metrics["gspo_clip_frac"] == 0.0
    assert loss.item() == pytest.approx(-expected_sequence_ratio)
    assert pi.grad is not None
    assert pi.grad.tolist() == pytest.approx(
        [-expected_sequence_ratio / 2, -expected_sequence_ratio / 2]
    )


def test_gspo_matches_paper_for_unequal_response_lengths() -> None:
    """Each sequence has equal weight, independent of its token count."""
    short = torch.zeros(2, requires_grad=True)
    long = torch.zeros(4, requires_grad=True)
    loss_fn = make_gspo_loss_fn(
        advantages=[1.0, 1.0],
        ref_logprobs=[],
        prompt_len=[1, 1],
        inf_logprobs=[[0.0] * 2, [0.0] * 4],
        old_policy_logprobs=[[0.0] * 2, [0.0] * 4],
    )

    loss, _ = loss_fn([], [short, long])
    # The trainer's num_sequences normalization divides this raw sum by two.
    (loss / 2).backward()

    assert loss.item() == pytest.approx(-2.0)
    assert short.grad is not None
    assert long.grad is not None
    assert short.grad.tolist() == pytest.approx([-0.25, -0.25])
    assert long.grad.tolist() == pytest.approx([-0.125] * 4)
    assert sum(abs(value) for value in short.grad.tolist()) == pytest.approx(0.5)
    assert sum(abs(value) for value in long.grad.tolist()) == pytest.approx(0.5)


def test_gspo_sequence_ratio_and_token_mean_ignore_masked_positions() -> None:
    """Tool and user spans do not affect the GSPO sequence objective."""
    old_policy = [0.0, 0.0, 0.0]
    pi = torch.tensor(
        [math.log(1.1), math.log(100.0), math.log(0.9)],
        requires_grad=True,
    )
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=[2, 3, 4], dtype="int64", shape=[3]
            ),
            "weights": tinker.TensorData(
                data=[1.0, 0.0, 1.0], dtype="float32", shape=[3]
            ),
        },
    )
    loss_fn = make_gspo_loss_fn(
        advantages=[1.0],
        ref_logprobs=[],
        prompt_len=1,
        inf_logprobs=[old_policy],
        old_policy_logprobs=[old_policy],
    )

    loss, metrics = loss_fn([datum], [pi])
    loss.backward()

    expected_sequence_ratio = math.sqrt(1.1 * 0.9)
    assert metrics["ppo_ratio_mean"] == pytest.approx(expected_sequence_ratio)
    assert loss.item() == pytest.approx(-expected_sequence_ratio)
    assert pi.grad is not None
    assert pi.grad.tolist() == pytest.approx(
        [-expected_sequence_ratio / 2, 0.0, -expected_sequence_ratio / 2]
    )


@pytest.mark.parametrize(
    ("ratio", "expected_low", "expected_high"),
    [
        (0.9996, 1.0, 0.0),
        (0.9998, 0.0, 0.0),
        (1.0003, 0.0, 0.0),
        (1.0005, 0.0, 1.0),
    ],
)
def test_gspo_reports_paper_asymmetric_clip_fractions(
    ratio: float, expected_low: float, expected_high: float
) -> None:
    pi = torch.tensor([math.log(ratio)], requires_grad=True)
    loss_fn = make_gspo_loss_fn(
        advantages=[1.0],
        ref_logprobs=[],
        prompt_len=1,
        inf_logprobs=[[0.0]],
        old_policy_logprobs=[[0.0]],
        gspo_config=GSPOConfig(
            clip_ratio_low=3e-4,
            clip_ratio_high=4e-4,
        ),
    )

    _, metrics = loss_fn([], [pi])

    assert metrics["gspo_clip_low_frac"] == expected_low
    assert metrics["gspo_clip_high_frac"] == expected_high
    assert metrics["gspo_clip_frac"] == expected_low + expected_high
    assert metrics["ppo_clip_frac"] == metrics["gspo_clip_frac"]


def test_gspo_reports_raw_inference_k3_without_changing_the_loss() -> None:
    pi_values = [-1.0, -1.2]
    raw_inf_values = [-1.1, -1.0]
    pi = torch.tensor(pi_values, requires_grad=True)
    loss_fn = make_gspo_loss_fn(
        advantages=[1.0],
        ref_logprobs=[],
        prompt_len=1,
        inf_logprobs=[pi_values],
        old_policy_logprobs=[pi_values],
        raw_inf_logprobs=[raw_inf_values],
    )

    loss, metrics = loss_fn([], [pi])

    diff = torch.tensor(pi_values) - torch.tensor(raw_inf_values)
    expected_k3 = (torch.exp(diff) - diff - 1.0).mean().item()
    assert metrics["raw_inference_logprob_coverage"] == 1.0
    assert metrics["inference_k1"] == pytest.approx(diff.mean().item())
    assert metrics["inference_k3"] == pytest.approx(expected_k3)
    assert metrics["inference_kld"] == pytest.approx(expected_k3)
    assert loss.item() == pytest.approx(-1.0)
