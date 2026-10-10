"""Rollout-denominator IGPO references, including masked tool tokens."""

import math

import pytest
import tinker
import torch

from training.utils.rl.algorithm.igpo import make_igpo_loss_fn


def test_rollout_ratio_and_per_token_advantages_match_loss_and_gradient():
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([10, 11, 12]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[11, 12, 13], dtype="int64", shape=[3]),
            "weights": tinker.TensorData(data=[1, 0, 1], dtype="int64", shape=[3]),
        },
    )
    policy = torch.tensor([-1.1, -10000.0, -1.0], requires_grad=True)
    loss_fn = make_igpo_loss_fn(
        per_token_advantages=[[2.0, -10.0, -1.0]],
        ref_logprobs=[],
        prompt_lens=[1],
        inf_logprobs=[[-1.0, None, -0.9]],
        kl_beta=0.0,
    )
    loss, metrics = loss_fn([datum], [policy])
    loss.backward()
    ratio = math.exp(-0.1)
    assert loss.item() == pytest.approx(-ratio)
    torch.testing.assert_close(policy.grad, torch.tensor([-2 * ratio, 0.0, ratio]))
    assert metrics["active_tokens"] == 2


@pytest.mark.parametrize("logprobs", [[], [-1.0], [None, -0.9]])
def test_missing_active_rollout_logprobs_are_rejected(logprobs):
    loss_fn = make_igpo_loss_fn(
        per_token_advantages=[[1.0, 1.0]],
        ref_logprobs=[],
        prompt_lens=[1],
        inf_logprobs=[logprobs],
        kl_beta=0.0,
    )
    with pytest.raises(ValueError, match="rollout_logprobs"):
        loss_fn([], [torch.tensor([-1.0, -0.9], requires_grad=True)])
