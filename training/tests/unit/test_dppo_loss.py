"""Tests for Divergence Proximal Policy Optimization."""

from __future__ import annotations

import math

import pytest
import tinker
import torch

from training.utils.rl.dppo import DPPOConfig, make_dppo_loss_fn


def _datum() -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=[1], dtype="int64", shape=[1]
            ),
            "weights": tinker.TensorData(
                data=[1], dtype="int64", shape=[1]
            ),
        },
    )


def _run(
    *,
    policy_prob: float,
    behavior_prob: float,
    advantage: float,
    config: DPPOConfig | None = None,
):
    policy_logprob = torch.tensor(
        [math.log(policy_prob)], requires_grad=True
    )
    behavior_logprob = math.log(behavior_prob)
    loss_fn = make_dppo_loss_fn(
        advantages=[advantage],
        ref_logprobs=[],
        inf_logprobs=[[behavior_logprob]],
        prompt_len=1,
        old_policy_logprobs=[[behavior_logprob]],
        dppo_config=config,
    )
    loss, metrics = loss_fn([_datum()], [policy_logprob])
    loss.backward()
    return loss.detach(), policy_logprob.grad, metrics


@pytest.mark.parametrize(
    ("policy_prob", "behavior_prob", "advantage"),
    [
        (0.30, 0.10, 1.0),
        (0.10, 0.30, -1.0),
    ],
)
def test_binary_tv_masks_large_advantage_improving_updates(
    policy_prob: float,
    behavior_prob: float,
    advantage: float,
) -> None:
    loss, grad, metrics = _run(
        policy_prob=policy_prob,
        behavior_prob=behavior_prob,
        advantage=advantage,
    )

    assert loss.item() == pytest.approx(0.0)
    assert grad.item() == pytest.approx(0.0)
    assert metrics["dppo_mask_frac"] == pytest.approx(1.0)
    assert metrics["dppo_divergence_mean"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("policy_prob", "behavior_prob", "advantage"),
    [
        (0.10, 0.30, 1.0),
        (0.30, 0.10, -1.0),
    ],
)
def test_binary_tv_preserves_large_updates_in_non_improving_direction(
    policy_prob: float,
    behavior_prob: float,
    advantage: float,
) -> None:
    loss, grad, metrics = _run(
        policy_prob=policy_prob,
        behavior_prob=behavior_prob,
        advantage=advantage,
    )
    ratio = policy_prob / behavior_prob

    assert loss.item() == pytest.approx(-ratio * advantage)
    assert grad.item() == pytest.approx(-ratio * advantage)
    assert metrics["dppo_mask_frac"] == pytest.approx(0.0)


def test_binary_kl_is_finite_and_can_mask() -> None:
    loss, grad, metrics = _run(
        policy_prob=0.8,
        behavior_prob=0.2,
        advantage=1.0,
        config=DPPOConfig(divergence="binary_kl", threshold=0.05),
    )

    assert loss.item() == pytest.approx(0.0)
    assert grad.item() == pytest.approx(0.0)
    assert metrics["dppo_divergence_mean"] > 0.05
