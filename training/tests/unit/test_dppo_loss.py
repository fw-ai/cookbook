"""Tests for Divergence Proximal Policy Optimization."""

from __future__ import annotations

import math

import pytest
import torch
from fireworks.training import sdk
from training.utils.rl.algorithm.dppo import DPPOConfig, make_loss_fn
from training.utils.rl.losses import PolicyBatch


def _datum() -> sdk.Datum:
    return sdk.Datum(
        model_input=sdk.ModelInput.from_ints([1]),
        loss_fn_inputs={
            "target_tokens": sdk.TensorData(data=[1], dtype="int64", shape=[1]),
            "weights": sdk.TensorData(data=[1], dtype="int64", shape=[1]),
        },
    )


def _run(
    *,
    policy_prob: float,
    behavior_prob: float,
    advantage: float,
    config: DPPOConfig | None = None,
    rollout_prob: float | None = None,
):
    """One-token DPPO step; ``behavior_prob`` is the anchor, ``rollout_prob`` the sampler."""
    policy_logprob = torch.tensor([math.log(policy_prob)], requires_grad=True)
    anchor = [[math.log(behavior_prob)]]
    rollout = anchor if rollout_prob is None else [[math.log(rollout_prob)]]
    batch = PolicyBatch(
        data=[_datum()],
        advantages=[advantage],
        prompt_lens=[1],
        rollout_logprobs=rollout,
        anchor_logprobs=anchor,
    )
    loss_fn = make_loss_fn(batch, config or DPPOConfig())
    loss, metrics = loss_fn(batch.data, [policy_logprob])
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
    assert metrics["dppo_clip_frac"] == pytest.approx(1.0)
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
    # ``dppo_clip_frac`` counts tokens past the threshold, masked or not.
    assert metrics["dppo_clip_frac"] == pytest.approx(1.0)


def test_binary_kl_is_finite_and_can_mask() -> None:
    loss, grad, metrics = _run(
        policy_prob=0.8,
        behavior_prob=0.2,
        advantage=1.0,
        config=DPPOConfig(divergence="binary_kl"),
    )

    assert loss.item() == pytest.approx(0.0)
    assert grad.item() == pytest.approx(0.0)
    assert metrics["dppo_divergence_mean"] > 0.05


def test_divergence_specific_default_thresholds() -> None:
    assert DPPOConfig(divergence="binary_tv").threshold == 0.15
    assert DPPOConfig(divergence="binary_kl").threshold == 0.05

    tv_loss, _, _ = _run(
        policy_prob=0.24,
        behavior_prob=0.1,
        advantage=1.0,
        config=DPPOConfig(divergence="binary_tv"),
    )
    kl_loss, _, _ = _run(
        policy_prob=0.24,
        behavior_prob=0.1,
        advantage=1.0,
        config=DPPOConfig(divergence="binary_kl"),
    )

    assert tv_loss.item() != pytest.approx(0.0)
    assert kl_loss.item() == pytest.approx(0.0)


def test_anchor_is_the_ratio_denominator_not_the_sampler() -> None:
    # Anchor 0.2 equals the policy, so nothing is masked even though the
    # sampler (0.01) is far away.
    loss, grad, metrics = _run(
        policy_prob=0.2,
        behavior_prob=0.2,
        rollout_prob=0.01,
        advantage=1.0,
    )
    assert loss.item() == pytest.approx(-1.0)
    assert grad.item() == pytest.approx(-1.0)
    assert metrics["dppo_clip_frac"] == pytest.approx(0.0)

    # With the sampler as the anchor the same update is masked.
    loss, grad, metrics = _run(
        policy_prob=0.2,
        behavior_prob=0.01,
        advantage=1.0,
    )
    assert loss.item() == pytest.approx(0.0)
    assert grad.item() == pytest.approx(0.0)
    assert metrics["dppo_clip_frac"] == pytest.approx(1.0)
