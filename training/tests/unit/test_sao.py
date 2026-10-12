"""SAO's DIS actor objective and adaptive GAE.

The oracles are small, independent reference implementations of SAO's DIS
loss (arXiv 2607.07508, Eqs. 1-3) and of an action-only GAE scan, so the
library's loss and advantages are checked against separately written math.
"""

from __future__ import annotations

import math

import pytest
import torch

from training.utils.rl.algorithm.sao import (
    adaptive_gae_lambdas,
    behavior_logprob_rows,
    make_dis_loss_fn,
)
from training.utils.rl.critic import token_gae


def oracle_dis_loss(current, behavior, advantages, mask, *, eps_low, eps_high):
    lp, old, adv = current[mask], behavior[mask], advantages[mask].detach()
    delta = lp.detach().double() - old.detach().double()
    keep = (delta > math.log1p(-eps_low)) & (delta < math.log1p(eps_high))
    weights = torch.zeros_like(delta)
    weights[keep] = delta[keep].exp()
    return -(weights.to(lp.dtype) * adv * lp).mean()


def oracle_gae(rewards, values, mask, *, gamma, lam):
    active = mask.nonzero().flatten().tolist()
    r = rewards.double().tolist()
    v = values.double().tolist()
    adv = [0.0] * len(r)
    next_value, next_adv = 0.0, 0.0
    for t in reversed(active):
        delta = r[t] + gamma * next_value - v[t]
        next_adv = delta + gamma * lam * next_adv
        adv[t] = next_adv
        next_value = v[t]
    return torch.tensor(adv, dtype=torch.float64)


def test_dis_loss_and_gradient_match_reference() -> None:
    generator = torch.Generator().manual_seed(0)
    masks = [
        torch.tensor([False, True, True, True, True, True]),
        torch.tensor([False, False, True, True]),
    ]
    behaviors = [
        torch.randn(6, generator=generator) - 2,
        torch.randn(4, generator=generator) - 2,
    ]
    # Push some tokens outside (0.7, 6) in both directions.
    currents = [
        behaviors[0] + torch.tensor([0.0, 0.1, -0.5, 2.5, -0.01, 0.3]),
        behaviors[1] + torch.tensor([0.0, 0.0, 1.0, -1.0]),
    ]
    advantages = [
        torch.tensor([0.0, 1.0, -2.0, 0.5, 3.0, -1.0]),
        torch.tensor([0.0, 0.0, -0.5, 2.0]),
    ]
    recipe_inputs = [c.clone().requires_grad_() for c in currents]
    loss_fn = make_dis_loss_fn(
        behavior_logprobs=behaviors,
        advantages=advantages,
        action_masks=masks,
        dis_low=0.3,
        dis_high=5.0,
    )
    loss, metrics = loss_fn([], recipe_inputs)
    loss.backward()

    oracle_inputs = [c.clone().requires_grad_() for c in currents]
    oracle = torch.stack(
        [
            oracle_dis_loss(c, b, a, m, eps_low=0.3, eps_high=5.0)
            for c, b, a, m in zip(oracle_inputs, behaviors, advantages, masks)
        ]
    ).mean()
    oracle.backward()

    assert torch.allclose(loss, oracle)
    for mine, theirs in zip(recipe_inputs, oracle_inputs):
        assert torch.allclose(mine.grad, theirs.grad)
    assert 0 < metrics["actor/rejected_token_fraction"] < 1


def test_dis_rejects_both_advantage_signs_and_keeps_denominator() -> None:
    mask = torch.tensor([True, True, True, True])
    # ratios: 1, e^-1 (< 0.7, rejected), e^2 (> 6, rejected), 1
    current = torch.tensor([0.0, -1.0, 2.0, 0.0], requires_grad=True)
    loss_fn = make_dis_loss_fn(
        behavior_logprobs=[torch.zeros(4)],
        advantages=[torch.tensor([1.0, 1.0, -1.0, -1.0])],
        action_masks=[mask],
        dis_low=0.3,
        dis_high=5.0,
    )

    loss, metrics = loss_fn([], [current])
    loss.backward()

    # Stop-gradient weight: dL/dlogp = -w * A / N with N = 4 (rejected kept).
    assert torch.allclose(current.grad, torch.tensor([-0.25, 0.0, 0.0, 0.25]))
    assert metrics["actor/rejected_token_fraction"] == 0.5


def test_dis_ratio_uses_sampler_logprobs() -> None:
    loss_fn = make_dis_loss_fn(
        behavior_logprobs=[torch.tensor([-1.2, -0.9])],
        advantages=[torch.tensor([1.0, 1.0])],
        action_masks=[torch.tensor([True, True])],
        dis_low=0.3,
        dis_high=5.0,
    )

    _loss, metrics = loss_fn([], [torch.tensor([-1.0, -1.0], requires_grad=True)])

    assert metrics["actor/behavior_logprob_mean_abs_delta"] == pytest.approx(0.15)
    assert metrics["actor/behavior_logprob_max_abs_delta"] == pytest.approx(0.2)


@pytest.mark.parametrize(("dis_low", "dis_high"), [(1.0, 5.0), (0.3, 0.0), (0.3, math.inf)])
def test_dis_rejects_invalid_bounds(dis_low, dis_high) -> None:
    with pytest.raises(ValueError, match="dis_low"):
        make_dis_loss_fn(
            behavior_logprobs=[],
            advantages=[],
            action_masks=[],
            dis_low=dis_low,
            dis_high=dis_high,
        )


def test_behavior_logprob_rows_validate_alignment_and_finiteness() -> None:
    masks = [torch.tensor([False, True])]
    assert torch.equal(
        behavior_logprob_rows([[None, -0.5]], masks)[0], torch.tensor([0.0, -0.5])
    )
    with pytest.raises(ValueError, match="sampler logprobs for every datum"):
        behavior_logprob_rows([], masks)
    with pytest.raises(ValueError, match="1 sampler logprobs for 2 target"):
        behavior_logprob_rows([[0.0]], masks)
    with pytest.raises(ValueError, match="non-finite"):
        behavior_logprob_rows([[0.0, float("nan")]], masks)
    with pytest.raises(ValueError, match="non-finite"):
        behavior_logprob_rows([[0.0, None]], masks)


def test_adaptive_lambda_per_response_length() -> None:
    masks = [torch.tensor([False] + [True] * 10), torch.tensor([True] * 16384)]

    lambdas = adaptive_gae_lambdas(masks, alpha=1.5)

    assert lambdas[0] == pytest.approx(1 - 1 / 15)
    assert lambdas[1] == pytest.approx(1 - 1 / (1.5 * 16384))
    with pytest.raises(ValueError, match="alpha"):
        adaptive_gae_lambdas(masks, alpha=0.0)


def test_long_horizon_gae_matches_float64_oracle() -> None:
    length = 16384
    mask = torch.ones(length, dtype=torch.bool)
    mask[:100] = False
    values = torch.linspace(0.1, 0.9, length, dtype=torch.float32)
    lam = adaptive_gae_lambdas([mask], alpha=1.5)[0]
    rewards = torch.zeros(length)
    rewards[-1] = 1.0

    advantages, returns = token_gae([1.0], [values], [mask], gamma=1.0, lam=[lam])
    _, critic_returns = token_gae([1.0], [values], [mask], gamma=1.0, lam=1.0)

    expected = oracle_gae(rewards, values, mask, gamma=1.0, lam=lam)
    assert torch.allclose(advantages[0].double(), expected, atol=1e-6)
    # gamma = lambda_critic = 1: every action token's target is the reward.
    assert torch.equal(critic_returns[0][mask], torch.ones(int(mask.sum())))
