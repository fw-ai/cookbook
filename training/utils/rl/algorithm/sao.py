"""SAO actor objective: decoupled importance sampling with adaptive GAE.

Single-rollout Asynchronous Optimization (arXiv 2607.07508) trains a
token-level actor against a separately trained critic
(:mod:`training.utils.rl.critic`). Per token the actor loss is
``-w * A * log pi`` with ``w = stopgrad(ratio * 1[1 - dis_low < ratio < 1 + dis_high])``
and ``ratio = pi / pi_sampler``: the anchor is the sampler's recorded
logprobs. Rejected tokens stay in each response's mean. The advantages are
token GAE with a per-response ``lambda = 1 - 1 / (alpha * T)``.

The trainer has no built-in DIS loss, so it runs as a client loss::

    loss_fn = make_dis_loss_fn(
        behavior_logprobs=behavior_logprob_rows(batch.rollout_logprobs, masks),
        advantages=advantages,
        action_masks=masks,
        dis_low=0.3,
        dis_high=5.0,
    )
    result = actor.forward_backward_custom(batch.data, loss_fn)
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch


def adaptive_gae_lambdas(
    action_masks: Sequence[torch.Tensor],
    *,
    alpha: float,
) -> List[float]:
    """Per-response ``lambda = 1 - 1 / (alpha * T)`` for T action tokens.

    It keeps about ``exp(-1 / alpha)`` of the terminal reward's direct weight
    across a whole response, where a fixed 0.95 decays to ~0 after a few
    hundred tokens.
    """
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Adaptive GAE alpha must be positive and finite.")
    lambdas = []
    for mask in action_masks:
        count = int(mask.sum())
        if count < 1:
            raise ValueError("Adaptive GAE needs at least one action token.")
        lambdas.append(max(0.0, 1.0 - 1.0 / (alpha * count)))
    return lambdas


def behavior_logprob_rows(
    rollout_logprobs: Sequence[Sequence[float | None]],
    action_masks: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Sampler logprobs aligned to ``target_tokens``, finite on action tokens."""
    if len(rollout_logprobs) != len(action_masks):
        raise ValueError(
            "DIS needs sampler logprobs for every datum; got "
            f"{len(rollout_logprobs)} rows for {len(action_masks)} datums."
        )
    rows = []
    for index, (row, mask) in enumerate(zip(rollout_logprobs, action_masks)):
        tensor = torch.tensor(
            [math.nan if value is None else float(value) for value in row],
            dtype=torch.float32,
        )
        if tensor.shape != mask.shape:
            raise ValueError(
                f"Datum {index} has {tensor.numel()} sampler logprobs for "
                f"{mask.numel()} target positions."
            )
        if not torch.isfinite(tensor[mask]).all():
            raise ValueError(f"Datum {index} has non-finite sampler logprobs.")
        rows.append(torch.where(mask, tensor, torch.zeros_like(tensor)))
    return rows


def validate_dis_bounds(dis_low: float, dis_high: float) -> None:
    if not 0 <= dis_low < 1 or not math.isfinite(dis_high) or dis_high <= 0:
        raise ValueError("DIS needs 0 <= dis_low < 1 and a positive finite dis_high.")


def make_dis_loss_fn(
    *,
    behavior_logprobs: Sequence[torch.Tensor],
    advantages: Sequence[torch.Tensor],
    action_masks: Sequence[torch.Tensor],
    dis_low: float,
    dis_high: float,
):
    """DIS loss for ``forward_backward_custom``: response means, then batch mean.

    Tokens outside ``(1 - dis_low, 1 + dis_high)`` are rejected for either
    advantage sign and stay in the per-response mean's denominator. The weight
    is not differentiated.
    """
    validate_dis_bounds(dis_low, dis_high)
    log_low = math.log1p(-dis_low)
    log_high = math.log1p(dis_high)

    def loss_fn(_data, current_logprobs):
        if not (
            len(current_logprobs)
            == len(behavior_logprobs)
            == len(advantages)
            == len(action_masks)
        ):
            raise ValueError("DIS policy inputs must align by response.")
        losses = []
        accepted = 0
        tokens = 0
        abs_delta_sum = 0.0
        max_abs_delta = 0.0
        for current, behavior, advantage, mask in zip(
            current_logprobs, behavior_logprobs, advantages, action_masks
        ):
            mask = mask.to(device=current.device)
            if not (current.shape == behavior.shape == advantage.shape == mask.shape):
                raise ValueError("DIS token arrays must align with each datum.")
            current_active = current[mask]
            behavior_active = behavior.to(device=current.device)[mask]
            advantage_active = advantage.to(
                device=current.device, dtype=current.dtype
            )[mask].detach()
            if not (
                torch.isfinite(current_active).all()
                and torch.isfinite(behavior_active).all()
                and torch.isfinite(advantage_active).all()
            ):
                raise ValueError("DIS action logprobs and advantages must be finite.")
            delta = current_active.detach().double() - behavior_active.double()
            keep = (delta > log_low) & (delta < log_high)
            # Exponentiate only accepted tokens so stale outliers cannot overflow.
            weights = torch.zeros_like(delta)
            weights[keep] = delta[keep].exp()
            losses.append(
                -(weights.to(current.dtype) * advantage_active * current_active).mean()
            )
            accepted += int(keep.sum())
            tokens += int(delta.numel())
            abs_delta_sum += float(delta.abs().sum())
            max_abs_delta = max(max_abs_delta, float(delta.abs().max()))
        loss = torch.stack(losses).mean()
        metrics = {
            "actor/loss": float(loss.detach()),
            "actor/accepted_fraction": accepted / tokens,
            "actor/rejected_token_fraction": 1.0 - accepted / tokens,
            "actor/behavior_logprob_mean_abs_delta": abs_delta_sum / tokens,
            "actor/behavior_logprob_max_abs_delta": max_abs_delta,
        }
        return loss, metrics

    return loss_fn
