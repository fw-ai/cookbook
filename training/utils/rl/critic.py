"""Projection-head critic for token-level actor-critic RL.

The critic is its own trainer, created with ``projection_head_dim=1``: a
dedicated projection module returns one value per token and the language-model
head is never used. This module turns policy datums into critic inputs,
computes token-level GAE from terminal rewards, trains the critic with a
clipped value loss through ``forward_backward_custom(..., output="projection")``
and reads values with ``forward_projection``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import tinker
import torch

from training.utils.rl.losses import PolicyBatch
from training.utils.timer import elapsed_timer

CRITIC_PROJECTION_HEAD_DIM = 1
"""Projection width of the critic trainer: one scalar value per token."""


@dataclass(frozen=True)
class CriticBatch:
    """Critic inputs for one rollout batch, aligned by response."""

    data: List[tinker.Datum]
    """Policy model inputs without language-model targets."""
    terminal_rewards: List[float]
    action_masks: List[torch.Tensor]
    """Boolean mask over each datum's target positions: the generated tokens."""


def critic_batch(batch: PolicyBatch) -> CriticBatch:
    """Critic inputs for single-turn responses, one datum per rollout run.

    Each response must be one terminal, contiguous run of generated tokens, so
    the terminal reward lands on its last token.
    """
    rewards: List[float] = []
    for group_index, group in enumerate(batch.groups):
        if len(group.data) != len(group.rewards):
            raise ValueError(
                "The critic expects one datum per rollout run. "
                f"Group {group_index} has {len(group.data)} datums and "
                f"{len(group.rewards)} rewards; multi-segment trajectories "
                "need an explicit segment-to-return mapping."
            )
        rewards.extend(float(reward) for reward in group.rewards)

    data: List[tinker.Datum] = []
    masks: List[torch.Tensor] = []
    for index, datum in enumerate(batch.data):
        weights = datum.loss_fn_inputs.get("weights")
        if weights is None:
            raise ValueError(f"Policy datum {index} is missing response weights.")
        mask = torch.tensor(weights.data, dtype=torch.bool)
        num_targets = len(datum.loss_fn_inputs["target_tokens"].data)
        if mask.numel() != num_targets:
            raise ValueError(
                f"Datum {index} has {mask.numel()} response weights for "
                f"{num_targets} target positions."
            )
        active = mask.nonzero(as_tuple=False).flatten().tolist()
        if not active:
            raise ValueError(f"Datum {index} has no active response tokens.")
        if active != list(range(active[0], num_targets)):
            raise ValueError(
                "The critic needs a terminal, contiguous response mask; "
                f"datum {index} has gaps or masked trailing tokens."
            )
        masks.append(mask)
        data.append(tinker.Datum(model_input=datum.model_input, loss_fn_inputs={}))
    return CriticBatch(data=data, terminal_rewards=rewards, action_masks=masks)


def projection_values(projection: torch.Tensor) -> torch.Tensor:
    """One value per token from a raw ``[tokens, 1]`` projection."""
    if projection.ndim != 2 or projection.shape[1] != CRITIC_PROJECTION_HEAD_DIM:
        raise ValueError(
            f"Critic projection has shape {tuple(projection.shape)}; expected "
            f"[tokens, {CRITIC_PROJECTION_HEAD_DIM}]."
        )
    return projection[:, 0]


def token_gae(
    terminal_rewards: Sequence[float],
    values: Sequence[torch.Tensor],
    action_masks: Sequence[torch.Tensor],
    *,
    gamma: float,
    lam: float | Sequence[float],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Token-level GAE advantages and lambda returns over action tokens.

    The terminal reward arrives on the last action token and the value after
    it is 0. ``lam`` is one lambda or one per response. The backward scan runs
    in float64: at 16K tokens an adaptive lambda is ~0.99996, and a float32
    recurrence loses that precision. Non-action positions get 0.
    """
    if isinstance(lam, (int, float)):
        lambdas = [float(lam)] * len(values)
    else:
        lambdas = [float(value) for value in lam]
    if not (len(terminal_rewards) == len(values) == len(action_masks) == len(lambdas)):
        raise ValueError("Rewards, values, action masks and lambdas must align by response.")
    advantages: List[torch.Tensor] = []
    returns: List[torch.Tensor] = []
    for reward, value, mask, response_lambda in zip(
        terminal_rewards, values, action_masks, lambdas
    ):
        positions = mask.nonzero(as_tuple=False).flatten().tolist()
        if not positions:
            raise ValueError("Every response needs at least one action token.")
        scan = value.detach().double().cpu().tolist()
        if not all(math.isfinite(scan[position]) for position in positions):
            raise ValueError("Critic values must be finite on action tokens.")
        response_advantages = [0.0] * len(scan)
        response_returns = [0.0] * len(scan)
        next_value = next_advantage = 0.0
        for position in reversed(positions):
            step_reward = float(reward) if position == positions[-1] else 0.0
            delta = step_reward + gamma * next_value - scan[position]
            next_advantage = delta + gamma * response_lambda * next_advantage
            response_advantages[position] = next_advantage
            response_returns[position] = next_advantage + scan[position]
            next_value = scan[position]
        advantages.append(value.new_tensor(response_advantages))
        returns.append(value.new_tensor(response_returns))
    return advantages, returns


def make_value_loss_fn(
    batch: CriticBatch,
    *,
    gamma: float,
    critic_lambda: float,
    value_clip: float,
    capture: Dict[str, Any],
    old_values: Sequence[torch.Tensor] | None = None,
):
    """Clipped value loss over raw projection outputs.

    Targets are lambda returns from ``old_values``, which also anchor the
    clip. Without them the call's own predictions are the old values; they are
    stored in ``capture["old_values"]`` so later updates on the same batch keep
    the same anchors and targets. Squared errors are averaged over each
    response's action tokens, then over responses.
    """

    def loss_fn(_data, projections):
        if len(projections) != len(batch.action_masks):
            raise ValueError("Critic projections must align with the batch responses.")
        predicted = []
        for projection, mask in zip(projections, batch.action_masks):
            value = projection_values(projection)
            if value.numel() != mask.numel():
                raise ValueError(
                    f"Critic returned {value.numel()} token rows for a "
                    f"{mask.numel()}-position response."
                )
            predicted.append(value)
        anchors = [
            value.detach() for value in (predicted if old_values is None else old_values)
        ]
        if len(anchors) != len(predicted) or any(
            old.shape != new.shape for old, new in zip(anchors, predicted)
        ):
            raise ValueError("Old values must align with critic predictions.")
        _, returns = token_gae(
            batch.terminal_rewards,
            anchors,
            batch.action_masks,
            gamma=gamma,
            lam=critic_lambda,
        )

        losses, active_values, active_targets = [], [], []
        for value, target, anchor, mask in zip(predicted, returns, anchors, batch.action_masks):
            target = target.to(device=value.device, dtype=value.dtype)
            anchor = anchor.to(device=value.device, dtype=value.dtype)
            mask = mask.to(device=value.device)
            clipped = anchor + torch.clamp(value - anchor, min=-value_clip, max=value_clip)
            error = torch.maximum((value - target).square(), (clipped - target).square())
            losses.append(error[mask].mean())
            active_values.append(value[mask])
            active_targets.append(target[mask])
        loss = torch.stack(losses).mean()

        capture["old_values"] = [anchor.cpu().clone() for anchor in anchors]
        capture["loss"] = float(loss.detach())
        with torch.no_grad():
            metrics = {
                "value/loss": float(loss.detach()),
                "value/prediction_mean": float(torch.cat(active_values).mean()),
                "value/return_mean": float(torch.cat(active_targets).mean()),
            }
        return loss, metrics

    return loss_fn


def predict_values(critic: Any, batch: CriticBatch) -> List[torch.Tensor]:
    """Forward-only read of the critic's current per-token values."""
    result = critic.forward_projection(batch.data)
    if len(result.loss_fn_outputs) != len(batch.data):
        raise ValueError(
            f"Critic returned {len(result.loss_fn_outputs)} projections for "
            f"{len(batch.data)} datums."
        )
    values = []
    for output, mask in zip(result.loss_fn_outputs, batch.action_masks):
        tensor_data = output.get("projection")
        if tensor_data is None:
            raise ValueError("Critic forward response is missing 'projection'.")
        projection = torch.tensor(tensor_data.data, dtype=torch.float32)
        if tensor_data.shape is not None:
            projection = projection.reshape(tensor_data.shape)
        value = projection_values(projection)
        if value.numel() != mask.numel():
            raise ValueError(
                f"Critic returned {value.numel()} token rows for a "
                f"{mask.numel()}-position response."
            )
        values.append(value)
    return values


@dataclass(frozen=True)
class CriticUpdate:
    fwd_bwd_results: List[Any]
    optim_result: Any
    value_losses: List[float]
    """Value loss of each update, measured before its optimizer step."""


def train_critic(
    critic: Any,
    batch: CriticBatch,
    *,
    updates: int,
    gamma: float,
    critic_lambda: float,
    value_clip: float,
    adam_params: tinker.AdamParams,
) -> CriticUpdate:
    """Run ``updates`` clipped value updates on one rollout batch.

    Every update's forward re-reads the critic, so it sees the parameters left
    by the previous update. Clip anchors and return targets stay fixed at the
    first update's values.
    """
    fwd_bwd_results: List[Any] = []
    value_losses: List[float] = []
    old_values = None
    optim_result = None
    for _ in range(updates):
        capture: Dict[str, Any] = {}
        loss_fn = make_value_loss_fn(
            batch,
            gamma=gamma,
            critic_lambda=critic_lambda,
            value_clip=value_clip,
            capture=capture,
            old_values=old_values,
        )
        with elapsed_timer("critic_fwd_bwd"):
            fwd_bwd_results.append(
                critic.forward_backward_custom(batch.data, loss_fn, output="projection")
            )
        with elapsed_timer("critic_optim_step"):
            optim_result = critic.optim_step(adam_params)
        old_values = capture["old_values"]
        value_losses.append(capture["loss"])
    return CriticUpdate(fwd_bwd_results, optim_result, value_losses)


def value_statistics(
    values: Sequence[torch.Tensor],
    terminal_rewards: Sequence[float],
    action_masks: Sequence[torch.Tensor],
    *,
    gamma: float = 1.0,
) -> Dict[str, float]:
    """Monte Carlo calibration of token values against discounted returns.

    Matching :func:`token_gae`, the k-th of n action tokens targets
    ``reward * gamma ** (n - 1 - k)``. Explained variance ignores a constant
    offset, so it is reported with MSE and bias; ``constant_baseline_mse`` is
    the best constant predictor's MSE.
    """
    predictions, targets, sequence_mse = [], [], []
    for value, reward, mask in zip(values, terminal_rewards, action_masks):
        active = value.double()[mask]
        steps_to_end = torch.arange(len(active) - 1, -1, -1, dtype=active.dtype)
        target = float(reward) * torch.pow(torch.tensor(float(gamma), dtype=active.dtype), steps_to_end)
        predictions.append(active)
        targets.append(target)
        sequence_mse.append(float((active - target).square().mean()))
    prediction, target = torch.cat(predictions), torch.cat(targets)
    variance = float(target.var(unbiased=False))
    stats = {
        "sequence_mse": sum(sequence_mse) / len(sequence_mse),
        "token_mse": float((prediction - target).square().mean()),
        "bias": float((prediction - target).mean()),
        "value_mean": float(prediction.mean()),
        "constant_baseline_mse": variance,
    }
    if variance > 1e-12:
        residual = float((target - prediction).var(unbiased=False))
        stats["explained_variance"] = 1.0 - residual / variance
    return stats


def advantage_metrics(
    advantages: Sequence[torch.Tensor],
    values: Sequence[torch.Tensor],
    action_masks: Sequence[torch.Tensor],
) -> Dict[str, float]:
    """Action-token statistics of the actor advantages and the values behind them."""
    active_advantages = torch.cat([a[m] for a, m in zip(advantages, action_masks)])
    active_values = torch.cat([v[m] for v, m in zip(values, action_masks)])
    return {
        "train/advantage_mean": float(active_advantages.mean()),
        "train/advantage_std": float(active_advantages.std(unbiased=False)),
        "train/advantage_rms": float(active_advantages.square().mean().sqrt()),
        "train/advantage_max_abs": float(active_advantages.abs().max()),
        "train/old_value_mean": float(active_values.mean()),
    }
