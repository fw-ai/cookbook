#!/usr/bin/env python3
"""Synchronous token-level PPO with an independent critic projection head.

Experimental: projection-head SAO is not validated for every model, so this
recipe lives with the other experimental loops.

The actor is a LoRA model and the critic is an independently trained model
of the same base on a second Fireworks trainer. The critic uses the Training API's
dedicated projection module, never the language model's vocabulary head. For
every rollout batch the recipe performs:

1. one critic value for every token in a single full-sequence pass;
2. token-level GAE and lambda returns from the pre-update critic values;
3. one clipped value update on the critic trainer;
4. after critic warmup, one clipped PPO update on the actor trainer.

The critic returns raw ``[tokens, D]`` projection outputs. ``D=1`` is scalar
regression; ``D>=2`` is decoded as a support expectation. The client computes
the continuous clipped value loss and returns ``dLoss/dProjection`` through
the SDK's projection custom-loss interface.

The default ``Config`` is the reference-PPO baseline above. ``sao_config()``
switches to SAO (Single-rollout Asynchronous Optimization, arXiv 2607.07508):

- DIS actor loss whose ratio uses the sampler's recorded logprobs, not a
  trainer recompute (reference PPO's ratio is 1 by construction);
- per-response adaptive GAE ``lambda = 1 - 1 / (alpha * T)``, scanned in float64;
- ``critic_updates`` critic steps per batch, re-reading values before each
  update and computing actor advantages from the updated critic;
- optional offline value pretraining with held-out selection;
- LoRA critics restricted to MLP adapters (``critic_train_attn=False``).

Dataset format and reward customization match ``recipes.rl_loop``.  The
default example expects rows with ``messages`` and ``ground_truth``.

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.experiment.ppo_value_head_loop
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import re
import signal
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tinker
import torch
from fireworks.training.sdk.training_spec import (
    LRSchedulerSpec,
    compute_lr,
    default_constant_schedule,
    normalize_lr_scheduler_spec,
)

from training.renderer import get_text_content
from training.utils import (
    CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO,
    DeployConfig,
    ReconnectableClient,
    TrainerConfig,
    WandBConfig,
    build_renderer,
    build_service_client,
    load_deployment_tokenizer,
    load_jsonl_dataset,
    log_metrics,
    log_metrics_json,
    phase_span,
    prepare_sampling_messages,
    read_api_extra_headers_env,
    setup_wandb,
    validate_config,
    wandb_finish,
)
from training.utils.checkpoints import TrainingCheckpoints
from training.utils.client import GradAccNormalization
from training.utils.dataloader import CursorDataLoader
from training.utils.rl.losses import PromptGroup, combine_prompt_groups
from training.utils.rl.metrics import compute_step_metrics
from training.utils.rl.rollout import (
    Rollout,
    model_input_to_token_ids,
    rollout_to_prompt_group,
    sampled_completion_to_rollout_run,
)
from training.utils.rl.sync_batch import collect_prompt_groups
from training.utils.termination import TerminatedBySignal
from training.utils.timer import elapsed_timer, flush_timing

logger = logging.getLogger(__name__)

QWEN3_4B = "accounts/fireworks/models/qwen3-4b"
QWEN3_4B_TOKENIZER = "Qwen/Qwen3-4B"
QWEN3_4B_LORA_SHAPE = "accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora"


@dataclass
class Config:
    log_path: str
    actor_base_model: str = QWEN3_4B
    critic_base_model: str = QWEN3_4B
    dataset: str | None = None

    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-7
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    value_loss_coef: float = 1.0
    value_clip: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    critic_lambda: float = 1.0
    normalize_advantages: bool = False
    critic_projection_head_dim: int = 1
    """Exact critic projection width. One is scalar regression."""
    critic_value_support: tuple[float, ...] | None = None
    """Ordered categorical support required when the projection width is > 1."""

    lr_scheduler: LRSchedulerSpec = field(default_factory=default_constant_schedule)
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    critic_warmup_batches: int = 0
    """Rollout batches that train only the critic before actor updates start.

    Counted in batches, so ``critic_updates=2`` means twice as many critic
    optimizer steps during warmup."""

    # -- SAO switches. The defaults are the reference-PPO baseline. --------
    policy_objective: str = "ppo"
    """``"ppo"`` (clipped surrogate) or ``"dis"`` (SAO decoupled importance
    sampling against the sampler's recorded logprobs)."""
    dis_low: float = 0.3
    dis_high: float = 5.0
    """DIS keeps tokens with ``1 - dis_low < ratio < 1 + dis_high``."""
    gae_mode: str = "fixed"
    """``"fixed"`` uses ``gae_lambda``; ``"adaptive"`` uses
    ``1 - 1 / (gae_alpha * T)`` per response of T action tokens."""
    gae_alpha: float = 1.5
    critic_updates: int = 1
    """Critic optimizer steps per rollout batch. Every step re-reads values."""
    actor_values: str = "pre_update"
    """Critic values for actor advantages: ``"pre_update"`` (reference PPO) or
    ``"post_update"`` (SAO: a fresh read after the batch's critic steps)."""
    critic_train_attn: bool = True
    critic_train_mlp: bool = True
    """LoRA critic adapter categories. ``critic_train_attn=False`` trains only
    MLP adapters (plus the projection head)."""
    value_pretrain_steps: int = 0
    """Offline critic-only steps on ``value_pretrain_data`` before online
    training. A resumed critic skips the remaining optional pretraining steps
    and continues online training from the recovered critic state."""
    value_pretrain_batch_size: int = 64
    value_pretrain_eval_interval: int = 8
    value_pretrain_seed: int = 20260920

    completions_per_prompt: int = 1
    prompt_groups_per_batch: int = 8
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    shuffle: bool = True
    seed: int = 0
    max_rows: int = 100
    max_seq_len: int | None = None
    lora_rank: int = 32
    lora_alpha: int | None = 32
    critic_lora_rank: int = 32
    critic_lora_alpha: int | None = 32
    renderer_name: str = ""

    grad_accumulation_normalization: GradAccNormalization | str | None = None
    grad_clip_norm: float = 1.0

    actor_trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(
            training_shape_id=QWEN3_4B_LORA_SHAPE,
        )
    )
    critic_trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(
            training_shape_id=QWEN3_4B_LORA_SHAPE,
        )
    )
    deployment: DeployConfig = field(
        default_factory=lambda: DeployConfig(
            tokenizer_model=QWEN3_4B_TOKENIZER,
        )
    )
    dcp_save_interval: int = 0
    weight_sync_timeout: int = 600
    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="ppo-value-head-tinker")
    )
    cleanup_on_exit: bool = True

    actor_init_from_checkpoint: str | None = None
    critic_init_from_checkpoint: str | None = None
    save_final_checkpoint: bool = True
    output_model_id: str | None = None


SAO_SETTINGS: dict[str, Any] = {
    "policy_objective": "dis",
    "dis_low": 0.3,
    "dis_high": 5.0,
    "gae_mode": "adaptive",
    "gae_alpha": 1.5,
    "gamma": 1.0,
    "critic_lambda": 1.0,
    "critic_updates": 2,
    "actor_values": "post_update",
    "critic_warmup_batches": 10,
    "critic_train_attn": False,
    "critic_train_mlp": True,
    "value_pretrain_steps": 64,
    "value_pretrain_batch_size": 64,
    "value_pretrain_eval_interval": 8,
    "actor_learning_rate": 1e-6,
    "critic_learning_rate": 5e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_eps": 1e-8,
    "weight_decay": 0.01,
    "grad_clip_norm": 1.0,
    "value_clip": 0.2,
    "normalize_advantages": False,
    "completions_per_prompt": 1,
    "prompt_groups_per_batch": 128,
    "max_completion_tokens": 16384,
    "max_seq_len": 32768,
    "temperature": 1.0,
}
"""SAO settings for single-turn math RL with dedicated LoRA trainers.

DIS bounds, adaptive GAE, critic update frequency, warmup and batch shape
follow the SAO paper's math-reasoning setup. The actor and critic are separate
LoRA trainers on the same base model, and the critic trains only MLP adapters
plus its projection head (``critic_train_attn=False``). Dataset and reward are
left to the caller."""


def sao_config(log_path: str, **overrides: Any) -> Config:
    """Return a ``Config`` with ``SAO_SETTINGS`` plus ``overrides``."""
    return Config(log_path=log_path, **{**SAO_SETTINGS, **overrides})


def extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    digits = re.search(r"(-?\d+)", match.group(1))
    return digits.group(1) if digits else None


def reward_fn(completion: str, row: dict) -> float:
    """Replace this function with the terminal reward for your task."""
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


def _response_text_for_grading(renderer, sampled) -> str:
    message, _termination = renderer.parse_response(
        sampled.full_tokens[sampled.prompt_len :]
    )
    return get_text_content(message)


def should_accept(prompt_group: PromptGroup) -> bool:
    """PPO does not require within-prompt reward variance."""
    return bool(prompt_group.data)


@dataclass(frozen=True)
class ProjectionValueBatch:
    """Critic inputs and response masks for per-token projection outputs."""

    data: list[tinker.Datum]
    action_masks: list[torch.Tensor]


def _datum_action_mask(datum: tinker.Datum) -> torch.Tensor:
    weights = datum.loss_fn_inputs.get("weights")
    if weights is None:
        raise ValueError("PPO policy datum is missing response weights.")
    return torch.tensor(weights.data, dtype=torch.bool)


def build_projection_value_batch(
    policy_data: list[tinker.Datum],
) -> ProjectionValueBatch:
    """Reuse full causal inputs while removing language-model loss targets."""
    data: list[tinker.Datum] = []
    masks: list[torch.Tensor] = []
    for datum_index, datum in enumerate(policy_data):
        mask = _datum_action_mask(datum)
        targets = datum.loss_fn_inputs.get("target_tokens")
        if targets is None:
            raise ValueError(f"Policy datum {datum_index} is missing target_tokens.")
        target_shape = list(targets.shape or [len(targets.data)])
        if len(target_shape) != 1:
            raise ValueError(
                "PPO policy target_tokens must be one-dimensional; "
                f"datum {datum_index} has shape {target_shape}."
            )
        target_count = int(target_shape[0])
        if mask.numel() != target_count:
            raise ValueError(
                f"Datum {datum_index} has {mask.numel()} response weights for "
                f"{target_count} target positions."
            )
        active_positions = mask.nonzero(as_tuple=False).flatten().tolist()
        if not active_positions:
            raise ValueError(f"Datum {datum_index} has no active response tokens.")
        if active_positions != list(range(active_positions[0], target_count)):
            raise ValueError(
                "Reference PPO requires a terminal, contiguous response mask; "
                f"datum {datum_index} has gaps or masked trailing tokens."
            )
        masks.append(mask)
        data.append(
            tinker.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={},
            )
        )
    return ProjectionValueBatch(data, masks)


def aligned_terminal_rewards(prompt_groups: list[PromptGroup]) -> list[float]:
    """Flatten one terminal reward per single-turn policy datum."""
    rewards: list[float] = []
    for group_index, group in enumerate(prompt_groups):
        if len(group.data) != len(group.rewards):
            raise ValueError(
                "Sequence PPO expects one datum per rollout run. "
                f"Group {group_index} has {len(group.data)} datums and "
                f"{len(group.rewards)} rewards; multi-segment trajectories "
                "need an explicit segment-to-return mapping."
            )
        rewards.extend(float(reward) for reward in group.rewards)
    return rewards


def managed_deployment_id(service: Any, *, enabled: bool) -> str | None:
    """Read deployment metadata only for SDK-managed sampler services."""
    return service.deployment_id if enabled else None


def validate_value_decoder(
    projection_head_dim: int,
    value_support: tuple[float, ...] | None,
) -> None:
    if type(projection_head_dim) is not int or projection_head_dim <= 0:
        raise ValueError("critic_projection_head_dim must be a positive integer.")
    if projection_head_dim == 1:
        if value_support is not None:
            raise ValueError(
                "critic_value_support is only valid for categorical heads."
            )
        return
    if value_support is None or len(value_support) != projection_head_dim:
        raise ValueError(
            "critic_value_support must contain one value per categorical "
            "projection dimension."
        )
    if not all(math.isfinite(value) for value in value_support):
        raise ValueError("critic_value_support values must be finite.")


def projection_values(
    projection: torch.Tensor,
    *,
    projection_head_dim: int,
    value_support: tuple[float, ...] | None,
) -> torch.Tensor:
    """Decode raw critic projection rows into one scalar value per token."""
    if projection.ndim != 2 or projection.shape[1] != projection_head_dim:
        raise ValueError(
            f"Critic projection has shape {tuple(projection.shape)}; expected "
            f"[tokens, {projection_head_dim}]."
        )
    if projection_head_dim == 1:
        return projection[:, 0]
    assert value_support is not None
    support = projection.new_tensor(value_support)
    return torch.softmax(projection, dim=-1).matmul(support)


def adaptive_gae_lambdas(
    action_masks: Sequence[torch.Tensor],
    *,
    alpha: float,
) -> list[float]:
    """SAO's per-response ``lambda = 1 - 1 / (alpha * T)`` for T action tokens.

    It keeps about ``exp(-1 / alpha)`` of the terminal reward's direct weight
    across a whole response, where a fixed 0.95 decays to ~0 after a few
    hundred tokens.
    """
    lambdas = []
    for mask in action_masks:
        count = int(mask.sum())
        if count < 1:
            raise ValueError("Adaptive GAE needs at least one action token.")
        lambdas.append(max(0.0, 1.0 - 1.0 / (alpha * count)))
    return lambdas


def actor_gae_lambdas(
    action_masks: Sequence[torch.Tensor],
    config: Config,
) -> list[float]:
    if config.gae_mode == "adaptive":
        return adaptive_gae_lambdas(action_masks, alpha=config.gae_alpha)
    return [config.gae_lambda] * len(action_masks)


def reference_targets(
    *,
    terminal_rewards: list[float],
    old_values: list[torch.Tensor],
    action_masks: list[torch.Tensor],
    gamma: float,
    gae_lambda: float | Sequence[float],
    critic_lambda: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute token-level GAE and lambda returns from the given values.

    ``gae_lambda`` is one actor lambda for every trajectory or one per
    trajectory. The backward scan runs in float64: at 16K tokens an adaptive
    lambda is ~0.99996, and a float32 recurrence loses that precision.
    """
    if not (len(terminal_rewards) == len(old_values) == len(action_masks)):
        raise ValueError("Rewards, values, and action masks must align by trajectory.")
    if isinstance(gae_lambda, (int, float)):
        actor_lambdas = [float(gae_lambda)] * len(old_values)
    else:
        actor_lambdas = [float(value) for value in gae_lambda]
        if len(actor_lambdas) != len(old_values):
            raise ValueError("Provide one actor GAE lambda per trajectory.")
    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for reward, values, mask, actor_lambda in zip(
        terminal_rewards,
        old_values,
        action_masks,
        actor_lambdas,
        strict=True,
    ):
        positions = mask.nonzero(as_tuple=False).flatten().tolist()
        if not positions:
            raise ValueError("Every PPO trajectory needs at least one action token.")
        scan_values = values.detach().double().cpu().tolist()
        if not all(math.isfinite(scan_values[position]) for position in positions):
            raise ValueError("Critic values must be finite on action tokens.")

        def _gae(lam: float) -> tuple[torch.Tensor, torch.Tensor]:
            result_adv = [0.0] * len(scan_values)
            result_ret = [0.0] * len(scan_values)
            next_value = 0.0
            next_advantage = 0.0
            for position in reversed(positions):
                step_reward = float(reward) if position == positions[-1] else 0.0
                delta = step_reward + gamma * next_value - scan_values[position]
                next_advantage = delta + gamma * lam * next_advantage
                result_adv[position] = next_advantage
                result_ret[position] = next_advantage + scan_values[position]
                next_value = scan_values[position]
            return values.new_tensor(result_adv), values.new_tensor(result_ret)

        actor_advantage, _ = _gae(actor_lambda)
        _, critic_return = _gae(critic_lambda)
        advantages.append(actor_advantage)
        returns.append(critic_return)
    return advantages, returns


def normalize_token_advantages(
    advantages: list[torch.Tensor],
    masks: list[torch.Tensor],
    *,
    enabled: bool,
) -> list[torch.Tensor]:
    if not enabled:
        return advantages
    active = torch.cat(
        [value[mask] for value, mask in zip(advantages, masks, strict=True)]
    )
    if active.numel() < 2 or active.std(unbiased=False) <= 1e-8:
        return advantages
    mean = active.mean()
    std = active.std(unbiased=False)
    return [
        torch.where(mask, (value - mean) / (std + 1e-8), value)
        for value, mask in zip(advantages, masks, strict=True)
    ]


def make_projection_value_loss_fn(
    *,
    terminal_rewards: list[float],
    action_masks: list[torch.Tensor],
    reference_values: list[torch.Tensor] | None = None,
    projection_head_dim: int,
    value_support: tuple[float, ...] | None,
    gamma: float,
    gae_lambda: float | Sequence[float],
    critic_lambda: float,
    normalize_advantages_enabled: bool,
    value_clip: float,
    value_loss_coef: float,
    capture: dict[str, Any],
):
    """Build clipped value loss directly over raw projection outputs.

    ``reference_values`` freezes PPO value anchors across critic updates on
    one rollout batch. The first update obtains them from its own forward.
    Backward sends the exact projection gradients through the Training SDK's
    ``output="projection"`` transport.
    """

    def loss_fn(_data, projections):
        if not (len(projections) == len(terminal_rewards) == len(action_masks)):
            raise ValueError(
                "Critic projections, rewards, and masks must align by response."
            )

        predicted_values = []
        for projection, mask in zip(projections, action_masks, strict=True):
            predicted = projection_values(
                projection,
                projection_head_dim=projection_head_dim,
                value_support=value_support,
            )
            if predicted.numel() != mask.numel():
                raise ValueError(
                    f"Critic returned {predicted.numel()} token rows for a "
                    f"{mask.numel()}-position trajectory."
                )
            predicted_values.append(predicted)

        if reference_values is None:
            old_values = [value.detach() for value in predicted_values]
        else:
            if len(reference_values) != len(predicted_values) or any(
                old.shape != predicted.shape
                for old, predicted in zip(reference_values, predicted_values)
            ):
                raise ValueError("Reference values must align with critic predictions.")
            old_values = [value.detach() for value in reference_values]
        advantages, returns = reference_targets(
            terminal_rewards=terminal_rewards,
            old_values=old_values,
            action_masks=action_masks,
            gamma=gamma,
            gae_lambda=gae_lambda,
            critic_lambda=critic_lambda,
        )
        advantages = normalize_token_advantages(
            advantages,
            action_masks,
            enabled=normalize_advantages_enabled,
        )

        per_response_losses = []
        active_predictions = []
        active_targets = []
        for predicted, targets, anchor, mask in zip(
            predicted_values,
            returns,
            old_values,
            action_masks,
            strict=True,
        ):
            targets = targets.to(device=predicted.device, dtype=predicted.dtype)
            anchor = anchor.to(device=predicted.device, dtype=predicted.dtype)
            mask = mask.to(device=predicted.device)
            squared_error = (predicted - targets).square()
            if value_clip > 0:
                clipped = anchor + torch.clamp(
                    predicted - anchor,
                    min=-value_clip,
                    max=value_clip,
                )
                squared_error = torch.maximum(
                    squared_error,
                    (clipped - targets).square(),
                )
            per_response_losses.append(squared_error[mask].mean())
            active_predictions.append(predicted[mask])
            active_targets.append(targets[mask])

        unscaled_loss = torch.stack(per_response_losses).mean()
        loss = value_loss_coef * unscaled_loss
        predicted_active = torch.cat(active_predictions)
        target_active = torch.cat(active_targets)

        capture["old_values"] = [value.detach().cpu().clone() for value in old_values]
        capture["advantages"] = [value.cpu() for value in advantages]
        capture["returns"] = [value.cpu() for value in returns]
        capture["values"] = predicted_active.detach().cpu()
        capture["loss"] = unscaled_loss.detach().cpu()
        with torch.no_grad():
            metrics = {
                "value/loss": float(unscaled_loss.item()),
                "value/prediction_mean": float(predicted_active.mean().item()),
                "value/return_mean": float(target_active.mean().item()),
            }
        return loss, metrics

    return loss_fn


def make_ppo_policy_loss_fn(
    *,
    old_logprobs: list[list[float]],
    advantages: list[torch.Tensor],
    action_masks: list[torch.Tensor],
    clip_low: float,
    clip_high: float,
):
    """Reference PPO: token mean per response, then response mean."""

    def loss_fn(_data, current_logprobs):
        if not (
            len(current_logprobs)
            == len(old_logprobs)
            == len(advantages)
            == len(action_masks)
        ):
            raise ValueError("PPO policy inputs must align by response.")
        losses = []
        clip_fractions = []
        approx_kls = []
        for current, old_row, advantage, mask in zip(
            current_logprobs,
            old_logprobs,
            advantages,
            action_masks,
            strict=True,
        ):
            old = torch.tensor(old_row, dtype=current.dtype, device=current.device)
            advantage = advantage.to(device=current.device, dtype=current.dtype)
            mask = mask.to(device=current.device)
            if not (current.shape == old.shape == advantage.shape == mask.shape):
                raise ValueError("PPO token arrays must align with each datum.")
            current_active = current[mask]
            old_active = old[mask].detach()
            advantage_active = advantage[mask].detach()
            ratio = torch.exp(current_active - old_active)
            unclipped = -ratio * advantage_active
            clipped = (
                -torch.clamp(
                    ratio,
                    min=1.0 - clip_low,
                    max=1.0 + clip_high,
                )
                * advantage_active
            )
            losses.append(torch.maximum(unclipped, clipped).mean())
            clip_fractions.append((clipped > unclipped).float().mean())
            approx_kls.append((old_active - current_active).mean())
        loss = torch.stack(losses).mean()
        metrics = {
            "actor/loss": float(loss.detach()),
            "actor/pg_clipfrac": float(torch.stack(clip_fractions).mean()),
            "actor/ppo_kl": float(torch.stack(approx_kls).mean().detach()),
        }
        return loss, metrics

    return loss_fn


def make_dis_policy_loss_fn(
    *,
    behavior_logprobs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    action_masks: list[torch.Tensor],
    dis_low: float,
    dis_high: float,
):
    """SAO decoupled importance sampling against the sampler's logprobs.

    Per token, ``w = stopgrad(ratio * 1[1 - dis_low < ratio < 1 + dis_high])``
    with ``ratio = exp(logp - logp_sampler)``, and the loss is
    ``-w * A * logp``. Tokens outside the interval are rejected for either
    advantage sign and stay in the per-response mean's denominator. The weight
    is not differentiated. Response means are then averaged over the batch.
    """
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
            current_logprobs,
            behavior_logprobs,
            advantages,
            action_masks,
            strict=True,
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


def behavior_logprob_rows(
    inf_logprobs: Sequence[Sequence[float]],
    action_masks: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Validate sampler logprobs aligned to ``target_tokens`` (``tokens[1:]``)."""
    if len(inf_logprobs) != len(action_masks):
        raise ValueError(
            "DIS needs sampler logprobs for every "
            f"datum; got {len(inf_logprobs)} rows for {len(action_masks)} datums."
        )
    rows = []
    for index, (row, mask) in enumerate(zip(inf_logprobs, action_masks, strict=True)):
        tensor = torch.tensor(list(row), dtype=torch.float32)
        if tensor.shape != mask.shape:
            raise ValueError(
                f"Datum {index} has {tensor.numel()} sampler logprobs for "
                f"{mask.numel()} target positions."
            )
        if not torch.isfinite(tensor[mask]).all():
            raise ValueError(f"Datum {index} has non-finite sampler logprobs.")
        rows.append(tensor)
    return rows


def critic_value_statistics(
    values: Sequence[torch.Tensor],
    terminal_rewards: Sequence[float],
    action_masks: Sequence[torch.Tensor],
) -> dict[str, float]:
    """Monte Carlo calibration of token values against terminal rewards.

    Explained variance ignores a constant offset, so it is reported with MSE
    and bias; ``constant_baseline_mse`` is the best constant predictor's MSE.
    """
    predictions, targets, sequence_mse = [], [], []
    for value, reward, mask in zip(values, terminal_rewards, action_masks, strict=True):
        active = value.double()[mask]
        target = torch.full_like(active, float(reward))
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


@dataclass(frozen=True)
class ValuePretrainData:
    """Frozen trajectories for offline critic pretraining.

    ``train`` feeds the critic updates; ``validation`` must come from prompts
    that never appear in ``train`` or in online training, and selects the
    checkpoint.
    """

    train: list[PromptGroup]
    validation: list[PromptGroup]


@dataclass(frozen=True)
class CriticBatch:
    data: list[tinker.Datum]
    terminal_rewards: list[float]
    action_masks: list[torch.Tensor]


def critic_batch_from_groups(groups: Sequence[PromptGroup]) -> CriticBatch:
    groups = list(groups)
    policy_data, *_unused = combine_prompt_groups(groups)
    value_batch = build_projection_value_batch(policy_data)
    return CriticBatch(
        data=value_batch.data,
        terminal_rewards=aligned_terminal_rewards(groups),
        action_masks=value_batch.action_masks,
    )


def forward_backward_projection_critic(
    critic,
    *,
    data: list[tinker.Datum],
    terminal_rewards: list[float],
    action_masks: list[torch.Tensor],
    config: Config,
    capture: dict[str, Any],
    actor_lambdas: Sequence[float] | None = None,
    reference_values: list[torch.Tensor] | None = None,
):
    """Run the critic only through the independent projection output path."""
    return critic.forward_backward_custom(
        data,
        make_projection_value_loss_fn(
            terminal_rewards=terminal_rewards,
            action_masks=action_masks,
            reference_values=reference_values,
            projection_head_dim=config.critic_projection_head_dim,
            value_support=config.critic_value_support,
            gamma=config.gamma,
            gae_lambda=(
                config.gae_lambda if actor_lambdas is None else list(actor_lambdas)
            ),
            critic_lambda=config.critic_lambda,
            normalize_advantages_enabled=config.normalize_advantages,
            value_clip=config.value_clip,
            value_loss_coef=config.value_loss_coef,
            capture=capture,
        ),
        output="projection",
    )


def predict_critic_values(
    critic,
    *,
    data: list[tinker.Datum],
    action_masks: Sequence[torch.Tensor],
    config: Config,
) -> list[torch.Tensor]:
    """Forward-only read of the critic's current per-token values."""
    result = critic.forward_projection(data)
    if len(result.loss_fn_outputs) != len(data):
        raise ValueError(
            f"Critic returned {len(result.loss_fn_outputs)} projections for "
            f"{len(data)} datums."
        )
    values = []
    for output, mask in zip(result.loss_fn_outputs, action_masks, strict=True):
        tensor_data = output.get("projection")
        if tensor_data is None:
            raise ValueError("Critic forward response is missing 'projection'.")
        projection = torch.tensor(tensor_data.data, dtype=torch.float32)
        if tensor_data.shape is not None:
            projection = projection.reshape(tensor_data.shape)
        value = projection_values(
            projection,
            projection_head_dim=config.critic_projection_head_dim,
            value_support=config.critic_value_support,
        )
        if value.numel() != mask.numel():
            raise ValueError(
                f"Critic returned {value.numel()} token rows for a "
                f"{mask.numel()}-position trajectory."
            )
        values.append(value)
    return values


@dataclass(frozen=True)
class CriticTrainingResult:
    advantages: list[torch.Tensor]
    actor_values: list[torch.Tensor]
    """Values the actor advantages were computed from."""
    pre_update_values: list[torch.Tensor]
    value_loss: float
    """Pre-update value loss of the batch's first critic step."""
    fwd_bwd_results: list[Any]
    optim_result: Any
    metrics: dict[str, float]


def train_critic_then_advantages(
    critic,
    *,
    batch: CriticBatch,
    actor_lambdas: Sequence[float],
    config: Config,
    adam_params: tinker.AdamParams,
    normalization: Any,
) -> CriticTrainingResult:
    """Run ``critic_updates`` critic steps, then derive actor advantages.

    Each step's custom-loss forward re-reads the critic's values, so every
    update sees the parameters left by the previous one. PPO value anchors
    and targets stay fixed at the first forward. ``post_update`` advantages
    add one forward-only read after the last step.
    """
    fwd_bwd_results = []
    captures: list[dict[str, Any]] = []
    optim_result = None
    metrics: dict[str, float] = {}
    for epoch in range(config.critic_updates):
        capture: dict[str, Any] = {}
        with elapsed_timer("critic_fwd_bwd"):
            result = forward_backward_projection_critic(
                critic,
                data=batch.data,
                terminal_rewards=batch.terminal_rewards,
                action_masks=batch.action_masks,
                config=config,
                capture=capture,
                actor_lambdas=actor_lambdas,
                reference_values=(captures[0]["old_values"] if captures else None),
            )
        with elapsed_timer("critic_optim_step"):
            optim_result = critic.optim_step(
                adam_params,
                grad_accumulation_normalization=normalization,
            )
        fwd_bwd_results.append(result)
        captures.append(capture)
        metrics[f"train/critic-value_loss/epoch_{epoch + 1}"] = float(capture["loss"])
    first = captures[0]
    if config.actor_values == "post_update":
        with elapsed_timer("critic_value_refresh"):
            actor_values = predict_critic_values(
                critic,
                data=batch.data,
                action_masks=batch.action_masks,
                config=config,
            )
        advantages, _ = reference_targets(
            terminal_rewards=batch.terminal_rewards,
            old_values=actor_values,
            action_masks=batch.action_masks,
            gamma=config.gamma,
            gae_lambda=list(actor_lambdas),
            critic_lambda=config.critic_lambda,
        )
        advantages = normalize_token_advantages(
            advantages,
            batch.action_masks,
            enabled=config.normalize_advantages,
        )
    else:
        actor_values = first["old_values"]
        advantages = first["advantages"]
    for key, value in critic_value_statistics(
        actor_values, batch.terminal_rewards, batch.action_masks
    ).items():
        metrics["train/critic_post/" + key] = value
    return CriticTrainingResult(
        advantages=advantages,
        actor_values=actor_values,
        pre_update_values=first["old_values"],
        value_loss=float(first["loss"]),
        fwd_bwd_results=fwd_bwd_results,
        optim_result=optim_result,
        metrics=metrics,
    )


def forward_backward_ppo_actor(
    actor,
    *,
    data: list[tinker.Datum],
    advantages: list[torch.Tensor],
    action_masks: list[torch.Tensor],
    clip_low: float,
    clip_high: float,
    behavior_logprobs: list[torch.Tensor] | None = None,
    dis_bounds: tuple[float, float] | None = None,
):
    """Run the actor through the ordinary language-model logprob path.

    One trainer forward supplies the current logprobs, and the custom loss
    reuses it through ``precomputed_forward``. For PPO the old logprobs are that
    same forward (reference PPO, ratio 1). ``dis_bounds`` switches to DIS, whose
    ratio uses the sampler's ``behavior_logprobs``.
    """
    old_policy_result = actor.forward(data, "cross_entropy")
    old_policy_logprobs = [
        output["logprobs"].data for output in old_policy_result.loss_fn_outputs
    ]
    if dis_bounds is not None:
        if behavior_logprobs is None:
            raise ValueError("DIS needs the sampler's behavior logprobs.")
        loss_fn = make_dis_policy_loss_fn(
            behavior_logprobs=behavior_logprobs,
            advantages=advantages,
            action_masks=action_masks,
            dis_low=dis_bounds[0],
            dis_high=dis_bounds[1],
        )
    else:
        loss_fn = make_ppo_policy_loss_fn(
            old_logprobs=old_policy_logprobs,
            advantages=advantages,
            action_masks=action_masks,
            clip_low=clip_low,
            clip_high=clip_high,
        )
    return actor.forward_backward_custom(
        data,
        loss_fn,
        precomputed_forward=old_policy_result,
    )


def run_value_pretraining(
    critic,
    critic_checkpoint: TrainingCheckpoints,
    *,
    pretrain_data: ValuePretrainData,
    config: Config,
    adam_params: tinker.AdamParams,
    normalization: Any,
) -> int:
    """Offline critic-only updates with held-out checkpoint selection.

    Validation sequence MSE is measured at step 0, every
    ``value_pretrain_eval_interval`` steps and at the end. Each improvement is
    saved as a resumable critic checkpoint; if the last step is not the best,
    the best checkpoint (weights, head and optimizer) is restored. Raises when
    no step beats the initial critic, before any actor training.
    """
    train = critic_batch_from_groups(pretrain_data.train)
    validation = critic_batch_from_groups(pretrain_data.validation)
    total = config.value_pretrain_steps
    order = list(range(len(train.data)))
    rng = random.Random(config.value_pretrain_seed)
    rng.shuffle(order)
    position = 0
    best_mse: float | None = None
    best_step = 0
    for step in range(total + 1):
        if step % config.value_pretrain_eval_interval == 0 or step == total:
            with elapsed_timer("value_pretrain_validation"):
                values = predict_critic_values(
                    critic,
                    data=validation.data,
                    action_masks=validation.action_masks,
                    config=config,
                )
            stats = critic_value_statistics(
                values, validation.terminal_rewards, validation.action_masks
            )
            log_metrics(
                {
                    "pretrain/step": step,
                    **{f"pretrain/validation/{k}": v for k, v in stats.items()},
                },
                step=step,
            )
            logger.info(
                "Value pretraining step %d | held-out sequence MSE %.4f "
                "(constant baseline %.4f)",
                step,
                stats["sequence_mse"],
                stats["constant_baseline_mse"],
            )
            if best_mse is None or stats["sequence_mse"] < best_mse:
                best_mse, best_step = stats["sequence_mse"], step
                if step > 0:
                    critic_checkpoint.save(
                        "value-pretrain-best",
                        resumable=True,
                        promotable=False,
                        data_consumed=0,
                    )
        if step == total:
            break
        indices = []
        for _ in range(config.value_pretrain_batch_size):
            if position == len(order):
                rng.shuffle(order)
                position = 0
            indices.append(order[position])
            position += 1
        capture: dict[str, Any] = {}
        step_started = time.monotonic()
        forward_backward_projection_critic(
            critic,
            data=[train.data[i] for i in indices],
            terminal_rewards=[train.terminal_rewards[i] for i in indices],
            action_masks=[train.action_masks[i] for i in indices],
            config=config,
            capture=capture,
        )
        fwd_bwd_seconds = time.monotonic() - step_started
        critic.optim_step(
            adam_params,
            grad_accumulation_normalization=normalization,
        )
        step_seconds = time.monotonic() - step_started
        log_metrics(
            {
                "pretrain/step": step + 1,
                "pretrain/train/value_loss": float(capture["loss"]),
                "pretrain/train/fwd_bwd_time_s": fwd_bwd_seconds,
                "pretrain/train/optim_step_time_s": step_seconds - fwd_bwd_seconds,
                "pretrain/train/step_wall_time_s": step_seconds,
                "pretrain/train/token_positions": sum(
                    train.action_masks[i].numel() for i in indices
                ),
            },
            step=step + 1,
        )
    if total and best_step == 0:
        raise RuntimeError(
            "Value pretraining did not improve held-out MSE; refusing actor training."
        )
    if best_step != total:
        # The newest resumable critic checkpoint is the best one saved above.
        restored = critic_checkpoint.resume(restore_optimizer=True)
        if restored is None:
            raise RuntimeError("Could not restore the best value-pretraining checkpoint.")
    logger.info("Value pretraining selected step %d (MSE %.4f)", best_step, best_mse)
    return best_step


def resume_rows_consumed(
    actor_resume: Any,
    critic_resume: Any,
    *,
    critic_warmup_batches: int,
) -> int:
    """Dataset cursor shared by the resumed actor and critic.

    The pair must share one cursor, except during critic warmup: periodic saves
    there checkpoint only the critic, and an actor with no updates still holds
    its initial weights, so it resumes at the critic's cursor.
    """
    actor_step = actor_resume.step if actor_resume else 0
    actor_rows = actor_resume.data_consumed if actor_resume else 0
    critic_step = critic_resume.step if critic_resume else 0
    critic_rows = critic_resume.data_consumed if critic_resume else 0
    actor_untouched = actor_step == 0
    if actor_rows != critic_rows and not (
        actor_untouched
        and actor_rows <= critic_rows
        and critic_step <= critic_warmup_batches
    ):
        raise ValueError(
            "Actor and critic checkpoints have different dataset cursors: "
            f"actor={actor_rows}, critic={critic_rows}. Resume an aligned pair."
        )
    return critic_rows


def save_periodic_checkpoints(
    actor_checkpoint: Any,
    critic_checkpoint: Any,
    *,
    actor_updated: bool,
    actor_step: int,
    rollout_batch: int,
    data_consumed: int,
) -> None:
    """Save resumable DCPs; during warmup only the critic has new state."""
    if actor_updated:
        actor_checkpoint.save(
            f"step-{actor_step}",
            resumable=True,
            promotable=False,
            data_consumed=data_consumed,
        )
    critic_checkpoint.save(
        f"step-{rollout_batch}",
        resumable=True,
        promotable=False,
        data_consumed=data_consumed,
    )


def validate_sao_settings(
    config: Config,
    value_pretrain_data: ValuePretrainData | None = None,
) -> None:
    """Static checks for the SAO switches (no I/O)."""
    cfg = config
    choices = {
        "policy_objective": ("ppo", "dis"),
        "gae_mode": ("fixed", "adaptive"),
        "actor_values": ("pre_update", "post_update"),
    }
    for name, allowed in choices.items():
        if getattr(cfg, name) not in allowed:
            raise ValueError(f"{name} must be one of {allowed}; got {getattr(cfg, name)!r}.")
    if not 0 <= cfg.dis_low < 1 or not math.isfinite(cfg.dis_high) or cfg.dis_high <= 0:
        raise ValueError("DIS needs 0 <= dis_low < 1 and a positive finite dis_high.")
    if not math.isfinite(cfg.gae_alpha) or cfg.gae_alpha <= 0:
        raise ValueError("gae_alpha must be positive and finite.")
    for name in ("critic_updates", "value_pretrain_batch_size", "value_pretrain_eval_interval"):
        value = getattr(cfg, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if type(cfg.value_pretrain_steps) is not int or cfg.value_pretrain_steps < 0:
        raise ValueError("value_pretrain_steps must be a non-negative integer.")
    if cfg.value_pretrain_steps and value_pretrain_data is None:
        raise ValueError("value_pretrain_steps > 0 needs value_pretrain_data= in main().")
    if value_pretrain_data is not None and not (
        value_pretrain_data.train and value_pretrain_data.validation
    ):
        raise ValueError("value_pretrain_data needs train and validation trajectories.")
    if not (cfg.critic_train_attn or cfg.critic_train_mlp):
        raise ValueError("Enable at least one of critic_train_attn / critic_train_mlp.")
    if not cfg.critic_train_attn and cfg.critic_lora_rank == 0:
        raise ValueError(
            "critic_train_attn=False selects LoRA adapter categories; it needs "
            "critic_lora_rank > 0. Full-parameter critics train every layer."
        )


def main(
    config: Config,
    *,
    sample_prompt_fn: Callable[..., Awaitable[PromptGroup | None]] | None = None,
    rows: list[dict] | None = None,
    value_pretrain_data: ValuePretrainData | None = None,
    reward: Callable[[str, dict], float] | None = None,
) -> dict[str, Any]:
    """Run reference PPO or SAO with separate actor and critic trainers.

    ``reward`` scores a sampled response against its source row. The default
    expects an integer inside ``<answer>...</answer>``.
    """
    cfg = config
    grade_response = reward if reward is not None else reward_fn
    bounded = ("gamma", "gae_lambda", "critic_lambda", "adam_beta1", "adam_beta2")
    for name in bounded:
        value = getattr(cfg, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1].")
    positive = (
        "actor_learning_rate",
        "critic_learning_rate",
        "adam_eps",
        "grad_clip_norm",
    )
    for name in positive:
        value = getattr(cfg, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    if not 0 <= cfg.eps_clip < 1 or cfg.eps_clip_high <= 0:
        raise ValueError("eps_clip must be in [0, 1); eps_clip_high must be positive.")
    if cfg.value_loss_coef < 0 or cfg.value_clip < 0 or cfg.weight_decay < 0:
        raise ValueError("Value coefficients and weight_decay must be non-negative.")
    if type(cfg.critic_warmup_batches) is not int or cfg.critic_warmup_batches < 0:
        raise ValueError("critic_warmup_batches must be a non-negative integer.")
    if cfg.completions_per_prompt < 1 or cfg.prompt_groups_per_batch < 1:
        raise ValueError("PPO rollout batch sizes must be positive.")
    if type(cfg.critic_lora_rank) is not int or cfg.critic_lora_rank < 0:
        raise ValueError("critic_lora_rank must be a non-negative integer.")
    validate_value_decoder(
        cfg.critic_projection_head_dim,
        cfg.critic_value_support,
    )
    validate_sao_settings(cfg, value_pretrain_data)
    if rows is None and not cfg.dataset:
        raise ValueError("Provide either cfg.dataset or rows= to main().")
    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required.")

    def _signal_handler(signum, _):
        name = signal.Signals(signum).name
        raise TerminatedBySignal(name)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(
        cfg.actor_base_model,
        cfg.dataset,
        deploy=cfg.deployment,
        output_model_id=cfg.output_model_id,
        require_dataset=(rows is None),
    )
    lr_scheduler = normalize_lr_scheduler_spec(cfg.lr_scheduler)
    setup_wandb(
        cfg.wandb,
        {
            "algorithm": "sao_ppo_projection_value_head",
            "actor_base_model": cfg.actor_base_model,
            "critic_base_model": cfg.critic_base_model,
            "actor_training_shape": cfg.actor_trainer.training_shape_id,
            "critic_training_shape": cfg.critic_trainer.training_shape_id,
            "actor_lr": cfg.actor_learning_rate,
            "critic_lr": cfg.critic_learning_rate,
            "gae_lambda": cfg.gae_lambda,
            "critic_lambda": cfg.critic_lambda,
            "critic_warmup_batches": cfg.critic_warmup_batches,
            "critic_projection_head_dim": cfg.critic_projection_head_dim,
            "critic_value_support": cfg.critic_value_support,
            "critic_lora_rank": cfg.critic_lora_rank,
            "policy_objective": cfg.policy_objective,
            "dis_low": cfg.dis_low,
            "dis_high": cfg.dis_high,
            "gae_mode": cfg.gae_mode,
            "gae_alpha": cfg.gae_alpha,
            "critic_updates": cfg.critic_updates,
            "actor_values": cfg.actor_values,
            "critic_train_attn": cfg.critic_train_attn,
            "critic_train_mlp": cfg.critic_train_mlp,
            "value_pretrain_steps": cfg.value_pretrain_steps,
        },
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    with ExitStack() as stack:
        uses_managed_sampler = sample_prompt_fn is None
        tokenizer = (
            load_deployment_tokenizer(cfg.deployment)
            if uses_managed_sampler
            else None
        )
        actor_service = build_service_client(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
            base_model=cfg.actor_base_model,
            tokenizer_model=cfg.deployment.tokenizer_model,
            max_lora_rank=cfg.lora_rank,
            max_context_length=cfg.max_seq_len,
            learning_rate=cfg.actor_learning_rate,
            trainer=cfg.actor_trainer,
            deployment=cfg.deployment if uses_managed_sampler else None,
            hotload_timeout_s=cfg.weight_sync_timeout,
            cleanup_trainer_on_close=cfg.cleanup_on_exit,
            cleanup_deployment_on_close=(
                CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO
                if cfg.cleanup_on_exit
                else None
            ),
            reference_required=False,
        )
        stack.callback(actor_service.close)
        critic_service = build_service_client(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
            base_model=cfg.critic_base_model,
            tokenizer_model=cfg.deployment.tokenizer_model,
            max_lora_rank=cfg.critic_lora_rank,
            projection_head_dim=cfg.critic_projection_head_dim,
            max_context_length=cfg.max_seq_len,
            learning_rate=cfg.critic_learning_rate,
            trainer=cfg.critic_trainer,
            deployment=None,
            cleanup_trainer_on_close=cfg.cleanup_on_exit,
            reference_required=False,
        )
        stack.callback(critic_service.close)

        actor_training_client = actor_service.create_training_client(
            cfg.actor_base_model,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
        )
        actor = ReconnectableClient.from_training_client(
            actor_training_client,
            base_model=cfg.actor_base_model,
            lora_rank=cfg.lora_rank,
            job_id=actor_service.trainer_job_id,
            service=actor_service,
        )
        critic_training_client = critic_service.create_training_client(
            cfg.critic_base_model,
            lora_rank=cfg.critic_lora_rank,
            lora_alpha=cfg.critic_lora_alpha,
            train_attn=cfg.critic_train_attn,
            train_mlp=cfg.critic_train_mlp,
            # Critic values come from the projection head, never the LM head.
            train_unembed=False,
        )
        critic = ReconnectableClient.from_training_client(
            critic_training_client,
            base_model=cfg.critic_base_model,
            lora_rank=cfg.critic_lora_rank,
            job_id=critic_service.trainer_job_id,
            service=critic_service,
        )

        sampler = None
        response_renderer = None
        if sample_prompt_fn is None:
            sampler = actor_service.create_deployment_sampler(tokenizer=tokenizer)
            response_renderer = build_renderer(
                tokenizer,
                cfg.deployment.tokenizer_model,
                cfg.renderer_name,
            )

        actor_checkpoint = TrainingCheckpoints(
            actor,
            actor_service,
            trainer_id=actor_service.trainer_job_id,
            log_path=str(Path(cfg.log_path) / "actor"),
            lora_rank=cfg.lora_rank,
        )
        critic_checkpoint = TrainingCheckpoints(
            critic,
            critic_service,
            trainer_id=critic_service.trainer_job_id,
            log_path=str(Path(cfg.log_path) / "critic"),
            lora_rank=cfg.critic_lora_rank,
        )
        restore_optimizer = getattr(
            cfg,
            "_restore_optimizer_from_init_checkpoint",
            True,
        )
        actor_resume = actor_checkpoint.resume(
            init_from_checkpoint=cfg.actor_init_from_checkpoint,
            restore_optimizer=restore_optimizer,
        )
        critic_resume = critic_checkpoint.resume(
            init_from_checkpoint=cfg.critic_init_from_checkpoint,
            restore_optimizer=restore_optimizer,
        )
        actor_step_offset = actor_resume.step if actor_resume else 0
        critic_step_offset = critic_resume.step if critic_resume else 0
        prior_rows_consumed = resume_rows_consumed(
            actor_resume,
            critic_resume,
            critic_warmup_batches=cfg.critic_warmup_batches,
        )

        if uses_managed_sampler:
            with elapsed_timer("weight_sync"):
                saved = actor.save_weights_for_sampler(
                    f"step-{actor_step_offset}",
                    checkpoint_type="base",
                )
                actor_service.hotload_sampler_snapshot(saved.path)
            flush_timing()

        if rows is None:
            rows = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        else:
            rows = list(rows)
        row_loader = CursorDataLoader(
            rows,
            start_cursor=prior_rows_consumed,
            epochs=cfg.epochs,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
        )
        row_iterator = iter(row_loader)
        remaining_rows = max(0, row_loader.total_items - prior_rows_consumed)
        outer_batches = math.ceil(remaining_rows / cfg.prompt_groups_per_batch)
        warmup_remaining = max(0, cfg.critic_warmup_batches - critic_step_offset)
        total_actor_steps = actor_step_offset + max(0, outer_batches - warmup_remaining)

        sample_kwargs: dict[str, Any] = {
            "max_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "top_p": 1.0,
            "top_k": 0,
            "max_seq_len": actor_service.max_context_length,
            "http_timeout": cfg.deployment.sample_timeout,
            "logprobs": True,
        }

        async def sample_one_prompt(
            row: dict,
            *,
            cursor_index: int,
        ) -> PromptGroup | None:
            if sample_prompt_fn is not None:
                return await sample_prompt_fn(row, cursor_index=cursor_index)

            messages = prepare_sampling_messages(row.get("messages", []))
            if not messages:
                return None
            model_input = response_renderer.build_generation_prompt(messages)
            prompt_token_ids = model_input_to_token_ids(model_input)
            try:
                sampled = await sampler.sample_with_prompt_tokens(
                    prompt_token_ids,
                    n=cfg.completions_per_prompt,
                    stop=response_renderer.get_stop_sequences(),
                    **sample_kwargs,
                )
            except Exception as error:
                logger.warning("Sampling row %d failed: %s", cursor_index, error)
                return None
            if not sampled or len(sampled) != cfg.completions_per_prompt:
                return None

            runs = []
            for sample in sampled:
                terminal_reward = grade_response(
                    _response_text_for_grading(response_renderer, sample),
                    row,
                )
                run = sampled_completion_to_rollout_run(sample, reward=terminal_reward)
                if run is None:
                    return None
                runs.append(run)
            return rollout_to_prompt_group(
                Rollout(runs=runs),
                advantage_fn=lambda rewards: list(rewards),
                with_reference=False,
            )

        def adam_params(learning_rate: float) -> tinker.AdamParams:
            return tinker.AdamParams(
                learning_rate=learning_rate,
                beta1=cfg.adam_beta1,
                beta2=cfg.adam_beta2,
                eps=cfg.adam_eps,
                weight_decay=cfg.weight_decay,
                grad_clip_norm=cfg.grad_clip_norm,
            )

        normalization = cfg.grad_accumulation_normalization or "none"
        critic_adam = adam_params(cfg.critic_learning_rate)

        pretrain_steps_taken = 0
        if cfg.value_pretrain_steps:
            if critic_resume is not None:
                logger.warning(
                    "Critic resumed from a checkpoint; skipping the configured "
                    "%d value-pretraining steps and continuing online training "
                    "from the recovered critic state. Use a fresh critic trainer "
                    "and log path to rerun value pretraining from the beginning.",
                    cfg.value_pretrain_steps,
                )
            else:
                assert value_pretrain_data is not None
                with phase_span("value_pretraining", category="train"):
                    run_value_pretraining(
                        critic,
                        critic_checkpoint,
                        pretrain_data=value_pretrain_data,
                        config=cfg,
                        adam_params=critic_adam,
                        normalization=normalization,
                    )
                pretrain_steps_taken = cfg.value_pretrain_steps
                flush_timing()

        async def run_training() -> tuple[int, int]:
            actor_step = actor_step_offset
            rollout_batch = critic_step_offset
            while True:
                batch_index = rollout_batch
                with phase_span(
                    "rollout_batch",
                    category="rollout",
                    attributes={"critic_batch": batch_index},
                ):
                    (
                        prompt_groups,
                        row_indices,
                        loop_stats,
                    ) = await collect_prompt_groups(
                        row_iterator,
                        target_size=cfg.prompt_groups_per_batch,
                        sample_prompt=sample_one_prompt,
                        should_accept=should_accept,
                    )
                if not row_indices:
                    break
                if not prompt_groups:
                    for index in row_indices:
                        row_loader.mark_resolved(index)
                    continue

                train_started = time.monotonic()
                policy_data, _adv, _ref, _lens, inf_logprobs, _raw = (
                    combine_prompt_groups(prompt_groups, include_raw=True)
                )
                terminal_rewards = aligned_terminal_rewards(prompt_groups)
                value_batch = build_projection_value_batch(policy_data)
                action_masks = value_batch.action_masks
                behavior = (
                    behavior_logprob_rows(inf_logprobs, action_masks)
                    if cfg.policy_objective == "dis"
                    else None
                )

                actor_lambdas = actor_gae_lambdas(action_masks, cfg)
                critic_outcome = train_critic_then_advantages(
                    critic,
                    batch=CriticBatch(
                        data=value_batch.data,
                        terminal_rewards=terminal_rewards,
                        action_masks=action_masks,
                    ),
                    actor_lambdas=actor_lambdas,
                    config=cfg,
                    adam_params=critic_adam,
                    normalization=normalization,
                )
                advantages = critic_outcome.advantages
                rollout_batch += 1

                actor_result = None
                actor_optim = None
                actor_updated = batch_index >= cfg.critic_warmup_batches
                actor_lr = 0.0
                if actor_updated:
                    with elapsed_timer("actor_fwd_bwd"):
                        actor_result = forward_backward_ppo_actor(
                            actor,
                            data=policy_data,
                            advantages=advantages,
                            action_masks=action_masks,
                            clip_low=cfg.eps_clip,
                            clip_high=cfg.eps_clip_high,
                            behavior_logprobs=behavior,
                            dis_bounds=(
                                (cfg.dis_low, cfg.dis_high)
                                if cfg.policy_objective == "dis"
                                else None
                            ),
                        )
                    next_actor_step = actor_step + 1
                    actor_lr = compute_lr(
                        lr_scheduler,
                        step=next_actor_step,
                        base_lr=cfg.actor_learning_rate,
                        total_steps=max(1, total_actor_steps),
                    )
                    with elapsed_timer("actor_optim_step"):
                        actor_optim = actor.optim_step(
                            adam_params(actor_lr),
                            grad_accumulation_normalization=normalization,
                        )
                    actor_step = next_actor_step
                    if uses_managed_sampler:
                        with elapsed_timer("weight_sync"):
                            saved = await asyncio.to_thread(
                                actor.save_weights_for_sampler,
                                f"step-{actor_step}",
                            )
                            await asyncio.to_thread(
                                actor_service.hotload_sampler_snapshot,
                                saved.path,
                            )

                for index in row_indices:
                    row_loader.mark_resolved(index)

                loop_stats["train_wall_time"] = time.monotonic() - train_started
                loop_stats["scheduler_step_wall_time"] = (
                    loop_stats["rollout_batch_wall_time"]
                    + loop_stats["train_wall_time"]
                )
                fwd_bwd_results = list(critic_outcome.fwd_bwd_results)
                if actor_result is not None:
                    fwd_bwd_results.append(actor_result)
                metrics = compute_step_metrics(
                    prompt_groups=prompt_groups,
                    fwd_bwd_results=fwd_bwd_results,
                    optim_result=actor_optim or critic_outcome.optim_result,
                    n_accum=len(fwd_bwd_results),
                    timing_metrics=flush_timing(),
                    loop_stats=loop_stats,
                )
                # Actor and critic updates are independent optimizer steps, not
                # one trainer's gradient-accumulation group.
                metrics.pop("train/effective_accumulation_steps", None)
                active_advantages = torch.cat(
                    [value[mask] for value, mask in zip(advantages, action_masks, strict=True)]
                )
                active_values = torch.cat(
                    [
                        value[mask]
                        for value, mask in zip(
                            critic_outcome.actor_values, action_masks, strict=True
                        )
                    ]
                )
                critic_updates = (
                    pretrain_steps_taken + rollout_batch * cfg.critic_updates
                )
                metrics.update(critic_outcome.metrics)
                metrics.update(
                    {
                        "train/step": rollout_batch,
                        "train/actor_updates": actor_step,
                        "train/critic_updates": critic_updates,
                        "train/actor_updated": int(actor_updated),
                        "train/actor_learning_rate": actor_lr,
                        "train/critic_learning_rate": cfg.critic_learning_rate,
                        "train/advantage_mean": float(active_advantages.mean()),
                        "train/advantage_std": float(
                            active_advantages.std(unbiased=False)
                        ),
                        "train/advantage_rms": float(
                            active_advantages.square().mean().sqrt()
                        ),
                        "train/advantage_max_abs": float(active_advantages.abs().max()),
                        "train/old_value_mean": float(active_values.mean()),
                        "train/gae_lambda_mean": sum(actor_lambdas) / len(actor_lambdas),
                        "train/gae_lambda_min": min(actor_lambdas),
                        "train/critic-value_loss": critic_outcome.value_loss,
                        "train/critic-sequences": len(value_batch.data),
                        "train/critic-token-positions": sum(
                            mask.numel() for mask in action_masks
                        ),
                        "train/critic-active-tokens": sum(
                            int(mask.sum()) for mask in action_masks
                        ),
                    }
                )
                reward = metrics.get("rollout/filtered_reward", 0.0)
                logger.info(
                    "PPO rollout batch %d | actor updates %d | reward %.3f | value %.3f",
                    rollout_batch,
                    actor_step,
                    reward,
                    float(active_values.mean()),
                )
                log_metrics_json(
                    rollout_batch,
                    reward=reward,
                    value=float(active_values.mean()),
                )
                log_metrics(metrics, step=rollout_batch)

                if (
                    cfg.dcp_save_interval > 0
                    and (rollout_batch - critic_step_offset) % cfg.dcp_save_interval == 0
                ):
                    save_periodic_checkpoints(
                        actor_checkpoint,
                        critic_checkpoint,
                        actor_updated=actor_updated,
                        actor_step=actor_step,
                        rollout_batch=rollout_batch,
                        data_consumed=row_loader.data_consumed,
                    )

            return actor_step, rollout_batch

        actor_step, critic_step = asyncio.run(run_training())

        has_actor_updates = actor_step > actor_step_offset
        has_critic_updates = critic_step > critic_step_offset
        has_advanced_dataset = row_loader.data_consumed > prior_rows_consumed
        if cfg.save_final_checkpoint and (has_critic_updates or has_advanced_dataset):
            actor_checkpoint.save(
                f"step-{actor_step}",
                resumable=True,
                promotable=has_actor_updates,
                data_consumed=row_loader.data_consumed,
            )
            critic_checkpoint.save(
                f"step-{critic_step}",
                resumable=True,
                promotable=False,
                data_consumed=row_loader.data_consumed,
            )
            if cfg.output_model_id and has_actor_updates:
                actor_checkpoint.promote_latest(
                    cfg.output_model_id,
                    cfg.actor_base_model,
                )

        wandb_finish(metrics_file=os.environ.get("COOKBOOK_METRICS_FILE"))
        return {
            "actor_steps": actor_step,
            "critic_steps": critic_step,
            "critic_updates": (
                pretrain_steps_taken
                + (critic_step - critic_step_offset) * cfg.critic_updates
            ),
            "actor_job_id": actor_service.trainer_job_id,
            "critic_job_id": critic_service.trainer_job_id,
            "deployment_id": managed_deployment_id(
                actor_service,
                enabled=uses_managed_sampler,
            ),
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main(
        Config(
            log_path="./ppo_logs",
            dataset=(
                "https://raw.githubusercontent.com/eval-protocol/python-sdk/"
                "main/development/gsm8k_sample.jsonl"
            ),
        )
    )
