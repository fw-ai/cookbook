"""Unclipped importance-sampling (IS) policy loss.

Matches the Tinker kernel::

    loss = -(exp(lp - anchor) * adv).sum()

The anchor is the rollout logprobs unless recomputed, so by default the ratio's
denominator is the recorded sampling distribution. ``ratio_log_cap`` bounds the
ratio for numerical stability; it is otherwise unclipped. Same module surface
as :mod:`training.utils.rl.algorithm.grpo`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from fireworks.training import sdk

from training.utils.client import GradAccNormalization
from training.utils.rl.common import (
    client_loss_metrics,
    policy_ratio_metrics,
    run_loss_loop,
)
from training.utils.rl.losses import PolicyBatch, build_policy_datums

LOSS_FN = "importance_sampling"
"""Trainer built-in unclipped importance-sampling loss."""

NORMALIZATION = GradAccNormalization.NUM_LOSS_TOKENS

PREFIX = "is"


@dataclass(frozen=True)
class ISConfig:
    """``ratio_log_cap`` clamps the client loss's log-ratio before ``exp``."""

    ratio_log_cap: float = 20.0

    def __post_init__(self) -> None:
        if self.ratio_log_cap < 0:
            raise ValueError("IS ratio_log_cap must be non-negative.")


def loss_fn_config(config: ISConfig) -> Dict[str, float]:
    """``loss_fn_config`` for :data:`LOSS_FN` (the kernel takes no settings)."""
    del config
    return {}


def build_datums(batch: PolicyBatch, config: ISConfig) -> List[sdk.Datum]:
    """Datums for the built-in :data:`LOSS_FN`."""
    del config
    return build_policy_datums(batch)


def metrics(batch: PolicyBatch, result: Any, config: ISConfig) -> Dict[str, float]:
    """``is_ratio_mean`` and ``is_kl`` for one trainer call."""
    del config
    return policy_ratio_metrics(batch, result, loss_name="importance_sampling", prefix=PREFIX)


def make_loss_fn(batch: PolicyBatch, config: ISConfig):
    """Client-side IS loss: ``forward_backward_custom(batch.data, make_loss_fn(batch, config))``."""

    def policy_fn(ctx):
        log_ratio = torch.clamp(ctx.resp_pi - ctx.resp_anchor, min=-config.ratio_log_cap, max=config.ratio_log_cap)
        ratio = torch.exp(log_ratio)
        active = ctx.resp_mask > 0.5
        return -(ratio * ctx.adv) * ctx.resp_mask, {"ratio_mean": ratio.detach()[active].mean().item()}

    def loss_fn(data: List[sdk.Datum], logprobs_list: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(batch, data, logprobs_list, "importance_sampling", policy_fn)
        return result.total_loss, client_loss_metrics(result, PREFIX)

    return loss_fn
