"""DRO (Distributionally Robust Optimization) policy loss.

Matches the Tinker kernel::

    loss = -(lp * adv - 0.5 * beta * (lp - anchor)^2).sum()

The quadratic penalty keeps the policy close to the anchor as a smooth,
continuous constraint instead of a hard clip, so DRO reports no clip fraction.
Same module surface as :mod:`training.utils.rl.algorithm.grpo`.
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

LOSS_FN = "dro"
"""Trainer built-in DRO loss."""

NORMALIZATION = GradAccNormalization.NUM_LOSS_TOKENS

PREFIX = "dro"


@dataclass(frozen=True)
class DROConfig:
    """``beta`` sets the strength of the quadratic anchor penalty."""

    beta: float = 0.05

    def __post_init__(self) -> None:
        if self.beta < 0:
            raise ValueError("DRO beta must be non-negative.")


def loss_fn_config(config: DROConfig) -> Dict[str, float]:
    """``loss_fn_config`` for :data:`LOSS_FN`."""
    return {"beta": config.beta}


def build_datums(batch: PolicyBatch, config: DROConfig) -> List[sdk.Datum]:
    """Datums for the built-in :data:`LOSS_FN`."""
    del config
    return build_policy_datums(batch)


def metrics(batch: PolicyBatch, result: Any, config: DROConfig) -> Dict[str, float]:
    """``dro_ratio_mean`` and ``dro_kl``."""
    del config  # The quadratic penalty has no trust-region boundary to report.
    return policy_ratio_metrics(batch, result, loss_name="dro", prefix=PREFIX)


def make_loss_fn(batch: PolicyBatch, config: DROConfig):
    """Client-side DRO loss on ``batch.data``; also reports ``dro_quad_penalty``."""

    def policy_fn(ctx):
        quad = (ctx.resp_pi - ctx.resp_anchor) ** 2
        gated_quad = torch.where(ctx.adv != 0, quad, torch.zeros_like(quad))
        quad_term = 0.5 * config.beta * gated_quad
        per_token_loss = -(ctx.resp_pi * ctx.adv - quad_term) * ctx.resp_mask
        active = ctx.resp_mask > 0.5
        return per_token_loss, {
            "quad_penalty": (quad_term * ctx.resp_mask).detach().sum().item(),
            "ratio_mean": torch.exp(ctx.pi_detached - ctx.resp_anchor)[active].mean().item(),
        }

    def loss_fn(data: List[sdk.Datum], logprobs_list: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(batch, data, logprobs_list, "dro", policy_fn)
        out = client_loss_metrics(result, PREFIX)
        nt = result.num_tokens
        out[f"{PREFIX}_quad_penalty"] = result.extra_sums.get("quad_penalty", 0.0) / nt if nt else 0.0
        return result.total_loss, out

    return loss_fn
