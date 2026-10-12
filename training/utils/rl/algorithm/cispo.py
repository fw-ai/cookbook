"""CISPO (Clipped Importance Sampling Policy Optimization).

Clips the ratio ``pi / anchor`` to ``[1 - eps_low, 1 + eps_high]`` and uses
the clipped ratio as a *detached* weight on the log-probability, matching the
Tinker kernel::

    clipped_ratio = clamp(ratio, 1 - eps_low, 1 + eps_high)
    per_token_loss = -(clipped_ratio.detach() * logprob * advantage)

Gradient flows through ``logprob``, not the ratio. Same module surface as
:mod:`training.utils.rl.algorithm.grpo`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import tinker

from training.utils.client import GradAccNormalization
from training.utils.rl.common import client_loss_metrics, policy_ratio_metrics, run_loss_loop
from training.utils.rl.losses import PolicyBatch, build_policy_datums

LOSS_FN = "cispo"
"""Trainer built-in CISPO loss."""

NORMALIZATION = GradAccNormalization.NUM_LOSS_TOKENS

PREFIX = "cispo"


@dataclass(frozen=True)
class CISPOConfig:
    """CISPO clipping thresholds; ``ratio_log_cap`` clamps the log-ratio before ``exp``."""

    eps_low: float = 0.2
    eps_high: float = 0.28
    ratio_log_cap: float = 20.0

    def __post_init__(self) -> None:
        if self.eps_low < 0 or self.eps_high < 0:
            raise ValueError("CISPO eps_low and eps_high must be non-negative.")
        if self.ratio_log_cap < 0:
            raise ValueError("CISPO ratio_log_cap must be non-negative.")


def loss_fn_config(config: CISPOConfig) -> Dict[str, float]:
    """``loss_fn_config`` for :data:`LOSS_FN`."""
    return {
        "clip_low_threshold": 1.0 - config.eps_low,
        "clip_high_threshold": 1.0 + config.eps_high,
        "ratio_log_cap": config.ratio_log_cap,
    }


def build_datums(batch: PolicyBatch, config: CISPOConfig) -> List[tinker.Datum]:
    """Datums for the built-in :data:`LOSS_FN`."""
    del config
    return build_policy_datums(batch)


def metrics(batch: PolicyBatch, result: Any, config: CISPOConfig) -> Dict[str, float]:
    """``cispo_ratio_mean``, ``cispo_clip_frac`` (clamped weights) and ``cispo_kl``."""
    low, high = 1.0 - config.eps_low, 1.0 + config.eps_high
    return policy_ratio_metrics(
        batch,
        result,
        loss_name="cispo",
        prefix=PREFIX,
        outside_trust_region=lambda ratio, *_: (ratio < low) | (ratio > high),
    )


def make_loss_fn(batch: PolicyBatch, config: CISPOConfig):
    """Client-side CISPO loss: ``forward_backward_custom(batch.data, make_loss_fn(batch, config))``."""

    def policy_fn(ctx):
        log_ratio = torch.clamp(
            ctx.resp_pi - ctx.resp_anchor, min=-config.ratio_log_cap, max=config.ratio_log_cap
        )
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - config.eps_low, 1.0 + config.eps_high)
        active = ctx.resp_mask > 0.5
        per_token_loss = -(clipped_ratio.detach() * ctx.resp_pi * ctx.adv) * ctx.resp_mask
        return per_token_loss, {
            "clip_frac": (clipped_ratio.detach()[active] != ratio.detach()[active]).float().mean().item(),
            "ratio_mean": ratio.detach()[active].mean().item(),
        }

    def loss_fn(
        data: List[tinker.Datum], logprobs_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(batch, data, logprobs_list, "cispo", policy_fn)
        return result.total_loss, client_loss_metrics(result, PREFIX)

    return loss_fn
