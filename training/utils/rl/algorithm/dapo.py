"""DAPO: PPO clipping with asymmetric bounds and optional dual clip.

Reference: https://arxiv.org/abs/2503.14476. Same module surface as
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

LOSS_FN = "dapo"
"""Trainer built-in DAPO loss."""

NORMALIZATION = GradAccNormalization.NUM_LOSS_TOKENS

PREFIX = "dapo"


@dataclass(frozen=True)
class DAPOConfig:
    """DAPO clipping thresholds.

    The ratio is clipped to ``[1 - eps_clip, 1 + eps_clip_high]``; equal
    values recover standard PPO clipping. ``eps_clip_c`` enables dual-clip PPO
    for negative-advantage tokens. ``ratio_log_cap`` clamps the log-ratio
    before ``exp`` for numerical stability.
    """

    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    eps_clip_c: float | None = None
    ratio_log_cap: float = 20.0

    def __post_init__(self) -> None:
        if self.eps_clip < 0 or self.eps_clip_high < 0:
            raise ValueError("DAPO eps_clip and eps_clip_high must be non-negative.")
        if self.eps_clip_c is not None and self.eps_clip_c <= 1.0:
            raise ValueError(f"DAPO dual-clip eps_clip_c must be > 1.0, got {self.eps_clip_c}.")
        if self.ratio_log_cap < 0:
            raise ValueError("DAPO ratio_log_cap must be non-negative.")


def loss_fn_config(config: DAPOConfig) -> Dict[str, float]:
    """``loss_fn_config`` for :data:`LOSS_FN`."""
    out = {
        "clip_low_threshold": 1.0 - config.eps_clip,
        "clip_high_threshold": 1.0 + config.eps_clip_high,
        "ratio_log_cap": config.ratio_log_cap,
    }
    if config.eps_clip_c is not None:
        out["eps_clip_c"] = config.eps_clip_c
    return out


def build_datums(batch: PolicyBatch, config: DAPOConfig) -> List[tinker.Datum]:
    """Datums for the built-in :data:`LOSS_FN`."""
    del config
    return build_policy_datums(batch)


def metrics(batch: PolicyBatch, result: Any, config: DAPOConfig) -> Dict[str, float]:
    """``dapo_ratio_mean``, ``dapo_clip_frac`` and ``dapo_kl`` for one trainer call."""
    low, high = 1.0 - config.eps_clip, 1.0 + config.eps_clip_high
    return policy_ratio_metrics(
        batch,
        result,
        loss_name="dapo",
        prefix=PREFIX,
        outside_trust_region=lambda ratio, *_: (ratio < low) | (ratio > high),
    )


def make_loss_fn(batch: PolicyBatch, config: DAPOConfig):
    """Client-side DAPO loss: ``forward_backward_custom(batch.data, make_loss_fn(batch, config))``."""

    def policy_fn(ctx):
        log_ratio = torch.clamp(
            ctx.resp_pi - ctx.resp_anchor, min=-config.ratio_log_cap, max=config.ratio_log_cap
        )
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, min=1.0 - config.eps_clip, max=1.0 + config.eps_clip_high)
        clipped_surrogate = torch.maximum(-ratio * ctx.adv, -clipped_ratio * ctx.adv)
        if config.eps_clip_c is not None:
            lower_clipped = torch.minimum(-config.eps_clip_c * ctx.adv, clipped_surrogate)
            per_token_loss = torch.where(ctx.adv < 0, lower_clipped, clipped_surrogate)
        else:
            per_token_loss = clipped_surrogate
        active = ctx.resp_mask > 0.5
        return per_token_loss * ctx.resp_mask, {
            "clip_frac": (clipped_ratio[active] != ratio[active]).float().mean().item(),
            "ratio_mean": ratio.detach()[active].mean().item(),
        }

    def loss_fn(
        data: List[tinker.Datum], logprobs_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(batch, data, logprobs_list, "dapo", policy_fn)
        return result.total_loss, client_loss_metrics(result, PREFIX)

    return loss_fn
