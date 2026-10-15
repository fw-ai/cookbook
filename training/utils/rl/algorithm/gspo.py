"""GSPO (Group Sequence Policy Optimization).

PPO-style clipping with a **sequence-level importance ratio** (geometric mean
of per-token ratios) against the batch anchor. Same module surface as
:mod:`training.utils.rl.algorithm.grpo`; the optimizer must normalize by
sequences (:data:`NORMALIZATION`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import torch
from fireworks.training import sdk

from training.utils.client import GradAccNormalization
from training.utils.rl.common import (
    client_loss_metrics,
    masked_mean,
    policy_ratio_metrics,
    run_loss_loop,
)
from training.utils.rl.losses import PolicyBatch, build_policy_datums

LOSS_FN = "gspo"
"""Trainer built-in GSPO loss."""

NORMALIZATION = GradAccNormalization.NUM_SEQUENCES
"""GSPO gives every response equal weight, so gradients average over sequences."""

PREFIX = "gspo"


@dataclass(frozen=True)
class GSPOConfig:
    """GSPO clipping configuration.

    ``clip_ratio_low`` / ``clip_ratio_high`` are the lower and upper clip
    epsilons. Set them equal for symmetric clipping.
    """

    clip_ratio_low: float = 3e-4
    clip_ratio_high: float = 4e-4
    seq_ratio_log_cap: float = 10.0
    token_reduction: Literal["mean", "sum"] = "mean"
    """``"sum"`` (client loss only) scales each response by its active length."""

    def __post_init__(self) -> None:
        if self.clip_ratio_low < 0 or self.clip_ratio_high < 0:
            raise ValueError("GSPO clip_ratio_low and clip_ratio_high must be non-negative.")
        if self.seq_ratio_log_cap < 0:
            raise ValueError("GSPO seq_ratio_log_cap must be non-negative.")
        if self.token_reduction not in {"mean", "sum"}:
            raise ValueError("GSPO token_reduction must be 'mean' or 'sum'.")


def loss_fn_config(config: GSPOConfig) -> Dict[str, float]:
    """``loss_fn_config`` for :data:`LOSS_FN` (token-mean reduction only)."""
    if config.token_reduction != "mean":
        raise ValueError(
            "built-in GSPO supports only token_reduction='mean'; use make_loss_fn "
            "with forward_backward_custom for token-sum."
        )
    return {
        "clip_low_threshold": 1.0 - config.clip_ratio_low,
        "clip_high_threshold": 1.0 + config.clip_ratio_high,
        "seq_ratio_log_cap": config.seq_ratio_log_cap,
    }


def build_datums(batch: PolicyBatch, config: GSPOConfig) -> List[sdk.Datum]:
    """Datums for the built-in :data:`LOSS_FN`; the response mask
    keeps zero-advantage responses in the sequence count."""
    del config
    return build_policy_datums(batch)


def metrics(batch: PolicyBatch, result: Any, config: GSPOConfig) -> Dict[str, float]:
    """Sequence-ratio ``gspo_ratio_mean``, ``gspo_clip_frac`` and ``gspo_kl``."""
    low, high = 1.0 - config.clip_ratio_low, 1.0 + config.clip_ratio_high
    return policy_ratio_metrics(
        batch,
        result,
        loss_name="gspo",
        prefix=PREFIX,
        outside_trust_region=lambda ratio, *_: (ratio < low) | (ratio > high),
        sequence_level=True,
    )


def make_loss_fn(batch: PolicyBatch, config: GSPOConfig):
    """Client-side GSPO loss: ``forward_backward_custom(batch.data, make_loss_fn(batch, config))``."""
    clip_low, clip_high = config.clip_ratio_low, config.clip_ratio_high

    def policy_fn(ctx):
        # The geometric-mean ratio is a sequence statistic shared by every
        # token, so masked positions must be excluded from the average: their
        # sampling logprobs are padded zeros and would corrupt the ratio for
        # the whole response.
        log_ratio = ctx.resp_pi - ctx.resp_anchor
        seq_log_ratio = masked_mean(log_ratio, ctx.resp_mask)
        log_seq_ratio = ctx.resp_pi - ctx.resp_pi.detach() + seq_log_ratio.detach()
        log_seq_ratio = torch.clamp(log_seq_ratio, max=config.seq_ratio_log_cap)
        seq_ratio = torch.exp(log_seq_ratio)

        clipped_seq_ratio = torch.clamp(seq_ratio, min=1.0 - clip_low, max=1.0 + clip_high)
        surr1 = -seq_ratio * ctx.adv
        surr2 = -clipped_seq_ratio * ctx.adv
        per_token_loss = torch.maximum(surr1, surr2) * ctx.resp_mask
        # The paper-aligned default gives every response equal weight. The
        # token-sum ablation intentionally scales each response by its active
        # length before optim_step(num_sequences) normalizes the batch.
        if config.token_reduction == "mean":
            per_token_loss = per_token_loss / (ctx.resp_mask > 0.5).sum()
        return per_token_loss, {
            "clip_frac": masked_mean((clipped_seq_ratio != seq_ratio).float(), ctx.resp_mask).item(),
            "ratio_mean": masked_mean(seq_ratio.detach(), ctx.resp_mask).item(),
        }

    def loss_fn(data: List[sdk.Datum], logprobs_list: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(batch, data, logprobs_list, "gspo", policy_fn)
        return result.total_loss, client_loss_metrics(result, PREFIX)

    return loss_fn
