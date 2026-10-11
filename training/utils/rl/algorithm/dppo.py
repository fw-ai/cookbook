"""Divergence Proximal Policy Optimization (DPPO) policy loss.

Implements the binary trust-region approximations from
https://arxiv.org/abs/2602.04879. Unlike PPO ratio clipping, DPPO masks an
advantage-improving token update only after its direct policy divergence from
the anchor exceeds a configured threshold. Same module surface as
:mod:`training.utils.rl.algorithm.grpo`; ``dppo_clip_frac`` is the share of
tokens past the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import tinker
import torch

from training.utils.client import GradAccNormalization
from training.utils.rl.common import (
    client_loss_metrics,
    masked_mean,
    policy_ratio_metrics,
    run_loss_loop,
)
from training.utils.rl.losses import PolicyBatch, build_policy_datums

LOSS_FN = "dppo"
"""Trainer built-in DPPO loss."""

NORMALIZATION = GradAccNormalization.NUM_LOSS_TOKENS

PREFIX = "dppo"

_DEFAULT_THRESHOLD = {
    "binary_tv": 0.15,
    "binary_kl": 0.05,
}


@dataclass(frozen=True)
class DPPOConfig:
    """DPPO binary-divergence trust region; ``threshold=None`` uses the divergence's default."""

    divergence: Literal["binary_tv", "binary_kl"] = "binary_tv"
    threshold: float | None = None
    ratio_log_cap: float = 20.0

    def __post_init__(self) -> None:
        if self.divergence not in _DEFAULT_THRESHOLD:
            raise ValueError("DPPO divergence must be 'binary_tv' or 'binary_kl'.")
        if self.threshold is None:
            object.__setattr__(self, "threshold", _DEFAULT_THRESHOLD[self.divergence])
        if self.threshold < 0:
            raise ValueError("DPPO threshold must be non-negative.")
        if self.ratio_log_cap < 0:
            raise ValueError("DPPO ratio_log_cap must be non-negative.")


def _binary_divergence(
    policy_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    kind: Literal["binary_tv", "binary_kl"],
) -> torch.Tensor:
    policy_prob = torch.exp(policy_logprobs)
    behavior_prob = torch.exp(behavior_logprobs)
    if kind == "binary_tv":
        return torch.abs(behavior_prob - policy_prob)

    eps = torch.finfo(policy_prob.dtype).eps
    policy_prob = policy_prob.clamp(min=eps, max=1.0 - eps)
    behavior_prob = behavior_prob.clamp(min=eps, max=1.0 - eps)
    return behavior_prob * torch.log(behavior_prob / policy_prob) + (
        1.0 - behavior_prob
    ) * torch.log((1.0 - behavior_prob) / (1.0 - policy_prob))


def loss_fn_config(config: DPPOConfig) -> Dict[str, float | str]:
    """``loss_fn_config`` for :data:`LOSS_FN`."""
    return {
        "divergence": config.divergence,
        "ratio_log_cap": config.ratio_log_cap,
        "threshold": float(config.threshold),
    }


def build_datums(batch: PolicyBatch, config: DPPOConfig) -> List[tinker.Datum]:
    """Datums for the built-in :data:`LOSS_FN`."""
    del config
    return build_policy_datums(batch)


def metrics(batch: PolicyBatch, result: Any, config: DPPOConfig) -> Dict[str, float]:
    """``dppo_ratio_mean``, ``dppo_kl`` and ``dppo_clip_frac`` (tokens past the threshold)."""
    threshold = float(config.threshold)

    def outside(_ratio, policy, anchor):
        return _binary_divergence(policy, anchor, config.divergence) > threshold

    return policy_ratio_metrics(
        batch, result, loss_name="dppo", prefix=PREFIX, outside_trust_region=outside
    )


def make_loss_fn(batch: PolicyBatch, config: DPPOConfig):
    """Client-side DPPO loss; also reports ``dppo_divergence_mean``."""

    def policy_fn(ctx):
        log_ratio = torch.clamp(
            ctx.resp_pi - ctx.resp_anchor, min=-config.ratio_log_cap, max=config.ratio_log_cap
        )
        ratio = torch.exp(log_ratio)
        divergence = _binary_divergence(ctx.resp_pi, ctx.resp_anchor, config.divergence)
        improving = ((ctx.adv > 0) & (ratio > 1.0)) | ((ctx.adv < 0) & (ratio < 1.0))
        outside = divergence.detach() > config.threshold
        trust_region_mask = ~(improving & outside)
        per_token_loss = -ratio * ctx.adv * trust_region_mask.detach() * ctx.resp_mask
        return per_token_loss, {
            "clip_frac": masked_mean(outside.float(), ctx.resp_mask).item(),
            "divergence_mean": masked_mean(divergence.detach(), ctx.resp_mask).item(),
            "ratio_mean": masked_mean(ratio.detach(), ctx.resp_mask).item(),
        }

    def loss_fn(
        data: List[tinker.Datum], logprobs_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(batch, data, logprobs_list, "dppo", policy_fn)
        out = client_loss_metrics(result, PREFIX)
        out[f"{PREFIX}_divergence_mean"] = result.extra_sums.get("divergence_mean", 0.0) / result.n_samples
        return result.total_loss, out

    return loss_fn
