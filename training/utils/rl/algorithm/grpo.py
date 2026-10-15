"""GRPO: PPO-clipped policy gradient on group-relative advantages.

Every objective module in this package has the same surface, so a fork swaps
objectives by changing one import::

    from training.utils.rl.algorithm import grpo as objective

    datums = objective.build_datums(batch, config)
    result = policy.forward_backward(
        datums, objective.LOSS_FN, loss_fn_config=objective.loss_fn_config(config)
    )
    result.metrics.update(objective.metrics(batch, result, config))
    policy.optim_step(adam, grad_accumulation_normalization=objective.NORMALIZATION, ...)

``make_loss_fn(batch, config)`` is the same objective as a client-side loss and
reports the same metric names. It runs on the batch's own datums, which carry
the loss mask: ``policy.forward_backward_custom(batch.data, make_loss_fn(batch,
config))``. Only the client
loss can add a reference-KL term (``kl_beta`` with ``batch.ref_logprobs``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from fireworks.training import sdk

from training.utils.client import GradAccNormalization
from training.utils.rl.common import SAFETY_CLAMP, policy_ratio_metrics, run_loss_loop
from training.utils.rl.losses import PolicyBatch, build_policy_datums

LOSS_FN = "ppo"
"""Trainer built-in loss that implements the clipped GRPO surrogate."""

NORMALIZATION = GradAccNormalization.NUM_LOSS_TOKENS
"""Token-mean gradients, so clipping and grad_norm do not scale with batch tokens."""

PREFIX = "ppo"
"""Metric prefix shared by the server and client paths."""


@dataclass(frozen=True)
class GRPOConfig:
    """PPO clipping range for the GRPO surrogate."""

    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    """Upper clip epsilon; ``None`` uses ``eps_clip``."""
    kl_beta: float = 0.0
    """Reference-KL weight for the client loss; needs ``batch.ref_logprobs``."""

    def __post_init__(self) -> None:
        if self.kl_beta < 0:
            raise ValueError("kl_beta must be non-negative.")
        if self.eps_clip < 0 or (self.eps_clip_high is not None and self.eps_clip_high < 0):
            raise ValueError("eps_clip and eps_clip_high must be non-negative.")

    @property
    def upper_eps(self) -> float:
        return self.eps_clip if self.eps_clip_high is None else self.eps_clip_high


def loss_fn_config(config: GRPOConfig) -> Dict[str, float]:
    """``loss_fn_config`` for :data:`LOSS_FN`."""
    if config.kl_beta:
        raise ValueError("built-in GRPO has no reference input; use make_loss_fn for kl_beta > 0.")
    return {
        "clip_low_threshold": 1.0 - config.eps_clip,
        "clip_high_threshold": 1.0 + config.upper_eps,
    }


def build_datums(batch: PolicyBatch, config: GRPOConfig) -> List[sdk.Datum]:
    """Datums for the built-in :data:`LOSS_FN`."""
    del config
    return build_policy_datums(batch)


def metrics(batch: PolicyBatch, result: Any, config: GRPOConfig) -> Dict[str, float]:
    """``ppo_ratio_mean``, ``ppo_clip_frac`` and ``ppo_kl`` for one trainer call."""
    low, high = 1.0 - config.eps_clip, 1.0 + config.upper_eps
    return policy_ratio_metrics(
        batch,
        result,
        loss_name="grpo",
        prefix=PREFIX,
        outside_trust_region=lambda ratio, *_: (ratio < low) | (ratio > high),
    )


def make_loss_fn(batch: PolicyBatch, config: GRPOConfig):
    """Client-side GRPO loss: ``forward_backward_custom(batch.data, make_loss_fn(batch, config))``."""
    eps_low, eps_high = config.eps_clip, config.upper_eps
    if config.kl_beta and not batch.ref_logprobs:
        raise ValueError("GRPO kl_beta > 0 needs batch.ref_logprobs.")

    def policy_fn(ctx):
        log_ratio = torch.clamp(ctx.resp_pi - ctx.resp_anchor, min=-SAFETY_CLAMP, max=SAFETY_CLAMP)
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, min=1.0 - eps_low, max=1.0 + eps_high)

        active = ctx.resp_mask > 0.5
        surr1 = -ratio * ctx.adv
        surr2 = -clipped_ratio * ctx.adv
        per_token_loss = torch.maximum(surr1, surr2)

        # Mean coefficient multiplying d(log pi) for this datum, including
        # PPO clipping. This is an inexpensive variance proxy across
        # samples, not per-parameter gradient covariance.
        unclipped_selected = surr1 >= surr2
        clipped_has_gradient = (ratio >= 1.0 - eps_low) & (ratio <= 1.0 + eps_high)
        ratio_with_gradient = torch.where(unclipped_selected | clipped_has_gradient, ratio, torch.zeros_like(ratio))
        pg_coefficient = (-ctx.adv * ratio_with_gradient * ctx.resp_mask)[active]
        pg_datum_coefficient = pg_coefficient.detach().float().mean()
        extra = {
            "clip_frac": (clipped_ratio[active] != ratio[active]).float().mean().item(),
            "ratio_mean": ratio.detach()[active].mean().item(),
            "resp_len": float(len(ctx.resp_pi)),
            "adv_term": (-ctx.adv * ctx.resp_pi * ctx.resp_mask).sum().item(),
            "pg_datum_coefficient_sum": pg_datum_coefficient.item(),
            "pg_datum_coefficient_sq_sum": pg_datum_coefficient.square().item(),
            "pg_datum_count": 1.0,
        }
        if ctx.resp_ref is not None and config.kl_beta:
            ref_log_ratio = ctx.resp_ref - ctx.resp_pi
            kl_penalty = config.kl_beta * (torch.exp(ref_log_ratio) - ref_log_ratio - 1.0)
            per_token_loss = per_token_loss + kl_penalty
            extra["kl_term"] = (kl_penalty * ctx.resp_mask).sum().item()
        return per_token_loss * ctx.resp_mask, extra

    def loss_fn(data: List[sdk.Datum], logprobs_list: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(batch, data, logprobs_list, "grpo", policy_fn)
        ns, nt = result.n_samples, result.num_tokens
        total_resp = int(result.extra_sums.get("resp_len", 0.0))
        out = dict(result.base_metrics)
        if "anchor_kl" in out:
            out[f"{PREFIX}_kl"] = out.pop("anchor_kl")
        out[f"{PREFIX}_clip_frac"] = result.extra_sums.get("clip_frac", 0.0) / ns
        out[f"{PREFIX}_ratio_mean"] = result.extra_sums.get("ratio_mean", 0.0) / ns
        out["active_tokens"] = nt
        out["total_resp_tokens"] = total_resp
        out["mask_ratio"] = nt / total_resp if total_resp > 0 else 0.0
        out["mean_adv_loss"] = result.extra_sums.get("adv_term", 0.0) / nt if nt > 0 else 0.0
        if "kl_term" in result.extra_sums:
            out["mean_kl_penalty"] = result.extra_sums["kl_term"] / nt if nt > 0 else 0.0
        out["mean_loss"] = result.total_loss.item() / nt if nt > 0 else 0.0
        pg_count = int(result.extra_sums.get("pg_datum_count", 0.0))
        if pg_count:
            pg_sum = result.extra_sums.get("pg_datum_coefficient_sum", 0.0)
            pg_sq_sum = result.extra_sums.get("pg_datum_coefficient_sq_sum", 0.0)
            pg_mean = pg_sum / pg_count
            pg_variance = max((pg_sq_sum - pg_sum * pg_mean) / (pg_count - 1), 0.0) if pg_count > 1 else 0.0
            pg_estimator_variance = pg_variance / pg_count
            out["policy_gradient/coefficient_variance_proxy"] = pg_variance
            out["policy_gradient/estimator_variance_proxy"] = pg_estimator_variance
            out["policy_gradient/estimator_std_error_proxy"] = pg_estimator_variance**0.5
            out["policy_gradient/sample_count"] = pg_count
        return result.total_loss, out

    return loss_fn
