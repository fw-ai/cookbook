"""Sampled reverse-KL term for on-policy distillation."""

from __future__ import annotations

import torch

from training.utils.rl.common import SampleContext


def add_sampled_reverse_kl(
    per_token_loss: torch.Tensor,
    ctx: SampleContext,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Add the stop-gradient reverse-KL estimator on active response tokens."""
    if beta <= 0:
        return per_token_loss, {}
    if ctx.resp_teacher is None:
        raise ValueError("opd_beta > 0 requires teacher_logprobs.")

    sampled_kl = ctx.pi_detached - ctx.resp_teacher
    opd_loss = beta * sampled_kl * ctx.resp_pi * ctx.tis_weight * ctx.resp_mask
    return per_token_loss + opd_loss, {
        "opd_kl_sum": (sampled_kl * ctx.resp_mask).sum().item(),
        "opd_tokens": ctx.resp_mask.sum().item(),
    }


def add_opd_metrics(metrics: dict[str, float], extra_sums: dict[str, float]) -> None:
    tokens = extra_sums.get("opd_tokens", 0.0)
    if tokens > 0:
        metrics["opd_kl_mean"] = extra_sums.get("opd_kl_sum", 0.0) / tokens
