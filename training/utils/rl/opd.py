"""Sampled reverse-KL term for on-policy distillation."""

from __future__ import annotations

import torch

from training.utils.rl.common import SampleContext


def add_sampled_reverse_kl(
    per_token_loss: torch.Tensor,
    ctx: SampleContext,
    beta: float,
    *,
    top_k: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Add sampled reverse-KL or sparse forward-KL on active response tokens."""
    if beta <= 0:
        return per_token_loss, {}
    if top_k > 0:
        if (
            ctx.resp_teacher_topk_probs is None
            or ctx.resp_student_topk_logprobs is None
        ):
            raise ValueError("top-K OPD requires sparse teacher targets.")
        teacher_probs = ctx.resp_teacher_topk_probs
        cross_entropy = -(
            teacher_probs * ctx.resp_student_topk_logprobs
        ).sum(dim=-1)
        teacher_entropy = -(
            teacher_probs
            * torch.log(torch.clamp(teacher_probs, min=1e-12))
        ).sum(dim=-1)
        forward_kl = cross_entropy - teacher_entropy
        opd_loss = beta * cross_entropy * ctx.tis_weight * ctx.resp_mask
        return per_token_loss + opd_loss, {
            "opd_forward_kl_sum": (forward_kl * ctx.resp_mask).sum().item(),
            "opd_cross_entropy_sum": (
                cross_entropy * ctx.resp_mask
            ).sum().item(),
            "opd_tokens": ctx.resp_mask.sum().item(),
        }
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
        if "opd_forward_kl_sum" in extra_sums:
            metrics["opd_forward_kl_mean"] = (
                extra_sums["opd_forward_kl_sum"] / tokens
            )
            metrics["opd_cross_entropy_mean"] = (
                extra_sums.get("opd_cross_entropy_sum", 0.0) / tokens
            )
            metrics["opd_kl_mean"] = metrics["opd_forward_kl_mean"]
        else:
            metrics["opd_kl_mean"] = extra_sums.get("opd_kl_sum", 0.0) / tokens
