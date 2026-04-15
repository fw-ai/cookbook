"""RLSD (RLVR with Self-Distillation) credit assignment.

Per-token credit weight from the evidence ratio between a privileged
teacher and the student (arXiv 2604.03128, "Self-Distilled RLVR").

Composable with any policy loss (GRPO, DAPO, GSPO, CISPO, etc.) —
the RLSD weight multiplies into the per-token advantage alongside the
existing TIS weight.

The teacher is the same model conditioned on privileged information
(e.g. a reference answer). Only the scalar log-probability of each
sampled token is needed — no full logits or KL divergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class RLSDConfig:
    """RLSD credit assignment configuration."""

    eps_w: float = 0.2
    """Clipping bound for evidence weights.  Limits max per-token credit
    deviation to ``[1 - eps_w, 1 + eps_w]``."""

    lam: float = 1.0
    """Interpolation coefficient.  ``0`` = uniform (RLSD disabled),
    ``1`` = full RLSD credit.  Can be ramped up during training."""


def compute_rlsd_weights(
    resp_student: torch.Tensor,
    resp_teacher: torch.Tensor,
    advantage_sign: float,
    config: RLSDConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute RLSD per-token credit weights from evidence ratios.

    Args:
        resp_student: Student (policy) log-probs for response tokens.
        resp_teacher: Teacher (privileged) log-probs for the same tokens.
            Must be the same length as ``resp_student``.
        advantage_sign: ``sign(A)`` for this trajectory (``1.0`` or ``-1.0``).
        config: RLSD configuration.

    Returns:
        ``(weights, metrics)`` where ``weights`` has the same shape as the
        inputs.  When ``A > 0``, tokens that the teacher supports get
        amplified credit; when ``A < 0``, the ratio is inverted so tokens
        the teacher disfavors get more blame.
    """
    delta = resp_teacher - resp_student
    w_raw = torch.exp(advantage_sign * delta)
    w_clipped = torch.clamp(w_raw, min=1.0 - config.eps_w, max=1.0 + config.eps_w)
    w = 1.0 + config.lam * (w_clipped - 1.0)

    metrics: dict[str, float] = {
        "rlsd/weight_mean": w.mean().item(),
        "rlsd/delta_mean": delta.mean().item(),
        "rlsd/delta_abs_mean": delta.abs().mean().item(),
        "rlsd/clip_frac": (w_clipped != w_raw).float().mean().item(),
    }
    return w, metrics
