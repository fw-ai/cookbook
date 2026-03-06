"""Importance sampling corrections for RL training.

Two corrections applied to every loss function:

1. **PPO IS ratio** -- ``exp(pi_theta - pi_prox)`` with PPO-style
   clipping.  The proximal logprobs are pre-computed via a real forward
   pass before the training loop.  In a 1:1 on-policy loop the ratio
   is 1 (no effect); it becomes non-trivial with off-policy or
   multi-minibatch training.

2. **TIS (Train-Inference IS) weight** -- ``exp(pi_prox - pi_old)``
   clamped at ``tis_cap``.  Corrects for the numerical gap between the
   training model and the inference deployment (FP8, quantization,
   different parallelism, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SAFETY_CLAMP = 20.0


@dataclass
class ISConfig:
    """Importance sampling correction configuration."""

    eps_clip: float = 0.2
    """PPO clip epsilon for the off-policy ratio (used by GRPO)."""
    eps_clip_high: float | None = None
    """Asymmetric upper clip bound for GRPO."""
    tis_cap: float = 5.0
    """Upper clamp for the TIS weight."""
    tis_level: str = "token"
    """'token': per-token IS weights.  'sequence': geometric mean
    of per-token ratios, broadcast to all tokens."""


def compute_tis_weight(
    resp_prox: torch.Tensor,
    resp_inf: torch.Tensor,
    config: ISConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute TIS weight: clamp(exp(prox - inf), max=tis_cap)."""
    tis_log = torch.clamp(resp_prox - resp_inf, min=-SAFETY_CLAMP, max=SAFETY_CLAMP)

    if config.tis_level == "sequence":
        tis_raw = torch.exp(tis_log.mean()).expand_as(tis_log)
    else:
        tis_raw = torch.exp(tis_log)

    tis_weight = torch.clamp(tis_raw, min=0.0, max=config.tis_cap)
    clip_frac = (tis_weight != tis_raw).float().mean().item()

    metrics: dict[str, float] = {
        "tis/weight_mean": tis_weight.mean().item(),
        "tis/clip_frac": clip_frac,
    }
    if config.tis_level == "sequence":
        metrics["tis/seq_ratio"] = tis_raw[0].item() if tis_raw.numel() > 0 else 1.0

    return tis_weight, metrics
