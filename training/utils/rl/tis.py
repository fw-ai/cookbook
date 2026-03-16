"""TIS (Train-Inference Importance Sampling) weight correction.

Corrects for the numerical gap between the training model and the
inference deployment (FP8, quantization, different parallelism, etc.)
by weighting each token with ``clamp(exp(prox - inf), max=cap)``.

Composable with any policy loss (GRPO, DAPO, GSPO, CISPO, IS, DRO).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SAFETY_CLAMP = 20.0


@dataclass
class TISConfig:
    """TIS weight configuration."""

    cap: float = 5.0
    """Upper clamp for the TIS weight."""
    level: str = "token"
    """'token': per-token IS weights.  'sequence': geometric mean
    of per-token ratios, broadcast to all tokens."""



def compute_tis_weight(
    resp_prox: torch.Tensor,
    resp_inf: torch.Tensor,
    config: TISConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute TIS weight: clamp(exp(prox - inf), max=cap)."""
    tis_log = torch.clamp(resp_prox - resp_inf, min=-SAFETY_CLAMP, max=SAFETY_CLAMP)

    if config.level == "sequence":
        tis_raw = torch.exp(tis_log.mean()).expand_as(tis_log)
    else:
        tis_raw = torch.exp(tis_log)

    tis_weight = torch.clamp(tis_raw, min=0.0, max=config.cap)
    clip_frac = (tis_weight != tis_raw).float().mean().item()

    metrics: dict[str, float] = {
        "tis/weight_mean": tis_weight.mean().item(),
        "tis/clip_frac": clip_frac,
    }
    if config.level == "sequence":
        metrics["tis/seq_ratio"] = tis_raw[0].item() if tis_raw.numel() > 0 else 1.0

    return tis_weight, metrics
