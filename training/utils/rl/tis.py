"""TIS (Train-Inference Importance Sampling) weight correction.

Corrects for the numerical gap between the training model and the
inference deployment (FP8, quantization, different parallelism, etc.)
by weighting each token with ``clamp(exp(old_policy - inf), max=cap)``.

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
    icepop_threshold: float | None = None
    """Optional token-level IcePop keep threshold for raw train-inference ratio.

    ``threshold=2`` keeps ratios inside ``[0.5, 2.0]`` and zeros out
    everything else.
    """

def compute_tis_weight(
    resp_old_policy: torch.Tensor,
    resp_inf: torch.Tensor,
    config: TISConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute TIS weight: clamp(exp(old_policy - inf), max=cap)."""
    if config.icepop_threshold is not None and config.level != "token":
        raise ValueError("icepop_threshold is only supported for token-level TIS.")
    if config.icepop_threshold is not None and config.icepop_threshold < 1.0:
        raise ValueError(
            f"TIS icepop_threshold must be >= 1.0, got {config.icepop_threshold}."
        )

    tis_log = torch.clamp(resp_old_policy - resp_inf, min=-SAFETY_CLAMP, max=SAFETY_CLAMP)

    if config.level == "sequence":
        tis_raw = torch.exp(tis_log.mean()).expand_as(tis_log)
    else:
        tis_raw = torch.exp(tis_log)

    tis_weight = torch.clamp(tis_raw, min=0.0, max=config.cap)
    clip_frac = (tis_weight != tis_raw).float().mean().item()

    if config.icepop_threshold is not None:
        lower = 1.0 / config.icepop_threshold
        upper = config.icepop_threshold
        keep = (tis_raw >= lower) & (tis_raw <= upper)
        tis_weight = torch.where(keep, tis_weight, torch.zeros_like(tis_weight))
        icepop_clip_frac = (~keep).float().mean().item()
    else:
        icepop_clip_frac = 0.0

    metrics: dict[str, float] = {
        "tis/weight_mean": tis_weight.mean().item(),
        "tis/clip_frac": clip_frac,
    }
    if config.icepop_threshold is not None:
        metrics["tis/icepop_clip_frac"] = icepop_clip_frac
    if config.level == "sequence":
        metrics["tis/seq_ratio"] = tis_raw[0].item() if tis_raw.numel() > 0 else 1.0

    return tis_weight, metrics
