"""DAPO (Dynamic Advantage Policy Optimization) loss for GRPO training.

Uses PPO-style clipped surrogate objective with asymmetric clipping bounds
and behavioral TIS weight correction.  The PPO ratio is computed against
pre-computed proximal logprobs.

Reference: https://arxiv.org/abs/2503.14476
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens, run_loss_loop
from training.utils.rl.tis import TISConfig


@dataclass
class DAPOConfig:
    """DAPO clipping thresholds.

    ``eps_clip`` is the lower clipping bound (ratio >= 1 - eps_clip).
    ``eps_clip_high`` is the upper clipping bound (ratio <= 1 + eps_clip_high).
    Setting them equal recovers standard PPO clipping.
    ``eps_clip_c`` enables optional dual-clip PPO for negative-advantage tokens.
    ``ratio_log_cap`` clamps log-ratio before exp() for numerical stability.
    """

    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    eps_clip_c: float | None = None
    ratio_log_cap: float = 20.0


def make_dapo_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    prox_logprobs: List[List[float]],
    dapo_config: DAPOConfig | None = None,
    tis_config: TISConfig | None = None,
) -> ...:
    """Build a DAPO loss closure with PPO-clipped ratio and behavioral TIS weight."""
    if dapo_config is None:
        dapo_config = DAPOConfig()
    if tis_config is None:
        tis_config = TISConfig()
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))

    def policy_fn(ctx):
        log_ratio = torch.clamp(
            ctx.resp_pi - ctx.resp_prox,
            min=-dapo_config.ratio_log_cap,
            max=dapo_config.ratio_log_cap,
        )
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, min=1.0 - dapo_config.eps_clip, max=1.0 + dapo_config.eps_clip_high)
        clip_frac = (clipped_ratio != ratio).float().mean().item()
        ratio_mean = ratio.detach().mean().item()

        surr1 = -ratio * ctx.adv
        surr2 = -clipped_ratio * ctx.adv
        clipped_surrogate = torch.maximum(surr1, surr2)

        if dapo_config.eps_clip_c is not None:
            if dapo_config.eps_clip_c <= 1.0:
                raise ValueError(f"DAPO dual-clip eps_clip_c must be > 1.0, got {dapo_config.eps_clip_c}.")
            surr3 = -dapo_config.eps_clip_c * ctx.adv
            lower_clipped = torch.minimum(surr3, clipped_surrogate)
            per_token_loss = torch.where(ctx.adv < 0, lower_clipped, clipped_surrogate)
        else:
            per_token_loss = clipped_surrogate

        per_token_loss = per_token_loss * ctx.tis_weight * ctx.resp_mask
        return per_token_loss, {"clip_frac": clip_frac, "ratio_mean": ratio_mean}

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(
            advantages, ref_logprobs, inf_logprobs, prompt_lens,
            prox_logprobs, tis_config, data, logprobs_list, policy_fn,
        )
        metrics = dict(result.base_metrics)
        ns = result.n_samples
        metrics["dapo_clip_frac"] = result.extra_sums.get("clip_frac", 0.0) / ns
        metrics["ppo_ratio_mean"] = result.extra_sums.get("ratio_mean", 0.0) / ns
        return result.total_loss, metrics

    return loss_fn
