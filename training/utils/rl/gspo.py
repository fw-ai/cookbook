"""GSPO (Group Sequence Policy Optimization) loss for GRPO training.

Implements PPO-style clipping with a **sequence-level importance ratio**
(geometric mean of per-token ratios) against pre-computed old-policy
logprobs, with behavioral TIS weight correction.

Example::

    loss_fn = make_gspo_loss_fn(..., gspo_config=GSPOConfig())
    policy.forward_backward_custom(data, loss_fn)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens, run_loss_loop
from training.utils.rl.observability import compute_inference_observability_metrics
from training.utils.rl.tis import TISConfig


@dataclass
class GSPOConfig:
    """GSPO clipping configuration.

    ``clip_ratio_low`` / ``clip_ratio_high`` control the lower and upper
    clip epsilons directly. Set them equal for symmetric clipping.
    """

    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    seq_ratio_log_cap: float = 10.0


def validate_gspo_config(config: GSPOConfig) -> None:
    """Validate GSPO settings before building the loss closure."""
    if config.clip_ratio_low < 0 or config.clip_ratio_high < 0:
        raise ValueError(
            "GSPO clip_ratio_low and clip_ratio_high must be non-negative."
        )
    if config.seq_ratio_log_cap < 0:
        raise ValueError("GSPO seq_ratio_log_cap must be non-negative.")


def make_gspo_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    old_policy_logprobs: List[List[float]],
    gspo_config: GSPOConfig | None = None,
    tis_config: TISConfig | None = None,
    raw_inf_logprobs: List[List[float]] | None = None,
) -> ...:
    """Build a GSPO loss closure with sequence-level PPO ratio and behavioral TIS weight."""
    if gspo_config is None:
        gspo_config = GSPOConfig()
    validate_gspo_config(gspo_config)
    if tis_config is None:
        tis_config = TISConfig()
    clip_low = gspo_config.clip_ratio_low
    clip_high = gspo_config.clip_ratio_high
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))

    def policy_fn(ctx):
        log_ratio = ctx.resp_pi - ctx.resp_old_policy
        active = ctx.resp_mask > 0.5
        active_count = active.sum().clamp_min(1).to(ctx.resp_pi.dtype)
        # GSPO assigns equal weight to each sequence and equal weight to each
        # generated token inside that sequence. ``num_sequences`` performs the
        # outer mean at optim_step; this division performs the inner token mean.
        seq_log_ratio = log_ratio[active].mean()
        log_seq_ratio = ctx.resp_pi - ctx.resp_pi.detach() + seq_log_ratio.detach()
        log_seq_ratio = torch.clamp(log_seq_ratio, max=gspo_config.seq_ratio_log_cap)
        seq_ratio = torch.exp(log_seq_ratio)

        clipped_seq_ratio = torch.clamp(
            seq_ratio, min=1.0 - clip_low, max=1.0 + clip_high
        )
        ratio_value = torch.exp(seq_log_ratio.detach())
        clip_low_frac = (ratio_value < 1.0 - clip_low).float().mean().item()
        clip_high_frac = (ratio_value > 1.0 + clip_high).float().mean().item()
        clip_frac = clip_low_frac + clip_high_frac
        ratio_mean = ratio_value.mean().item()

        surr1 = -seq_ratio * ctx.adv
        surr2 = -clipped_seq_ratio * ctx.adv
        per_token_loss = (
            torch.maximum(surr1, surr2) * ctx.tis_weight * ctx.resp_mask / active_count
        )
        return per_token_loss, {
            "clip_frac": clip_frac,
            "clip_low_frac": clip_low_frac,
            "clip_high_frac": clip_high_frac,
            "ratio_mean": ratio_mean,
        }

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(
            advantages,
            ref_logprobs,
            inf_logprobs,
            prompt_lens,
            old_policy_logprobs,
            tis_config,
            data,
            logprobs_list,
            "gspo",
            policy_fn,
        )
        metrics = dict(result.base_metrics)
        ns = result.n_samples
        clip_frac = result.extra_sums.get("clip_frac", 0.0) / ns
        ratio_mean = result.extra_sums.get("ratio_mean", 0.0) / ns
        metrics.update(
            {
                # Keep the generic PPO aliases for existing dashboards while
                # exposing GSPO-specific names that make the sequence-level
                # meaning explicit.
                "ppo_clip_frac": clip_frac,
                "ppo_ratio_mean": ratio_mean,
                "gspo_clip_frac": clip_frac,
                "gspo_clip_low_frac": result.extra_sums.get("clip_low_frac", 0.0) / ns,
                "gspo_clip_high_frac": result.extra_sums.get("clip_high_frac", 0.0)
                / ns,
                "gspo_sequence_ratio_mean": ratio_mean,
            }
        )
        if raw_inf_logprobs is not None:
            metrics.update(
                compute_inference_observability_metrics(
                    data,
                    logprobs_list,
                    raw_inf_logprobs,
                    prompt_lens,
                    "gspo",
                )
            )
        return result.total_loss, metrics

    return loss_fn
