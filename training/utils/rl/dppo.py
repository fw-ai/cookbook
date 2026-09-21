"""Divergence Proximal Policy Optimization (DPPO) policy loss.

Implements the binary trust-region approximations from
https://arxiv.org/abs/2602.04879. Unlike PPO ratio clipping, DPPO masks an
advantage-improving token update only after its direct policy divergence from
the rollout behavior policy exceeds a configured threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Union

import tinker
import torch

from training.utils.rl.common import (
    _normalize_prompt_lens,
    masked_mean,
    run_loss_loop,
)
from training.utils.rl.tis import TISConfig


@dataclass
class DPPOConfig:
    """DPPO binary-divergence trust-region configuration."""

    divergence: Literal["binary_tv", "binary_kl"] = "binary_tv"
    threshold: float = 0.15
    ratio_log_cap: float = 20.0


def validate_dppo_config(config: DPPOConfig) -> None:
    """Validate DPPO settings before building the loss closure."""
    if config.divergence not in {"binary_tv", "binary_kl"}:
        raise ValueError(
            "DPPO divergence must be 'binary_tv' or 'binary_kl'."
        )
    if config.threshold < 0:
        raise ValueError("DPPO threshold must be non-negative.")
    if config.ratio_log_cap < 0:
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


def make_dppo_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    old_policy_logprobs: List[List[float]],
    dppo_config: DPPOConfig | None = None,
    tis_config: TISConfig | None = None,
) -> ...:
    """Build DPPO with a binary-TV/KL divergence trust-region mask."""
    if dppo_config is None:
        dppo_config = DPPOConfig()
    validate_dppo_config(dppo_config)
    if tis_config is None:
        tis_config = TISConfig()
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))

    def policy_fn(ctx):
        log_ratio = torch.clamp(
            ctx.resp_pi - ctx.resp_old_policy,
            min=-dppo_config.ratio_log_cap,
            max=dppo_config.ratio_log_cap,
        )
        ratio = torch.exp(log_ratio)
        divergence = _binary_divergence(
            ctx.resp_pi,
            ctx.resp_old_policy,
            dppo_config.divergence,
        )
        improving = ((ctx.adv > 0) & (ratio > 1.0)) | (
            (ctx.adv < 0) & (ratio < 1.0)
        )
        trust_region_mask = ~(
            improving & (divergence > dppo_config.threshold)
        )
        per_token_loss = (
            -ratio
            * ctx.adv
            * trust_region_mask.detach()
            * ctx.tis_weight
            * ctx.resp_mask
        )
        return per_token_loss, {
            "mask_frac": masked_mean(
                (~trust_region_mask).float(), ctx.resp_mask
            ).item(),
            "divergence_mean": masked_mean(
                divergence.detach(), ctx.resp_mask
            ).item(),
            "ratio_mean": masked_mean(
                ratio.detach(), ctx.resp_mask
            ).item(),
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
            "dppo",
            policy_fn,
        )
        metrics = dict(result.base_metrics)
        ns = result.n_samples
        metrics["dppo_mask_frac"] = (
            result.extra_sums.get("mask_frac", 0.0) / ns
        )
        metrics["dppo_divergence_mean"] = (
            result.extra_sums.get("divergence_mean", 0.0) / ns
        )
        metrics["ppo_ratio_mean"] = (
            result.extra_sums.get("ratio_mean", 0.0) / ns
        )
        return result.total_loss, metrics

    return loss_fn
