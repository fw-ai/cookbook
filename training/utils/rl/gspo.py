"""GSPO (Group Sequence Policy Optimization) loss for GRPO training.

Implements PPO-style clipping with a **sequence-level importance ratio**
(geometric mean of per-token ratios) against pre-computed proximal
logprobs, with behavioral TIS weight correction.

Example::

    Config(policy_loss="gspo", gspo=GSPOConfig())
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens, run_loss_loop
from training.utils.rl.spec import LossSpec
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
    kl_beta: float = 0.001


def make_gspo_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    prox_logprobs: List[List[float]],
    gspo_config: GSPOConfig | None = None,
    tis_config: TISConfig | None = None,
) -> ...:
    """Build a GSPO loss closure with sequence-level PPO ratio and behavioral TIS weight."""
    if gspo_config is None:
        gspo_config = GSPOConfig()
    if tis_config is None:
        tis_config = TISConfig()
    clip_low = gspo_config.clip_ratio_low
    clip_high = gspo_config.clip_ratio_high
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))

    def policy_fn(ctx):
        log_ratio = ctx.resp_pi - ctx.resp_prox
        seq_log_ratio = log_ratio.mean()
        log_seq_ratio = ctx.resp_pi - ctx.resp_pi.detach() + seq_log_ratio.detach()
        log_seq_ratio = torch.clamp(log_seq_ratio, max=gspo_config.seq_ratio_log_cap)
        seq_ratio = torch.exp(log_seq_ratio)

        clipped_seq_ratio = torch.clamp(seq_ratio, min=1.0 - clip_low, max=1.0 + clip_high)
        clip_frac = (clipped_seq_ratio != seq_ratio).float().mean().item()
        ratio_mean = seq_ratio.detach().mean().item()

        surr1 = -seq_ratio * ctx.adv
        surr2 = -clipped_seq_ratio * ctx.adv
        per_token_loss = torch.maximum(surr1, surr2) * ctx.tis_weight * ctx.resp_mask
        return per_token_loss, {"clip_frac": clip_frac, "ratio_mean": ratio_mean}

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(
            advantages, ref_logprobs, inf_logprobs, prompt_lens,
            prox_logprobs, tis_config, data, logprobs_list, "gspo", policy_fn,
        )
        metrics = dict(result.base_metrics)
        ns = result.n_samples
        metrics["gspo_clip_frac"] = result.extra_sums.get("clip_frac", 0.0) / ns
        metrics["ppo_ratio_mean"] = result.extra_sums.get("ratio_mean", 0.0) / ns
        return result.total_loss, metrics

    return loss_fn


def _builtin_config(
    *, gspo_config: GSPOConfig | None = None, **_kw: Any,
) -> tuple[str, dict[str, Any]]:
    cfg = gspo_config or GSPOConfig()
    return "gspo", {
        "clip_low_threshold": 1.0 - cfg.clip_ratio_low,
        "clip_high_threshold": 1.0 + cfg.clip_ratio_high,
        "seq_ratio_log_cap": cfg.seq_ratio_log_cap,
    }


def _make_loss(
    *,
    advantages: List[float],
    ref_logprobs: List[List[float]],
    prompt_lens: List[int],
    inf_logprobs: List[List[float]],
    prox_logprobs: List[List[float]],
    gspo_config: GSPOConfig | None,
    tis_config: TISConfig,
    **_kw: Any,
) -> Any:
    return make_gspo_loss_fn(
        advantages,
        ref_logprobs,
        inf_logprobs,
        prompt_lens,
        prox_logprobs,
        gspo_config,
        tis_config=tis_config,
    )


LOSS_SPEC = LossSpec(
    name="gspo",
    make_loss_fn=_make_loss,
    builtin_config_builder=_builtin_config,
)
