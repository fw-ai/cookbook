"""CISPO (Clipped Importance Sampling Policy Optimization) loss.

Clips the importance-sampling ratio ``pi/pi_prox`` to ``[1-eps_low,
1+eps_high]`` and uses the clipped ratio as a *detached* weight on the
log-probability.  This matches the Tinker kernel formula::

    clipped_ratio = clamp(ratio, 1 - eps_low, 1 + eps_high)
    per_token_loss = -(clipped_ratio.detach() * logprob * advantage) * tis_weight

Gradient flows through ``logprob`` (not through ``ratio``), and the
clipped ratio caps the effective learning signal for tokens whose
policy has already moved far from the proximal checkpoint.  TIS weight
corrects for the train-inference gap.
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
class CISPOConfig:
    """CISPO clipping thresholds.

    The ratio ``pi/pi_prox`` is clamped to ``[1 - eps_low, 1 + eps_high]``.
    ``ratio_log_cap`` clamps log-ratio before exp() for numerical stability.
    """

    eps_low: float = 0.2
    eps_high: float = 0.28
    ratio_log_cap: float = 20.0


def make_cispo_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    prox_logprobs: List[List[float]],
    cispo_config: CISPOConfig | None = None,
    tis_config: TISConfig | None = None,
) -> ...:
    """Build a CISPO loss closure with ratio clipping and behavioral TIS weight."""
    if cispo_config is None:
        cispo_config = CISPOConfig()
    if tis_config is None:
        tis_config = TISConfig()
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))

    def policy_fn(ctx):
        log_ratio = torch.clamp(
            ctx.resp_pi - ctx.resp_prox,
            min=-cispo_config.ratio_log_cap,
            max=cispo_config.ratio_log_cap,
        )
        ratio = torch.exp(log_ratio)

        clip_lo = 1.0 - cispo_config.eps_low
        clip_hi = 1.0 + cispo_config.eps_high
        clipped_ratio = torch.clamp(ratio, clip_lo, clip_hi)
        clip_frac = (clipped_ratio.detach() != ratio.detach()).float().mean().item()
        ratio_mean = ratio.detach().mean().item()

        per_token_loss = -(clipped_ratio.detach() * ctx.resp_pi * ctx.adv) * ctx.tis_weight * ctx.resp_mask
        return per_token_loss, {"clip_frac": clip_frac, "ratio_mean": ratio_mean}

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(
            advantages, ref_logprobs, inf_logprobs, prompt_lens,
            prox_logprobs, tis_config, data, logprobs_list, "cispo", policy_fn,
        )
        metrics = dict(result.base_metrics)
        ns = result.n_samples
        metrics["cispo_clip_frac"] = result.extra_sums.get("clip_frac", 0.0) / ns
        metrics["ppo_ratio_mean"] = result.extra_sums.get("ratio_mean", 0.0) / ns
        return result.total_loss, metrics

    return loss_fn


def _builtin_config(
    *, cispo_config: CISPOConfig | None = None, **_kw: Any,
) -> tuple[str, dict[str, Any]]:
    cfg = cispo_config or CISPOConfig()
    return "cispo", {
        "clip_low_threshold": 1.0 - cfg.eps_low,
        "clip_high_threshold": 1.0 + cfg.eps_high,
        "ratio_log_cap": cfg.ratio_log_cap,
    }


def _client_loss_factory(
    *,
    advantages: List[float],
    ref_logprobs: List[List[float]],
    prompt_lens: List[int],
    inf_logprobs: List[List[float]],
    prox_logprobs: List[List[float]],
    cispo_config: CISPOConfig | None,
    tis_config: TISConfig,
    **_kw: Any,
) -> Any:
    return make_cispo_loss_fn(
        advantages,
        ref_logprobs,
        inf_logprobs,
        prompt_lens,
        prox_logprobs,
        cispo_config,
        tis_config=tis_config,
    )


LOSS_SPEC = LossSpec(
    name="cispo",
    client_loss_factory=_client_loss_factory,
    builtin_config_builder=_builtin_config,
)
