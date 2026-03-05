"""DAPO (Dynamic Advantage Policy Optimization) loss for GRPO training.

Uses PPO-style clipped surrogate objective with asymmetric clipping bounds
and behavioral IS weight correction.  The PPO ratio is computed against
pre-computed proximal logprobs.

Reference: https://arxiv.org/abs/2503.14476
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens
from training.utils.rl.importance_sampling import (
    SAFETY_CLAMP,
    DecoupledConfig,
    compute_behave_weight,
)


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
    decoupled_config: DecoupledConfig | None = None,
) -> ...:
    """Build a DAPO loss closure with PPO-clipped ratio and behavioral IS weight."""
    if dapo_config is None:
        dapo_config = DAPOConfig()
    if decoupled_config is None:
        decoupled_config = DecoupledConfig()
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, requires_grad=True)
        total_kl = 0.0
        total_inf_diff = 0.0
        total_inf_kld = 0.0
        inf_num_samples = 0
        num_tokens = 0
        clip_frac_sum = 0.0
        behave_metrics_agg: Dict[str, float] = {}

        for i, pi_logprobs in enumerate(logprobs_list):
            adv = advantages[i]
            ref_lp = ref_logprobs[i]
            inf_lp = inf_logprobs[i]
            prox_lp = prox_logprobs[i]
            response_start = max(0, prompt_lens[i] - 1)

            resp_pi = pi_logprobs[response_start:]
            resp_len = len(resp_pi)
            if resp_len == 0:
                continue

            resp_ref = torch.tensor(
                [ref_lp[response_start + j] if (response_start + j) < len(ref_lp) else 0.0 for j in range(resp_len)],
                dtype=resp_pi.dtype, device=resp_pi.device,
            )
            pi_detached = resp_pi.detach()

            resp_inf = torch.tensor(
                inf_lp[response_start:response_start + resp_len],
                dtype=resp_pi.dtype, device=resp_pi.device,
            )
            resp_prox = torch.tensor(
                prox_lp[response_start:response_start + resp_len],
                dtype=resp_pi.dtype, device=resp_pi.device,
            )

            inf_log_diff = pi_detached - resp_inf
            total_inf_diff += inf_log_diff.abs().mean().item()
            total_inf_kld += (torch.exp(inf_log_diff) - inf_log_diff - 1.0).mean().item()
            inf_num_samples += 1

            log_ratio = torch.clamp(resp_pi - resp_prox, min=-dapo_config.ratio_log_cap, max=dapo_config.ratio_log_cap)
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, min=1.0 - dapo_config.eps_clip, max=1.0 + dapo_config.eps_clip_high)
            clip_frac_sum += (clipped_ratio != ratio).float().mean().item()

            behave_weight, bm = compute_behave_weight(resp_prox, resp_inf, decoupled_config)
            for k, v in bm.items():
                behave_metrics_agg[k] = behave_metrics_agg.get(k, 0.0) + v

            adv_t = torch.as_tensor(adv, dtype=resp_pi.dtype, device=resp_pi.device)
            surr1 = -ratio * adv_t
            surr2 = -clipped_ratio * adv_t
            clipped_surrogate = torch.maximum(surr1, surr2)
            if dapo_config.eps_clip_c is not None:
                if dapo_config.eps_clip_c <= 1.0:
                    raise ValueError(f"DAPO dual-clip eps_clip_c must be > 1.0, got {dapo_config.eps_clip_c}.")
                surr3 = -dapo_config.eps_clip_c * adv_t
                lower_clipped = torch.minimum(surr3, clipped_surrogate)
                per_token_loss = torch.where(adv_t < 0, lower_clipped, clipped_surrogate)
            else:
                per_token_loss = clipped_surrogate
            per_token_loss = per_token_loss * behave_weight

            total_loss = total_loss + per_token_loss.sum()
            total_kl += (pi_detached - resp_ref).sum().item()
            num_tokens += resp_len

        n_samples = max(len(logprobs_list), 1)
        metrics: Dict[str, float] = {
            "mean_kl": total_kl / num_tokens if num_tokens > 0 else 0.0,
            "dapo_clip_frac": clip_frac_sum / n_samples,
        }
        if inf_num_samples > 0:
            metrics["inference_diff"] = total_inf_diff / inf_num_samples
            metrics["inference_kld"] = total_inf_kld / inf_num_samples
        for k, v in behave_metrics_agg.items():
            metrics[k] = v / n_samples
        return total_loss, metrics

    return loss_fn
