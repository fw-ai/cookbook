"""GSPO (Group Sequence Policy Optimization) loss for GRPO training.

Implements PPO-style clipping with a **sequence-level importance ratio**
(geometric mean of per-token ratios) against pre-computed proximal
logprobs, with behavioral IS weight correction.

Example::

    Config(policy_loss="gspo", gspo=GSPOConfig(clip_ratio=0.2))
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens
from training.utils.rl.importance_sampling import (
    DecoupledConfig,
    compute_behave_weight,
)


@dataclass
class GSPOConfig:
    """GSPO clipping configuration.

    ``clip_ratio`` is the symmetric fallback clip epsilon.
    Set ``clip_ratio_low`` / ``clip_ratio_high`` for asymmetric clipping.
    """

    clip_ratio: float = 0.2
    clip_ratio_low: float | None = None
    clip_ratio_high: float | None = None
    seq_ratio_log_cap: float = 10.0
    kl_beta: float = 0.001


def make_gspo_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    prox_logprobs: List[List[float]],
    gspo_config: GSPOConfig | None = None,
    decoupled_config: DecoupledConfig | None = None,
) -> ...:
    """Build a GSPO loss closure with sequence-level PPO ratio and behavioral IS weight."""
    if gspo_config is None:
        gspo_config = GSPOConfig()
    if decoupled_config is None:
        decoupled_config = DecoupledConfig()
    clip_low = gspo_config.clip_ratio if gspo_config.clip_ratio_low is None else gspo_config.clip_ratio_low
    clip_high = gspo_config.clip_ratio if gspo_config.clip_ratio_high is None else gspo_config.clip_ratio_high
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

            log_ratio = resp_pi - resp_prox
            seq_log_ratio = log_ratio.mean()
            log_seq_ratio = resp_pi - resp_pi.detach() + seq_log_ratio.detach()
            log_seq_ratio = torch.clamp(log_seq_ratio, max=gspo_config.seq_ratio_log_cap)
            seq_ratio = torch.exp(log_seq_ratio)

            clipped_seq_ratio = torch.clamp(seq_ratio, min=1.0 - clip_low, max=1.0 + clip_high)
            clip_frac_sum += (clipped_seq_ratio != seq_ratio).float().mean().item()

            behave_weight, bm = compute_behave_weight(resp_prox, resp_inf, decoupled_config)
            for k, v in bm.items():
                behave_metrics_agg[k] = behave_metrics_agg.get(k, 0.0) + v

            adv_t = torch.as_tensor(adv, dtype=resp_pi.dtype, device=resp_pi.device)
            surr1 = -seq_ratio * adv_t
            surr2 = -clipped_seq_ratio * adv_t
            per_token_loss = torch.maximum(surr1, surr2) * behave_weight

            total_loss = total_loss + per_token_loss.sum()
            total_kl += (pi_detached - resp_ref).sum().item()
            num_tokens += resp_len

        n_samples = max(len(logprobs_list), 1)
        metrics: Dict[str, float] = {
            "mean_kl": total_kl / num_tokens if num_tokens > 0 else 0.0,
            "gspo_clip_frac": clip_frac_sum / n_samples,
        }
        if inf_num_samples > 0:
            metrics["inference_diff"] = total_inf_diff / inf_num_samples
            metrics["inference_kld"] = total_inf_kld / inf_num_samples
        for k, v in behave_metrics_agg.items():
            metrics[k] = v / n_samples
        return total_loss, metrics

    return loss_fn
