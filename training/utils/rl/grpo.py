"""GRPO (Group Relative Policy Optimization) loss for RL training.

Uses PPO-style clipped surrogate objective with behavioral IS weight
correction.  The PPO ratio is computed against pre-computed proximal
logprobs (from a forward pass before training), and the behavioral
weight corrects for the train-inference gap.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens
from training.utils.rl.importance_sampling import (
    SAFETY_CLAMP,
    ISConfig,
    compute_tis_weight,
)


def make_grpo_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    inf_logprobs: List[List[float]],
    prox_logprobs: List[List[float]],
    kl_beta: float = 0.001,
    is_config: ISConfig | None = None,
) -> ...:
    """GRPO loss with PPO-clipped ratio and behavioral IS weight.

    ``prox_logprobs`` are pre-computed by a forward pass before training.
    The PPO ratio ``exp(pi_theta - prox)`` is clipped by
    ``ISConfig.eps_clip``.  The behavioral weight
    ``exp(prox - inf)`` corrects for train-inference mismatch.
    """
    if is_config is None:
        is_config = ISConfig()
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))
    eps_high = is_config.eps_clip if is_config.eps_clip_high is None else is_config.eps_clip_high

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
        ppo_ratio_mean_sum = 0.0
        tis_metrics_agg: Dict[str, float] = {}

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

            log_ratio = torch.clamp(resp_pi - resp_prox, min=-SAFETY_CLAMP, max=SAFETY_CLAMP)
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, min=1.0 - is_config.eps_clip, max=1.0 + eps_high)
            clip_frac_sum += (clipped_ratio != ratio).float().mean().item()
            ppo_ratio_mean_sum += ratio.detach().mean().item()

            tis_weight, bm = compute_tis_weight(resp_prox, resp_inf, is_config)
            for k, v in bm.items():
                tis_metrics_agg[k] = tis_metrics_agg.get(k, 0.0) + v

            adv_t = torch.as_tensor(adv, dtype=resp_pi.dtype, device=resp_pi.device)
            surr1 = -ratio * adv_t
            surr2 = -clipped_ratio * adv_t
            per_token_loss = torch.maximum(surr1, surr2) * tis_weight
            kl_penalty = kl_beta * (pi_detached - resp_ref)
            per_token_loss = per_token_loss + kl_penalty

            total_loss = total_loss + per_token_loss.sum()
            total_kl += (pi_detached - resp_ref).sum().item()
            num_tokens += resp_len

        n_samples = max(len(logprobs_list), 1)
        metrics: Dict[str, float] = {
            "mean_kl": total_kl / num_tokens if num_tokens > 0 else 0.0,
            "ppo_clip_frac": clip_frac_sum / n_samples,
            "ppo_ratio_mean": ppo_ratio_mean_sum / n_samples,
        }
        if inf_num_samples > 0:
            metrics["inference_diff"] = total_inf_diff / inf_num_samples
            metrics["inference_kld"] = total_inf_kld / inf_num_samples
        for k, v in tis_metrics_agg.items():
            metrics[k] = v / n_samples
        return total_loss, metrics

    return loss_fn
