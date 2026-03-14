"""CISPO (Clipped Importance Sampling Policy Optimization) loss.

Clips the importance-sampling ratio ``pi/pi_prox`` to ``[1-eps_low,
1+eps_high]`` and uses the clipped ratio as a *detached* weight on the
log-probability.  This matches the Tinker kernel formula::

    clipped_ratio = clamp(ratio, 1 - eps_low, 1 + eps_high)
    per_token_loss = -(clipped_ratio.detach() * logprob * advantage) * tis_weight

Gradient flows through ``logprob`` (not through ``ratio``), and the
clipped ratio caps the effective learning signal for tokens whose
policy has already moved far from the proximal checkpoint.  Behavioral
IS weight corrects for the train-inference gap.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens, _get_loss_mask
from training.utils.rl.importance_sampling import (
    SAFETY_CLAMP,
    ISConfig,
    compute_tis_weight,
)


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
    is_config: ISConfig | None = None,
) -> ...:
    """Build a CISPO loss closure with ratio clipping and behavioral IS weight."""
    if cispo_config is None:
        cispo_config = CISPOConfig()
    if is_config is None:
        is_config = ISConfig()
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
        ppo_ratio_mean_sum = 0.0
        clip_frac_count = 0
        tis_metrics_agg: Dict[str, float] = {}

        for i, pi_logprobs in enumerate(logprobs_list):
            adv = advantages[i]
            ref_lp = ref_logprobs[i] if ref_logprobs else []
            inf_lp = inf_logprobs[i]
            prox_lp = prox_logprobs[i]
            response_start = max(0, prompt_lens[i] - 1)

            resp_pi = pi_logprobs[response_start:]
            resp_len = len(resp_pi)
            if resp_len == 0:
                continue

            if i < len(data):
                resp_mask = _get_loss_mask(
                    data[i], response_start, resp_len, resp_pi.dtype, resp_pi.device,
                )
            else:
                resp_mask = torch.ones(resp_len, dtype=resp_pi.dtype, device=resp_pi.device)
            active = resp_mask > 0.5
            active_count = int(active.sum().item())
            if active_count == 0:
                continue

            resp_ref = torch.tensor(
                [ref_lp[response_start + j] if (response_start + j) < len(ref_lp) else 0.0 for j in range(resp_len)],
                dtype=resp_pi.dtype, device=resp_pi.device,
            )
            pi_detached = resp_pi.detach()

            if not inf_lp:
                raise ValueError(
                    f"CISPO requires inference logprobs for sample {i} but got empty list. "
                    f"Ensure logprobs=True is set when using policy_loss='cispo'."
                )
            if len(inf_lp) < response_start + resp_len:
                raise ValueError(
                    f"CISPO requires at least {response_start + resp_len} inference logprobs "
                    f"for sample {i}, got {len(inf_lp)}."
                )

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

            log_ratio = torch.clamp(resp_pi - resp_prox, min=-cispo_config.ratio_log_cap, max=cispo_config.ratio_log_cap)
            ratio = torch.exp(log_ratio)

            clip_lo = 1.0 - cispo_config.eps_low
            clip_hi = 1.0 + cispo_config.eps_high
            clipped_ratio = torch.clamp(ratio, clip_lo, clip_hi)
            clip_frac_sum += (clipped_ratio.detach() != ratio.detach()).float().mean().item()
            ppo_ratio_mean_sum += ratio.detach().mean().item()
            clip_frac_count += 1

            tis_weight, bm = compute_tis_weight(resp_prox, resp_inf, is_config)
            for k, v in bm.items():
                tis_metrics_agg[k] = tis_metrics_agg.get(k, 0.0) + v

            adv_t = torch.as_tensor(adv, dtype=resp_pi.dtype, device=resp_pi.device)
            per_token_loss = -(clipped_ratio.detach() * resp_pi * adv_t) * tis_weight * resp_mask

            total_loss = total_loss + per_token_loss.sum()
            total_kl += ((pi_detached - resp_ref) * resp_mask).sum().item()
            num_tokens += active_count

        n_samples = max(len(logprobs_list), 1)
        metrics: Dict[str, float] = {
            "mean_kl": total_kl / num_tokens if num_tokens > 0 else 0.0,
            "cispo_clip_frac": clip_frac_sum / clip_frac_count if clip_frac_count > 0 else 0.0,
            "ppo_ratio_mean": ppo_ratio_mean_sum / n_samples,
        }
        if inf_num_samples > 0:
            metrics["inference_diff"] = total_inf_diff / inf_num_samples
            metrics["inference_kld"] = total_inf_kld / inf_num_samples
        for k, v in tis_metrics_agg.items():
            metrics[k] = v / n_samples
        return total_loss, metrics

    return loss_fn
