"""DRO (Distributionally Robust Optimization) policy loss.

Matches the Tinker kernel formula::

    loss = -(lp * adv - 0.5 * beta * (lp - slp)^2).sum()

The quadratic penalty ``0.5 * beta * (lp - slp)^2`` constrains the policy
at *all* positions (including where ``adv=0``) to stay close to the
proximal checkpoint.  This differs from PPO-style clipping by providing a
smooth, continuous penalty rather than a hard clip boundary.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens, _get_loss_mask
from training.utils.rl.importance_sampling import (
    ISConfig,
    compute_tis_weight,
)


@dataclass
class DROConfig:
    """DRO loss configuration.

    ``beta`` controls the strength of the quadratic proximity penalty.
    """

    beta: float = 0.05


def make_dro_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    prox_logprobs: List[List[float]],
    dro_config: DROConfig | None = None,
    is_config: ISConfig | None = None,
) -> ...:
    """Build a DRO loss closure with quadratic proximity penalty."""
    if dro_config is None:
        dro_config = DROConfig()
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
        total_quad_penalty = 0.0
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

            tis_weight, bm = compute_tis_weight(resp_prox, resp_inf, is_config)
            for k, v in bm.items():
                tis_metrics_agg[k] = tis_metrics_agg.get(k, 0.0) + v

            adv_t = torch.as_tensor(adv, dtype=resp_pi.dtype, device=resp_pi.device)
            quad = (resp_pi - resp_prox) ** 2
            resp_active = adv_t != 0
            gated_quad = torch.where(resp_active, quad, torch.zeros_like(quad))
            linear_term = resp_pi * adv_t * tis_weight
            quad_term = 0.5 * dro_config.beta * gated_quad
            per_token_loss = -(linear_term - quad_term) * resp_mask

            total_loss = total_loss + per_token_loss.sum()
            total_kl += ((pi_detached - resp_ref) * resp_mask).sum().item()
            total_quad_penalty += (quad_term * resp_mask).detach().sum().item()
            num_tokens += active_count

        n_samples = max(len(logprobs_list), 1)
        metrics: Dict[str, float] = {
            "mean_kl": total_kl / num_tokens if num_tokens > 0 else 0.0,
            "dro_quad_penalty": total_quad_penalty / num_tokens if num_tokens > 0 else 0.0,
        }
        if inf_num_samples > 0:
            metrics["inference_diff"] = total_inf_diff / inf_num_samples
            metrics["inference_kld"] = total_inf_kld / inf_num_samples
        for k, v in tis_metrics_agg.items():
            metrics[k] = v / n_samples
        return total_loss, metrics

    return loss_fn
