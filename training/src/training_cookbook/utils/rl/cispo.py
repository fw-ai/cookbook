"""CISPO (Clipped Importance Sampling Policy Optimization) loss for GRPO training.

Clips importance sampling weights (the ratio pi/pi_old) rather than the
PPO surrogate objective.  Tokens where the ratio has moved too far in a
destabilizing direction are masked out entirely, while all remaining
tokens contribute full gradients.  This "always use all tokens" property
yields better sample efficiency than PPO/DAPO clipping empirically.

The masking rule (Eq. 7 in the MiniMax-M1 paper):
    M_{i,t} = 0  if  A > 0 and r > 1 + eps_high   (already-boosted token)
    M_{i,t} = 0  if  A < 0 and r < 1 - eps_low     (already-suppressed token)
    M_{i,t} = 1  otherwise

Per-token loss: M_{i,t} * (-r_{i,t} * A_i)

TIS can be composed on top via ``tis_weights_fn`` for additional
train-inference mismatch correction.

Reference: https://arxiv.org/abs/2506.13585 (Section 3.1)

Example::

    Config(policy_loss="cispo", cispo=CISPOConfig(eps_low=0.2, eps_high=0.28))
    Config(policy_loss="cispo", tis_enabled=True)  # CISPO + TIS
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union, Callable
from dataclasses import dataclass

import torch
import tinker

from training_cookbook.utils.rl.common import _normalize_prompt_lens


@dataclass
class CISPOConfig:
    """CISPO clipping thresholds.

    ``eps_low`` controls suppression masking: tokens with negative advantage
    and ratio < 1 - eps_low are masked (already sufficiently suppressed).

    ``eps_high`` controls boosting masking: tokens with positive advantage
    and ratio > 1 + eps_high are masked (already sufficiently boosted).

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
    cispo_config: CISPOConfig | None = None,
    tis_weights_fn: Callable | None = None,
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """Build a CISPO loss closure.

    Computes importance-sampled policy gradient with IS-weight clipping:
    the ratio ``pi/pi_old`` is used directly (not clipped), but tokens
    where the ratio violates the CISPO mask are zeroed out entirely.

    ``prompt_len`` may be a single int or a per-datum list for multi-prompt
    batched calls.

    *inf_logprobs* is always required (rollout/old-policy logprobs for ratio).
    Pass *tis_weights_fn* to apply additional TIS correction on top.
    """
    if cispo_config is None:
        cispo_config = CISPOConfig()
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, requires_grad=True)
        total_kl = 0.0
        total_rho = 0.0
        total_inf_diff = 0.0
        total_inf_kld = 0.0
        inf_num_samples = 0
        num_tokens = 0
        mask_frac_sum = 0.0
        mask_frac_count = 0
        agg_tis: Dict[str, float] = {}

        for i, pi_logprobs in enumerate(logprobs_list):
            adv = advantages[i]
            ref_lp = ref_logprobs[i]
            inf_lp = inf_logprobs[i]
            response_start = max(0, prompt_lens[i] - 1)

            resp_pi = pi_logprobs[response_start:]
            resp_len = len(resp_pi)
            if resp_len == 0:
                continue

            resp_ref = torch.tensor(
                [ref_lp[response_start + j] if (response_start + j) < len(ref_lp) else 0.0 for j in range(resp_len)],
                dtype=resp_pi.dtype,
                device=resp_pi.device,
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
                inf_lp[response_start : response_start + resp_len],
                dtype=resp_pi.dtype,
                device=resp_pi.device,
            )
            inf_log_diff = pi_detached - resp_inf
            total_inf_diff += inf_log_diff.abs().mean().item()
            total_inf_kld += (torch.exp(inf_log_diff) - inf_log_diff - 1.0).mean().item()
            inf_num_samples += 1

            # Importance ratio: r = pi / pi_old = exp(log_pi - log_pi_old)
            log_ratio = torch.clamp(
                resp_pi - resp_inf,
                min=-cispo_config.ratio_log_cap,
                max=cispo_config.ratio_log_cap,
            )
            ratio = torch.exp(log_ratio)

            # CISPO mask (Eq. 7): zero out tokens that have already moved
            # far enough in the direction the advantage pushes.
            ratio_detached = ratio.detach()
            if adv > 0:
                mask = (ratio_detached <= 1.0 + cispo_config.eps_high).float()
            elif adv < 0:
                mask = (ratio_detached >= 1.0 - cispo_config.eps_low).float()
            else:
                mask = torch.ones_like(ratio_detached)

            mask_frac_sum += 1.0 - mask.mean().item()
            mask_frac_count += 1

            adv_t = torch.as_tensor(adv, dtype=resp_pi.dtype, device=resp_pi.device)
            per_token_loss = mask * (-ratio * adv_t)

            if tis_weights_fn:
                weights, tis_metrics = tis_weights_fn(pi_detached, i)
                per_token_loss = per_token_loss * weights
                total_rho += weights.sum().item()
                for k, v in tis_metrics.items():
                    agg_tis[k] = agg_tis.get(k, 0.0) + v

            total_loss = total_loss + per_token_loss.sum()
            total_kl += (pi_detached - resp_ref).sum().item()
            num_tokens += resp_len

        metrics: Dict[str, float] = {
            "mean_kl": total_kl / num_tokens if num_tokens > 0 else 0.0,
            "cispo_mask_frac": mask_frac_sum / mask_frac_count if mask_frac_count > 0 else 0.0,
        }
        if inf_num_samples > 0:
            metrics["inference_diff"] = total_inf_diff / inf_num_samples
            metrics["inference_kld"] = total_inf_kld / inf_num_samples
        if tis_weights_fn:
            metrics["mean_importance_ratio"] = total_rho / num_tokens if num_tokens > 0 else 1.0
            n_samples = len(logprobs_list) or 1
            for k, v in agg_tis.items():
                metrics[k] = v / n_samples
        return total_loss, metrics

    return loss_fn
