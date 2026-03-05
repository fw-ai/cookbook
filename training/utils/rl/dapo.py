"""DAPO (Dynamic Advantage Policy Optimization) loss for GRPO training.

Uses PPO-style clipped surrogate objective with asymmetric clipping bounds:
the lower bound (eps_clip) and upper bound (eps_clip_high) can differ.
No explicit KL penalty -- divergence is controlled solely via clipping.

TIS can be composed on top via ``tis_weights_fn`` for additional
train-inference mismatch correction.

Reference: https://arxiv.org/abs/2503.14476

Example::

    Config(policy_loss="dapo", dapo=DAPOConfig(eps_clip=0.2, eps_clip_high=0.28))
    Config(policy_loss="dapo", tis_enabled=True)  # DAPO + TIS
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union, Callable
from dataclasses import dataclass

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens


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
    dapo_config: DAPOConfig | None = None,
    tis_weights_fn: Callable | None = None,
    decoupled_fn: Callable | None = None,
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """Build a DAPO loss closure.

    Computes the PPO clipped surrogate objective with asymmetric bounds.
    The importance ratio ``pi/pi_old`` is clipped to
    ``[1 - eps_clip, 1 + eps_clip_high]``.

    ``prompt_len`` may be a single int or a per-datum list for multi-prompt
    batched calls.

    *inf_logprobs* is always required (used for the PPO ratio).
    Pass *tis_weights_fn* to apply additional TIS correction on top.
    Pass *decoupled_fn* to use AReaL-style decoupled IS corrections.
    When set, the PPO ratio uses pi_prox instead of pi_old, and
    behavioral IS weight is applied on top.
    """
    if dapo_config is None:
        dapo_config = DAPOConfig()
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
        clip_frac_sum = 0.0
        clip_frac_count = 0
        agg_tis: Dict[str, float] = {}
        agg_decoupled: Dict[str, float] = {}

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
                    f"DAPO requires inference logprobs for sample {i} but got empty list. "
                    f"Ensure logprobs=True is set when using policy_loss='dapo'."
                )
            if len(inf_lp) < response_start + resp_len:
                raise ValueError(
                    f"DAPO requires at least {response_start + resp_len} inference logprobs "
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

            if decoupled_fn is not None:
                ppo_ratio, ppo_clipped, behave_weight, dec_metrics = decoupled_fn(resp_pi, resp_inf, i)
                adv_t = torch.as_tensor(adv, dtype=resp_pi.dtype, device=resp_pi.device)
                surr1 = -ppo_ratio * adv_t
                surr2 = -ppo_clipped * adv_t
                clipped_surrogate = torch.maximum(surr1, surr2)
                if dapo_config.eps_clip_c is not None:
                    if dapo_config.eps_clip_c <= 1.0:
                        raise ValueError(
                            f"DAPO dual-clip bound eps_clip_c must be > 1.0, got {dapo_config.eps_clip_c}."
                        )
                    surr3 = -dapo_config.eps_clip_c * adv_t
                    lower_clipped = torch.minimum(surr3, clipped_surrogate)
                    per_token_loss = torch.where(adv_t < 0, lower_clipped, clipped_surrogate)
                else:
                    per_token_loss = clipped_surrogate
                per_token_loss = per_token_loss * behave_weight
                for k, v in dec_metrics.items():
                    agg_decoupled[k] = agg_decoupled.get(k, 0.0) + v
            else:
                log_ratio = torch.clamp(
                    resp_pi - resp_inf,
                    min=-dapo_config.ratio_log_cap,
                    max=dapo_config.ratio_log_cap,
                )
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(
                    ratio,
                    min=1.0 - dapo_config.eps_clip,
                    max=1.0 + dapo_config.eps_clip_high,
                )
                clip_frac_sum += (clipped_ratio != ratio).float().mean().item()
                clip_frac_count += 1

                adv_t = torch.as_tensor(adv, dtype=resp_pi.dtype, device=resp_pi.device)
                surr1 = -ratio * adv_t
                surr2 = -clipped_ratio * adv_t
                clipped_surrogate = torch.maximum(surr1, surr2)
                if dapo_config.eps_clip_c is not None:
                    if dapo_config.eps_clip_c <= 1.0:
                        raise ValueError(
                            f"DAPO dual-clip bound eps_clip_c must be > 1.0, got {dapo_config.eps_clip_c}."
                        )
                    surr3 = -dapo_config.eps_clip_c * adv_t
                    lower_clipped = torch.minimum(surr3, clipped_surrogate)
                    per_token_loss = torch.where(adv_t < 0, lower_clipped, clipped_surrogate)
                else:
                    per_token_loss = clipped_surrogate

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
            "dapo_clip_frac": clip_frac_sum / clip_frac_count if clip_frac_count > 0 else 0.0,
        }
        if inf_num_samples > 0:
            metrics["inference_diff"] = total_inf_diff / inf_num_samples
            metrics["inference_kld"] = total_inf_kld / inf_num_samples
        if tis_weights_fn and not decoupled_fn:
            metrics["mean_importance_ratio"] = total_rho / num_tokens if num_tokens > 0 else 1.0
            n_samples = len(logprobs_list) or 1
            for k, v in agg_tis.items():
                metrics[k] = v / n_samples
        if decoupled_fn:
            n_samples = len(logprobs_list) or 1
            for k, v in agg_decoupled.items():
                metrics[k] = v / n_samples
        return total_loss, metrics

    return loss_fn
