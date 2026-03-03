"""GRPO (Group Relative Policy Optimization) loss for RL training."""

from __future__ import annotations

from typing import Dict, List, Tuple, Union, Callable

import torch
import tinker

from fireworks.training.cookbook.utils.rl.common import _normalize_prompt_lens


def make_grpo_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    inf_logprobs: List[List[float]],
    kl_beta: float = 0.001,
    tis_weights_fn: Callable | None = None,
) -> Callable[[List[tinker.Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]:
    """GRPO policy-gradient loss with KL penalty against a reference model.

    ``prompt_len`` may be a single int (all datums share the same prompt
    length) or a per-datum list for multi-prompt batched calls.
    ``inf_logprobs`` is required to compute train/inference divergence metrics.

    Pass *tis_weights_fn* (from :func:`make_tis_weights_fn`) to apply
    TIS train-inference mismatch correction on top of the base loss.
    """
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

            per_token_loss = (-adv + kl_beta) * resp_pi

            if tis_weights_fn:
                weights, tis_metrics = tis_weights_fn(pi_detached, i)
                per_token_loss = per_token_loss * weights
                total_rho += weights.sum().item()
                for k, v in tis_metrics.items():
                    agg_tis[k] = agg_tis.get(k, 0.0) + v

            if not inf_lp:
                raise ValueError(
                    f"GRPO requires inference logprobs for sample {i} but got empty list. "
                    f"Ensure logprobs=True is set."
                )
            if len(inf_lp) < response_start + resp_len:
                raise ValueError(
                    f"GRPO requires at least {response_start + resp_len} inference logprobs "
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

            total_loss = total_loss + per_token_loss.sum()
            total_kl += (pi_detached - resp_ref).sum().item()
            num_tokens += resp_len

        metrics: Dict[str, float] = {
            "mean_kl": total_kl / num_tokens if num_tokens > 0 else 0.0,
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
