from __future__ import annotations

from typing import Dict, List

import torch
import tinker

from training.utils.rl.common import (
    _coerce_response_logprobs,
    _get_loss_mask,
    validate_inference_logprobs_for_sample,
)


def compute_inference_observability_metrics(
    data: List[tinker.Datum],
    logprobs_list: List[torch.Tensor],
    raw_inf_logprobs: List[List[float]] | None,
    prompt_lens: List[int],
    policy_loss: str,
) -> Dict[str, float]:
    if not raw_inf_logprobs:
        return {}

    total_inf_diff = 0.0
    total_inf_kld = 0.0
    raw_inf_num_samples = 0

    for i, pi_logprobs in enumerate(logprobs_list):
        response_start = max(0, prompt_lens[i] - 1)
        resp_pi = pi_logprobs[response_start:]
        resp_len = len(resp_pi)
        if resp_len == 0:
            continue

        if i < len(data):
            resp_mask = _get_loss_mask(
                data[i],
                response_start,
                resp_len,
                resp_pi.dtype,
                resp_pi.device,
            )
        else:
            resp_mask = torch.ones(resp_len, dtype=resp_pi.dtype, device=resp_pi.device)
        active = resp_mask > 0.5
        if int(active.sum().item()) == 0:
            continue

        raw_inf_lp = raw_inf_logprobs[i] if i < len(raw_inf_logprobs) else []
        if not raw_inf_lp:
            continue
        validate_inference_logprobs_for_sample(
            policy_loss,
            i,
            raw_inf_lp,
            response_start + resp_len,
            source="raw inference",
        )
        raw_inf_values = _coerce_response_logprobs(
            raw_inf_lp[response_start : response_start + resp_len],
            active,
            policy_loss=policy_loss,
            sample_idx=i,
            source="raw inference",
        )
        resp_raw_inf = torch.tensor(
            raw_inf_values,
            dtype=resp_pi.dtype,
            device=resp_pi.device,
        )
        inf_log_diff = resp_pi.detach()[active] - resp_raw_inf[active]
        total_inf_diff += inf_log_diff.abs().mean().item()
        total_inf_kld += (torch.exp(inf_log_diff) - inf_log_diff - 1.0).mean().item()
        raw_inf_num_samples += 1

    if raw_inf_num_samples == 0:
        return {}
    return {
        "inference_diff": total_inf_diff / raw_inf_num_samples,
        "inference_kld": total_inf_kld / raw_inf_num_samples,
    }
