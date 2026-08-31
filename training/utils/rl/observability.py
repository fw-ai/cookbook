"""Optional train/inference drift metrics for direct client losses."""

from __future__ import annotations

from typing import Dict, List

import tinker
import torch

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
    """Report inference k1 and k3 drift from inference to the trainer policy."""
    raw_inf_logprobs = raw_inf_logprobs or []

    total_k1 = 0.0
    total_k3 = 0.0
    raw_inf_num_samples = 0
    expected_active_tokens = 0
    compared_active_tokens = 0

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
            resp_mask = torch.ones(
                resp_len,
                dtype=resp_pi.dtype,
                device=resp_pi.device,
            )
        active = resp_mask > 0.5
        active_tokens = int(active.sum().item())
        if active_tokens == 0:
            continue
        expected_active_tokens += active_tokens

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
        total_k1 += inf_log_diff.mean().item()
        total_k3 += (torch.exp(inf_log_diff) - inf_log_diff - 1.0).mean().item()
        raw_inf_num_samples += 1
        compared_active_tokens += active_tokens

    if expected_active_tokens == 0 or compared_active_tokens == 0:
        return {}
    return {
        "raw_inference_logprob_coverage": (
            compared_active_tokens / expected_active_tokens
        ),
        "inference_k1": total_k1 / raw_inf_num_samples,
        "inference_k3": total_k3 / raw_inf_num_samples,
    }


def compute_server_grpo_observability_metrics(
    data: List[tinker.Datum],
    policy_logprobs: List[torch.Tensor],
    old_policy_logprobs: List[List[float]],
    raw_inf_logprobs: List[List[float]] | None,
    prompt_lens: List[int],
    *,
    eps_clip: float,
    eps_clip_high: float | None,
    ratio_log_cap: float = 20.0,
) -> Dict[str, float]:
    """Reconstruct client-visible diagnostics for the built-in PPO kernel.

    The trainer owns the differentiable loss. Its response still contains the
    exact policy logprobs used by that loss, so diagnostics can be computed
    without another forward pass.
    """
    n = len(data)
    aligned = {
        "policy_logprobs": len(policy_logprobs),
        "old_policy_logprobs": len(old_policy_logprobs),
        "prompt_lens": len(prompt_lens),
    }
    mismatched = {name: size for name, size in aligned.items() if size != n}
    if mismatched:
        details = ", ".join(f"{name}={size}" for name, size in mismatched.items())
        raise ValueError(
            f"Server-side GRPO metrics require {n} aligned rows; {details}."
        )

    eps_high = eps_clip if eps_clip_high is None else eps_clip_high
    clip_total = 0.0
    ratio_total = 0.0
    reported_samples = 0
    active_tokens = 0
    total_response_tokens = 0

    for i, (datum, pi_logprobs, old_policy_row, prompt_len) in enumerate(
        zip(
            data,
            policy_logprobs,
            old_policy_logprobs,
            prompt_lens,
            strict=True,
        )
    ):
        response_start = max(0, prompt_len - 1)
        resp_pi = pi_logprobs[response_start:]
        resp_len = len(resp_pi)
        total_response_tokens += resp_len
        if resp_len == 0:
            continue

        resp_mask = _get_loss_mask(
            datum,
            response_start,
            resp_len,
            resp_pi.dtype,
            resp_pi.device,
        )
        active = resp_mask > 0.5
        sample_active_tokens = int(active.sum().item())
        active_tokens += sample_active_tokens
        if sample_active_tokens == 0:
            continue

        validate_inference_logprobs_for_sample(
            "grpo",
            i,
            old_policy_row,
            response_start + resp_len,
            source="old_policy_logprobs",
        )
        resp_old_values = _coerce_response_logprobs(
            old_policy_row[response_start : response_start + resp_len],
            active,
            policy_loss="grpo",
            sample_idx=i,
            source="old_policy_logprobs",
        )
        resp_old_policy = torch.tensor(
            resp_old_values,
            dtype=resp_pi.dtype,
            device=resp_pi.device,
        )
        log_ratio = torch.clamp(
            resp_pi.detach()[active] - resp_old_policy[active],
            min=-ratio_log_cap,
            max=ratio_log_cap,
        )
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio,
            min=1.0 - eps_clip,
            max=1.0 + eps_high,
        )
        clip_total += (clipped_ratio != ratio).float().mean().item()
        ratio_total += ratio.mean().item()
        reported_samples += 1

    metrics: Dict[str, float] = {
        "active_tokens": float(active_tokens),
        "total_resp_tokens": float(total_response_tokens),
    }
    if reported_samples:
        metrics.update(
            {
                "ppo_clip_frac": clip_total / reported_samples,
                "ppo_ratio_mean": ratio_total / reported_samples,
            }
        )
    metrics.update(
        compute_inference_observability_metrics(
            data,
            policy_logprobs,
            raw_inf_logprobs,
            prompt_lens,
            "grpo",
        )
    )
    return metrics
