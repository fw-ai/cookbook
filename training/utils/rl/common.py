"""Shared helpers used by RL loss variants."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
import tinker

from training.utils.rl.tis import TISConfig, compute_tis_weight


def _normalize_prompt_lens(prompt_len: Union[int, List[int]], n: int) -> List[int]:
    """Accept ``int`` (single prompt_len for all datums) or ``List[int]``."""
    if isinstance(prompt_len, int):
        return [prompt_len] * n
    prompt_lens = list(prompt_len)
    if len(prompt_lens) != n:
        raise ValueError(f"Expected {n} prompt lengths, got {len(prompt_lens)}.")
    return prompt_lens


def _get_loss_mask(
    datum: tinker.Datum,
    response_start: int,
    resp_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Extract per-position loss mask from ``loss_fn_inputs["weights"]``.

    Returns a tensor of shape ``[resp_len]`` sliced from ``response_start``.
    Falls back to ``loss_fn_inputs["loss_mask"]`` for legacy datums and
    finally to all-ones (no masking) when neither is present.

    The trainer SDK rejects any ``loss_fn_inputs`` key other than
    ``{"target_tokens", "weights"}`` for ``forward_backward_custom``, so
    new producers MUST write the per-token mask under ``"weights"``.
    """
    mask_td = datum.loss_fn_inputs.get("weights") or datum.loss_fn_inputs.get("loss_mask")
    if mask_td is not None:
        mask_vals = mask_td.data[response_start : response_start + resp_len]
        if len(mask_vals) < resp_len:
            mask_vals = list(mask_vals) + [0.0] * (resp_len - len(mask_vals))
        return torch.tensor(mask_vals, dtype=dtype, device=device)
    return torch.ones(resp_len, dtype=dtype, device=device)


def _format_policy_loss_label(policy_loss: str) -> str:
    """Human-readable label for error messages."""
    if policy_loss == "importance_sampling":
        return policy_loss
    return policy_loss.upper()


def _coerce_logprobs_to_float(
    values: List[Any],
    *,
    expected_len: int,
    source: str,
    sample_idx: int,
    coordinates: str,
) -> List[float]:
    if len(values) != expected_len:
        raise RuntimeError(
            f"{source} for sample {sample_idx} has {coordinates} length {len(values)}, "
            f"expected {expected_len} {coordinates} logprobs."
        )
    if any(value is None for value in values):
        raise RuntimeError(f"{source} for sample {sample_idx} has null {coordinates} logprob.")
    return [float(value) for value in values]


def align_sample_logprobs_to_target_tokens(
    sampled: Any,
    *,
    attr: str,
    source: str,
    sample_idx: int,
    required: bool,
) -> List[float] | None:
    """Normalize sampler logprobs into ``target_tokens`` coordinates.

    Echoed samples are already target-aligned. Non-echo samples are
    completion-only and need prompt-prefix padding.
    """
    values = getattr(sampled, attr, None)
    if values is None or not values:
        if required:
            raise RuntimeError(f"{source} required but sample {sample_idx} has none.")
        return None
    target_len = max(0, len(sampled.full_tokens) - 1)
    values = list(values)
    if getattr(sampled, "logprobs_echoed", False):
        return _coerce_logprobs_to_float(
            values,
            expected_len=target_len,
            source=source,
            sample_idx=sample_idx,
            coordinates="target-aligned",
        )

    response_start = min(max(0, int(sampled.prompt_len) - 1), target_len)
    response_len = target_len - response_start
    completion_logprobs = _coerce_logprobs_to_float(
        values,
        expected_len=response_len,
        source=source,
        sample_idx=sample_idx,
        coordinates="completion",
    )
    return [0.0] * response_start + completion_logprobs


def validate_inference_logprobs_for_sample(
    policy_loss: str,
    sample_idx: int,
    inf_lp: List[Any],
    required: int,
    *,
    source: str = "rollout_logprobs",
) -> None:
    """Ensure one sample has logprobs for response tokens."""
    policy_label = _format_policy_loss_label(policy_loss)
    if not inf_lp:
        raise ValueError(
            f"{policy_label} requires {source} for sample {sample_idx} but got empty list. "
            f"Ensure logprobs=True is set when using policy_loss='{policy_loss}'."
        )

    if len(inf_lp) < required:
        raise ValueError(
            f"{policy_label} requires at least {required} values in {source} "
            f"for sample {sample_idx}, got {len(inf_lp)}."
        )


def _coerce_response_logprobs(
    values: List[Any],
    active: torch.Tensor,
    *,
    policy_loss: str,
    sample_idx: int,
    source: str,
) -> List[float]:
    policy_label = _format_policy_loss_label(policy_loss)
    result: List[float] = []
    for pos, value in enumerate(values):
        if value is None:
            if bool(active[pos].item()):
                raise ValueError(
                    f"{policy_label} requires a non-null value in {source} for "
                    f"sample {sample_idx} active response position {pos}."
                )
            result.append(0.0)
        else:
            result.append(float(value))
    return result


@dataclass
class SampleContext:
    """Pre-computed tensors for a single sample in the RL loss loop.

    All response tensors have shape ``[resp_len]`` and are guaranteed to be
    the same length.
    """

    resp_pi: torch.Tensor
    """Policy logprobs for response tokens (has grad)."""
    pi_detached: torch.Tensor
    """Detached policy logprobs."""
    resp_ref: torch.Tensor
    """Reference model logprobs for response tokens."""
    resp_old_policy: torch.Tensor
    """Old-policy forward-pass logprobs for response tokens."""
    resp_inf: torch.Tensor
    """Rollout logprobs for response tokens."""
    resp_mask: torch.Tensor
    """Per-token loss mask (1.0 = active, 0.0 = masked)."""
    adv: torch.Tensor
    """Scalar advantage value (as a 0-d tensor)."""
    tis_weight: torch.Tensor
    """TIS importance weight per token."""


@dataclass
class LossLoopResult:
    """Output of :func:`run_loss_loop`."""

    total_loss: torch.Tensor
    base_metrics: Dict[str, float]
    extra_sums: Dict[str, float]
    num_tokens: int
    n_samples: int


PolicyFn = Callable[[SampleContext], Tuple[torch.Tensor, Dict[str, float]]]
"""``(ctx) -> (per_token_loss, extra_metrics)``

The returned ``per_token_loss`` tensor should already incorporate
``ctx.tis_weight`` and ``ctx.resp_mask`` however the policy requires.
Values in ``extra_metrics`` are summed across samples into
:attr:`LossLoopResult.extra_sums`; the caller is responsible for
averaging.
"""


def run_loss_loop(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_lens: List[int],
    old_policy_logprobs: List[List[float]],
    tis_config: TISConfig,
    data: List[tinker.Datum],
    logprobs_list: List[torch.Tensor],
    policy_loss: str,
    policy_fn: PolicyFn,
) -> LossLoopResult:
    """Shared loss loop: tensor setup, TIS weight, loss metrics, KL.

    Iterates over ``logprobs_list``, builds a :class:`SampleContext` for each
    sample, and delegates per-token loss computation to ``policy_fn``.
    loss/TIS ratios use ``inf_logprobs`` and ``old_policy_logprobs``.
    """
    total_loss = torch.tensor(0.0, requires_grad=True)
    total_kl = 0.0
    total_ppo_kl = 0.0
    total_ref_kl = 0.0
    ref_num_samples = 0
    behavior_num_samples = 0
    num_tokens = 0
    tis_metrics_agg: Dict[str, float] = {}
    extra_sums: Dict[str, float] = {}

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
        active_count = int(active.sum().item())
        if active_count == 0:
            continue

        ref_lp = ref_logprobs[i] if ref_logprobs else []
        resp_ref = torch.tensor(
            [ref_lp[response_start + j] if (response_start + j) < len(ref_lp) else 0.0 for j in range(resp_len)],
            dtype=resp_pi.dtype,
            device=resp_pi.device,
        )
        pi_detached = resp_pi.detach()

        inf_lp = inf_logprobs[i] if i < len(inf_logprobs) else []
        validate_inference_logprobs_for_sample(
            policy_loss,
            i,
            inf_lp,
            response_start + resp_len,
            source="rollout_logprobs",
        )
        resp_inf_values = _coerce_response_logprobs(
            inf_lp[response_start : response_start + resp_len],
            active,
            policy_loss=policy_loss,
            sample_idx=i,
            source="rollout_logprobs",
        )
        resp_inf = torch.tensor(
            resp_inf_values,
            dtype=resp_pi.dtype,
            device=resp_pi.device,
        )
        old_policy_lp = old_policy_logprobs[i]
        resp_old_policy = torch.tensor(
            [old_policy_lp[response_start + j] if (response_start + j) < len(old_policy_lp) else 0.0 for j in range(resp_len)],
            dtype=resp_pi.dtype,
            device=resp_pi.device,
        )

        # Filter to loss_mask>0 positions: masked bridge/tool tokens otherwise
        # contaminate sequence-level TIS weight (matches slime/AReaL behavior).
        active_pi = pi_detached[active]
        active_ref = resp_ref[active]
        active_inf = resp_inf[active]
        active_old_policy = resp_old_policy[active]

        ppo_log_diff = active_pi - active_old_policy
        total_ppo_kl += (torch.exp(ppo_log_diff) - ppo_log_diff - 1.0).mean().item()
        if ref_lp:
            ref_log_diff = active_ref - active_pi
            total_ref_kl += (torch.exp(ref_log_diff) - ref_log_diff - 1.0).mean().item()
            ref_num_samples += 1
        behavior_num_samples += 1

        tis_weight_active, bm = compute_tis_weight(active_old_policy, active_inf, tis_config)
        # Identity (1.0) at masked positions: zeroes under ``resp_mask`` for
        # masked-multiplied losses, no-op weight for ``dro``.
        tis_weight = torch.ones(resp_len, dtype=resp_pi.dtype, device=resp_pi.device)
        tis_weight[active] = tis_weight_active.to(resp_pi.dtype)
        for k, v in bm.items():
            tis_metrics_agg[k] = tis_metrics_agg.get(k, 0.0) + v

        adv_t = torch.as_tensor(advantages[i], dtype=resp_pi.dtype, device=resp_pi.device)

        ctx = SampleContext(
            resp_pi=resp_pi,
            pi_detached=pi_detached,
            resp_ref=resp_ref,
            resp_old_policy=resp_old_policy,
            resp_inf=resp_inf,
            resp_mask=resp_mask,
            adv=adv_t,
            tis_weight=tis_weight,
        )
        per_token_loss, extra = policy_fn(ctx)

        total_loss = total_loss + per_token_loss.sum()
        total_kl += ((pi_detached - resp_ref) * resp_mask).sum().item()
        num_tokens += active_count
        for k, v in extra.items():
            extra_sums[k] = extra_sums.get(k, 0.0) + v

    n_samples = max(behavior_num_samples, 1)
    base_metrics: Dict[str, float] = {
        "mean_kl": total_kl / num_tokens if num_tokens > 0 else 0.0,
    }
    if behavior_num_samples > 0:
        base_metrics["ppo_kl"] = total_ppo_kl / behavior_num_samples
    if ref_num_samples > 0:
        base_metrics["ref_kl"] = total_ref_kl / ref_num_samples
    for k, v in tis_metrics_agg.items():
        base_metrics[k] = v / n_samples

    return LossLoopResult(
        total_loss=total_loss,
        base_metrics=base_metrics,
        extra_sums=extra_sums,
        num_tokens=num_tokens,
        n_samples=n_samples,
    )
