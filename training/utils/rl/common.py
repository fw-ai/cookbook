"""Shared helpers used by RL loss variants."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
import tinker

SAFETY_CLAMP = 20.0


def _normalize_prompt_lens(prompt_len: Union[int, List[int]], n: int) -> List[int]:
    """Accept ``int`` (single prompt_len for all datums) or ``List[int]``."""
    if isinstance(prompt_len, int):
        return [prompt_len] * n
    prompt_lens = list(prompt_len)
    if len(prompt_lens) != n:
        raise ValueError(f"Expected {n} prompt lengths, got {len(prompt_lens)}.")
    return prompt_lens


def _masked_sum(
    values: torch.Tensor,  # (resp_len,)
    mask: torch.Tensor,  # (resp_len,)
) -> torch.Tensor:  # (1,)
    """Compute the sum of `values` over elements selected by `mask`."""
    # If NaNs exist out of mask, replace NaNs in values with a value that
    # won't affect the sum (e.g., 0 for masked regions)
    valid_values = torch.where(mask > 0.5, values, 0.0)
    return (valid_values * mask).sum()


def masked_mean(
    values: torch.Tensor,  # (resp_len,)
    mask: torch.Tensor,  # (resp_len,)
) -> torch.Tensor:  # (1,)
    """
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.

    Returns:
        Tensor: Masked mean, reduced over all elements.
    """
    s = _masked_sum(values, mask)
    return s / (mask.sum() + 1e-8)


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

    Echoed samples are already target-aligned.  Prompt tokens are conditioned
    inputs rather than sampled outputs, so serving may report null
    ``sampling_logprob`` values for their ``prompt_len - 1`` target positions.
    Normalize those unsampled positions to zero while still requiring every
    generated-token behavior logprob.  Non-echo samples are completion-only
    and need the same zero prompt-prefix padding.
    """
    values = getattr(sampled, attr, None)
    if values is None or not values:
        if required:
            raise RuntimeError(f"{source} required but sample {sample_idx} has none.")
        return None
    sequence_len = len(sampled.full_tokens)
    prompt_len = int(sampled.prompt_len)
    if not 0 <= prompt_len <= sequence_len:
        raise RuntimeError(
            f"{source} for sample {sample_idx} has invalid prompt_len {prompt_len} "
            f"for {sequence_len} tokens."
        )
    target_len = max(0, sequence_len - 1)
    response_start = max(0, prompt_len - 1)
    values = list(values)
    if getattr(sampled, "logprobs_echoed", False):
        if attr == "sampling_logprobs":
            if len(values) != target_len:
                raise RuntimeError(
                    f"{source} for sample {sample_idx} has target-aligned length "
                    f"{len(values)}, expected {target_len} target-aligned logprobs."
                )
            prompt_values = [
                0.0 if value is None else float(value)
                for value in values[:response_start]
            ]
            completion_values = _coerce_logprobs_to_float(
                values[response_start:],
                expected_len=target_len - response_start,
                source=source,
                sample_idx=sample_idx,
                coordinates="completion",
            )
            return prompt_values + completion_values
        return _coerce_logprobs_to_float(
            values,
            expected_len=target_len,
            source=source,
            sample_idx=sample_idx,
            coordinates="target-aligned",
        )

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
    resp_ref: torch.Tensor | None
    """Reference model logprobs, or ``None`` when no reference is available."""
    resp_old_policy: torch.Tensor
    """Fixed clipping/trust-region anchor, independent of rollout probabilities."""
    resp_inf: torch.Tensor
    """Rollout logprobs for response tokens."""
    resp_mask: torch.Tensor
    """Per-token loss mask (1.0 = active, 0.0 = masked)."""
    adv: torch.Tensor
    """Scalar advantage value (as a 0-d tensor)."""


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
``ctx.resp_mask`` however the policy requires.
Values in ``extra_metrics`` are summed across samples into
:attr:`LossLoopResult.extra_sums`; the caller is responsible for
averaging.
"""


def run_loss_loop(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_lens: List[int],
    data: List[tinker.Datum],
    logprobs_list: List[torch.Tensor],
    policy_loss: str,
    policy_fn: PolicyFn,
    *,
    old_policy_logprobs: List[List[float]] | None = None,
) -> LossLoopResult:
    """Shared loss loop: tensor setup, response masks, loss metrics and KL.

    Iterates over ``logprobs_list``, builds a :class:`SampleContext` for each
    sample, and delegates per-token loss computation to ``policy_fn``.
    Keep the fixed policy anchor separate from recorded rollout probabilities.
    """
    if old_policy_logprobs is not None and len(old_policy_logprobs) != len(
        logprobs_list
    ):
        raise ValueError("old_policy_logprobs must have one row per datum")
    total_loss = torch.tensor(0.0, requires_grad=True)
    total_kl = 0.0
    total_ppo_kl = 0.0
    total_ref_kl = 0.0
    ref_num_samples = 0
    behavior_num_samples = 0
    num_tokens = 0
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
        resp_ref = (
            torch.tensor(
                [ref_lp[response_start + j] if (response_start + j) < len(ref_lp) else 0.0 for j in range(resp_len)],
                dtype=resp_pi.dtype,
                device=resp_pi.device,
            )
            if ref_lp
            else None
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
        resp_old_policy = resp_inf
        if old_policy_logprobs is not None:
            old_lp = old_policy_logprobs[i]
            validate_inference_logprobs_for_sample(
                policy_loss,
                i,
                old_lp,
                response_start + resp_len,
                source="old_policy_logprobs",
            )
            resp_old_policy = torch.tensor(
                _coerce_response_logprobs(
                    old_lp[response_start : response_start + resp_len],
                    active,
                    policy_loss=policy_loss,
                    sample_idx=i,
                    source="old_policy_logprobs",
                ),
                dtype=resp_pi.dtype,
                device=resp_pi.device,
            )

        active_pi = pi_detached[active]
        active_ref = resp_ref[active] if resp_ref is not None else None

        ppo_log_diff = active_pi - resp_old_policy[active]
        total_ppo_kl += (torch.exp(ppo_log_diff) - ppo_log_diff - 1.0).mean().item()
        if active_ref is not None:
            ref_log_diff = active_ref - active_pi
            total_ref_kl += (torch.exp(ref_log_diff) - ref_log_diff - 1.0).mean().item()
            ref_num_samples += 1
        behavior_num_samples += 1

        adv_t = torch.as_tensor(advantages[i], dtype=resp_pi.dtype, device=resp_pi.device)

        ctx = SampleContext(
            resp_pi=resp_pi,
            pi_detached=pi_detached,
            resp_ref=resp_ref,
            resp_inf=resp_inf,
            resp_old_policy=resp_old_policy,
            resp_mask=resp_mask,
            adv=adv_t,
        )
        per_token_loss, extra = policy_fn(ctx)

        total_loss = total_loss + per_token_loss.sum()
        if resp_ref is not None:
            total_kl += ((pi_detached - resp_ref) * resp_mask).sum().item()
        num_tokens += active_count
        for k, v in extra.items():
            extra_sums[k] = extra_sums.get(k, 0.0) + v

    n_samples = max(behavior_num_samples, 1)
    base_metrics: Dict[str, float] = {}
    if behavior_num_samples > 0:
        base_metrics["ppo_kl"] = total_ppo_kl / behavior_num_samples
    if ref_num_samples > 0:
        base_metrics["mean_kl"] = total_kl / num_tokens if num_tokens > 0 else 0.0
        base_metrics["ref_kl"] = total_ref_kl / ref_num_samples

    return LossLoopResult(
        total_loss=total_loss,
        base_metrics=base_metrics,
        extra_sums=extra_sums,
        num_tokens=num_tokens,
        n_samples=n_samples,
    )
