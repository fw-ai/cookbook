"""Shared helpers used by RL loss variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import torch
from fireworks.training import sdk

SAFETY_CLAMP = 20.0


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
    datum: sdk.Datum,
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
            f"{source} for sample {sample_idx} has invalid prompt_len {prompt_len} " f"for {sequence_len} tokens."
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
            prompt_values = [0.0 if value is None else float(value) for value in values[:response_start]]
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
    resp_anchor: torch.Tensor
    """Fixed clipping/trust-region anchor (``PolicyBatch.anchor_logprobs``)."""
    resp_inf: torch.Tensor
    """Rollout logprobs for response tokens."""
    resp_mask: torch.Tensor
    """Per-token loss mask (1.0 = active, 0.0 = masked)."""
    adv: torch.Tensor
    """Advantage: a 0-d tensor, or ``[resp_len]`` with ``PolicyBatch.token_advantages``."""


@dataclass
class LossLoopResult:
    """Output of :func:`run_loss_loop`."""

    total_loss: torch.Tensor
    base_metrics: Dict[str, float]
    """``anchor_kl`` (k3 policy/anchor, per datum) and, with reference rows,
    ``ref_kl`` / ``mean_kl``."""
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
    batch: Any,
    data: List[sdk.Datum],
    logprobs_list: List[torch.Tensor],
    policy_loss: str,
    policy_fn: PolicyFn,
) -> LossLoopResult:
    """Shared client-loss loop over a :class:`~training.utils.rl.losses.PolicyBatch`.

    Builds a :class:`SampleContext` per datum (response masks, rollout logprobs,
    the batch anchor, optional reference rows and advantages) and delegates the
    per-token loss to ``policy_fn``.
    """
    n = len(logprobs_list)
    rows = {
        "advantages": len(batch.advantages),
        "rollout_logprobs": len(batch.rollout_logprobs),
        "anchor_logprobs": len(batch.anchor_logprobs),
        "prompt_lens": len(batch.prompt_lens),
    }
    if batch.token_advantages is not None:
        rows["token_advantages"] = len(batch.token_advantages)
    if batch.ref_logprobs:
        rows["ref_logprobs"] = len(batch.ref_logprobs)
    mismatched = {name: size for name, size in rows.items() if size != n}
    if mismatched:
        details = ", ".join(f"{name}={size}" for name, size in mismatched.items())
        raise ValueError(f"{_format_policy_loss_label(policy_loss)} loss needs {n} aligned rows; {details}.")

    total_loss = torch.tensor(0.0, requires_grad=True)
    total_kl = 0.0
    total_anchor_kl = 0.0
    total_ref_kl = 0.0
    ref_num_samples = 0
    behavior_num_samples = 0
    num_tokens = 0
    extra_sums: Dict[str, float] = {}

    def response_row(rows: List[List[float]], i: int, start: int, length: int, active, source: str):
        row = rows[i]
        validate_inference_logprobs_for_sample(policy_loss, i, row, start + length, source=source)
        return _coerce_response_logprobs(
            row[start : start + length], active, policy_loss=policy_loss, sample_idx=i, source=source
        )

    for i, pi_logprobs in enumerate(logprobs_list):
        response_start = max(0, batch.prompt_lens[i] - 1)
        resp_pi = pi_logprobs[response_start:]
        resp_len = len(resp_pi)
        if resp_len == 0:
            continue
        if i < len(data):
            resp_mask = _get_loss_mask(data[i], response_start, resp_len, resp_pi.dtype, resp_pi.device)
        else:
            resp_mask = torch.ones(resp_len, dtype=resp_pi.dtype, device=resp_pi.device)
        active = resp_mask > 0.5
        active_count = int(active.sum().item())
        if active_count == 0:
            continue

        def as_tensor(values):
            return torch.tensor(values, dtype=resp_pi.dtype, device=resp_pi.device)

        resp_inf = as_tensor(
            response_row(batch.rollout_logprobs, i, response_start, resp_len, active, "rollout_logprobs")
        )
        resp_anchor = as_tensor(
            response_row(batch.anchor_logprobs, i, response_start, resp_len, active, "anchor_logprobs")
        )
        resp_ref = None
        if batch.ref_logprobs:
            ref_lp = batch.ref_logprobs[i]
            resp_ref = as_tensor(
                [ref_lp[response_start + j] if response_start + j < len(ref_lp) else 0.0 for j in range(resp_len)]
            )
        pi_detached = resp_pi.detach()

        anchor_log_diff = pi_detached[active] - resp_anchor[active]
        total_anchor_kl += (torch.exp(anchor_log_diff) - anchor_log_diff - 1.0).mean().item()
        if resp_ref is not None:
            ref_log_diff = resp_ref[active] - pi_detached[active]
            total_ref_kl += (torch.exp(ref_log_diff) - ref_log_diff - 1.0).mean().item()
            total_kl += ((pi_detached - resp_ref) * resp_mask).sum().item()
            ref_num_samples += 1
        behavior_num_samples += 1

        if batch.token_advantages is not None:
            adv_t = as_tensor(list(batch.token_advantages[i][response_start : response_start + resp_len]))
        else:
            adv_t = torch.as_tensor(batch.advantages[i], dtype=resp_pi.dtype, device=resp_pi.device)

        ctx = SampleContext(
            resp_pi=resp_pi,
            pi_detached=pi_detached,
            resp_ref=resp_ref,
            resp_inf=resp_inf,
            resp_anchor=resp_anchor,
            resp_mask=resp_mask,
            adv=adv_t,
        )
        per_token_loss, extra = policy_fn(ctx)

        total_loss = total_loss + per_token_loss.sum()
        num_tokens += active_count
        for k, v in extra.items():
            extra_sums[k] = extra_sums.get(k, 0.0) + v

    n_samples = max(behavior_num_samples, 1)
    base_metrics: Dict[str, float] = {}
    if behavior_num_samples > 0:
        base_metrics["anchor_kl"] = total_anchor_kl / behavior_num_samples
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


def client_loss_metrics(result: LossLoopResult, prefix: str) -> Dict[str, float]:
    """Name a client loss's metrics like its server-side ``metrics`` function.

    ``{prefix}_kl`` (k3 policy/anchor), ``{prefix}_ratio_mean`` and, when the
    policy function reports ``clip_frac``, ``{prefix}_clip_frac``; both are
    averaged per datum. Reference-KL metrics pass through unchanged.
    """
    out = dict(result.base_metrics)
    if "anchor_kl" in out:
        out[f"{prefix}_kl"] = out.pop("anchor_kl")
    ns = result.n_samples
    if "ratio_mean" in result.extra_sums:
        out[f"{prefix}_ratio_mean"] = result.extra_sums["ratio_mean"] / ns
    if "clip_frac" in result.extra_sums:
        out[f"{prefix}_clip_frac"] = result.extra_sums["clip_frac"] / ns
    return out


# ---------------------------------------------------------------------------
# Metric primitives shared by the objective modules in ``algorithm``
# ---------------------------------------------------------------------------


def result_logprobs(result: Any) -> List[torch.Tensor]:
    """Per-datum target logprobs returned by a trainer forward or forward_backward."""
    return [output["logprobs"].to_torch() for output in result.loss_fn_outputs]


@dataclass(frozen=True)
class ResponseLogprobs:
    """Active response-token logprobs of one datum, policy beside a reference row."""

    index: int
    policy: torch.Tensor
    reference: torch.Tensor
    response_tokens: int


def iter_response_logprobs(
    data: List[sdk.Datum],
    policy_logprobs: List[torch.Tensor],
    reference_logprobs: List[List[float]],
    prompt_lens: List[int],
    *,
    loss_name: str,
    source: str,
):
    """Yield active-token policy and reference logprobs for each datum.

    ``reference_logprobs`` is target-aligned (a rollout or anchor row per
    datum). Datums without active response tokens are skipped; misaligned
    rows raise instead of producing a silently wrong metric.
    """
    sizes = {
        "policy_logprobs": len(policy_logprobs),
        source: len(reference_logprobs),
        "prompt_lens": len(prompt_lens),
    }
    mismatched = {name: size for name, size in sizes.items() if size != len(data)}
    if mismatched:
        details = ", ".join(f"{name}={size}" for name, size in mismatched.items())
        raise ValueError(
            f"{_format_policy_loss_label(loss_name)} metrics require " f"{len(data)} aligned rows; {details}."
        )
    for i, (datum, pi, reference_row, prompt_len) in enumerate(
        zip(data, policy_logprobs, reference_logprobs, prompt_lens, strict=True)
    ):
        response_start = max(0, prompt_len - 1)
        resp_pi = pi[response_start:].detach()
        resp_len = len(resp_pi)
        if resp_len == 0:
            continue
        active = _get_loss_mask(datum, response_start, resp_len, resp_pi.dtype, resp_pi.device) > 0.5
        if not bool(active.any()):
            continue
        validate_inference_logprobs_for_sample(loss_name, i, reference_row, response_start + resp_len, source=source)
        reference = torch.tensor(
            _coerce_response_logprobs(
                reference_row[response_start : response_start + resp_len],
                active,
                policy_loss=loss_name,
                sample_idx=i,
                source=source,
            ),
            dtype=resp_pi.dtype,
            device=resp_pi.device,
        )
        yield ResponseLogprobs(i, resp_pi[active], reference[active], resp_len)


def inference_drift_metrics(batch: Any, result: Any) -> Dict[str, float]:
    """k1/k3 drift from raw serving logprobs to the trainer policy for one call.

    ``batch`` is a :class:`~training.utils.rl.losses.PolicyBatch`; ``result``
    is the trainer call's result (its per-datum target logprobs). Each sequence
    has equal weight. Rows without raw serving logprobs are skipped, and
    ``raw_inference_logprob_coverage`` reports the share of active tokens
    compared; it is ``0.0`` when serving returned none, so "no data" is
    distinguishable from "no drift".
    """
    policy_logprobs = result_logprobs(result)
    rows = batch.raw_inference_logprobs or []
    total_k1 = total_k3 = 0.0
    compared_samples = expected_tokens = compared_tokens = 0
    for i, pi in enumerate(policy_logprobs):
        response_start = max(0, batch.prompt_lens[i] - 1)
        resp_pi = pi[response_start:].detach()
        resp_len = len(resp_pi)
        if resp_len == 0:
            continue
        mask = (
            _get_loss_mask(batch.data[i], response_start, resp_len, resp_pi.dtype, resp_pi.device)
            if i < len(batch.data)
            else torch.ones(resp_len, dtype=resp_pi.dtype, device=resp_pi.device)
        )
        active = mask > 0.5
        active_tokens = int(active.sum().item())
        if active_tokens == 0:
            continue
        expected_tokens += active_tokens
        raw = rows[i] if i < len(rows) else []
        if not raw:
            continue
        validate_inference_logprobs_for_sample(
            "inference_drift", i, raw, response_start + resp_len, source="raw inference"
        )
        resp_raw = torch.tensor(
            _coerce_response_logprobs(
                raw[response_start : response_start + resp_len],
                active,
                policy_loss="inference_drift",
                sample_idx=i,
                source="raw inference",
            ),
            dtype=resp_pi.dtype,
            device=resp_pi.device,
        )
        diff = resp_pi[active] - resp_raw[active]
        total_k1 += diff.mean().item()
        total_k3 += (torch.exp(diff) - diff - 1.0).mean().item()
        compared_samples += 1
        compared_tokens += active_tokens
    if not expected_tokens:
        return {}
    if not compared_tokens:
        return {"raw_inference_logprob_coverage": 0.0}
    return {
        "raw_inference_logprob_coverage": compared_tokens / expected_tokens,
        "inference_k1": total_k1 / compared_samples,
        "inference_k3": total_k3 / compared_samples,
    }


def policy_ratio_metrics(
    batch: Any,
    result: Any,
    *,
    loss_name: str,
    prefix: str,
    outside_trust_region: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    sequence_level: bool = False,
) -> Dict[str, float]:
    """Trust-region diagnostics for a server-side policy loss, from its returned logprobs.

    ``batch`` is a :class:`training.utils.rl.losses.PolicyBatch`. Ratios are
    policy over ``batch.anchor_logprobs``, per token or, with
    ``sequence_level``, per response from the mean log-ratio. Each objective
    module passes the rule for its own trust region, ``(ratio,
    policy_logprobs, anchor_logprobs) -> bool mask`` over active tokens (or one
    response), and gets:

    - ``{prefix}_ratio_mean``: mean ratio, averaged per datum.
    - ``{prefix}_clip_frac``: share of ratios outside the trust region.
    - ``{prefix}_kl``: k3 estimate of the policy/anchor KL, averaged per datum.
    - ``active_tokens`` / ``total_resp_tokens``: masked and total response tokens.

    Sampler/trainer drift is a separate call, :func:`inference_drift_metrics`.
    """
    policy = result_logprobs(result)
    ratio_sum = clip_sum = kl_sum = 0.0
    samples = active_tokens = 0
    for pair in iter_response_logprobs(
        batch.data,
        policy,
        batch.anchor_logprobs,
        batch.prompt_lens,
        loss_name=loss_name,
        source="anchor_logprobs",
    ):
        log_ratio = torch.clamp(pair.policy - pair.reference, -SAFETY_CLAMP, SAFETY_CLAMP)
        kl_sum += (torch.exp(log_ratio) - log_ratio - 1.0).mean().item()
        ratio = torch.exp(log_ratio.mean() if sequence_level else log_ratio)
        ratio_sum += ratio.mean().item()
        if outside_trust_region is not None:
            outside = outside_trust_region(ratio, pair.policy, pair.reference)
            clip_sum += outside.float().mean().item()
        samples += 1
        active_tokens += len(log_ratio)
    metrics: Dict[str, float] = {
        "active_tokens": float(active_tokens),
        "total_resp_tokens": float(sum(max(0, len(pi) - max(0, n - 1)) for pi, n in zip(policy, batch.prompt_lens))),
    }
    if samples:
        metrics[f"{prefix}_ratio_mean"] = ratio_sum / samples
        metrics[f"{prefix}_kl"] = kl_sum / samples
        if outside_trust_region is not None:
            metrics[f"{prefix}_clip_frac"] = clip_sum / samples
    return metrics
