"""Shared helpers used by RL loss variants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
import tinker

if TYPE_CHECKING:
    from training.utils.rl.tis import TISConfig


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
    """Extract per-position loss mask from ``loss_fn_inputs["loss_mask"]``.

    Returns a tensor of shape ``[resp_len]`` sliced from ``response_start``.
    Falls back to all-ones (no masking) when the datum has no ``loss_mask``.
    """
    mask_td = datum.loss_fn_inputs.get("loss_mask")
    if mask_td is not None:
        mask_vals = mask_td.data[response_start : response_start + resp_len]
        if len(mask_vals) < resp_len:
            mask_vals = list(mask_vals) + [0.0] * (resp_len - len(mask_vals))
        return torch.tensor(mask_vals, dtype=dtype, device=device)
    return torch.ones(resp_len, dtype=dtype, device=device)


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
    resp_prox: torch.Tensor
    """Proximal forward-pass logprobs for response tokens."""
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
    prox_logprobs: List[List[float]],
    tis_config: TISConfig,
    data: List[tinker.Datum],
    logprobs_list: List[torch.Tensor],
    policy_fn: PolicyFn,
) -> LossLoopResult:
    """Shared loss loop: tensor setup, TIS weight, inference metrics, KL.

    Iterates over ``logprobs_list``, builds a :class:`SampleContext` for each
    sample, and delegates per-token loss computation to ``policy_fn``.
    """
    from training.utils.rl.tis import compute_tis_weight

    total_loss = torch.tensor(0.0, requires_grad=True)
    total_kl = 0.0
    total_inf_diff = 0.0
    total_inf_kld = 0.0
    inf_num_samples = 0
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
                data[i], response_start, resp_len, resp_pi.dtype, resp_pi.device,
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
            dtype=resp_pi.dtype, device=resp_pi.device,
        )
        pi_detached = resp_pi.detach()

        inf_lp = inf_logprobs[i] if inf_logprobs else []
        resp_inf = torch.tensor(
            [inf_lp[response_start + j] if (response_start + j) < len(inf_lp) else 0.0 for j in range(resp_len)],
            dtype=resp_pi.dtype, device=resp_pi.device,
        )
        prox_lp = prox_logprobs[i]
        resp_prox = torch.tensor(
            [prox_lp[response_start + j] if (response_start + j) < len(prox_lp) else 0.0 for j in range(resp_len)],
            dtype=resp_pi.dtype, device=resp_pi.device,
        )

        inf_log_diff = pi_detached - resp_inf
        total_inf_diff += inf_log_diff.abs().mean().item()
        total_inf_kld += (torch.exp(inf_log_diff) - inf_log_diff - 1.0).mean().item()
        inf_num_samples += 1

        tis_weight, bm = compute_tis_weight(resp_prox, resp_inf, tis_config)
        for k, v in bm.items():
            tis_metrics_agg[k] = tis_metrics_agg.get(k, 0.0) + v

        adv_t = torch.as_tensor(advantages[i], dtype=resp_pi.dtype, device=resp_pi.device)

        ctx = SampleContext(
            resp_pi=resp_pi, pi_detached=pi_detached,
            resp_ref=resp_ref, resp_prox=resp_prox,
            resp_mask=resp_mask, adv=adv_t, tis_weight=tis_weight,
        )
        per_token_loss, extra = policy_fn(ctx)

        total_loss = total_loss + per_token_loss.sum()
        total_kl += ((pi_detached - resp_ref) * resp_mask).sum().item()
        num_tokens += active_count
        for k, v in extra.items():
            extra_sums[k] = extra_sums.get(k, 0.0) + v

    n_samples = max(inf_num_samples, 1)
    base_metrics: Dict[str, float] = {
        "mean_kl": total_kl / num_tokens if num_tokens > 0 else 0.0,
    }
    if inf_num_samples > 0:
        base_metrics["inference_diff"] = total_inf_diff / inf_num_samples
        base_metrics["inference_kld"] = total_inf_kld / inf_num_samples
    for k, v in tis_metrics_agg.items():
        base_metrics[k] = v / n_samples

    return LossLoopResult(
        total_loss=total_loss,
        base_metrics=base_metrics,
        extra_sums=extra_sums,
        num_tokens=num_tokens,
        n_samples=n_samples,
    )
