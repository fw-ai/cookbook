"""Unclipped Importance Sampling (IS) policy loss.

Unclipped importance-sampling objective matching the Tinker kernel::

    loss = -(exp(lp - rollout) * adv).sum()

The ratio ``exp(pi - rollout)`` is capped by ``ratio_log_cap`` for numerical
stability but is otherwise unclipped. Its denominator is the recorded
sampling distribution.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens, run_loss_loop
from training.utils.rl.algorithm.base import ClientObjective, PolicyLoss


def validate_is_config(*, ratio_log_cap: float) -> None:
    """Validate importance-sampling settings before building the loss closure."""
    if ratio_log_cap < 0:
        raise ValueError("IS ratio_log_cap must be non-negative.")


def make_is_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    ratio_log_cap: float = 20.0,
) -> ...:
    """Build an IS loss closure with unclipped ratio."""
    validate_is_config(ratio_log_cap=ratio_log_cap)
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))

    def policy_fn(ctx):
        log_ratio = torch.clamp(
            ctx.resp_pi - ctx.resp_inf,
            min=-ratio_log_cap,
            max=ratio_log_cap,
        )
        ratio = torch.exp(log_ratio)
        per_token_loss = -(ratio * ctx.adv) * ctx.resp_mask
        return per_token_loss, {"is_ratio_mean": ratio.detach().mean().item()}

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(
            advantages,
            ref_logprobs,
            inf_logprobs,
            prompt_lens,
            data,
            logprobs_list,
            "importance_sampling",
            policy_fn,
        )
        metrics = dict(result.base_metrics)
        metrics["is_ratio_mean"] = result.extra_sums.get("is_ratio_mean", 0.0) / result.n_samples
        return result.total_loss, metrics

    return loss_fn


POLICY_LOSS = PolicyLoss(
    name="importance_sampling",
    client=lambda inputs, _options: ClientObjective(
        inputs.data,
        make_is_loss_fn(
            advantages=inputs.advantages,
            ref_logprobs=inputs.ref_logprobs,
            inf_logprobs=inputs.rollout_logprobs,
            prompt_len=inputs.prompt_lens,
        ),
    ),
    builtin_name="importance_sampling",
    loss_fn_config=lambda _options: {},
    uses_anchor=False,
)
