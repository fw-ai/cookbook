"""Unclipped Importance Sampling (IS) policy loss.

Unclipped importance-sampling objective matching the Tinker kernel::

    loss = -(exp(lp - slp) * adv).sum()

The ratio ``exp(pi - prox)`` is capped by ``ratio_log_cap`` for numerical
stability but is otherwise unclipped.  TIS weight corrects for the
train-inference gap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import torch
import tinker

from training.utils.rl.spec import LossSpec
from training.utils.rl.tis import TISConfig


def make_is_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    prox_logprobs: List[List[float]],
    ratio_log_cap: float = 20.0,
    tis_config: TISConfig | None = None,
) -> ...:
    """Build an IS loss closure with unclipped ratio and behavioral TIS weight."""
    from training.utils.rl.common import _normalize_prompt_lens, run_loss_loop

    if tis_config is None:
        tis_config = TISConfig()
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))

    def policy_fn(ctx):
        log_ratio = torch.clamp(
            ctx.resp_pi - ctx.resp_prox,
            min=-ratio_log_cap,
            max=ratio_log_cap,
        )
        ratio = torch.exp(log_ratio)
        per_token_loss = -(ratio * ctx.adv) * ctx.tis_weight * ctx.resp_mask
        return per_token_loss, {"is_ratio_mean": ratio.detach().mean().item()}

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(
            advantages, ref_logprobs, inf_logprobs, prompt_lens,
            prox_logprobs, tis_config, data, logprobs_list, "importance_sampling", policy_fn,
        )
        metrics = dict(result.base_metrics)
        metrics["is_ratio_mean"] = result.extra_sums.get("is_ratio_mean", 0.0) / result.n_samples
        return result.total_loss, metrics

    return loss_fn


def _builtin_config(*, ratio_log_cap: float = 20.0, **_kw: Any) -> tuple[str, dict[str, Any]]:
    return "importance_sampling", {
        "ratio_log_cap": ratio_log_cap,
    }


def _make_loss(
    *,
    advantages: List[float],
    ref_logprobs: List[List[float]],
    prompt_lens: List[int],
    inf_logprobs: List[List[float]],
    prox_logprobs: List[List[float]],
    tis_config: TISConfig,
    ratio_log_cap: float = 20.0,
    **_kw: Any,
) -> Any:
    return make_is_loss_fn(
        advantages,
        ref_logprobs,
        inf_logprobs,
        prompt_lens,
        prox_logprobs,
        ratio_log_cap=ratio_log_cap,
        tis_config=tis_config,
    )


LOSS_SPEC = LossSpec(
    name="importance_sampling",
    make_loss_fn=_make_loss,
    builtin_config_builder=_builtin_config,
)
