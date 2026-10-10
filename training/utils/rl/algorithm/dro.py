"""DRO (Distributionally Robust Optimization) policy loss.

Matches the Tinker kernel formula::

    loss = -(lp * adv - 0.5 * beta * (lp - rollout)^2).sum()

The quadratic penalty ``0.5 * beta * (lp - rollout)^2`` constrains the policy
at *all* positions (including where ``adv=0``) to stay close to the
old-policy snapshot.  This differs from PPO-style clipping by providing a
smooth, continuous penalty rather than a hard clip boundary.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens, run_loss_loop
from training.utils.rl.algorithm.base import ClientObjective, PolicyLoss, client_loss_kwargs


@dataclass
class DROConfig:
    """DRO loss configuration.

    ``beta`` controls the strength of the quadratic old-policy penalty.
    """

    beta: float = 0.05


def validate_dro_config(config: DROConfig) -> None:
    """Validate DRO settings before building the loss closure."""
    if config.beta < 0:
        raise ValueError("DRO beta must be non-negative.")


def make_dro_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    dro_config: DROConfig | None = None,
    *,
    old_policy_logprobs: List[List[float]] | None = None,
) -> ...:
    """Build a DRO loss closure with quadratic old-policy penalty."""
    if dro_config is None:
        dro_config = DROConfig()
    validate_dro_config(dro_config)
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))

    def policy_fn(ctx):
        quad = (ctx.resp_pi - ctx.resp_old_policy) ** 2
        resp_active = ctx.adv != 0
        gated_quad = torch.where(resp_active, quad, torch.zeros_like(quad))
        linear_term = ctx.resp_pi * ctx.adv
        quad_term = 0.5 * dro_config.beta * gated_quad
        per_token_loss = -(linear_term - quad_term) * ctx.resp_mask

        return per_token_loss, {
            "quad_penalty": (quad_term * ctx.resp_mask).detach().sum().item(),
        }

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
            "dro",
            policy_fn,
            old_policy_logprobs=old_policy_logprobs,
        )
        metrics = dict(result.base_metrics)
        nt = result.num_tokens
        metrics["dro_quad_penalty"] = result.extra_sums.get("quad_penalty", 0.0) / nt if nt > 0 else 0.0
        return result.total_loss, metrics

    return loss_fn


POLICY_LOSS = PolicyLoss(
    name="dro",
    client=lambda inputs, options: ClientObjective(
        inputs.data,
        make_dro_loss_fn(**client_loss_kwargs(inputs), dro_config=options),
    ),
    builtin_name="dro",
    loss_fn_config=lambda options: {"beta": options.beta},
    validate=validate_dro_config,
)
