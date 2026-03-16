"""DRO (Distributionally Robust Optimization) policy loss.

Matches the Tinker kernel formula::

    loss = -(lp * adv - 0.5 * beta * (lp - slp)^2).sum()

The quadratic penalty ``0.5 * beta * (lp - slp)^2`` constrains the policy
at *all* positions (including where ``adv=0``) to stay close to the
proximal checkpoint.  This differs from PPO-style clipping by providing a
smooth, continuous penalty rather than a hard clip boundary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens, run_loss_loop
from training.utils.rl.spec import LossSpec
from training.utils.rl.tis import TISConfig


@dataclass
class DROConfig:
    """DRO loss configuration.

    ``beta`` controls the strength of the quadratic proximity penalty.
    """

    beta: float = 0.05


def make_dro_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    prox_logprobs: List[List[float]],
    dro_config: DROConfig | None = None,
    tis_config: TISConfig | None = None,
) -> ...:
    """Build a DRO loss closure with quadratic proximity penalty."""
    if dro_config is None:
        dro_config = DROConfig()
    if tis_config is None:
        tis_config = TISConfig()
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))

    def policy_fn(ctx):
        quad = (ctx.resp_pi - ctx.resp_prox) ** 2
        resp_active = ctx.adv != 0
        gated_quad = torch.where(resp_active, quad, torch.zeros_like(quad))
        linear_term = ctx.resp_pi * ctx.adv * ctx.tis_weight
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
            advantages, ref_logprobs, inf_logprobs, prompt_lens,
            prox_logprobs, tis_config, data, logprobs_list, "dro", policy_fn,
        )
        metrics = dict(result.base_metrics)
        nt = result.num_tokens
        metrics["dro_quad_penalty"] = result.extra_sums.get("quad_penalty", 0.0) / nt if nt > 0 else 0.0
        return result.total_loss, metrics

    return loss_fn


def _builtin_config(
    *, dro_config: DROConfig | None = None, **_kw: Any,
) -> tuple[str, dict[str, Any]]:
    cfg = dro_config or DROConfig()
    return "dro", {
        "beta": cfg.beta,
    }


def _make_loss(
    *,
    advantages: List[float],
    ref_logprobs: List[List[float]],
    prompt_lens: List[int],
    inf_logprobs: List[List[float]],
    prox_logprobs: List[List[float]],
    dro_config: DROConfig | None,
    tis_config: TISConfig,
    **_kw: Any,
) -> Any:
    return make_dro_loss_fn(
        advantages,
        ref_logprobs,
        inf_logprobs,
        prompt_lens,
        prox_logprobs,
        dro_config,
        tis_config=tis_config,
    )


LOSS_SPEC = LossSpec(
    name="dro",
    make_loss_fn=_make_loss,
    builtin_config_builder=_builtin_config,
)
