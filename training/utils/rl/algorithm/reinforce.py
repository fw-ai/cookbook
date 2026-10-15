"""REINFORCE with a direct importance ratio ``p_theta / q`` (client loss only).

Per-token loss for sample *i* at response position *t*::

    L_it = (-A_i * exp(log pi - log q) + kl_beta * (log pi - log pi_ref)) * mask(t)

``q`` is the rollout (sampling) distribution. The trainer has no built-in
REINFORCE, so this module has no ``LOSS_FN``:
``policy.forward_backward_custom(build_datums(batch, config), make_loss_fn(batch, config))``. Pair it with a
mean-only ``advantage_fn`` for group-centered rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from fireworks.training import sdk

from training.utils.client import GradAccNormalization
from training.utils.rl.common import SAFETY_CLAMP, client_loss_metrics, run_loss_loop
from training.utils.rl.losses import PolicyBatch

NORMALIZATION = GradAccNormalization.NUM_LOSS_TOKENS

PREFIX = "reinforce"


@dataclass(frozen=True)
class ReinforceConfig:
    """``kl_beta`` > 0 adds a reference penalty and needs ``batch.ref_logprobs``."""

    kl_beta: float = 0.0

    def __post_init__(self) -> None:
        if self.kl_beta < 0:
            raise ValueError("REINFORCE kl_beta must be non-negative.")


def build_datums(batch: PolicyBatch, config: ReinforceConfig) -> List[sdk.Datum]:
    """Datums for :func:`make_loss_fn`: the batch's own datums, which carry the loss mask."""
    del config
    return list(batch.data)


def make_loss_fn(batch: PolicyBatch, config: ReinforceConfig):
    """Client-side REINFORCE loss over :func:`build_datums`."""
    if config.kl_beta and not batch.ref_logprobs:
        raise ValueError("REINFORCE kl_beta > 0 needs batch.ref_logprobs.")

    def policy_fn(ctx):
        log_ratio = torch.clamp(ctx.resp_pi - ctx.resp_inf, min=-SAFETY_CLAMP, max=SAFETY_CLAMP)
        ratio = torch.exp(log_ratio)
        per_token_loss = -ratio * ctx.adv
        if config.kl_beta:
            per_token_loss = per_token_loss + config.kl_beta * (ctx.pi_detached - ctx.resp_ref)
        active = ctx.resp_mask > 0.5
        return per_token_loss * ctx.resp_mask, {
            "resp_len": float(len(ctx.resp_pi)),
            "ratio_mean": ratio.detach()[active].mean().item(),
            "adv_term": (-ctx.adv * ratio.detach() * ctx.resp_mask).sum().item(),
        }

    def loss_fn(data: List[sdk.Datum], logprobs_list: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(batch, data, logprobs_list, "reinforce", policy_fn)
        out = client_loss_metrics(result, PREFIX)
        nt = result.num_tokens
        total_resp = int(result.extra_sums.get("resp_len", 0.0))
        out["active_tokens"] = nt
        out["total_resp_tokens"] = total_resp
        out["mask_ratio"] = nt / total_resp if total_resp > 0 else 0.0
        out["mean_adv_loss"] = result.extra_sums.get("adv_term", 0.0) / nt if nt > 0 else 0.0
        out["mean_loss"] = result.total_loss.item() / nt if nt > 0 else 0.0
        return result.total_loss, out

    return loss_fn
