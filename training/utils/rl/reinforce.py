"""REINFORCE loss with importance-sampling correction.

Uses the IS ratio ``p_theta / q = exp(log pi - log q_inf)`` directly,
without PPO-style clipping or TIS decomposition.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens, run_loss_loop
from training.utils.rl.spec import LossSpec
from training.utils.rl.tis import SAFETY_CLAMP, TISConfig


def make_reinforce_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    prompt_lens: Union[int, List[int]],
    inf_logprobs: List[List[float]],
    prox_logprobs: List[List[float]],
    kl_beta: float = 0.0,
    tis_config: TISConfig | None = None,
):
    """REINFORCE loss with direct IS ratio ``p_theta / q``.

    Per-token loss for sample *i* at response position *t*::

        L_it = (-A_i * exp(log pi - log q_inf)
                + beta * (log pi - log pi_ref)) * mask(t)

    Args:
        advantages: Per-sample advantage values.
        ref_logprobs: Per-sample reference log-probability sequences.
        prompt_lens: Prompt token length(s); scalar broadcasts to all samples.
        inf_logprobs: Per-sample inference deployment log-probabilities.
        prox_logprobs: Per-sample proximal (pre-training forward) log-probabilities.
        kl_beta: KL penalty coefficient (0 disables KL term).
        tis_config: TIS weight configuration.
    """
    if tis_config is None:
        tis_config = TISConfig()
    prompt_lens_list = _normalize_prompt_lens(prompt_lens, len(advantages))

    def policy_fn(ctx):
        log_ratio = torch.clamp(ctx.resp_pi - ctx.resp_inf, min=-SAFETY_CLAMP, max=SAFETY_CLAMP)
        ratio = torch.exp(log_ratio)
        per_token_loss = -ratio * ctx.adv
        if kl_beta > 0.0:
            per_token_loss = per_token_loss + kl_beta * (ctx.pi_detached - ctx.resp_ref)
        per_token_loss = per_token_loss * ctx.resp_mask

        active = ctx.resp_mask > 0.5
        return per_token_loss, {
            "resp_len": float(len(ctx.resp_pi)),
            "ratio_mean": ratio.detach()[active].mean().item(),
            "adv_term": (-ctx.adv * ratio.detach() * ctx.resp_mask).sum().item(),
            "kl_term": (kl_beta * ctx.pi_detached * ctx.resp_mask).sum().item(),
        }

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(
            advantages,
            ref_logprobs,
            inf_logprobs,
            prompt_lens_list,
            prox_logprobs,
            tis_config,
            data,
            logprobs_list,
            "reinforce",
            policy_fn,
        )
        nt = result.num_tokens
        total_resp = int(result.extra_sums.get("resp_len", 0.0))

        ns = result.n_samples
        metrics = dict(result.base_metrics)
        metrics["active_tokens"] = nt
        metrics["total_resp_tokens"] = total_resp
        metrics["mask_ratio"] = nt / total_resp if total_resp > 0 else 0.0
        metrics["is_ratio_mean"] = result.extra_sums.get("ratio_mean", 0.0) / ns if ns > 0 else 0.0
        metrics["mean_adv_loss"] = result.extra_sums.get("adv_term", 0.0) / nt if nt > 0 else 0.0
        metrics["mean_kl_penalty"] = result.extra_sums.get("kl_term", 0.0) / nt if nt > 0 else 0.0
        metrics["mean_loss"] = result.total_loss.item() / nt if nt > 0 else 0.0
        return result.total_loss, metrics

    return loss_fn


def _client_loss_factory(
    *,
    advantages: List[float],
    ref_logprobs: List[List[float]],
    prompt_lens: List[int],
    inf_logprobs: List[List[float]],
    prox_logprobs: List[List[float]],
    kl_beta: float,
    tis_config: TISConfig,
    **_kw: Any,
) -> Any:
    return make_reinforce_loss_fn(
        advantages,
        ref_logprobs,
        prompt_lens,
        inf_logprobs=inf_logprobs,
        prox_logprobs=prox_logprobs,
        kl_beta=kl_beta,
        tis_config=tis_config,
    )


LOSS_SPEC = LossSpec(
    name="reinforce",
    client_loss_factory=_client_loss_factory,
    builtin_config_builder=None,
)
