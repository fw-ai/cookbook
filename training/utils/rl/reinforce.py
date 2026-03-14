"""Vanilla REINFORCE (Williams, 1992) loss for RL training.

Uses raw policy log-probabilities directly without PPO-style clipping
or importance sampling correction, making it the simplest policy-gradient
loss variant.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens, _get_loss_mask
from training.utils.rl.spec import LossSpec


def make_reinforce_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    prompt_lens: Union[int, List[int]],
    kl_beta: float = 0.0,
):
    """Vanilla REINFORCE loss: ``-advantage * log pi`` + optional KL penalty.

    Per-token loss for sample *i* at response position *t*::

        L_it = -A_i * log pi(a_t | s_t) + beta * (log pi(a_t | s_t) - log pi_ref(a_t | s_t))

    Args:
        advantages: Per-sample advantage values.
        ref_logprobs: Per-sample reference log-probability sequences.
        prompt_lens: Prompt token length(s); scalar broadcasts to all samples.
        kl_beta: KL penalty coefficient (0 disables KL term).
    """
    prompt_lens_list = _normalize_prompt_lens(prompt_lens, len(advantages))

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, requires_grad=True)
        total_kl = 0.0
        num_tokens = 0
        total_resp_tokens = 0
        total_adv_term = 0.0
        total_kl_term = 0.0

        for i, pi_logprobs in enumerate(logprobs_list):
            adv = advantages[i]
            response_start = max(0, prompt_lens_list[i] - 1)

            resp_pi = pi_logprobs[response_start:]
            resp_len = len(resp_pi)
            if resp_len == 0:
                continue

            resp_mask = _get_loss_mask(
                data[i], response_start, resp_len, resp_pi.dtype, resp_pi.device,
            )
            active = resp_mask > 0.5
            active_count = int(active.sum().item())
            if active_count == 0:
                continue

            adv_t = torch.as_tensor(adv, dtype=resp_pi.dtype, device=resp_pi.device)
            per_token_loss = -adv_t * resp_pi

            if kl_beta > 0.0 and i < len(ref_logprobs) and ref_logprobs[i]:
                ref_lp = ref_logprobs[i]
                resp_ref = torch.tensor(
                    [ref_lp[response_start + j] if (response_start + j) < len(ref_lp) else 0.0
                     for j in range(resp_len)],
                    dtype=resp_pi.dtype, device=resp_pi.device,
                )
                kl_penalty = kl_beta * (resp_pi - resp_ref)
                per_token_loss = per_token_loss + kl_penalty

                pi_detached = resp_pi.detach()
                kl_per_token = (pi_detached - resp_ref) * resp_mask
                total_kl += kl_per_token.sum().item()
                total_kl_term += (kl_beta * pi_detached * resp_mask).sum().item()

            per_token_loss = per_token_loss * resp_mask
            total_loss = total_loss + per_token_loss.sum()
            total_adv_term += (-adv * resp_pi.detach() * resp_mask).sum().item()
            num_tokens += active_count
            total_resp_tokens += resp_len

        metrics: Dict[str, float] = {
            "active_tokens": num_tokens,
            "total_resp_tokens": total_resp_tokens,
            "mask_ratio": num_tokens / total_resp_tokens if total_resp_tokens > 0 else 0.0,
            "mean_adv_loss": total_adv_term / num_tokens if num_tokens > 0 else 0.0,
            "mean_loss": total_loss.item() / num_tokens if num_tokens > 0 else 0.0,
        }
        if kl_beta > 0.0:
            metrics["mean_kl"] = total_kl / num_tokens if num_tokens > 0 else 0.0
            metrics["mean_kl_penalty"] = total_kl_term / num_tokens if num_tokens > 0 else 0.0

        return total_loss, metrics

    return loss_fn


def _client_loss_factory(
    *,
    advantages: List[float],
    ref_logprobs: List[List[float]],
    prompt_lens: List[int],
    kl_beta: float,
    **_kw: Any,
) -> Any:
    return make_reinforce_loss_fn(
        advantages, ref_logprobs, prompt_lens, kl_beta=kl_beta,
    )


LOSS_SPEC = LossSpec(
    name="reinforce",
    client_loss_factory=_client_loss_factory,
    builtin_config_builder=None,
)
