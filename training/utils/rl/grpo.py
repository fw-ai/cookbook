"""GRPO (Group Relative Policy Optimization) loss for RL training.

Uses PPO-style clipped surrogate objective with behavioral TIS weight
correction.  The PPO ratio is computed against pre-computed old-policy
logprobs (from a forward pass before training), and the TIS weight
corrects for the train-inference gap. Optional reference regularization uses
the differentiable k3 KL estimator.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import tinker

from training.utils.rl.common import _normalize_prompt_lens, run_loss_loop
from training.utils.rl.observability import compute_inference_observability_metrics
from training.utils.rl.tis import SAFETY_CLAMP, TISConfig


def validate_grpo_config(
    *,
    kl_beta: float,
    eps_clip: float,
    eps_clip_high: float | None,
    reference_training_shape_id: str | None = None,
    reference_job_id: str | None = None,
    reference_configured: bool = False,
    anchor_logp: str | None = None,
    ppo_n_minibatches: int | None = None,
) -> None:
    """Validate GRPO loss and recipe settings before provisioning or training."""
    if kl_beta < 0:
        raise ValueError("kl_beta must be non-negative.")
    if kl_beta == 0 and (
        reference_configured
        or reference_training_shape_id is not None
        or reference_job_id is not None
    ):
        raise ValueError(
            "Reference trainer settings require kl_beta > 0; they would be unused."
        )
    if eps_clip < 0 or (eps_clip_high is not None and eps_clip_high < 0):
        raise ValueError("eps_clip and eps_clip_high must be non-negative.")
    if anchor_logp is not None and anchor_logp not in {"old_policy", "rollout"}:
        raise ValueError(
            "anchor_logp must be 'old_policy' or 'rollout', got "
            f"{anchor_logp!r}."
        )
    if ppo_n_minibatches is not None and ppo_n_minibatches < 1:
        raise ValueError("ppo_n_minibatches must be >= 1.")


def make_grpo_loss_fn(
    advantages: List[float],
    ref_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    inf_logprobs: List[List[float]],
    old_policy_logprobs: List[List[float]],
    kl_beta: float = 0.001,
    eps_clip: float = 0.2,
    eps_clip_high: float | None = None,
    tis_config: TISConfig | None = None,
    raw_inf_logprobs: List[List[float]] | None = None,
) -> ...:
    """GRPO loss with PPO-clipped ratio and behavioral TIS weight.

    ``old_policy_logprobs`` are pre-computed by a forward pass before training.
    The PPO ratio ``exp(pi_theta - old_policy)`` is clipped by ``eps_clip``.
    The TIS weight ``exp(old_policy - inf)`` corrects for train-inference mismatch.
    Reference KL uses ``exp(ref - pi) - (ref - pi) - 1``.
    """
    validate_grpo_config(
        kl_beta=kl_beta,
        eps_clip=eps_clip,
        eps_clip_high=eps_clip_high,
    )
    if tis_config is None:
        tis_config = TISConfig()
    prompt_lens = _normalize_prompt_lens(prompt_len, len(advantages))
    _eps_high = eps_clip if eps_clip_high is None else eps_clip_high

    def policy_fn(ctx):
        log_ratio = torch.clamp(ctx.resp_pi - ctx.resp_old_policy, min=-SAFETY_CLAMP, max=SAFETY_CLAMP)
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + _eps_high)

        active = ctx.resp_mask > 0.5
        clip_frac = (clipped_ratio[active] != ratio[active]).float().mean().item()
        ratio_mean = ratio.detach()[active].mean().item()

        surr1 = -ratio * ctx.adv
        surr2 = -clipped_ratio * ctx.adv
        ref_log_ratio = ctx.resp_ref - ctx.resp_pi
        ref_kl = torch.exp(ref_log_ratio) - ref_log_ratio - 1.0
        kl_penalty = kl_beta * ref_kl
        per_token_loss = (torch.maximum(surr1, surr2) * ctx.tis_weight + kl_penalty) * ctx.resp_mask

        return per_token_loss, {
            "clip_frac": clip_frac,
            "ratio_mean": ratio_mean,
            "resp_len": float(len(ctx.resp_pi)),
            "adv_term": (-ctx.adv * ctx.resp_pi * ctx.resp_mask).sum().item(),
            "kl_term": (kl_penalty * ctx.resp_mask).sum().item(),
        }

    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result = run_loss_loop(
            advantages, ref_logprobs, inf_logprobs, prompt_lens,
            old_policy_logprobs, tis_config, data, logprobs_list, "grpo", policy_fn,
        )
        ns = result.n_samples
        nt = result.num_tokens
        total_resp = int(result.extra_sums.get("resp_len", 0.0))

        metrics = dict(result.base_metrics)
        metrics["ppo_clip_frac"] = result.extra_sums.get("clip_frac", 0.0) / ns
        metrics["ppo_ratio_mean"] = result.extra_sums.get("ratio_mean", 0.0) / ns
        metrics["active_tokens"] = nt
        metrics["total_resp_tokens"] = total_resp
        metrics["mask_ratio"] = nt / total_resp if total_resp > 0 else 0.0
        metrics["mean_adv_loss"] = result.extra_sums.get("adv_term", 0.0) / nt if nt > 0 else 0.0
        metrics["mean_kl_penalty"] = result.extra_sums.get("kl_term", 0.0) / nt if nt > 0 else 0.0
        metrics["mean_loss"] = result.total_loss.item() / nt if nt > 0 else 0.0
        if raw_inf_logprobs is not None:
            metrics.update(
                compute_inference_observability_metrics(
                    data,
                    logprobs_list,
                    raw_inf_logprobs,
                    prompt_lens,
                    "grpo",
                )
            )
        return result.total_loss, metrics

    return loss_fn
