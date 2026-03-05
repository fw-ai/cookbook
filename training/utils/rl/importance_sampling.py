"""Importance sampling corrections for RL training.

Two orthogonal correction mechanisms:

1. **TIS (Truncated Importance Sampling)** -- legacy single-ratio
   correction: ``clamp(exp(train_lp - rollout_lp), low, high)``.
   Enabled via ``tis_enabled=True``.

2. **Decoupled IS** -- AReaL-style two-correction approach
   (https://arxiv.org/abs/2505.24298):

   * **PPO off-policy ratio** between current policy (pi_theta) and a
     proximal policy (pi_prox), with PPO-style clipping.
   * **Behavioral IS weight** between proximal policy and rollout
     (pi_old), with capping.
   * Proximal policy is approximated via log-linear interpolation
     (no extra forward pass).

   Enabled via ``decoupled=DecoupledConfig()``.

Usage::

    Config(policy_loss="grpo", tis_enabled=True, tis=ISConfig(clip_high=10.0))
    Config(policy_loss="grpo", decoupled=DecoupledConfig())
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union, Callable
from dataclasses import dataclass

import torch

from training.utils.rl.common import _normalize_prompt_lens

SAFETY_CLAMP = 20.0
"""Clamp log-ratio to [-SAFETY_CLAMP, SAFETY_CLAMP] before exp() to
prevent inf/NaN.  Matches VERL's ``SAFETY_BOUND``."""

TISWeightsFn = Callable[
    [torch.Tensor, int],
    Tuple[torch.Tensor, Dict[str, float]],
]
"""Per-sample TIS weights function: ``(pi_detached, sample_idx) -> (weights, metrics)``."""


@dataclass
class ISConfig:
    """TIS (Truncated Importance Sampling) configuration."""

    clip_high: float = 2.0
    clip_low: float = 0.0


@dataclass
class DecoupledConfig:
    """AReaL-style decoupled IS correction configuration.

    Splits the single IS ratio into two corrections:
    1. PPO ratio: exp(pi_theta - pi_prox) with eps_clip
    2. Behavioral weight: exp(pi_prox - pi_old) with behave_cap
    """

    eps_clip: float = 0.2
    """Symmetric PPO clip epsilon for the off-policy ratio."""
    eps_clip_high: float | None = None
    """Asymmetric upper clip bound.  When set, lower=eps_clip, upper=eps_clip_high."""
    behave_cap: float = 5.0
    """Upper cap for the behavioral IS weight."""
    behave_mode: str = "token_truncate"
    """'token_truncate': clamp to [0, cap].  'token_mask': zero out where > cap."""


def make_tis_weights_fn(
    inf_logprobs: List[List[float]],
    prompt_len: Union[int, List[int]],
    tis_config: ISConfig | None = None,
) -> TISWeightsFn:
    """Create a per-sample TIS weights function (vanilla clamped IS).

    Computes ``weights = clamp(exp(train_lp - rollout_lp), low, high)``
    per response token -- the same formula used in common open-source RL stacks.

    ``prompt_len`` may be a single int or a per-datum list for multi-prompt
    batched calls.

    Returns a callable ``(pi_detached, sample_idx) -> (weights, metrics)``
    suitable for passing to any loss function's ``tis_weights_fn`` parameter.
    """
    if tis_config is None:
        tis_config = ISConfig()
    prompt_lens = _normalize_prompt_lens(prompt_len, len(inf_logprobs))

    def weights_fn(
        pi_detached: torch.Tensor,
        sample_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        inf_lp = inf_logprobs[sample_idx]
        if not inf_lp:
            raise ValueError(
                f"TIS requires inference logprobs for sample {sample_idx} but got empty list. "
                f"Ensure logprobs=True is set when tis_enabled=True."
            )
        response_start = max(0, prompt_lens[sample_idx] - 1)
        resp_len = len(pi_detached)
        if len(inf_lp) < response_start + resp_len:
            raise ValueError(
                f"TIS requires at least {response_start + resp_len} inference logprobs "
                f"for sample {sample_idx}, got {len(inf_lp)}."
            )
        resp_inf = torch.tensor(
            inf_lp[response_start : response_start + resp_len],
            dtype=pi_detached.dtype,
            device=pi_detached.device,
        )

        log_ratio = torch.clamp(pi_detached - resp_inf, min=-SAFETY_CLAMP, max=SAFETY_CLAMP)
        rho = torch.exp(log_ratio)
        weights = torch.clamp(rho, min=tis_config.clip_low, max=tis_config.clip_high)
        clip_frac = (weights != rho).float().mean().item()

        metrics = {
            "tis_mean_ratio": rho.mean().item(),
            "tis_max_ratio": rho.max().item(),
            "tis_clip_frac": clip_frac,
        }
        return weights, metrics

    return weights_fn


# ---------------------------------------------------------------------------
# Decoupled IS correction (AReaL-style)
# ---------------------------------------------------------------------------


def _compute_proximal_logprobs(
    pi_theta: torch.Tensor,
    pi_old: torch.Tensor,
    generation_step: int,
    current_step: int,
) -> Tuple[torch.Tensor, float]:
    """Log-linear approximation of proximal policy logprobs.

    Follows AReaL's loglinear method (arXiv:2505.24298, functional.py):
      v_prox = current_step - 1
      alpha = clamp((v_prox - generation_step) / (current_step - generation_step), 0, 1)
      log_p_prox = (1 - alpha) * log_p_old + alpha * log_p_theta

    Returns (proximal_logprobs, alpha).
    """
    version_diff = current_step - generation_step
    if version_diff <= 0:
        return pi_old.clone(), 0.0

    v_prox = current_step - 1
    alpha = max(0.0, min(1.0, (v_prox - generation_step) / version_diff))
    pi_prox = (1.0 - alpha) * pi_old + alpha * pi_theta
    return pi_prox, alpha


def compute_decoupled_corrections(
    pi_theta: torch.Tensor,
    pi_old: torch.Tensor,
    generation_step: int,
    current_step: int,
    config: DecoupledConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Compute both decoupled IS corrections for a single sample.

    Returns (ppo_ratio, behave_weight, metrics) where:
      ppo_ratio = exp(pi_theta - pi_prox), clipped by eps_clip  (WITH gradient)
      behave_weight = exp(pi_prox - pi_old), capped             (detached)
    """
    pi_prox, alpha = _compute_proximal_logprobs(
        pi_theta.detach(), pi_old, generation_step, current_step,
    )

    # PPO off-policy ratio (carries gradient through pi_theta)
    log_ratio = torch.clamp(pi_theta - pi_prox, min=-SAFETY_CLAMP, max=SAFETY_CLAMP)
    ppo_ratio = torch.exp(log_ratio)
    eps_high = config.eps_clip if config.eps_clip_high is None else config.eps_clip_high
    ppo_clipped = torch.clamp(ppo_ratio, min=1.0 - config.eps_clip, max=1.0 + eps_high)
    ppo_clip_frac = (ppo_clipped != ppo_ratio).float().mean().item()

    # Behavioral IS weight (detached -- no gradient)
    behave_log = torch.clamp(pi_prox - pi_old, min=-SAFETY_CLAMP, max=SAFETY_CLAMP)
    behave_raw = torch.exp(behave_log)
    if config.behave_mode == "token_mask":
        behave_weight = torch.where(
            behave_raw > config.behave_cap,
            torch.zeros_like(behave_raw),
            behave_raw,
        )
    else:
        behave_weight = torch.clamp(behave_raw, min=0.0, max=config.behave_cap)
    behave_clip_frac = (behave_weight != behave_raw).float().mean().item()

    metrics = {
        "decoupled/ppo_ratio_mean": ppo_ratio.detach().mean().item(),
        "decoupled/ppo_clip_frac": ppo_clip_frac,
        "decoupled/behave_weight_mean": behave_weight.mean().item(),
        "decoupled/behave_clip_frac": behave_clip_frac,
        "decoupled/alpha": alpha,
    }
    return ppo_ratio, ppo_clipped, behave_weight, metrics
