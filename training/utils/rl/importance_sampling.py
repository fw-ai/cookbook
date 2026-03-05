"""Decoupled importance sampling corrections for RL training.

AReaL-style two-correction approach (https://arxiv.org/abs/2505.24298):

* **PPO off-policy ratio** between current policy (pi_theta) and
  proximal policy (pi_prox), with PPO-style clipping.  The proximal
  logprobs are pre-computed via a real forward pass before the training
  loop (like VERL's ``compute_log_prob``).

* **Behavioral IS weight** between proximal policy and rollout
  (pi_old / inf_logprobs), with capping.  Corrects for the
  train-inference numerical gap (FP8, quantization, etc.).

With ``ppo_n_minibatches=1``, ``pi_prox == pi_theta`` so the PPO ratio
is 1 and only the behavioral weight matters.  With ``ppo_n_minibatches>1``,
weights drift across minibatches and the PPO ratio becomes non-trivial.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SAFETY_CLAMP = 20.0
"""Clamp log-ratio to [-SAFETY_CLAMP, SAFETY_CLAMP] before exp() to
prevent inf/NaN.  Matches VERL's ``SAFETY_BOUND``."""


@dataclass
class DecoupledConfig:
    """AReaL-style decoupled IS correction configuration.

    Controls the behavioral IS weight between proximal and rollout logprobs.
    The PPO ratio clipping is configured per-loss (DAPO, GSPO, CISPO configs).
    For GRPO, the PPO clip bounds are ``eps_clip`` / ``eps_clip_high``.
    """

    eps_clip: float = 0.2
    """Symmetric PPO clip epsilon for the off-policy ratio (used by GRPO)."""
    eps_clip_high: float | None = None
    """Asymmetric upper clip bound for GRPO.  When set, lower=eps_clip, upper=eps_clip_high."""
    behave_cap: float = 5.0
    """Upper cap for the behavioral IS weight."""
    behave_mode: str = "token_truncate"
    """'token_truncate': clamp to [0, cap].  'token_mask': zero out where > cap."""


def compute_behave_weight(
    resp_prox: torch.Tensor,
    resp_inf: torch.Tensor,
    config: DecoupledConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the behavioral IS weight: exp(prox - inf), capped.

    Args:
        resp_prox: proximal policy logprobs (response tokens only)
        resp_inf: inference/rollout logprobs (response tokens only)
        config: capping configuration

    Returns:
        (behave_weight, metrics) where behave_weight is detached.
    """
    behave_log = torch.clamp(resp_prox - resp_inf, min=-SAFETY_CLAMP, max=SAFETY_CLAMP)
    behave_raw = torch.exp(behave_log)
    if config.behave_mode == "token_mask":
        behave_weight = torch.where(
            behave_raw > config.behave_cap,
            torch.zeros_like(behave_raw),
            behave_raw,
        )
    else:
        behave_weight = torch.clamp(behave_raw, min=0.0, max=config.behave_cap)
    clip_frac = (behave_weight != behave_raw).float().mean().item()

    metrics = {
        "behave/weight_mean": behave_weight.mean().item(),
        "behave/clip_frac": clip_frac,
    }
    return behave_weight, metrics
