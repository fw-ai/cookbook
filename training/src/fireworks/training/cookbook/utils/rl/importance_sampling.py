"""Truncated Importance Sampling (TIS) for train-inference mismatch correction.

Provides per-token importance weighting that can be composed with **any**
base policy loss (GRPO, DAPO, GSPO, etc.).  The architecture follows
an orthogonal design: base loss computes per-token loss, TIS
multiplies by clipped importance weights, then the result is summed.

Usage::

    Config(policy_loss="grpo", tis_enabled=True, tis=ISConfig(clip_high=10.0))
    Config(policy_loss="dapo", tis_enabled=True)
    Config(policy_loss="gspo", tis_enabled=True)

To write your own TIS, replace ``make_tis_weights_fn`` with a function
that returns the same ``(pi_detached, sample_idx) -> (weights, metrics)``
signature and pass it to the loss via ``tis_weights_fn=``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union, Callable
from dataclasses import dataclass

import torch

from fireworks.training.cookbook.utils.rl.common import _normalize_prompt_lens

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
