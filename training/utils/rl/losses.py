"""RL loss dispatch and rollout data types."""

from __future__ import annotations

from typing import Any, List, Tuple, Callable
from dataclasses import field, dataclass

import tinker

from training.utils.rl.importance_sampling import ISConfig


@dataclass
class PromptGroup:
    """Processed data from one prompt's rollout, ready for training."""

    data: List[tinker.Datum]
    advantages: List[float]
    ref_logprobs: List[List[float]]
    prompt_len: int
    rewards: List[float]
    ref_data: List[tinker.Datum] = field(default_factory=list)
    """Reference-only datums (no routing matrices)."""
    inf_logprobs: List[List[float]] = field(default_factory=list)
    completion_lens: List[int] = field(default_factory=list)
    """Per-sample completion lengths in tokens."""
    truncated: List[bool] = field(default_factory=list)
    """Per-sample flag: True if completion hit max_completion_tokens."""


def combine_prompt_groups(
    groups: List[PromptGroup],
) -> Tuple[List[tinker.Datum], List[float], List[List[float]], List[int], List[List[float]]]:
    """Flatten a list of PromptGroups into combined arrays for a fwd_bwd call.

    Returns (data, advantages, ref_logprobs, prompt_lens, inf_logprobs).
    """
    data: List[tinker.Datum] = []
    advantages: List[float] = []
    ref_logprobs: List[List[float]] = []
    prompt_lens: List[int] = []
    inf_logprobs: List[List[float]] = []

    for pg in groups:
        data.extend(pg.data)
        advantages.extend(pg.advantages)
        ref_logprobs.extend(pg.ref_logprobs)
        prompt_lens.extend([pg.prompt_len] * len(pg.data))
        inf_logprobs.extend(pg.inf_logprobs)

    return data, advantages, ref_logprobs, prompt_lens, inf_logprobs


LossFnBuilder = Callable[..., Any]
"""Signature for the loss builder returned by ``build_loss_fn``."""


def build_loss_fn(
    policy_loss: str,
    kl_beta: float,
    dapo_config: Any = None,
    gspo_config: Any = None,
    cispo_config: Any = None,
    is_config: ISConfig | None = None,
) -> LossFnBuilder:
    """Create a loss builder that dispatches to grpo/dapo/gspo/cispo.

    Returns a callable:
      (advantages, ref_logprobs, prompt_lens, inf_logprobs, prox_logprobs) -> loss_fn
    """
    if is_config is None:
        is_config = ISConfig()

    from training.utils.rl.dapo import make_dapo_loss_fn
    from training.utils.rl.grpo import make_grpo_loss_fn
    from training.utils.rl.gspo import make_gspo_loss_fn
    from training.utils.rl.cispo import make_cispo_loss_fn

    def build(
        advantages: List[float],
        ref_logprobs: List[List[float]],
        prompt_lens: List[int],
        inf_logprobs: List[List[float]],
        prox_logprobs: List[List[float]],
    ) -> Any:
        if policy_loss == "dapo":
            return make_dapo_loss_fn(
                advantages, ref_logprobs, inf_logprobs,
                prompt_lens, prox_logprobs,
                dapo_config, is_config=is_config,
            )
        if policy_loss == "gspo":
            return make_gspo_loss_fn(
                advantages, ref_logprobs, inf_logprobs,
                prompt_lens, prox_logprobs,
                gspo_config, is_config=is_config,
            )
        if policy_loss == "cispo":
            return make_cispo_loss_fn(
                advantages, ref_logprobs, inf_logprobs,
                prompt_lens, prox_logprobs,
                cispo_config, is_config=is_config,
            )
        if policy_loss == "grpo":
            return make_grpo_loss_fn(
                advantages, ref_logprobs,
                prompt_lens, inf_logprobs=inf_logprobs,
                prox_logprobs=prox_logprobs,
                kl_beta=kl_beta, is_config=is_config,
            )
        raise ValueError(
            f"Unsupported policy_loss '{policy_loss}'. Expected one of: grpo, dapo, gspo, cispo."
        )

    return build
