"""RL loss dispatch and rollout data types."""

from __future__ import annotations

from typing import Any, List, Tuple, Callable
from dataclasses import field, dataclass

import tinker


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
    generation_step: int | None = None
    """Training step when this group's rollout was submitted (async mode)."""


def combine_prompt_groups(
    groups: List[PromptGroup],
) -> Tuple[List[tinker.Datum], List[float], List[List[float]], List[int], List[List[float]], List[int | None]]:
    """Flatten a list of PromptGroups into combined arrays for a fwd_bwd call.

    Returns (data, advantages, ref_logprobs, prompt_lens, inf_logprobs, generation_steps).
    """
    data: List[tinker.Datum] = []
    advantages: List[float] = []
    ref_logprobs: List[List[float]] = []
    prompt_lens: List[int] = []
    inf_logprobs: List[List[float]] = []
    generation_steps: List[int | None] = []

    for pg in groups:
        data.extend(pg.data)
        advantages.extend(pg.advantages)
        ref_logprobs.extend(pg.ref_logprobs)
        prompt_lens.extend([pg.prompt_len] * len(pg.data))
        inf_logprobs.extend(pg.inf_logprobs)
        generation_steps.extend([pg.generation_step] * len(pg.data))

    return data, advantages, ref_logprobs, prompt_lens, inf_logprobs, generation_steps


LossFnBuilder = Callable[..., Any]
"""Signature for the loss builder returned by ``build_loss_fn``."""


def build_loss_fn(
    policy_loss: str,
    kl_beta: float,
    tis_enabled: bool = False,
    tis_config: Any = None,
    dapo_config: Any = None,
    gspo_config: Any = None,
    cispo_config: Any = None,
    decoupled_config: Any = None,
) -> LossFnBuilder:
    """Create a loss builder that dispatches to grpo/dapo/gspo/cispo.

    Returns a callable that accepts (advantages, ref_logprobs, prompt_lens,
    inf_logprobs, generation_steps, current_step) and returns a tinker loss_fn.
    """
    from training.utils.rl.dapo import make_dapo_loss_fn
    from training.utils.rl.grpo import make_grpo_loss_fn
    from training.utils.rl.gspo import make_gspo_loss_fn
    from training.utils.rl.cispo import make_cispo_loss_fn
    from training.utils.rl.importance_sampling import (
        make_tis_weights_fn,
        compute_decoupled_corrections,
    )

    def build(
        advantages: List[float],
        ref_logprobs: List[List[float]],
        prompt_lens: List[int],
        inf_logprobs: List[List[float]],
        generation_steps: List[int | None] | None = None,
        current_step: int | None = None,
    ) -> Any:
        tis_wf = None
        if tis_enabled and tis_config is not None and decoupled_config is None:
            tis_wf = make_tis_weights_fn(inf_logprobs, prompt_lens, tis_config)

        decoupled_fn = None
        if decoupled_config is not None and generation_steps is not None and current_step is not None:
            from training.utils.rl.common import _normalize_prompt_lens

            normalized_prompt_lens = _normalize_prompt_lens(prompt_lens, len(advantages))

            def _decoupled_fn(resp_pi, resp_inf, sample_idx):
                gen_step = generation_steps[sample_idx]
                if gen_step is None:
                    gen_step = current_step
                return compute_decoupled_corrections(
                    resp_pi, resp_inf, gen_step, current_step, decoupled_config,
                )

            decoupled_fn = _decoupled_fn

        if policy_loss == "dapo":
            return make_dapo_loss_fn(
                advantages, ref_logprobs, inf_logprobs,
                prompt_lens, dapo_config, tis_weights_fn=tis_wf,
                decoupled_fn=decoupled_fn,
            )
        if policy_loss == "gspo":
            return make_gspo_loss_fn(
                advantages, ref_logprobs, inf_logprobs,
                prompt_lens, gspo_config, tis_weights_fn=tis_wf,
                decoupled_fn=decoupled_fn,
            )
        if policy_loss == "cispo":
            return make_cispo_loss_fn(
                advantages, ref_logprobs, inf_logprobs,
                prompt_lens, cispo_config, tis_weights_fn=tis_wf,
                decoupled_fn=decoupled_fn,
            )
        if policy_loss == "grpo":
            return make_grpo_loss_fn(
                advantages, ref_logprobs,
                prompt_lens, inf_logprobs=inf_logprobs, kl_beta=kl_beta,
                tis_weights_fn=tis_wf, decoupled_fn=decoupled_fn,
            )
        raise ValueError(
            f"Unsupported policy_loss '{policy_loss}'. Expected one of: grpo, dapo, gspo, cispo."
        )

    return build
