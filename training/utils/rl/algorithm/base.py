"""Policy-loss definitions: formulas and settings, never trainer calls.

Each objective module owns its typed options, its portable client loss, and
the translation to a trainer built-in loss. Recipes resolve one definition at
startup and keep forward, backward and optimizer calls in the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal

import tinker

LossExecution = Literal["client", "builtin"]


@dataclass(frozen=True)
class PolicyLossInputs:
    """Aligned per-datum rows for one training chunk."""

    data: List[tinker.Datum]
    advantages: List[float]
    ref_logprobs: List[List[float]]
    prompt_lens: List[int]
    rollout_logprobs: List[List[float]]
    """Recorded sampling logprobs aligned to ``target_tokens``."""
    anchor_logprobs: List[List[float]] | None = None
    """Fixed trainer snapshot; ``None`` anchors on ``rollout_logprobs``."""
    raw_inference_logprobs: List[List[float]] | None = None
    """Raw model logprobs for drift diagnostics only."""
    sampler_topk_token_ids: List[List[List[int]]] | None = None
    sampler_topk_logprobs: List[List[List[float]]] | None = None


@dataclass(frozen=True)
class ClientObjective:
    """Datums and a differentiable loss closure for ``forward_backward_custom``."""

    data: List[tinker.Datum]
    loss_fn: Callable
    needs_forward: bool = False
    """``data`` differs from the chunk datums and needs its own forward pass."""


@dataclass(frozen=True)
class PolicyLoss:
    """Static definition of one policy objective."""

    name: str
    client: Callable[[PolicyLossInputs, Any], ClientObjective]
    """``(inputs, options) -> ClientObjective`` for the portable custom-loss path."""
    builtin_name: str | None = None
    """Trainer ``loss_fn``; ``None`` when no built-in objective exists."""
    loss_fn_config: Callable[[Any], Dict[str, float]] | None = None
    """``options -> loss_fn_config`` for the built-in objective."""
    builtin_metrics: Callable[[PolicyLossInputs, list, Any], Dict[str, float]] | None = None
    """``(inputs, trainer_logprobs, options) -> metrics`` after a built-in step."""
    validate: Callable[[Any], None] | None = None
    sampling_kwargs: Callable[[Any], Dict[str, Any]] | None = None
    """``options -> sampler kwargs`` the objective needs recorded at rollout time."""
    normalization: str | None = None
    """Required optimizer normalization; ``None`` uses the recipe setting."""
    uses_anchor: bool = True
    """``False`` always uses rollout probabilities as the denominator."""
    allows_kl: bool = False
    """Whether a reference model (``kl_beta > 0``) may be configured."""


def client_loss_kwargs(inputs: PolicyLossInputs) -> Dict[str, Any]:
    """Keyword arguments shared by the ``make_*_loss_fn`` builders."""
    return dict(
        advantages=inputs.advantages,
        ref_logprobs=inputs.ref_logprobs,
        prompt_len=inputs.prompt_lens,
        inf_logprobs=inputs.rollout_logprobs,
        old_policy_logprobs=inputs.anchor_logprobs,
    )
