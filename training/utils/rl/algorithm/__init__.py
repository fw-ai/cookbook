"""RL policy objectives, one module per algorithm.

Modules hold formulas and their settings only. :func:`resolve_policy_loss`
validates one objective once at startup; recipes then call the trainer with
the resolved definition and never branch on algorithm names again.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Literal, get_args

from training.utils.rl.algorithm.base import (
    ClientObjective,
    LossExecution,
    PolicyLoss,
    PolicyLossInputs,
)
from training.utils.rl.algorithm.cispo import POLICY_LOSS as CISPO
from training.utils.rl.algorithm.dapo import POLICY_LOSS as DAPO
from training.utils.rl.algorithm.dppo import POLICY_LOSS as DPPO
from training.utils.rl.algorithm.dro import POLICY_LOSS as DRO
from training.utils.rl.algorithm.grpo import POLICY_LOSS as GRPO
from training.utils.rl.algorithm.gspo import POLICY_LOSS as GSPO
from training.utils.rl.algorithm.importance_sampling import (
    POLICY_LOSS as IMPORTANCE_SAMPLING,
)
from training.utils.rl.algorithm.score_centering import POLICY_LOSS as SCORE_CENTERING

__all__ = [
    "ClientObjective",
    "LossExecution",
    "POLICY_LOSSES",
    "PolicyLoss",
    "PolicyLossInputs",
    "ResolvedPolicyLoss",
    "TargetLogprobSupport",
    "resolve_policy_loss",
    "supported_builtin_loss_fns",
]

POLICY_LOSSES: Dict[str, PolicyLoss] = {
    loss.name: loss
    for loss in (
        GRPO,
        GSPO,
        DAPO,
        DRO,
        CISPO,
        DPPO,
        IMPORTANCE_SAMPLING,
        SCORE_CENTERING,
    )
}


TargetLogprobSupport = Literal["full_vocabulary", "sampling_support"]
"""Vocabulary over which the trainer normalizes target-policy logprobs."""


def supported_builtin_loss_fns() -> frozenset[str]:
    """Built-in loss names accepted by the installed SDK request types."""
    import fireworks.training.sdk  # noqa: F401  (registers Fireworks loss names)
    import tinker.types

    return frozenset(get_args(tinker.types.LossFnType))


@dataclass(frozen=True)
class ResolvedPolicyLoss:
    """One validated objective; immutable for the lifetime of a run."""

    definition: PolicyLoss
    options: Any
    execution: LossExecution
    anchor: str
    """``"old_policy"`` or ``"rollout"`` after applying the objective's needs."""
    normalization: Any
    loss_fn_config: Dict[str, Any] | None
    """Built-in trainer settings; ``None`` for client execution."""
    target_logprob_support: TargetLogprobSupport = "full_vocabulary"

    @property
    def sampling_kwargs(self) -> Dict[str, Any]:
        if self.definition.sampling_kwargs is None:
            return {}
        return self.definition.sampling_kwargs(self.options)

    @property
    def builtin(self) -> bool:
        return self.execution == "builtin"

    @property
    def trainer_loss(self) -> str:
        if self.builtin:
            return f"server_{self.definition.builtin_name}"
        return f"client_{self.definition.name}"

    def metadata(self) -> Dict[str, Any]:
        """Stable policy-loss metadata shared by logs and launch manifests."""
        options = asdict(self.options) if is_dataclass(self.options) else {}
        return {
            "trainer_loss": self.trainer_loss,
            "loss_execution": self.execution,
            "policy_loss": self.definition.name,
            "anchor_logp": self.anchor,
            "target_logprob_support": self.target_logprob_support,
            "loss_fn_config": self.loss_fn_config,
            **{
                name: options if name == self.definition.name else None
                for name in POLICY_LOSSES
            },
        }


def resolve_policy_loss(
    name: str,
    *,
    options: Any,
    execution: str,
    anchor: str,
    kl_beta: float,
    grad_accumulation_normalization: Any = None,
    target_logprob_support: str = "full_vocabulary",
) -> ResolvedPolicyLoss:
    """Validate one objective and its execution path before any trainer call."""
    if name not in POLICY_LOSSES:
        raise ValueError(
            f"unsupported policy_loss {name!r}; expected one of {sorted(POLICY_LOSSES)}"
        )
    definition = POLICY_LOSSES[name]
    if execution not in get_args(LossExecution):
        raise ValueError("loss_execution must be 'client' or 'builtin'")
    if anchor not in {"old_policy", "rollout"}:
        raise ValueError("anchor_logp must be 'old_policy' or 'rollout'")
    if target_logprob_support not in get_args(TargetLogprobSupport):
        raise ValueError(
            "target_logprob_support must be 'full_vocabulary' or 'sampling_support'"
        )
    anchor = anchor if definition.uses_anchor else "rollout"
    if definition.validate is not None:
        definition.validate(options)
    if kl_beta != 0 and not definition.allows_kl:
        raise ValueError(f"policy_loss={name!r} requires kl_beta=0.")

    loss_fn_config = None
    if execution == "builtin":
        if definition.builtin_name is None:
            raise ValueError(f"policy_loss={name!r} has no built-in objective")
        if definition.builtin_name not in supported_builtin_loss_fns():
            raise ValueError(
                f"built-in loss {definition.builtin_name!r} is not supported by the "
                "installed SDK; use loss_execution='client' or upgrade the SDK"
            )
        if kl_beta != 0:
            raise ValueError("loss_execution='builtin' requires kl_beta=0.")
        loss_fn_config = definition.loss_fn_config(options)
    if target_logprob_support == "sampling_support":
        if execution != "builtin":
            raise ValueError(
                "target_logprob_support='sampling_support' requires loss_execution='builtin'"
            )
        if anchor != "rollout":
            # A snapshot forward is normalized over the full vocabulary.
            raise ValueError(
                "target_logprob_support='sampling_support' requires anchor_logp='rollout'"
            )
        loss_fn_config = {**loss_fn_config, "target_logprob_support": target_logprob_support}

    normalization = grad_accumulation_normalization
    if definition.normalization is not None:
        requested = getattr(normalization, "value", normalization)
        if requested not in (None, definition.normalization):
            raise ValueError(
                f"policy_loss={name!r} requires grad_accumulation_normalization="
                f"{definition.normalization!r}; got {requested!r}"
            )
        normalization = definition.normalization

    return ResolvedPolicyLoss(
        definition=definition,
        options=options,
        execution=execution,
        anchor=anchor,
        normalization=normalization,
        loss_fn_config=loss_fn_config,
        target_logprob_support=target_logprob_support,
    )
