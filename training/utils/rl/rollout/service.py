"""Service-agnostic rollout protocol.

The async RL recipe has exactly one extension point -- ``rollout_fn`` --
and everything inside it is user territory.  In practice most users plug
in *some* remote service (an agent framework, a RAG-with-verifier stack,
an LLM-as-judge loop, eval-protocol, ...).  This module defines the
shape that lives between the service adapter and the generic packer in
:mod:`training.utils.rl.rollout`, so the integration split is:
*service adapter* on one side, *trainer packing* on the other, with
neither importing the other's dependencies.

No external deps.  In particular, this module does **not** import
``eval_protocol`` or any SDK -- pick it up from a plain service class
and the cookbook has no opinion on which one.

Token-native contract
---------------------

A :class:`RolloutPayload` is **token-native**: every :class:`TurnRecord`
carries ``token_ids`` and every assistant turn carries per-token
``logprobs``, both straight from the same inference call that generated
them (slime/AReaL convention).  The trainer never re-tokenizes text;
re-tokenization silently misaligns the loss mask and inference
logprobs.  Services that today only emit text (e.g. EP's
``RemoteRolloutProcessor``) must grow a token-native trace before they
can drive RL training; see
``https://github.com/fw-ai/fireworks/issues/23756``.

Use :class:`training.utils.rl.rollout.TrajectoryAssembler`
to build :class:`RolloutPayload` from multi-turn engine calls.  It
carries the AReaL prefix-equality invariant for free and sets
``_assembled=True`` so the packer skips its defensive check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Literal, Optional, Protocol


__all__ = [
    "TurnRecord",
    "RolloutPayload",
    "RolloutService",
]


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class TurnRecord:
    """One conversational turn, token-native.

    ``token_ids`` is required on every turn.  ``logprobs`` is required
    on assistant turns and must align 1:1 with ``token_ids``; both must
    come from the same inference call that generated them.  ``text`` is
    optional and used only for human-readable logging.
    """

    role: Role
    text: str = ""
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None
    finish_reason: Optional[str] = None


@dataclass
class RolloutPayload:
    """One completion's worth of rollout data, service-emitted.

    ``total_reward`` carries the "server wins" convention: when set, the
    trainer trusts it; when ``None``, the packer expects the caller to
    attach a reward (e.g. by grading downstream).  This deprecates
    in-band reward sentinels.
    """

    turns: List[TurnRecord]
    total_reward: Optional[float] = None
    tokenizer_id: Optional[str] = None
    """Identifier for the tokenizer that produced ``token_ids``.  When
    set, the packer asserts it matches the trainer's tokenizer so a
    mismatched-vocab integration fails loud instead of training on the
    wrong token IDs."""
    finish_reason: str = "stop"
    extras: dict = field(default_factory=dict)
    _assembled: bool = False
    """Set by :class:`TrajectoryAssembler.to_payload`.  Tells the packer
    that the per-turn ``token_ids`` were stitched with the prefix-equality
    invariant already enforced, so the defensive consistency check can be
    skipped.  Hand-built payloads default to ``False`` and get the check."""


class RolloutService(Protocol):
    """What the cookbook needs from any rollout backend.

    Given a dataset row's prompt messages, return ``n`` completed
    rollouts.  Everything else -- multi-turn loops, tool calls,
    retries, grading -- is the service's business.
    """

    async def rollout(
        self,
        messages: List[dict],
        *,
        n: int,
        sample_kwargs: dict[str, Any],
        row: dict,
    ) -> List[RolloutPayload]: ...


RolloutServiceCallable = Callable[
    [List[dict], int, dict[str, Any], dict],
    Awaitable[List[RolloutPayload]],
]
"""Plain-function form of :class:`RolloutService`.  Either shape works
with :func:`training.utils.rl.rollout.make_remote_rollout_fn`."""
