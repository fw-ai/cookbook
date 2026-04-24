"""Service-agnostic rollout protocol.

The async RL recipe has exactly one extension point -- ``rollout_fn`` --
and everything inside it is user territory.  In practice most users plug
in *some* remote service (an agent framework, a RAG-with-verifier stack,
an LLM-as-judge loop, eval-protocol, ...).  This module defines the
shape that lives between the service adapter and the generic
"tokenize + pack into Rollout" helper in :mod:`training.utils.rl.text_rollout`,
so the integration split is: *service adapter* on one side, *trainer
packing* on the other, with neither importing the other's dependencies.

No external deps.  In particular, this module does **not** import
``eval_protocol`` or any SDK -- pick it up from a plain service class
and the cookbook has no opinion on which one.

Forward-compat contract
-----------------------

A :class:`RolloutPayload` carries either text (with the trainer doing
echo-rescore + template-based loss-mask derivation) or token-native
turn records (``token_ids`` + per-token ``logprobs`` + ``role``).  The
token-native path short-circuits the echo re-score and is the target
shape once upstream services start emitting it; see
``https://github.com/fw-ai/fireworks/issues/23512`` for the EP ask.
Until then, leave ``TurnRecord.token_ids`` / ``logprobs`` as ``None`` and
the text helper will fall back.
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
    """One conversational turn, text- or token-native.

    A service that only has the message text fills ``text`` and leaves
    ``token_ids`` / ``logprobs`` as ``None``; the packer will re-tokenize
    against the current policy and run an ``echo=True`` re-score.

    A service that captured token-level data (e.g. on the server side
    with ``logprobs=True``) fills ``token_ids`` everywhere and
    ``logprobs`` on assistant turns -- the packer trusts it as-is and
    skips the round-trip.
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
    """Identifier for the tokenizer used to produce ``token_ids``.  When
    the packer tokenizes locally it compares against the policy's
    tokenizer; mismatch triggers a warning and a fallback to re-tokenize
    from ``text``."""
    finish_reason: str = "stop"
    extras: dict = field(default_factory=dict)


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
with :func:`training.utils.rl.text_rollout.make_text_rollout_fn`."""
