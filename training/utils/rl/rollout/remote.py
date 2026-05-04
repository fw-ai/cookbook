"""Generic packer: :class:`RolloutService` output -> :class:`RolloutSample`.

The recipe needs ``rollout_fn(sample_prompt) -> RolloutSample | None``,
called once per sample (each dataset row fans out to
``completions_per_prompt`` calls).  The most common integration pattern
is "some remote service produces completion data; the trainer accepts
token-level data verbatim".  This helper collapses the common case to
one call::

    rollout_fn = make_remote_rollout_fn(service, sample_kwargs=...)

Token-native only
-----------------

Every :class:`TurnRecord` MUST carry ``token_ids``, and assistant
turns MUST carry ``logprobs`` aligned with ``token_ids``.  The packer
concatenates the per-turn tokens, derives ``loss_mask`` from roles
(``1`` on assistant, ``0`` elsewhere), and trusts the supplied
per-token ``logprobs`` as-is.

Re-tokenizing assistant text after the fact silently breaks two
things: (a) the loss mask drifts off the BPE boundary the engine
actually generated, and (b) the per-token logprobs no longer align
with the tokens fed to the trainer.  AReaL and slime both refuse to
do it; this packer follows the same rule.

Reward
------

``payload.total_reward`` is authoritative when set ("server wins").
When ``None``, pass ``reward_fn=...`` to grade client-side.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, List, Optional, Union

from training.utils.rl.rollout.types import RolloutSample
from training.utils.rl.rollout.service import (
    RolloutPayload,
    RolloutService,
    RolloutServiceCallable,
)


__all__ = [
    "make_remote_rollout_fn",
    "pack_payload_to_sample",
]


logger = logging.getLogger(__name__)


ServiceLike = Union[RolloutService, RolloutServiceCallable]
RewardFn = Callable[[dict, RolloutPayload], Awaitable[float]]


async def _call_service(
    service: ServiceLike,
    messages: List[dict],
    *,
    n: int,
    sample_kwargs: dict[str, Any],
    row: dict,
) -> List[RolloutPayload]:
    if hasattr(service, "rollout"):
        return await service.rollout(  # type: ignore[union-attr]
            messages, n=n, sample_kwargs=sample_kwargs, row=row,
        )
    return await service(messages, n, sample_kwargs, row)  # type: ignore[misc, operator]


def make_remote_rollout_fn(
    service: ServiceLike,
    *,
    sample_kwargs: dict[str, Any] | None = None,
    tokenizer_id: str | None = None,
    reward_fn: Optional[RewardFn] = None,
    messages_key: str = "messages",
    allow_empty_messages: bool = False,
):
    """Build a per-sample ``rollout_fn`` that calls ``service`` once per
    sample and packs the resulting payload.

    The framework fans each row out to ``completions_per_prompt`` parallel
    samples; this helper requests ``n=1`` from the service per call.  When
    a payload carries ``total_reward=None``, ``reward_fn`` grades it
    client-side.

    Parameters
    ----------
    service
        The :class:`RolloutService` (or compatible callable) producing
        :class:`RolloutPayload` lists.
    sample_kwargs
        Forwarded to ``service.rollout`` per call.  Typically the
        ``RolloutSetup.sample_kwargs`` passed by the recipe.
    tokenizer_id
        Optional policy tokenizer identifier.  When set, payloads with
        a mismatched ``tokenizer_id`` raise rather than silently train
        on misaligned token IDs.
    allow_empty_messages
        When ``False`` (default), the rollout returns ``None`` whenever
        ``sample_prompt[messages_key]`` is empty or missing -- the right
        guard for chat-style services.  When ``True``, empty messages
        are forwarded through (e.g. env-driven domains where the env
        emits the seed observation inside the service).
    """

    sk = dict(sample_kwargs or {})

    async def rollout_fn(sample_prompt: dict) -> RolloutSample | None:
        messages = sample_prompt.get(messages_key) or []
        if not messages and not allow_empty_messages:
            return None

        # The service-side keyword stays ``row=`` (user-supplied services
        # and reward fns expect that name); only the closure param renames.
        payloads = await _call_service(
            service,
            messages,
            n=1,
            sample_kwargs=dict(sk),
            row=sample_prompt,
        )

        if not payloads:
            return None
        if len(payloads) > 1:
            logger.warning(
                "service returned %d payloads for n=1; using the first",
                len(payloads),
            )

        try:
            return await pack_payload_to_sample(
                payloads[0],
                tokenizer_id=tokenizer_id,
                reward_fn=reward_fn,
                row=sample_prompt,
            )
        except _PackError as exc:
            logger.warning("dropping payload: %s", exc)
            return None

    return rollout_fn


# ---------------------------------------------------------------------------
# Payload -> RolloutSample
# ---------------------------------------------------------------------------


class _PackError(RuntimeError):
    pass


async def pack_payload_to_sample(
    payload: RolloutPayload,
    *,
    tokenizer_id: str | None = None,
    reward_fn: Optional[RewardFn] = None,
    row: Optional[dict] = None,
) -> RolloutSample:
    """Normalise one :class:`RolloutPayload` into one :class:`RolloutSample`.

    Token-native only: every turn must carry ``token_ids``, and every
    assistant turn must carry ``logprobs`` aligned with ``token_ids``.
    Raises :class:`_PackError` with a user-actionable message on any
    structural problem.

    ``tokenizer_id`` is the policy tokenizer identifier; when both it
    and ``payload.tokenizer_id`` are set, they must match.
    """
    if not payload.turns:
        raise _PackError("payload has no turns")
    if (
        payload.tokenizer_id
        and tokenizer_id
        and payload.tokenizer_id != tokenizer_id
    ):
        raise _PackError(
            "payload tokenizer_id "
            f"{payload.tokenizer_id!r} does not match policy tokenizer "
            f"{tokenizer_id!r}",
        )

    missing = [i for i, t in enumerate(payload.turns) if t.token_ids is None]
    if missing:
        raise _PackError(
            f"turns {missing} missing token_ids; this packer is token-native "
            "only.  Have the upstream service emit per-turn token_ids and "
            "per-token logprobs from the same call that generated them "
            "(see slime / AReaL).  Re-tokenizing text post-hoc silently "
            "misaligns the loss mask and inference logprobs.",
        )

    # Defensive checks for hand-built payloads (``_assembled=False``).
    if not getattr(payload, "_assembled", False):
        empty_turns = [i for i, t in enumerate(payload.turns) if not t.token_ids]
        if empty_turns:
            raise _PackError(
                f"hand-built payload has empty turn(s) at indices {empty_turns}; "
                "every turn must carry at least one token id.  An empty "
                "intermediate turn typically means the service mis-rendered "
                "(re-tokenized) a turn and emitted a stale or empty span.  "
                "If this is intentional, drop the turn from the payload "
                "instead of leaving an empty token_ids list.",
            )
        if payload.turns[-1].role != "assistant":
            raise _PackError(
                "hand-built payload must end with an assistant turn (got "
                f"role={payload.turns[-1].role!r}).  The trainer's loss is "
                "computed over the final assistant span; a non-assistant "
                "tail typically means a gap turn was appended after the "
                "last engine call by mistake.",
            )

    tokens, logprobs, loss_mask = _pack_token_native(payload)

    reward = payload.total_reward
    if reward is None:
        if reward_fn is None:
            raise _PackError(
                "payload.total_reward is None and no reward_fn was provided",
            )
        reward = float(await reward_fn(row or {}, payload))

    last_assistant = next(
        (t for t in reversed(payload.turns) if t.role == "assistant"), None,
    )
    text = last_assistant.text if last_assistant else ""
    finish_reason = (
        (last_assistant.finish_reason if last_assistant else None)
        or payload.finish_reason
        or "stop"
    )

    return RolloutSample(
        tokens=tokens,
        logprobs=logprobs,
        loss_mask=loss_mask,
        reward=float(reward),
        finish_reason=finish_reason,
        text=text,
    )


def _pack_token_native(
    payload: RolloutPayload,
) -> tuple[List[int], List[float], List[int]]:
    tokens: List[int] = []
    logprobs: List[float] = []
    loss_mask: List[int] = []
    for t in payload.turns:
        assert t.token_ids is not None  # checked by caller
        n = len(t.token_ids)
        if t.role == "assistant":
            if t.logprobs is None or len(t.logprobs) != n:
                raise _PackError(
                    f"assistant turn needs per-token logprobs aligned with "
                    f"token_ids (got {0 if t.logprobs is None else len(t.logprobs)} "
                    f"for {n} tokens)",
                )
            lp = [float(x) for x in t.logprobs]
            mask = [1] * n
        else:
            lp = [0.0] * n
            mask = [0] * n
        tokens.extend(t.token_ids)
        logprobs.extend(lp)
        loss_mask.extend(mask)

    if not any(m > 0 for m in loss_mask):
        raise _PackError("no assistant tokens in payload; nothing to train on")
    if len(tokens) < 2:
        raise _PackError("payload shorter than 2 tokens")
    return tokens, logprobs, loss_mask
