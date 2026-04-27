"""Generic packer: :class:`RolloutService` output -> :class:`Rollout`.

The recipe needs a ``rollout_fn(row, ctx) -> Rollout | None``.  The most
common integration pattern is "some remote service produces completion
data; the trainer accepts token-level data verbatim".  This helper
collapses the common case to one call::

    rollout_fn = make_text_rollout_fn(service)

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
do it; this packer follows the same rule.  Services that today only
have text -- including EP's RemoteRolloutProcessor -- need to grow a
token-native trace before they can drive RL training; see
``https://github.com/fw-ai/fireworks/issues/23512``.

Reward
------

``payload.total_reward`` is authoritative when set ("server wins").
When ``None``, pass ``reward_fn=...`` to grade client-side.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, List, Optional, Union

from training.utils.rl.rollout import Rollout, RolloutSample
from training.utils.rl.rollout_service import (
    RolloutPayload,
    RolloutService,
    RolloutServiceCallable,
    TurnRecord,
)


__all__ = [
    "make_text_rollout_fn",
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
    fn = service.rollout if hasattr(service, "rollout") else service
    return await fn(messages, n=n, sample_kwargs=sample_kwargs, row=row)  # type: ignore[misc]


def make_text_rollout_fn(
    service: ServiceLike,
    *,
    reward_fn: Optional[RewardFn] = None,
    messages_key: str = "messages",
):
    """Build a ``rollout_fn`` that calls ``service`` and packs the result.

    ``service`` returns ``list[RolloutPayload]`` of length
    ``ctx.completions_per_prompt``.  When a payload carries
    ``total_reward=None``, ``reward_fn`` is called; if neither is
    provided the pack fails loud.
    """

    async def rollout_fn(row: dict, ctx) -> Rollout | None:
        messages = row.get(messages_key) or []
        if not messages:
            return None

        try:
            payloads = await _call_service(
                service,
                messages,
                n=ctx.completions_per_prompt,
                sample_kwargs=dict(ctx.sample_kwargs),
                row=row,
            )
        except Exception as exc:
            logger.warning("rollout service failed: %s", exc)
            return None

        if len(payloads) != ctx.completions_per_prompt:
            logger.warning(
                "service returned %d payloads, expected %d",
                len(payloads), ctx.completions_per_prompt,
            )

        version = ctx.current_version()
        samples: List[RolloutSample] = []
        for payload in payloads:
            try:
                sample = await pack_payload_to_sample(
                    payload,
                    ctx=ctx,
                    version=version,
                    reward_fn=reward_fn,
                    row=row,
                )
            except _PackError as exc:
                logger.warning("dropping payload: %s", exc)
                return None
            samples.append(sample)

        if not samples:
            return None
        return Rollout(samples=samples, row_meta={"row_id": row.get("id")})

    return rollout_fn


# ---------------------------------------------------------------------------
# Payload -> RolloutSample
# ---------------------------------------------------------------------------


class _PackError(RuntimeError):
    pass


async def pack_payload_to_sample(
    payload: RolloutPayload,
    *,
    ctx,
    version: int,
    reward_fn: Optional[RewardFn] = None,
    row: Optional[dict] = None,
) -> RolloutSample:
    """Normalise one :class:`RolloutPayload` into one :class:`RolloutSample`.

    Token-native only: every turn must carry ``token_ids``, and every
    assistant turn must carry ``logprobs`` aligned with ``token_ids``.
    Raises :class:`_PackError` with a user-actionable message on any
    structural problem.
    """
    if not payload.turns:
        raise _PackError("payload has no turns")

    missing = [i for i, t in enumerate(payload.turns) if t.token_ids is None]
    if missing:
        raise _PackError(
            f"turns {missing} missing token_ids; this packer is token-native "
            "only.  Have the upstream service emit per-turn token_ids and "
            "per-token logprobs from the same call that generated them "
            "(see slime / AReaL).  Re-tokenizing text post-hoc silently "
            "misaligns the loss mask and inference logprobs.",
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
        versions=[version] * len(tokens),
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
