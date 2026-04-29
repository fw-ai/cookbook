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
``https://github.com/fw-ai/fireworks/issues/23756``.

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
    # ``RolloutService.rollout(self, messages, *, n=, sample_kwargs=, row=)``
    # documents kwargs.  ``RolloutServiceCallable`` is typed as
    # ``Callable[[List[dict], int, dict, dict], ...]`` — i.e. positional.
    # Calling a plain callable with kwargs broke any user whose params
    # weren't named exactly ``n`` / ``sample_kwargs`` / ``row``; honor
    # each form's contract.
    if hasattr(service, "rollout"):
        return await service.rollout(  # type: ignore[union-attr]
            messages, n=n, sample_kwargs=sample_kwargs, row=row,
        )
    return await service(messages, n, sample_kwargs, row)  # type: ignore[misc, operator]


def make_text_rollout_fn(
    service: ServiceLike,
    *,
    reward_fn: Optional[RewardFn] = None,
    messages_key: str = "messages",
    allow_empty_messages: bool = False,
):
    """Build a ``rollout_fn`` that calls ``service`` and packs the result.

    ``service`` returns ``list[RolloutPayload]`` of length
    ``ctx.completions_per_prompt``.  When a payload carries
    ``total_reward=None``, ``reward_fn`` is called; if neither is
    provided the pack fails loud.

    Parameters
    ----------
    allow_empty_messages
        When ``False`` (default), ``rollout_fn`` returns ``None`` whenever
        ``row[messages_key]`` is empty or missing.  This is the correct
        guard for services that need a non-empty seed conversation
        (typical chat-style remote rollouts).  When ``True``, the helper
        forwards the empty list through to ``service.rollout`` — useful
        for env-driven domains (e.g. FrozenLake) where the env emits the
        first observation inside the processor and the seed conversation
        is genuinely empty.

    The resulting :class:`Rollout`'s ``row_meta`` carries ``payload_extras``:
    a list of ``payload.extras`` dicts indexed parallel to ``samples`` so
    domain-specific per-payload metadata (e.g. step rewards, IGPO
    bookkeeping) survives the cookbook packing path without each
    consumer having to re-implement the helper.
    """

    async def rollout_fn(row: dict, ctx) -> Rollout | None:
        messages = row.get(messages_key) or []
        if not messages and not allow_empty_messages:
            return None

        # Snapshot the policy version BEFORE awaiting the remote service.
        # In async RL ``weight_sync_fn`` advances ``ctx.current_version()``
        # concurrently with rollouts; reading it after the await would
        # tag a payload sampled at version N as N+1 if a hotload landed
        # mid-call.  That understates rollout staleness and lets overly
        # stale samples bypass ``max_head_offpolicy_versions`` (or get
        # the wrong IS correction).
        version = ctx.current_version()
        # Service exceptions propagate.  ``async_rl_loop`` counts a
        # returned ``None`` as ``sample_fail`` and folds it into
        # ``data_consumed``, so swallowing a service outage / hard
        # integration bug here would persist a resume cursor that
        # silently skips the broken rows on the next run.  If the
        # caller has a transient-error policy, they implement it on
        # the service side (retries, circuit breaker) — this helper
        # surfaces the failure so the run aborts rather than
        # checkpointing rows as consumed at step 0.
        payloads = await _call_service(
            service,
            messages,
            n=ctx.completions_per_prompt,
            sample_kwargs=dict(ctx.sample_kwargs),
            row=row,
        )

        if len(payloads) != ctx.completions_per_prompt:
            logger.warning(
                "service returned %d payloads, expected %d",
                len(payloads), ctx.completions_per_prompt,
            )
        samples: List[RolloutSample] = []
        payload_extras: List[dict] = []
        # Drop only the malformed payload, not the whole rollout group.
        # The helper accepts variable group sizes (the length-mismatch
        # branch above only logs), so a single bad completion from a
        # flaky service should leave the surviving completions trainable
        # — the previous behavior turned every transient ``_PackError``
        # into full-group loss and noticeably reduced throughput on
        # unreliable rollout backends.
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
                continue
            samples.append(sample)
            payload_extras.append(dict(payload.extras))

        if not samples:
            return None
        row_meta: dict = {
            "row_id": row.get("id"),
            "payload_extras": payload_extras,
        }
        return Rollout(samples=samples, row_meta=row_meta)

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
    expected_tokenizer_id = getattr(ctx, "tokenizer_id", None)
    if (
        payload.tokenizer_id
        and expected_tokenizer_id
        and payload.tokenizer_id != expected_tokenizer_id
    ):
        raise _PackError(
            "payload tokenizer_id "
            f"{payload.tokenizer_id!r} does not match policy tokenizer "
            f"{expected_tokenizer_id!r}",
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
    # ``TrajectoryAssembler`` already enforces prefix-equality and per-call
    # token_ids/logprobs alignment, so payloads it emits skip these checks.
    # Hand-built payloads (e.g. from a remote service that builds turns
    # directly) need extra structural validation so a re-tokenized
    # intermediate turn or an empty turn fails at the rollout boundary
    # rather than training on misaligned assistant logprobs.
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
        # The trainer needs a terminal assistant span to train on.  An
        # assembler-emitted payload always ends with an assistant turn
        # (its ``to_payload`` enforces this); for hand-built payloads we
        # check it here so a service that accidentally drops the final
        # assistant turn fails fast.
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
