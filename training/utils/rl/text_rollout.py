"""Generic packer: :class:`RolloutService` output -> :class:`Rollout`.

The recipe needs a ``rollout_fn(row, ctx) -> Rollout | None``.  The most
common integration pattern is "some remote service produces completion
data, the trainer tokenizes and re-scores logprobs".  This helper
collapses the common case to one call::

    rollout_fn = make_text_rollout_fn(service)

Supported shapes
----------------

**Token-native (fast path).**  When every :class:`TurnRecord` in a
payload has ``token_ids`` populated, the packer concatenates the per-turn
tokens, derives ``loss_mask`` from roles (``1`` on assistant, ``0``
elsewhere), trusts the supplied per-token ``logprobs``, and skips the
inference round-trip.  This is the shape to target once upstream
services capture token-level traces; see issue 23512.

**Text-only single-turn (fallback).**  When turns carry text only, the
packer expects **exactly one** assistant turn at the end of the
conversation.  It applies the policy's chat template twice (prompt
only, then prompt+assistant) to recover the completion's token span,
issues one ``echo=True, max_tokens=1`` request to score the completion
under the current policy, and sets ``loss_mask = [0]*prompt_len +
[1]*comp_len``.  Multi-turn text-only is brittle across tokenizer
templates (cf. Qwen3 trailing-newline and GLM boundary quirks) and is
rejected with a clear error -- integrate at the token level instead.

Reward
------

``payload.total_reward`` is authoritative when set ("server wins").
When ``None``, pass ``reward_fn=...`` to grade client-side.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, List, Optional, Union

import requests

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
    provided the reward defaults to ``0.0`` (and the trainer will filter
    the group out if ``dynamic_filter_fn`` rejects zero-variance).
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
                    prompt_messages=messages,
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
    prompt_messages: List[dict],
    ctx,
    version: int,
    reward_fn: Optional[RewardFn] = None,
    row: Optional[dict] = None,
) -> RolloutSample:
    """Normalise one :class:`RolloutPayload` into one :class:`RolloutSample`.

    Token-native when every turn has ``token_ids``; text-only
    single-assistant-turn otherwise.  Raises :class:`_PackError` with a
    user-actionable message on any structural problem.
    """
    if not payload.turns:
        raise _PackError("payload has no turns")

    token_native = all(t.token_ids is not None for t in payload.turns)
    any_token_native = any(t.token_ids is not None for t in payload.turns)
    if any_token_native and not token_native:
        raise _PackError(
            "payload mixes token-native and text-only turns; either provide "
            "token_ids on every turn or none",
        )

    if token_native:
        tokens, logprobs, loss_mask = _pack_token_native(payload)
    else:
        tokens, logprobs, loss_mask = await _pack_text_only(
            payload, prompt_messages=prompt_messages, ctx=ctx,
        )

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


async def _pack_text_only(
    payload: RolloutPayload,
    *,
    prompt_messages: List[dict],
    ctx,
) -> tuple[List[int], List[float], List[int]]:
    assistant_turns = [t for t in payload.turns if t.role == "assistant"]
    if len(assistant_turns) != 1 or payload.turns[-1].role != "assistant":
        raise _PackError(
            "text-only payloads must end with exactly one assistant turn; "
            "multi-turn services must emit token_ids on every turn (see "
            "rollout_service.TurnRecord)",
        )

    completion_text = assistant_turns[0].text
    if not completion_text:
        raise _PackError("assistant turn has empty text")

    tokenizer = ctx.tokenizer
    prompt_ids: List[int] = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True,
    )
    full_ids: List[int] = tokenizer.apply_chat_template(
        [*prompt_messages, {"role": "assistant", "content": completion_text}],
        tokenize=True, add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise _PackError(
            "chat template not prefix-preserving; adding the assistant turn "
            "changed the prompt tokens (tokenizer template bug or a "
            "system-message rewrite)",
        )

    prompt_len = len(prompt_ids)
    comp_len = len(full_ids) - prompt_len
    if comp_len <= 0:
        raise _PackError("assistant turn produced zero new tokens")

    logprobs = await asyncio.to_thread(
        _echo_rescore,
        full_ids,
        inference_url=ctx.inference_url,
        api_key=ctx.api_key,
        model=ctx.model,
    )
    loss_mask = [0] * prompt_len + [1] * comp_len
    return full_ids, logprobs, loss_mask


def _normalize_completions_url(inference_url: str) -> str:
    url = inference_url.rstrip("/")
    for suffix in ("/inference/v1", "/inference"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return f"{url}/inference/v1/completions"


def _echo_rescore(
    tokens: List[int],
    *,
    inference_url: str,
    api_key: str,
    model: str,
) -> List[float]:
    """Score ``tokens`` under the current policy via ``echo=True, max_tokens=1``.

    Returns a length-``len(tokens)`` list with ``0.0`` at position 0 (no
    prior context) and per-token logprobs at positions 1..N.  Called
    off-loop via :func:`asyncio.to_thread` so ``requests`` doesn't block
    the event loop.
    """
    resp = requests.post(
        _normalize_completions_url(inference_url),
        json={
            "model": model,
            "prompt": tokens,
            "echo": True,
            "max_tokens": 1,
            "logprobs": 0,
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    )
    resp.raise_for_status()
    token_lp = resp.json()["choices"][0]["logprobs"]["token_logprobs"]
    out: List[float] = [0.0 if x is None else float(x) for x in token_lp]
    if len(out) < len(tokens):
        out = out + [0.0] * (len(tokens) - len(out))
    return out[: len(tokens)]
