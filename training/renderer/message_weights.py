"""Per-message training-weight helpers shared by renderers and the render path.

The Fireworks SFT schema marks which turns carry loss with a per-message
``weight``; Tinker's schema uses ``trainable``. ``training.utils.supervised``
resolves either into a ``trainable`` flag on every message of the row and
renders with ``TrainOnWhat.CUSTOMIZED``, which weights each message according
to its own flag.

Two invariants keep that mode honest. Both live here because renderers and the
render dispatcher each need one of them.

Template context is never a target
    A renderer may synthesize messages while rendering: an empty system block
    Nemotron-3 and MiniMax-M2 always emit, Gemma 4's thinking marker, gpt-oss's
    reasoning-effort preamble, Mistral's baked-in default prompt. Those come
    from the prompt template rather than the dataset, so they must never carry
    loss. ``CUSTOMIZED`` also requires every rendered message to declare
    ``trainable``, so a synthesized message without the flag fails the render
    outright. :func:`untrained_synthesized_context` supplies it.

Weights only ever remove loss
    ``weight=1`` restates the default (see ``_resolve_trainable``), so the loss
    a weighted row carries is the loss the unweighted row carries intersected
    with the flags. :func:`render_masked_example` computes that intersection
    directly. :func:`equivalent_builtin_train_on_what` finds the cases where a
    single built-in mode already selects it, which keeps the common weighted row
    on the renderer's well-trodden built-in path and off ``CUSTOMIZED``.

    "The loss the unweighted row carries" is the renderer's own decision, and a
    renderer that overrides ``build_supervised_example`` may read a mode more
    narrowly than the base does — Kimi ends its "last assistant turn" at the
    last non-tool-call assistant rather than at the last user message. So the
    message-level reasoning here uses the base renderer's rule, and the
    authoritative narrowing is the token-level intersection in
    :func:`render_masked_example`.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import tinker
import torch
from training._vendor.tinker_cookbook_0_4_3.renderers.base import TrainOnWhat

# Modes whose target set is some subset of the assistant messages, ordered so
# that the mode an unweighted row would have used wins when several of them
# select the same messages (the render path maps ALL_ASSISTANT_MESSAGES to
# LAST_ASSISTANT_TURN per unrolled turn).
_ASSISTANT_TARGET_MODES = (
    TrainOnWhat.LAST_ASSISTANT_TURN,
    TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    TrainOnWhat.ALL_ASSISTANT_MESSAGES,
)


def _role(message: Any) -> str | None:
    if isinstance(message, Mapping):
        role = message.get("role")
        if isinstance(role, str):
            return role
    return None


def uses_per_message_weights(messages: Sequence[Any]) -> bool:
    """Whether this conversation carries resolved per-message weights."""
    return any(
        isinstance(message, Mapping) and "trainable" in message for message in messages
    )


def untrained_synthesized_context(messages: Sequence[Any]) -> list[Any]:
    """Declare any message that carries no ``trainable`` flag untrained.

    Call this wherever a renderer synthesizes template messages, on the list it
    is about to render. Every message of the dataset row already arrives with a
    resolved flag, so the ones missing it are exactly the ones the renderer just
    added, and template context carries no loss.

    A conversation that uses no per-message weights is returned untouched:
    renderers reject a ``trainable`` field outside ``CUSTOMIZED``.
    """
    if not uses_per_message_weights(messages):
        return list(messages)
    return [
        message
        if not isinstance(message, Mapping) or "trainable" in message
        else {**message, "trainable": False}
        for message in messages
    ]


def without_trainable_flags(messages: Sequence[Any]) -> list[Any]:
    """Drop ``trainable`` so a non-``CUSTOMIZED`` mode accepts the conversation."""
    return [
        {key: value for key, value in message.items() if key != "trainable"}
        if isinstance(message, Mapping) and "trainable" in message
        else message
        for message in messages
    ]


def _mode_targets(messages: Sequence[Any], mode: TrainOnWhat) -> frozenset[int]:
    """Indices ``mode`` assigns loss to under the base ``Renderer``'s rule.

    A renderer that overrides ``build_supervised_example`` may narrow a mode
    further, so treat this as an upper bound rather than the last word.
    """
    last_user_idx = max(
        (idx for idx, message in enumerate(messages) if _role(message) == "user"),
        default=-1,
    )
    last_idx = len(messages) - 1
    targets = set()
    for idx, message in enumerate(messages):
        role = _role(message)
        match mode:
            case TrainOnWhat.LAST_ASSISTANT_MESSAGE:
                selected = role == "assistant" and idx == last_idx
            case TrainOnWhat.LAST_ASSISTANT_TURN:
                selected = role == "assistant" and idx > last_user_idx
            case TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                selected = role == "assistant"
            case TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES:
                selected = role in ("user", "system")
            case TrainOnWhat.ALL_MESSAGES | TrainOnWhat.ALL_TOKENS:
                selected = True
            case TrainOnWhat.CUSTOMIZED:
                selected = bool(
                    isinstance(message, Mapping) and message.get("trainable", False)
                )
            case _:
                raise ValueError(f"Unknown train_on_what: {mode}")
        if selected:
            targets.add(idx)
    return frozenset(targets)


def stable_chunk_sentinel(chunk: Any) -> int:
    """A negative pseudo token id standing for one non-text chunk.

    Derived from the chunk's content, never from its position, so the same image
    keeps the same id however the surrounding text happens to be chunked.
    """
    if isinstance(chunk, tinker.types.ImageAssetPointerChunk):
        payload = (
            f"{chunk.type}:{chunk.location}:{chunk.format}:{chunk.expected_tokens}"
        ).encode()
    elif isinstance(chunk, tinker.types.ImageChunk):
        payload = b"|".join(
            [
                chunk.type.encode(),
                chunk.format.encode(),
                str(chunk.expected_tokens).encode(),
                bytes(chunk.data),
            ]
        )
    else:  # pragma: no cover - defensive branch for future chunk types
        payload = repr(chunk).encode()

    digest = hashlib.sha1(payload).digest()
    return -(int.from_bytes(digest[:8], "big") + 1)


def _rendered_positions(rendered_input: Any) -> list[int]:
    """Per-position identity of a rendered sequence.

    Flattened deliberately: chunk boundaries are not part of the identity,
    because a renderer may split one chunk in two to weight part of it
    differently (GLM masks the injected ``<think>`` token that opens a trainable
    assistant turn) without moving a single token. That is also why a non-text
    chunk is identified by its content rather than by where it sits in the chunk
    list — splitting a text chunk elsewhere must not make an image look
    different.

    Accepts a bare token sequence as well, matching what the render path already
    tolerates from a renderer that returns token ids instead of a ``ModelInput``.
    """
    chunks = getattr(rendered_input, "chunks", None)
    if chunks is None:
        tokens = (
            rendered_input.tolist()
            if hasattr(rendered_input, "tolist")
            else rendered_input
        )
        return [int(token) for token in tokens]

    positions: list[int] = []
    for chunk in chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            positions.extend(int(token) for token in chunk.tokens)
        else:
            positions.extend([stable_chunk_sentinel(chunk)] * int(chunk.length))
    return positions


def _flagged_targets(messages: Sequence[Any]) -> frozenset[int]:
    return frozenset(
        idx
        for idx, message in enumerate(messages)
        if isinstance(message, Mapping) and bool(message.get("trainable", False))
    )


def equivalent_builtin_train_on_what(
    messages: Sequence[Any],
    default_train_on_what: TrainOnWhat,
) -> TrainOnWhat | None:
    """A built-in mode that selects exactly the loss a weighted row should carry.

    The target is the flagged messages that ``default_train_on_what`` — the mode
    the row would render with if it carried no weights — also selects. Matching
    against that intersection rather than against the flags alone is what keeps
    a mask from widening loss: flags that select an assistant turn the requested
    mode leaves alone must not pull it back in.

    Returns ``None`` when no built-in mode selects that set — a middle tool call
    masked inside an otherwise trained turn, say — which needs ``CUSTOMIZED``
    and :func:`render_masked_example`.
    """
    if not uses_per_message_weights(messages):
        return None
    target = _flagged_targets(messages) & _mode_targets(messages, default_train_on_what)
    for mode in _ASSISTANT_TARGET_MODES:
        if _mode_targets(messages, mode) == target:
            return mode
    return None


def render_masked_example(
    renderer: Any,
    messages: Sequence[Any],
    *,
    default_train_on_what: TrainOnWhat,
) -> tuple[Any, torch.Tensor]:
    """Render per-message weights as a mask over the renderer's default loss.

    Use this for the masks :func:`equivalent_builtin_train_on_what` cannot
    express. ``CUSTOMIZED`` asks each renderer to weight messages by their own
    flag, which for a renderer whose built-in modes are narrower than "every
    assistant after the last user" can weight MORE than the unweighted row
    would: Kimi treats back-to-back assistants as separate turns, so masking the
    middle one would otherwise start training the first. Intersecting with the
    unweighted render keeps ``weight`` a mask.

    ``train_on_what`` selects weights and never changes the rendered tokens, so
    the two weight vectors are positionally aligned. That is checked rather than
    assumed: a renderer that carried per-render state on ``self`` instead of on
    the messages would silently misalign the two, which would corrupt loss
    placement rather than fail. This costs a second render, which is why the
    equivalent-mode path handles the common masks instead.
    """
    flagged_input, flagged_weights = renderer.build_supervised_example(
        list(messages),
        train_on_what=TrainOnWhat.CUSTOMIZED,
    )
    default_input, default_weights = renderer.build_supervised_example(
        without_trainable_flags(messages),
        train_on_what=default_train_on_what,
    )
    if _rendered_positions(flagged_input) != _rendered_positions(default_input):
        raise RuntimeError(
            f"{type(renderer).__name__} rendered different tokens under CUSTOMIZED and "
            f"under {default_train_on_what.value}; rendering the same conversation must "
            "not depend on the loss mode or on renderer state left by an earlier render"
        )
    return flagged_input, torch.minimum(flagged_weights, default_weights)
