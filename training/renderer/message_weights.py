"""Per-message training-weight helpers shared by the renderers.

The Fireworks SFT schema marks which turns carry loss with a per-message
``weight``; Tinker's schema uses ``trainable``. ``training.utils.supervised``
resolves either into a ``trainable`` flag on every message of the row and
renders with ``TrainOnWhat.CUSTOMIZED``, which weights each message according
to its own flag.

That mode carries one requirement the dataset alone cannot satisfy: every
message the renderer renders must declare ``trainable``. A renderer may
synthesize messages while rendering — an empty system block Nemotron-3 and
MiniMax-M2 always emit, Gemma 4's thinking marker, gpt-oss's reasoning-effort
preamble, Mistral's baked-in default prompt, Kimi's default system message —
and those are created long after the dataset's flags were resolved. They come
from the prompt template rather than the dataset, so they must never carry
loss, and they must say so. :func:`untrained_synthesized_context` is where each
renderer says it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


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
