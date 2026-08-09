"""Per-message training-weight helpers shared by renderers and the render path.

The Fireworks SFT schema marks which turns carry loss with a per-message
``weight``; Tinker's schema uses ``trainable``. ``training.utils.supervised``
resolves either into a ``trainable`` flag on every message of the row and
renders with ``TrainOnWhat.CUSTOMIZED``, which weights each message according
to its own flag.

Two invariants keep that mode honest. Both live here because renderers and the
render dispatcher each need one of them.

Template context is never a target
    A renderer may synthesize messages while rendering — an empty system block
    Nemotron-3 and MiniMax-M2 always emit, Gemma 4's thinking marker, gpt-oss's
    reasoning-effort preamble, Mistral's baked-in default prompt, Kimi's default
    system message — long after the dataset's flags were resolved. They come
    from the prompt template rather than the dataset, so they must never carry
    loss, and ``CUSTOMIZED`` additionally requires them to say so.
    :func:`untrained_synthesized_context` is where each renderer says it.

Flags that restate the default change nothing
    ``weight=1`` restates the default (see ``_resolve_trainable``), so when the
    messages the flags select are exactly the messages some built-in mode
    selects, that mode is equivalent, and rendering with it keeps a weighted row
    byte-identical to the same row without weights — on each renderer's
    well-trodden built-in path rather than its ``CUSTOMIZED`` branch.
    :func:`equivalent_builtin_train_on_what` finds it.

    Which messages a mode selects is the renderer's own decision, and a renderer
    that overrides ``build_supervised_example`` may read a mode more narrowly
    than the base does — Kimi ends its "last assistant turn" at the last
    non-tool-call assistant rather than at the last user message. The reasoning
    here uses the base renderer's rule, which is an upper bound.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tinker_cookbook.renderers.base import TrainOnWhat

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


def flags_clamped_to_mode(
    messages: Sequence[Any],
    train_on_what: TrainOnWhat,
) -> list[Any]:
    """Clear flags on messages ``train_on_what`` does not select.

    A weight can only withhold loss, so a flag on a message the requested mode
    never targets must not pull it back in. Use this for the masks
    :func:`equivalent_builtin_train_on_what` cannot express: they render with
    ``CUSTOMIZED``, which honours every flag verbatim regardless of the mode the
    caller actually asked for.
    """
    targets = _mode_targets(messages, train_on_what)
    return [
        message
        if not isinstance(message, Mapping) or idx in targets
        else {**message, "trainable": False}
        for idx, message in enumerate(messages)
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
    masked inside an otherwise trained turn, say — which needs ``CUSTOMIZED``.
    """
    if not uses_per_message_weights(messages):
        return None
    target = _flagged_targets(messages) & _mode_targets(messages, default_train_on_what)
    for mode in _ASSISTANT_TARGET_MODES:
        if _mode_targets(messages, mode) == target:
            return mode
    return None
