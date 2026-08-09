"""Mixin: disaggregate multi-turn SFT data into per-user-turn supervised
examples, with non-trainable assistants filtered out.

Background
----------

HuggingFace chat templates for thinking models (Qwen3, Qwen3.5,
DeepSeek-V3 thinking, GLM-5.1, gpt-oss reasoning, Kimi K2/K2.5,
Nemotron3) strip historical ``<think>`` blocks from prior assistant
turns during inference while retaining the model-defined current trajectory
(often every assistant after the last real user query, not merely the final
assistant message). The shipped ``apply_chat_template`` commonly defaults to this
"strip-from-history" behavior. History flags use different polarity across
vendors (for example ``clear_thinking=False`` versus
``preserve_thinking=True``); the semantic capability registry maps them to
concrete renderer variants. ``enable_thinking`` is a separate generation axis
and is not a history-mode switch.

A naive multi-turn ``ALL_ASSISTANT_MESSAGES`` SFT pipeline that renders
the full conversation as one datum and weights every assistant turn
trains the model on the wrong target distribution: historical turns
appear as ``<think></think>{ans}`` (empty think) while the last turn
keeps ``<think>{cot}</think>{ans}``. The model learns to emit empty
thinking 99% of the time. Customer reports of "5/10 missing CoT" in
cookbook-trained thinking models trace back to exactly this.

Approach
--------

Walk the user-message indices and, for each per-user-turn prefix,
render an independent supervised example trained on
``LAST_ASSISTANT_TURN``. Each example's prompt context byte-equals what
HF ``apply_chat_template`` (default args) would produce for the same
prefix, so training tokens stay aligned with what the model sees at
inference. Training cost grows ~N²/2 in conversation length, in
exchange for inference parity.

Non-trainable round filter
~~~~~~~~~~~~~~~~~~~~~~~~~~

If the prefix's terminal assistant turn is marked non-trainable
(``trainable=False`` or ``weight=0``), the entire datum is skipped —
the user explicitly told us not to train this answer, so emitting a
datum that would weight it (under ``LAST_ASSISTANT_TURN``) would
violate the user's intent. The non-trainable assistant remains in the
prefix as context for any *later* trainable round. This matches the
V1 SFT trainer's ``_split_at_thinking_boundaries`` filter (it skips
yielding a round whose terminal assistant has ``weight != 1``).

Per-message weights (``CUSTOMIZED``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A row that carries any ``weight`` / ``trainable`` field is rendered with
``CUSTOMIZED`` so per-message flags are honored. ``CUSTOMIZED`` weights
every trainable message wherever it appears, which on a prefix render
would re-train each earlier assistant turn once per later split — with
its thinking already stripped from history, the exact train/inference
mismatch this split exists to avoid.

Each prefix is therefore reduced to its own terminal turn: every message
before that turn is demoted to context, so an assistant turn is trained
in exactly the one datum where it is the target. The flags that survive
select only messages of the terminal turn, and whenever they select
exactly what a built-in mode selects the prefix renders with that mode
instead, keeping a weighted row byte-identical to the same row without
weights. Flags that mask part of the terminal turn in a way no built-in
mode expresses keep ``CUSTOMIZED``.

Booleanization of ``weight`` follows
``training/utils/supervised.py::_resolve_trainable``: ``trainable``
wins if present, otherwise ``bool(weight)``, otherwise the assistant
is trainable by default. Inlined here to avoid a circular import (the
``utils.supervised`` module already imports from
``training.renderer.*``).

This mirrors the upstream Kimi K2 implementation
(``tinker_cookbook/renderers/kimi_k2.py:335``) and adds the
non-trainable-round filter.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

from tinker_cookbook.renderers.base import TrainOnWhat

from training.renderer.message_weights import (
    equivalent_builtin_train_on_what,
    flags_clamped_to_mode,
    uses_per_message_weights,
    without_trainable_flags,
)


def _is_trainable_assistant(message: Mapping[str, Any]) -> bool:
    """Mirror of ``_resolve_trainable`` for assistant messages, inlined to
    avoid a circular import. ``trainable`` field wins if present;
    otherwise ``bool(weight)``; otherwise True (assistant default)."""
    trainable = message.get("trainable")
    if trainable is not None:
        return bool(trainable)
    weight = message.get("weight")
    if weight is not None:
        return bool(weight)
    return True


def _terminal_assistant(prefix: Sequence[Any]) -> Mapping[str, Any] | None:
    """Walk the prefix backward and return the last assistant message,
    or ``None`` if the prefix has none."""
    for msg in reversed(prefix):
        if isinstance(msg, Mapping) and msg.get("role") == "assistant":
            return msg
    return None


def _terminal_turn_start(prefix: Sequence[Any]) -> int:
    """Index of the first message after the prefix's last user message.

    That message onward is the terminal turn: the one target this datum
    exists to train.
    """
    for idx in range(len(prefix) - 1, -1, -1):
        msg = prefix[idx]
        if isinstance(msg, Mapping) and msg.get("role") == "user":
            return idx + 1
    return 0


def _history_demoted_to_context(prefix: Sequence[Any]) -> list[Any]:
    """Keep per-message flags on the terminal turn, mark history as context."""
    terminal_start = _terminal_turn_start(prefix)
    return [
        msg
        if idx >= terminal_start or not isinstance(msg, Mapping)
        else {**msg, "trainable": False}
        for idx, msg in enumerate(prefix)
    ]


class DisaggregateMultiTurnMixin:
    """Provide a multi-turn-safe ``build_supervised_examples`` for renderers
    that don't satisfy the sequence extension property.

    Mix in BEFORE the upstream renderer class to override the base
    implementation::

        class Qwen3SplitRenderer(DisaggregateMultiTurnMixin, Qwen3Renderer):
            pass
    """

    def build_supervised_examples(
        self,
        messages: list[Any],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_TURN,
    ):
        # Extension fast-path: if the renderer satisfies the sequence
        # extension property (e.g. caller opted into PRESERVED history mode
        # on a renderer that supports it), the singular render is correct
        # and disaggregating would N²-blow up token cost for no win.
        if self.has_extension_property:
            return [
                self.build_supervised_example(messages, train_on_what=train_on_what)
            ]

        # Single-target modes don't need split — the singular path renders
        # the full conversation with weights only on the last segment, so
        # historical strip behavior is irrelevant (no loss flows there).
        if train_on_what in (
            TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            TrainOnWhat.LAST_ASSISTANT_TURN,
        ):
            return [
                self.build_supervised_example(messages, train_on_what=train_on_what)
            ]

        user_message_idxs = [
            idx for idx, message in enumerate(messages) if message["role"] == "user"
        ]

        if train_on_what not in (
            TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            TrainOnWhat.CUSTOMIZED,
        ):
            warnings.warn(
                "Using train_on_what=ALL_MESSAGES/ALL_TOKENS/"
                "ALL_USER_AND_SYSTEM_MESSAGES with a renderer that "
                "does not satisfy the extension property "
                "(has_extension_property=False). The same train_on_what mode "
                "is applied to each per-user-turn split.",
                UserWarning,
                stacklevel=2,
            )

        examples = []
        for next_user_idx in [*user_message_idxs[1:], len(messages)]:
            prefix = messages[:next_user_idx]
            terminal = _terminal_assistant(prefix)
            # Skip rounds whose terminal assistant the user marked
            # non-trainable. A LAST_ASSISTANT_TURN render of this prefix
            # would weight that assistant's tokens and train them anyway,
            # contradicting the user's intent. The non-trainable assistant
            # is preserved in the prefix of LATER trainable rounds as
            # context.
            if terminal is not None and not _is_trainable_assistant(terminal):
                continue
            if train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                mode = TrainOnWhat.LAST_ASSISTANT_TURN
            elif train_on_what == TrainOnWhat.CUSTOMIZED and uses_per_message_weights(
                prefix
            ):
                prefix = _history_demoted_to_context(prefix)
                # A fully trainable terminal turn restates LAST_ASSISTANT_TURN
                # and a terminal turn masked down to its final answer restates
                # LAST_ASSISTANT_MESSAGE; either way the weighted row renders
                # identically to the same row without weights. Only a mask no
                # built-in mode expresses needs per-message weights.
                builtin_mode = equivalent_builtin_train_on_what(
                    prefix,
                    TrainOnWhat.LAST_ASSISTANT_TURN,
                )
                if builtin_mode is None:
                    mode = TrainOnWhat.CUSTOMIZED
                    # Demotion clears the flags BEFORE the terminal turn.
                    # Inside it, a flag on something LAST_ASSISTANT_TURN would
                    # not train — a tool result carrying weight: 1 — must not
                    # pick up loss either.
                    prefix = flags_clamped_to_mode(
                        prefix,
                        TrainOnWhat.LAST_ASSISTANT_TURN,
                    )
                else:
                    mode = builtin_mode
                    prefix = without_trainable_flags(prefix)
            else:
                mode = train_on_what
            examples.append(self.build_supervised_example(prefix, train_on_what=mode))

        return examples
