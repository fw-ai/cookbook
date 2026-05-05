"""Mixin: disaggregate multi-turn SFT data into per-user-turn supervised
examples, with non-trainable assistants filtered out.

Background
----------

HuggingFace chat templates for thinking models (Qwen3, Qwen3.5,
DeepSeek-V3 thinking, GLM-5.1, gpt-oss reasoning, Kimi K2/K2.5,
Nemotron3) strip historical ``<think>`` blocks from prior assistant
turns during inference: only the *last* assistant turn carries its real
reasoning trace. The shipped ``apply_chat_template`` defaults to this
"strip-from-history" behavior — opt-in flags
(``clear_thinking=False``, ``enable_thinking=False``) toggle alternate
paths but the strip path is what every standard inference stack
(vLLM, transformers, Fireworks serverless) feeds the model.

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
        # extension property (e.g. caller opted into preserve-history mode
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

        if train_on_what != TrainOnWhat.ALL_ASSISTANT_MESSAGES:
            warnings.warn(
                "Using train_on_what=ALL_MESSAGES/ALL_TOKENS/"
                "ALL_USER_AND_SYSTEM_MESSAGES/CUSTOMIZED with a renderer that "
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
            mode = (
                TrainOnWhat.LAST_ASSISTANT_TURN
                if train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES
                else train_on_what
            )
            examples.append(
                self.build_supervised_example(prefix, train_on_what=mode)
            )

        return examples
