"""Mixin: disaggregate multi-turn ALL_ASSISTANT_MESSAGES SFT data into
per-user-turn supervised examples.

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

This mirrors the upstream Kimi K2 implementation
(``tinker_cookbook/renderers/kimi_k2.py:335``) and the local GLM5
implementation it was modelled after.
"""

from __future__ import annotations

import warnings
from typing import Any

from tinker_cookbook.renderers.base import TrainOnWhat


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
            mode = (
                TrainOnWhat.LAST_ASSISTANT_TURN
                if train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES
                else train_on_what
            )
            examples.append(
                self.build_supervised_example(prefix, train_on_what=mode)
            )

        return examples
