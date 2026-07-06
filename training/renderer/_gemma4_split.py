"""Local Gemma 4 renderers with multi-turn SFT disaggregate support.

Reasoning emission is history-gated (only after the final user in the
rendered prefix). ``gemma4`` matches official jinja (``tool_calls`` required);
``gemma4_thinking`` also supervises plain ``reasoning_content`` turns.
A naive full-transcript render therefore drops reasoning for earlier assistant
tool turns while still weighting their tokens — the same failure mode Qwen3/GLM5
hit before ``DisaggregateMultiTurnMixin``.

Re-register ``gemma4`` / ``gemma4_thinking`` with split subclasses that mix
in ``DisaggregateMultiTurnMixin`` so multi-turn SFT disaggregates per user
turn. Each prefix byte-equals HF ``apply_chat_template`` for that prefix, so
the terminal assistant in each datum keeps its thought channel.
"""

from __future__ import annotations

from tinker_cookbook.renderers import register_renderer

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin
from training.renderer.gemma4 import Gemma4Renderer


class Gemma4SplitRenderer(DisaggregateMultiTurnMixin, Gemma4Renderer):
    @property
    def has_extension_property(self) -> bool:
        return False


class Gemma4ThinkingSplitRenderer(DisaggregateMultiTurnMixin, Gemma4Renderer):
    def __init__(self, tokenizer, image_processor=None):
        del image_processor
        super().__init__(tokenizer, enable_thinking=True)

    @property
    def has_extension_property(self) -> bool:
        return False


register_renderer("gemma4", lambda tok, ip=None: Gemma4SplitRenderer(tok))
register_renderer(
    "gemma4_thinking",
    lambda tok, ip=None: Gemma4ThinkingSplitRenderer(tok),
)
