"""Loss placement at the think-block boundary of a prefilled assistant turn.

A marker carries loss only when the model has to produce it. The generation
suffix prefills ``<think>\\n``, so at inference the model starts *inside* the
think block and never emits the opening tag; ``</think>`` is not prefilled, so
stopping the think phase is the model's job and has to train. Upstream
``Qwen3_5Renderer`` splits the assistant turn the other way around on both
counts, and ``Nemotron3Renderer`` inherits that split:

- with reasoning content, the rendered output opens with ``<think>\\n``, which
  puts the prefilled tag in the trainable span;
- without it, the empty ``<think>...</think>`` wrapper is emitted as the header
  suffix, which masks the closing tag the model still has to generate.

This mixin re-splits the rendered chunks to move that boundary. Token ids and
their order never change, so chat-template parity is untouched — only which
side of the header/output line each marker falls on.

Apply it to thinking-enabled renderers only. Disable-thinking variants prefill
the whole ``<think></think>`` wrapper in the generation suffix, so masking both
markers is already correct there; DeepSeek V4's chat mode masks both for the
same reason.
"""

from __future__ import annotations

import tinker
from tinker_cookbook.renderers.base import Message, RenderContext, RenderedMessage

_THINK_OPEN_PREFILL = "<think>\n"
_THINK_CLOSE = "</think>"


class ThinkPrefillWeightsMixin:
    """Mask the generation-prefilled ``<think>\\n`` but train ``</think>``."""

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        rendered = super().render_message(message, ctx)
        if message.get("role") != "assistant":
            return rendered

        header_tokens = list(rendered.header.tokens) if rendered.header else []
        close_tokens = self.tokenizer.encode(_THINK_CLOSE, add_special_tokens=False)
        if len(close_tokens) == 1 and close_tokens[0] in header_tokens:
            split = header_tokens.index(close_tokens[0])
            rendered = RenderedMessage(
                header=tinker.EncodedTextChunk(tokens=header_tokens[:split]),
                output=[
                    tinker.EncodedTextChunk(tokens=header_tokens[split:]),
                    *rendered.output,
                ],
                stop_overlap=rendered.stop_overlap,
            )
            header_tokens = header_tokens[:split]

        prefill = self.tokenizer.encode(
            _THINK_OPEN_PREFILL, add_special_tokens=False
        )
        if rendered.output and isinstance(rendered.output[0], tinker.EncodedTextChunk):
            output_tokens = list(rendered.output[0].tokens)
            if output_tokens[: len(prefill)] == prefill:
                rendered = RenderedMessage(
                    header=tinker.EncodedTextChunk(tokens=header_tokens + prefill),
                    output=[
                        tinker.EncodedTextChunk(tokens=output_tokens[len(prefill) :]),
                        *rendered.output[1:],
                    ],
                    stop_overlap=rendered.stop_overlap,
                )
        return rendered
