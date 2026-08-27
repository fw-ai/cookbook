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

Thinking-enabled renderers use :class:`ThinkPrefillWeightsMixin` to move the
two markers onto the correct sides of the loss boundary. Disable-thinking
renderers use :class:`DisableThinkingWeightsMixin` instead: their generation
suffix prefills the whole empty wrapper, so both marker tokens are removed from
the final loss even when an input trajectory contains explicit reasoning.
"""

from __future__ import annotations

import tinker
import torch
from tinker_cookbook.renderers.base import Message, RenderContext, RenderedMessage

_THINK_OPEN = "<think>"
_THINK_OPEN_PREFILL = f"{_THINK_OPEN}\n"
_THINK_CLOSE = "</think>"


class DisableThinkingWeightsMixin:
    """Mask both markers supplied by a disable-thinking generation suffix.

    Official templates retain explicit reasoning in existing assistant
    messages even when generation thinking is disabled. Keep those bytes for
    template parity, but never train either marker because inference supplies
    the complete empty ``<think>...</think>`` wrapper.
    """

    def build_supervised_example(
        self,
        messages: list[Message],
        *args: object,
        **kwargs: object,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        model_input, weights = super().build_supervised_example(
            messages, *args, **kwargs
        )
        token_ids = list(model_input.to_ints())
        masked = weights.clone()
        for marker in (_THINK_OPEN, _THINK_CLOSE):
            marker_tokens = self.tokenizer.encode(marker, add_special_tokens=False)
            if not marker_tokens:
                continue
            for start in range(len(token_ids) - len(marker_tokens) + 1):
                if token_ids[start : start + len(marker_tokens)] == marker_tokens:
                    masked[start : start + len(marker_tokens)] = 0
        return model_input, masked


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
