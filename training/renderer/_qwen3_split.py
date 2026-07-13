"""Local Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.7 renderers.

Upstream ``tinker_cookbook.renderers.qwen3`` and
``tinker_cookbook.renderers.qwen3_5`` ship without a
``build_supervised_examples`` override. Combined with
``has_extension_property=False`` (the default for thinking-mode
variants), the cookbook SFT dispatcher routes multi-turn
``ALL_ASSISTANT_MESSAGES`` rendering to the upstream base implementation
which raises ``NotImplementedError``.

This module re-registers the upstream renderer names with local
subclasses that mix in ``DisaggregateMultiTurnMixin``. Importing this
module (eagerly via ``training/renderer/__init__.py``) installs the
override before any caller resolves a renderer via ``get_renderer``.
"""

from __future__ import annotations

import dataclasses

import tinker
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    TrainOnWhat,
)
from tinker_cookbook.renderers.qwen3 import (
    Qwen3DisableThinkingRenderer,
    Qwen3Renderer,
    Qwen3VLInstructRenderer,
    Qwen3VLRenderer,
)
from tinker_cookbook.renderers.qwen3_5 import (
    Qwen3_5DisableThinkingRenderer,
    Qwen3_5Renderer,
)

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin


class Qwen3SplitRenderer(DisaggregateMultiTurnMixin, Qwen3Renderer):
    pass


class Qwen3DisableThinkingSplitRenderer(
    DisaggregateMultiTurnMixin, Qwen3DisableThinkingRenderer
):
    pass


class Qwen3VLSplitRenderer(DisaggregateMultiTurnMixin, Qwen3VLRenderer):
    pass


class Qwen3VLInstructSplitRenderer(DisaggregateMultiTurnMixin, Qwen3VLInstructRenderer):
    pass


class Qwen3_5SplitRenderer(DisaggregateMultiTurnMixin, Qwen3_5Renderer):
    pass


class Qwen3_5DisableThinkingSplitRenderer(
    DisaggregateMultiTurnMixin, Qwen3_5DisableThinkingRenderer
):
    pass


# Qwen3.6 ships with the same vocab, special tokens, and tokenizer as
# Qwen3.5. The chat template adds one new opt-in feature:
# `preserve_thinking`. Default invocation (`preserve_thinking`
# undefined or false) produces output byte-identical to Qwen3.5's
# template, so the Qwen3.5 renderer family is correct for the default
# non-interleave-thinking case on Qwen3.6 models.
#
# Three Qwen3.6 renderer variants:
#
#   `qwen3_6`                    — default; alias of Qwen3.5 renderer
#                                    (`strip_thinking_from_history=True`).
#                                    Matches HF apply_chat_template default
#                                    args. Use for normal SFT/DPO/RL where
#                                    historical thinking should be stripped.
#
#   `qwen3_6_preserve_thinking`  — interleave thinking enabled
#                                    (`strip_thinking_from_history=False`).
#                                    Matches HF apply_chat_template with
#                                    `preserve_thinking=true`. Use for:
#                                      * Multi-turn RL training (gets the
#                                        renderer extension property — each
#                                        observation is a prefix extension
#                                        of the previous, no recompute).
#                                      * Long-horizon agent SFT where prior
#                                        reasoning is load-bearing at
#                                        inference time.
#
#   `qwen3_6_disable_thinking`   — empty think block in generation prompt
#                                    (`enable_thinking=false`). Alias of
#                                    Qwen3.5 disable-thinking renderer.
#                                    Use for non-thinking SFT (text2sql,
#                                    structured output, format compliance);
#                                    customers must also pass
#                                    `enable_thinking=false` at inference
#                                    time to keep train↔inference parity.


class Qwen3_6SplitRenderer(Qwen3_5SplitRenderer):
    pass


class Qwen3_6DisableThinkingSplitRenderer(Qwen3_5DisableThinkingSplitRenderer):
    pass


class Qwen3_6PreserveThinkingSplitRenderer(Qwen3_5SplitRenderer):
    """Qwen3.6 with interleave thinking — historical assistant `<think>`
    blocks preserved across turns.

    Forwards `strip_thinking_from_history=False` to the upstream
    `Qwen3_5Renderer` (whose Qwen3 base accepts the kwarg). With
    preserve-thinking on, the renderer satisfies the *extension property*:
    a shorter prefix of a multi-turn conversation tokenizes to a prefix of
    the full conversation, so multi-turn RL avoids prefix recomputation per
    step.

    Token-level output matches HF `apply_chat_template` invoked with
    `preserve_thinking=true`, including the "empty thinking block"
    behavior for historical assistants without reasoning content (see
    `_assistant_header_suffix` override below).
    """

    def __init__(self, tokenizer, image_processor=None):
        super().__init__(
            tokenizer,
            image_processor=image_processor,
            strip_thinking_from_history=False,
        )

    def _assistant_header_suffix(self, message, ctx):
        """Match HF chat template `preserve_thinking=true` byte-for-byte.

        The official Qwen3.6 chat template, when `preserve_thinking=true`
        is set, takes the IF branch unconditionally for every assistant
        turn (historical OR trailing)::

            {{- '<|im_start|>' + role + '\\n<think>\\n'
                + reasoning_content + '\\n</think>\\n\\n'
                + content }}

        For assistants without `reasoning_content`, this emits the
        empty wrapper `<think>\\n\\n</think>\\n\\n` before the visible
        content. Upstream `Qwen3_5Renderer._assistant_header_suffix`
        only emits the wrapper for the trailing generation-prompt slot
        (`ctx.idx > ctx.last_user_index`) — see the `if ctx.idx <=
        ctx.last_user_index: return ""` early exit there. This override
        removes that early exit so the wrapper logic also applies to
        historical assistant turns, matching HF's output for the same
        conversation when `preserve_thinking=true`.

        The body is otherwise identical to upstream's: detect whether
        the assistant message carries a `thinking` part (in which case
        `_preprocess_message_parts` emits the real `<think>...
        </think>` block, so the suffix returns ""), otherwise emit the
        empty wrapper.

        Train-inference parity is the rationale: customers training with
        this renderer and serving the result with the official Qwen3.6
        chat template at inference (vLLM default, etc.) will see the
        same tokens at both stages.

        This is the "empty thinking blocks spam context"
        pattern documented at
        https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates.
        Adopting it here is a deliberate trade-off in favour of
        train-inference parity over token efficiency. Customers serving
        with froggeric or another bug-fixed template at inference would
        see drift; in that case use the default `qwen3_5` renderer
        family instead, which doesn't emit empty wrappers.
        """
        content = message.get("content", "")
        has_think = False
        if isinstance(content, list):
            has_think = any(
                isinstance(p, dict) and p.get("type") == "thinking" for p in content
            )
        elif isinstance(content, str):
            has_think = "<think>" in content
        return "" if has_think else "<think>\n\n</think>\n\n"


class _Qwen3_7TemplateMixin:
    """Qwen3.7 template deltas on top of the Qwen3.6 renderer family.

    Qwen3.7 keeps the Qwen3.6 vocabulary, special tokens, thinking blocks,
    XML tool-call format, parser, and stop token. Its stock template differs
    in three places that the upstream Qwen3.5 renderer does not model:

    * An assistant remains in the active query/tool chain until the next real
      user query, so its thinking stays visible while tool results are added.
    * Consecutive tool results share one ``user`` envelope.
    * Tool declarations retain their OpenAI
      ``{"type":"function","function":...}`` wrapper.

    Everything else deliberately delegates to Qwen3.6.
    """

    @staticmethod
    def _visible_text(message: Message) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )

    def _format_tool_calls_chunks(self, message: Message):
        """Match Qwen3.7's conditional separator before the first tool call."""
        assert "tool_calls" in message, "tool_calls are required to format tool calls"
        separator = "\n\n" if self._visible_text(message).strip() else ""
        return [
            {
                "type": "text",
                "text": separator
                + "\n".join(
                    self._format_tool_call_xml(tool_call)
                    for tool_call in message["tool_calls"]
                ),
            }
        ]

    def _render_tool_response(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        """Render consecutive tool results in one HF-compatible user turn.

        ``RenderContext`` has the previous message but no look-ahead. A tool
        group is opened by its first result, continued without another header,
        and closed either at the conversation tail or by the following
        non-tool message's header.
        """
        first_in_group = not (
            ctx.prev_message is not None and ctx.prev_message.get("role") == "tool"
        )
        header_text = ""
        if first_in_group:
            maybe_newline = "\n" if ctx.idx > 0 else ""
            header_text = f"{maybe_newline}<|im_start|>user"

        content = self._visible_text(message).strip()
        output_text = f"\n<tool_response>\n{content}\n</tool_response>"
        if ctx.is_last:
            output_text += "<|im_end|>"

        header = None
        if header_text:
            header = tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(
                    header_text,
                    add_special_tokens=False,
                )
            )
        return RenderedMessage(
            header=header,
            output=[
                tinker.types.EncodedTextChunk(
                    tokens=self.tokenizer.encode(
                        output_text,
                        add_special_tokens=False,
                    )
                )
            ],
            stop_overlap=(
                tinker.types.EncodedTextChunk(
                    tokens=self.tokenizer.encode("\n", add_special_tokens=False)
                )
                if ctx.is_last
                else None
            ),
        )

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        if message["role"] == "tool":
            return self._render_tool_response(message, ctx)

        # HF's ``last_query_index`` is the last real user query, not the last
        # message. Mark assistants in an active tool chain as current so the
        # inherited renderer retains their thinking in default mode.
        inherited_ctx = ctx
        if message["role"] == "assistant":
            inherited_ctx = dataclasses.replace(
                ctx,
                is_last=ctx.idx > ctx.last_user_index,
            )
        rendered = super().render_message(message, inherited_ctx)

        # A non-tool message closes the preceding tool-result group. The
        # inherited header already starts with the newline after <|im_end|>.
        if ctx.prev_message is not None and ctx.prev_message.get("role") == "tool":
            end_tokens = self.tokenizer.encode(
                "<|im_end|>",
                add_special_tokens=False,
            )
            old_header_tokens = list(rendered.header.tokens) if rendered.header else []
            rendered = RenderedMessage(
                header=tinker.types.EncodedTextChunk(
                    tokens=end_tokens + old_header_tokens
                ),
                output=rendered.output,
                stop_overlap=rendered.stop_overlap,
            )
        if ctx.is_last:
            rendered = RenderedMessage(
                header=rendered.header,
                output=rendered.output,
                stop_overlap=tinker.types.EncodedTextChunk(
                    tokens=self.tokenizer.encode("\n", add_special_tokens=False)
                ),
            )
        return rendered

    def create_conversation_prefix_with_tools(
        self,
        tools,
        system_prompt: str = "",
    ):
        """Keep OpenAI wrappers exactly as Qwen3.7's HF template serializes them."""
        wrapped_tools = [{"type": "function", "function": tool} for tool in tools]
        return super().create_conversation_prefix_with_tools(
            wrapped_tools,
            system_prompt=system_prompt,
        )

    def build_supervised_example(
        self,
        messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ):
        """Add HF's final newline as a masked, template-injected token.

        Generation prompts get this newline from the following assistant
        header. A closed conversation has no following header, while the HF
        template still emits ``<|im_end|>\n``. Keep the extra token out of the
        loss because it is not part of the model's ``<|im_end|>`` stop signal.
        """
        model_input, weights = super().build_supervised_example(
            messages,
            train_on_what=train_on_what,
        )
        newline_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
        if newline_tokens:
            weights[-len(newline_tokens) :] = 0
        return model_input, weights


class Qwen3_7SplitRenderer(_Qwen3_7TemplateMixin, Qwen3_6SplitRenderer):
    pass


class Qwen3_7DisableThinkingSplitRenderer(
    _Qwen3_7TemplateMixin,
    Qwen3_6DisableThinkingSplitRenderer,
):
    pass


class Qwen3_7PreserveThinkingSplitRenderer(
    _Qwen3_7TemplateMixin,
    Qwen3_6PreserveThinkingSplitRenderer,
):
    pass


register_renderer("qwen3", lambda tok, ip=None: Qwen3SplitRenderer(tok))
register_renderer(
    "qwen3_disable_thinking",
    lambda tok, ip=None: Qwen3DisableThinkingSplitRenderer(tok),
)
register_renderer(
    "qwen3_vl",
    lambda tok, ip=None: Qwen3VLSplitRenderer(tok, image_processor=ip),
)
register_renderer(
    "qwen3_vl_instruct",
    lambda tok, ip=None: Qwen3VLInstructSplitRenderer(tok, image_processor=ip),
)
register_renderer(
    "qwen3_5",
    lambda tok, ip=None: Qwen3_5SplitRenderer(tok, image_processor=ip),
)
register_renderer(
    "qwen3_5_disable_thinking",
    lambda tok, ip=None: Qwen3_5DisableThinkingSplitRenderer(tok, image_processor=ip),
)
register_renderer(
    "qwen3_6",
    lambda tok, ip=None: Qwen3_6SplitRenderer(tok, image_processor=ip),
)
register_renderer(
    "qwen3_6_disable_thinking",
    lambda tok, ip=None: Qwen3_6DisableThinkingSplitRenderer(tok, image_processor=ip),
)
register_renderer(
    "qwen3_6_preserve_thinking",
    lambda tok, ip=None: Qwen3_6PreserveThinkingSplitRenderer(tok, image_processor=ip),
)
register_renderer(
    "qwen3_7",
    lambda tok, ip=None: Qwen3_7SplitRenderer(tok, image_processor=ip),
)
register_renderer(
    "qwen3_7_disable_thinking",
    lambda tok, ip=None: Qwen3_7DisableThinkingSplitRenderer(
        tok,
        image_processor=ip,
    ),
)
register_renderer(
    "qwen3_7_preserve_thinking",
    lambda tok, ip=None: Qwen3_7PreserveThinkingSplitRenderer(
        tok,
        image_processor=ip,
    ),
)
