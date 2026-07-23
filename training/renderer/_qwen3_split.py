"""Local Qwen3 / Qwen3.5 / Qwen3.6 renderers with multi-turn SFT disaggregate support.

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

import json
from dataclasses import replace
from typing import Any, Mapping, cast

from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    TextPart,
    ToolCall,
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


_PREWRAPPED_TOOL_RESPONSE = "_qwen_prewrapped_tool_response"


def _text_content_for_boundary(content: Any) -> str | None:
    """Return text used by the official template's last-query scan.

    The template runs ``render_content(..., false)|trim`` before deciding
    whether a user-role message is actually a pre-wrapped tool response.  A
    string or a multipart message containing only text has the same unambiguous
    representation here.  Mixed image/video content is necessarily a real user
    query, so return ``None`` rather than attempting to emulate vision tokens.
    """

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, Mapping) or part.get("type") != "text":
            return None
        text_parts.append(str(part.get("text", "")))
    return "".join(text_parts)


def _is_prewrapped_tool_response(message: Mapping[str, Any]) -> bool:
    """Match Qwen3.5/3.6's definition of a non-query user message."""

    if message.get("role") != "user":
        return False
    content = _text_content_for_boundary(message.get("content"))
    if content is None:
        return False
    content = content.strip()
    return content.startswith("<tool_response>") and content.endswith(
        "</tool_response>"
    )


def _mark_prewrapped_tool_responses(messages: list[Message]) -> list[Message]:
    """Exclude pre-wrapped tool responses from user-boundary calculations.

    Tinker's base renderer calculates ``last_user_index`` and the local
    disaggregation mixin calculates split points from the raw ``role``.  The
    official Qwen templates instead ignore a user-role message whose complete
    content is a ``<tool_response>...</tool_response>`` wrapper.  Mark a copied
    message as a tool for those calculations, then restore its original user
    role in the Qwen3.5 template-parity mixin so its already-wrapped bytes are
    emitted unchanged.
    """

    marked_messages: list[Message] = []
    for message in messages:
        if not _is_prewrapped_tool_response(message):
            marked_messages.append(message)
            continue
        marked = dict(message)
        marked["role"] = "tool"
        marked[_PREWRAPPED_TOOL_RESPONSE] = True
        marked_messages.append(cast(Message, marked))
    return marked_messages


class _Qwen3_5TemplateParityMixin:
    """Apply Qwen3.5/3.6 history/tool parity across thinking modes.

    Upstream Qwen3VL strips thinking from every non-final assistant via
    ``ctx.is_last``.  The official Qwen3.5/3.6 templates instead retain every
    assistant after the last real user query, including an assistant tool call
    and the assistant that follows its tool result.  Both thinking-enabled and
    disable-thinking renderers need this history behavior; ``enable_thinking``
    only changes the generation suffix.
    """

    def build_generation_prompt(self, messages: list[Message], *args, **kwargs):
        return super().build_generation_prompt(
            _mark_prewrapped_tool_responses(messages), *args, **kwargs
        )

    def build_supervised_example(self, messages: list[Message], *args, **kwargs):
        return super().build_supervised_example(
            _mark_prewrapped_tool_responses(messages), *args, **kwargs
        )

    def build_supervised_examples(self, messages: list[Message], *args, **kwargs):
        return super().build_supervised_examples(
            _mark_prewrapped_tool_responses(messages), *args, **kwargs
        )

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        if message.get(_PREWRAPPED_TOOL_RESPONSE):
            restored = dict(message)
            restored["role"] = "user"
            restored.pop(_PREWRAPPED_TOOL_RESPONSE, None)
            message = cast(Message, restored)

        if self.strip_thinking_from_history and message["role"] == "assistant":
            after_last_user = ctx.last_user_index == -1 or ctx.idx > ctx.last_user_index
            ctx = replace(ctx, is_last=after_last_user)
        return super().render_message(message, ctx)

    def _format_tool_calls_chunks(self, message: Message) -> list[TextPart]:
        """Match the conditional separator before Qwen's first tool call."""

        chunks = super()._format_tool_calls_chunks(message)
        if _has_visible_assistant_content(message):
            return chunks
        return [
            TextPart(
                type="text",
                text=chunks[0]["text"].removeprefix("\n\n"),
            )
        ]


class Qwen3_5SplitRenderer(
    DisaggregateMultiTurnMixin,
    Qwen3_5Renderer,
):
    """Legacy concrete ``qwen3_5`` behavior."""


class Qwen3_5DisableThinkingSplitRenderer(
    DisaggregateMultiTurnMixin,
    Qwen3_5DisableThinkingRenderer,
):
    """Legacy concrete ``qwen3_5_disable_thinking`` behavior."""


class Qwen3_5InterleavedRenderer(
    _Qwen3_5TemplateParityMixin,
    DisaggregateMultiTurnMixin,
    Qwen3_5Renderer,
):
    """Correct last-real-user boundary for Qwen3.5 INTERLEAVED mode."""


class Qwen3_5DisableThinkingInterleavedRenderer(
    _Qwen3_5TemplateParityMixin,
    DisaggregateMultiTurnMixin,
    Qwen3_5DisableThinkingRenderer,
):
    """Correct boundary with thinking generation independently disabled."""


def _has_visible_assistant_content(message: Message) -> bool:
    content = message.get("content", "")
    if isinstance(content, str):
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[-1]
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "thinking":
                continue
            if part.get("type") == "text":
                if str(part.get("text", "")).strip():
                    return True
                continue
            return True
        return False
    return bool(content)


class _Qwen3_6ToolArgumentsMixin:
    """Apply Qwen3.6's JSON serialization rule for tool arguments.

    Qwen3.5 stringifies scalar values (``True``/``None``), while Qwen3.6
    leaves strings unquoted and serializes every non-string value through
    Jinja's ``tojson`` filter (``true``/``null``).  Thinking-history mode and
    thinking enablement are independent of this formatting difference, so all
    Qwen3.6 renderer variants share the override.
    """

    def _format_tool_call_xml(self, tool_call: ToolCall) -> str:
        args = (
            json.loads(tool_call.function.arguments)
            if tool_call.function.arguments
            else {}
        )
        if not isinstance(args, Mapping):
            raise ValueError("Qwen3.6 tool-call arguments must decode to an object")

        lines = [f"<tool_call>\n<function={tool_call.function.name}>"]
        for param_name, param_value in args.items():
            value_str = (
                param_value
                if isinstance(param_value, str)
                else json.dumps(param_value, ensure_ascii=False)
            )
            lines.append(f"<parameter={param_name}>\n{value_str}\n</parameter>")
        lines.append("</function>\n</tool_call>")
        return "\n".join(lines)


# Qwen3.6 shares Qwen3.5's vocab, special tokens, and INTERLEAVED history boundary.
# Its template independently adds opt-in ``preserve_thinking`` and JSON
# serialization for non-string tool arguments; the latter is handled by the
# mixin above rather than treating the complete templates as byte-identical.
#
# Three Qwen3.6 renderer variants:
#
#   `qwen3_6_interleaved`        — default semantic-mode implementation
#                                    (`strip_thinking_from_history=True`).
#                                    Matches HF apply_chat_template default
#                                    args. Use for normal SFT/DPO/RL where
#                                    historical thinking should be stripped.
#
#   `qwen3_6_preserved`          — preserve thinking enabled
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
#   `qwen3_6_disable_thinking_interleaved` — empty think block in generation prompt
#                                    (`enable_thinking=false`). Alias of
#                                    Qwen3.5 disable-thinking renderer.
#                                    Use for non-thinking SFT (text2sql,
#                                    structured output, format compliance);
#                                    customers must also pass
#                                    `enable_thinking=false` at inference
#                                    time to keep train↔inference parity.


class Qwen3_6SplitRenderer(Qwen3_5SplitRenderer):
    """Legacy concrete ``qwen3_6`` behavior."""


class Qwen3_6DisableThinkingSplitRenderer(Qwen3_5DisableThinkingSplitRenderer):
    """Legacy concrete ``qwen3_6_disable_thinking`` behavior."""


class Qwen3_6InterleavedRenderer(
    _Qwen3_6ToolArgumentsMixin,
    Qwen3_5InterleavedRenderer,
):
    """Correct Qwen3.6 INTERLEAVED boundary and tool serialization."""


class Qwen3_6DisableThinkingInterleavedRenderer(
    _Qwen3_6ToolArgumentsMixin,
    Qwen3_5DisableThinkingInterleavedRenderer,
):
    """Correct Qwen3.6 boundary with thinking generation disabled."""


class _Qwen3_6PreserveThinkingMixin:
    """Qwen3.6 PRESERVED mode: historical assistant `<think>` blocks remain.

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


class Qwen3_6PreserveThinkingSplitRenderer(
    _Qwen3_6PreserveThinkingMixin,
    Qwen3_6SplitRenderer,
):
    """Legacy concrete ``qwen3_6_preserve_thinking`` behavior."""


class Qwen3_6PreservedRenderer(
    _Qwen3_6PreserveThinkingMixin,
    Qwen3_6InterleavedRenderer,
):
    """Correct Qwen3.6 PRESERVED boundary and tool serialization."""


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
    "qwen3_5_interleaved",
    lambda tok, ip=None: Qwen3_5InterleavedRenderer(tok, image_processor=ip),
)
register_renderer(
    "qwen3_5_disable_thinking_interleaved",
    lambda tok, ip=None: Qwen3_5DisableThinkingInterleavedRenderer(
        tok, image_processor=ip
    ),
)
register_renderer(
    "qwen3_6_interleaved",
    lambda tok, ip=None: Qwen3_6InterleavedRenderer(tok, image_processor=ip),
)
register_renderer(
    "qwen3_6_disable_thinking_interleaved",
    lambda tok, ip=None: Qwen3_6DisableThinkingInterleavedRenderer(
        tok, image_processor=ip
    ),
)
register_renderer(
    "qwen3_6_preserved",
    lambda tok, ip=None: Qwen3_6PreservedRenderer(tok, image_processor=ip),
)
