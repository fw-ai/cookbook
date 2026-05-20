"""Renderer for DeepSeek-V4-Flash chat templates.

This is a port of the upstream Python encoder shipped at
``deepseek-ai/DeepSeek-V4-Flash/encoding/encoding_dsv4.py`` (license: MIT)
into the ``tinker_cookbook`` renderer abstraction.

DeepSeek-V4 uses a hand-rolled DSML-flavored format rather than a Jinja
chat template:

- ``<｜begin▁of▁sentence｜>`` once at the very start of the conversation.
- System messages emit their content with no role tag.
- User messages emit ``<｜User｜>{content}`` (or, after merging, a
  ``\\n\\n``-joined sequence of text blocks and
  ``<tool_result>...</tool_result>`` blocks for prior tool responses).
- Assistant messages emit ``{thinking_block}{content}{tool_calls}<EOS>``.
  Thinking is a literal ``<think>{reasoning}</think>`` and is always
  preceded by a ``<｜Assistant｜>`` role tag emitted by the *boundary*
  between the previous user-or-developer message and this assistant turn.

The boundary suffix has three forms:

* ``<｜Assistant｜><think>``  — thinking-mode terminal turn, OR thinking
  mode without history-stripping.
* ``<｜Assistant｜></think>`` — thinking-mode historical turn under
  ``strip_thinking_from_history=True`` (matches the encoder's
  ``drop_thinking=True`` default), OR any chat-mode turn.

The encoder also auto-disables history thinking-stripping whenever any
message in the conversation carries a ``tools`` field. We mirror that
behaviour so renderer output matches the encoder byte-for-byte.

Tool calls serialize as a DSML block::

    \\n\\n<｜DSML｜tool_calls>
    <｜DSML｜invoke name="{name}">
    <｜DSML｜parameter name="{k}" string="true|false">{v}</｜DSML｜parameter>
    ...
    </｜DSML｜invoke>
    ...
    </｜DSML｜tool_calls>

with ``string="true"`` for raw string parameters (emitted verbatim) and
``string="false"`` for everything else (JSON-encoded with
``ensure_ascii=False``).

Tool messages are merged into the *preceding* user message as
``<tool_result>...</tool_result>`` blocks and reordered to match the
order of the preceding assistant's ``tool_calls`` (matching the encoder's
``merge_tool_messages`` + ``sort_tool_results_by_call_order`` pipeline).

Out of scope for this renderer (call sites that use these features should
keep using the upstream encoder directly):

- ``developer`` role
- ``latest_reminder`` role
- Internal task tokens (``action`` / ``query`` / ``authority`` / ``domain``
  / ``title`` / ``read_url``)
- ``wo_eos`` flag
- ``reasoning_effort`` preamble
- ``context`` (encoded prefix) parameter

Verified against ``encoding_dsv4.encode_messages`` byte-for-byte by the
unit tests in ``test_deepseek_v4_renderer.py``.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any, Literal

import tinker
import torch
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    TrainOnWhat,
    UnparsedToolCall,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

# ── Special tokens (must match encoding_dsv4.py exactly) ────────────────────
# NOTE: ``｜`` is U+FF5C FULLWIDTH VERTICAL LINE (not ASCII ``|``).
#       ``▁`` is U+2581 LOWER ONE EIGHTH BLOCK (not ASCII ``_``).
_BOS_TEXT = "<｜begin▁of▁sentence｜>"
_EOS_TEXT = "<｜end▁of▁sentence｜>"
_USER_SP = "<｜User｜>"
_ASSISTANT_SP = "<｜Assistant｜>"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_DSML = "｜DSML｜"

_TOOL_CALLS_OPEN = f"<{_DSML}tool_calls>"
_TOOL_CALLS_CLOSE = f"</{_DSML}tool_calls>"
_TOOL_CALLS_BOUNDARY = f"\n\n<{_DSML}tool_calls"

# ── Tools / response_format prompt sections ─────────────────────────────────

_TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}tool_calls>" block like the following:

<{dsml}tool_calls>
<{dsml}invoke name="$TOOL_NAME">
<{dsml}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}parameter>
...
</{dsml}invoke>
<{dsml}invoke name="$TOOL_NAME2">
...
</{dsml}invoke>
</{dsml}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think_open}), you MUST output your complete reasoning inside {think_open}...{think_close} BEFORE any tool calls or final response.

Otherwise, output directly after {think_close} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

_RESPONSE_FORMAT_TEMPLATE = (
    "## Response Format:\n\n"
    "You MUST strictly adhere to the following schema to reply:\n{schema}"
)

# ── Parsing regexes (mirror encoding_dsv4.parse_tool_calls) ─────────────────

_TOOL_CALLS_BLOCK_RE = re.compile(
    rf"\n\n<{re.escape(_DSML)}tool_calls>\n(.*?)\n</{re.escape(_DSML)}tool_calls>",
    re.DOTALL,
)
_INVOKE_RE = re.compile(
    rf'<{re.escape(_DSML)}invoke name="([^"]+)">\n(.*?)\n</{re.escape(_DSML)}invoke>',
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    rf'<{re.escape(_DSML)}parameter name="([^"]+)" string="(true|false)">'
    rf"(.*?)"
    rf"</{re.escape(_DSML)}parameter>",
    re.DOTALL,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _to_json(value: Any) -> str:
    """Match ``encoding_dsv4.to_json`` (ensure_ascii=False, ASCII fallback)."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return json.dumps(value, ensure_ascii=True)


def _visible_text(content: Any) -> str:
    """Flatten string-or-structured content to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
            elif (
                isinstance(item, Mapping)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                out.append(item["text"])
        return "".join(out)
    return str(content)


def _extract_reasoning_and_text(content: Any) -> tuple[str, str]:
    """Split assistant ``content`` into ``(reasoning, visible_text)``."""
    if isinstance(content, str):
        if "</think>" not in content:
            return "", content
        head, _, tail = content.partition("</think>")
        reasoning = head.split("<think>", 1)[-1].strip("\n")
        visible = tail.lstrip("\n")
        return reasoning, visible
    if isinstance(content, list):
        reasoning_parts: list[str] = []
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "thinking" and isinstance(part.get("thinking"), str):
                reasoning_parts.append(part["thinking"])
            elif part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "".join(reasoning_parts), "".join(text_parts)
    return "", str(content)


def _normalize_tool_arguments(raw: str) -> dict[str, Any]:
    """Tinker stores tool args as a JSON string; decode it once for rendering.

    Mirrors the encoder's defensive fallback: if ``raw`` doesn't parse as
    a JSON object, wrap it as ``{"arguments": raw}`` so we never crash on
    malformed model output.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"arguments": raw}
    if not isinstance(parsed, Mapping):
        return {"arguments": raw}
    return dict(parsed)


def _format_tool_calls(tool_calls: list[ToolCall]) -> str:
    """Render a list of tool calls as a single DSML ``<…tool_calls>`` block."""
    invokes: list[str] = []
    for tc in tool_calls:
        args = _normalize_tool_arguments(tc.function.arguments)
        param_lines = [
            f'<{_DSML}parameter name="{k}" string="'
            f'{"true" if isinstance(v, str) else "false"}">'
            f"{v if isinstance(v, str) else _to_json(v)}"
            f"</{_DSML}parameter>"
            for k, v in args.items()
        ]
        invokes.append(
            f'<{_DSML}invoke name="{tc.function.name}">\n'
            + "\n".join(param_lines)
            + f"\n</{_DSML}invoke>"
        )
    body = "\n".join(invokes)
    return f"\n\n{_TOOL_CALLS_OPEN}\n{body}\n{_TOOL_CALLS_CLOSE}"


def _render_tools_section(tools: list[Mapping[str, Any]]) -> str:
    """Render OpenAI-format tool schemas into the ``## Tools`` block.

    Mirrors ``encoding_dsv4.render_tools`` exactly: each ``tool`` is
    expected to be ``{"type": "function", "function": {...}}``; we extract
    the inner function dict, JSON-dump one per line, and substitute into
    ``TOOLS_TEMPLATE``.
    """
    function_dicts = [tool["function"] for tool in tools]
    return _TOOLS_TEMPLATE.format(
        dsml=_DSML,
        think_open=_THINK_OPEN,
        think_close=_THINK_CLOSE,
        tool_schemas="\n".join(_to_json(t) for t in function_dicts),
    )


def _has_any_tools(messages: list[Message]) -> bool:
    return any(m.get("tools") for m in messages)


# ── Preprocessing (mirror encoder's merge + sort + drop pipeline) ───────────


def _merge_tool_messages(messages: list[Message]) -> list[Message]:
    """Fold ``role=tool`` messages into the previous user's ``content_blocks``.

    Mirrors ``encoding_dsv4.merge_tool_messages``. User messages always get
    a ``content_blocks`` field added (even when there's no tool to merge
    in) so subsequent passes can rely on the field's presence.
    """
    merged: list[Message] = []
    for msg in messages:
        msg = copy.deepcopy(dict(msg))
        role = msg.get("role")

        if role == "tool":
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
            ):
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append({"role": "user", "content_blocks": [tool_block]})
            continue

        if role == "user":
            text_block = {"type": "text", "text": msg.get("content", "")}
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].append(text_block)
            else:
                new_msg: dict[str, Any] = {
                    "role": "user",
                    "content": msg.get("content", ""),
                    "content_blocks": [text_block],
                }
                for key in ("task", "wo_eos", "mask"):
                    if key in msg:
                        new_msg[key] = msg[key]
                merged.append(new_msg)
            continue

        merged.append(msg)
    return merged


def _sort_tool_results_by_call_order(messages: list[Message]) -> list[Message]:
    """Reorder ``tool_result`` blocks to match the preceding tool-call order.

    Mirrors ``encoding_dsv4.sort_tool_results_by_call_order``. Mutates
    ``messages`` in place (consistent with the encoder); callers should
    ``deepcopy`` first if they need the originals untouched.
    """
    last_order: dict[str, int] = {}
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            last_order = {}
            for idx, tc in enumerate(msg["tool_calls"]):
                # Tinker ToolCall objects expose ``.id``; OpenAI dict shape
                # nests it under ``id``/``function.id``.
                tc_id = ""
                if isinstance(tc, Mapping):
                    tc_id = tc.get("id") or tc.get("function", {}).get("id", "")
                else:
                    tc_id = getattr(tc, "id", "") or ""
                if tc_id:
                    last_order[tc_id] = idx
            continue

        if role == "user" and msg.get("content_blocks"):
            tool_blocks = [
                b for b in msg["content_blocks"] if b.get("type") == "tool_result"
            ]
            if len(tool_blocks) > 1 and last_order:
                sorted_blocks = sorted(
                    tool_blocks,
                    key=lambda b: last_order.get(b.get("tool_use_id", ""), 0),
                )
                cursor = 0
                new_blocks: list[Any] = []
                for block in msg["content_blocks"]:
                    if block.get("type") == "tool_result":
                        new_blocks.append(sorted_blocks[cursor])
                        cursor += 1
                    else:
                        new_blocks.append(block)
                msg["content_blocks"] = new_blocks
    return messages


def _drop_thinking_from_history(messages: list[Message]) -> list[Message]:
    """Strip ``reasoning_content`` and structured ``thinking`` parts from
    historical assistant turns (mirrors ``_drop_thinking_messages``).

    A "historical" assistant is one whose index is strictly less than the
    last user index. Assistant turns at or after the last user index keep
    their reasoning intact.
    """
    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in ("user", "developer"):
            last_user_idx = idx
            break

    result: list[Message] = []
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant" and idx < last_user_idx:
            stripped = copy.copy(dict(msg))
            stripped.pop("reasoning_content", None)
            content = stripped.get("content")
            if isinstance(content, list):
                stripped["content"] = [
                    p
                    for p in content
                    if not (isinstance(p, Mapping) and p.get("type") == "thinking")
                ]
            elif isinstance(content, str) and "</think>" in content:
                stripped["content"] = content.split("</think>", 1)[-1].lstrip("\n")
            result.append(stripped)
        else:
            result.append(msg)
    return result


# ── Renderer ────────────────────────────────────────────────────────────────


ThinkingMode = Literal["chat", "thinking"]


class DeepseekV4Renderer(Renderer):
    """Renderer for ``deepseek-ai/DeepSeek-V4-Flash`` instruct models."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        thinking_mode: ThinkingMode = "thinking",
        strip_thinking_from_history: bool = True,
    ) -> None:
        super().__init__(tokenizer)
        if thinking_mode not in ("chat", "thinking"):
            raise ValueError(f"Invalid thinking_mode: {thinking_mode!r}")
        self.thinking_mode: ThinkingMode = thinking_mode
        self.strip_thinking_from_history = strip_thinking_from_history
        # Set per-call by ``_preprocess`` so ``render_message`` can pick the
        # right assistant header without re-checking the message list.
        self._effective_strip = strip_thinking_from_history

    # ---- public Renderer API --------------------------------------------------

    @property
    def has_extension_property(self) -> bool:
        """True when historical and terminal assistant turns render identically.

        Chat mode never emits a thinking block, so it always satisfies the
        extension property. Thinking mode only satisfies it when we keep
        history reasoning intact.
        """
        return self.thinking_mode == "chat" or not self.strip_thinking_from_history

    def get_stop_sequences(self) -> list[int]:
        return [self._eos_token]

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        return super().build_generation_prompt(
            self._preprocess(messages),
            role=role,
            prefill=prefill,
        )

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        return super().build_supervised_example(
            self._preprocess(messages),
            train_on_what=train_on_what,
        )

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]
        if role == "system":
            return self._render_system(message)
        if role == "user":
            return self._render_user(message)
        if role == "assistant":
            return self._render_assistant(message, ctx)
        raise ValueError(
            f"DeepseekV4Renderer: unsupported role {role!r}. "
            f"`developer`, `latest_reminder`, `tool` (un-merged), and task roles "
            f"are intentionally out of scope; preprocess them upstream."
        )

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """Decode ``response`` and split into reasoning / content / tool_calls.

        Mirrors ``encoding_dsv4.parse_message_from_completion_text`` enough
        to roundtrip the rendered output, but stays forgiving for partial
        completions: returns ``ok=False`` when the EOS hasn't shown up yet.
        """
        text = self.tokenizer.decode(response)
        eos_idx = text.find(_EOS_TEXT)
        ok = eos_idx >= 0
        if ok:
            text = text[:eos_idx]

        reasoning = ""
        if self.thinking_mode == "thinking" and _THINK_CLOSE in text:
            head, _, text = text.partition(_THINK_CLOSE)
            reasoning = head.removeprefix(_THINK_OPEN).strip("\n")

        tool_calls: list[ToolCall] = []
        unparsed: list[UnparsedToolCall] = []
        block_match = _TOOL_CALLS_BLOCK_RE.search(text)
        if block_match:
            content = text[: block_match.start()]
            for invoke in _INVOKE_RE.finditer(block_match.group(1)):
                name = invoke.group(1)
                args: dict[str, Any] = {}
                for param in _PARAMETER_RE.finditer(invoke.group(2)):
                    key, is_str, value = param.group(1), param.group(2), param.group(3)
                    args[key] = value if is_str == "true" else _parse_value(value)
                tool_calls.append(
                    ToolCall(
                        function=ToolCall.FunctionBody(
                            name=name,
                            arguments=_to_json(args),
                        )
                    )
                )
            trailing = text[block_match.end() :]
            if trailing.strip():
                unparsed.append(
                    UnparsedToolCall(
                        raw_text=trailing,
                        error="Unexpected content after tool_calls block",
                    )
                )
        else:
            content = text

        message = Message(role="assistant", content=content)
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        if unparsed:
            message["unparsed_tool_calls"] = unparsed
        return message, ok

    # ---- internal helpers -----------------------------------------------------

    def _preprocess(self, messages: list[Message]) -> list[Message]:
        merged = _merge_tool_messages(messages)
        merged = _sort_tool_results_by_call_order(merged)
        # Encoder rule: any message with `tools` defined disables drop_thinking.
        self._effective_strip = self.strip_thinking_from_history and not _has_any_tools(
            merged
        )
        if self.thinking_mode == "thinking" and self._effective_strip:
            merged = _drop_thinking_from_history(merged)
        return merged

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode(_BOS_TEXT, add_special_tokens=False)

    def _encode_single_special(self, token_str: str) -> int:
        token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(
                f"DeepseekV4Renderer expected {token_str!r} to encode as one "
                f"token, got {token_ids}."
            )
        return int(token_ids[0])

    @property
    def _eos_token(self) -> int:
        return self._encode_single_special(_EOS_TEXT)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ---- render_message branches ---------------------------------------------

    def _render_system(self, message: Message) -> RenderedMessage:
        # System has no role tag — content goes directly into the prompt.
        body = _visible_text(message.get("content"))
        tools = message.get("tools")
        response_format = message.get("response_format")
        if tools:
            body += "\n\n" + _render_tools_section(list(tools))
        if response_format:
            body += "\n\n" + _RESPONSE_FORMAT_TEMPLATE.format(
                schema=_to_json(response_format)
            )
        header = tinker.types.EncodedTextChunk(tokens=[])
        output = [tinker.types.EncodedTextChunk(tokens=self._encode(body))]
        return RenderedMessage(header=header, output=output)

    def _render_user(self, message: Message) -> RenderedMessage:
        # Header is the role tag; output is the content blocks (or raw content).
        header = tinker.types.EncodedTextChunk(tokens=self._encode(_USER_SP))
        body = self._render_user_body(message)
        output = [tinker.types.EncodedTextChunk(tokens=self._encode(body))]
        return RenderedMessage(header=header, output=output)

    def _render_user_body(self, message: Message) -> str:
        content_blocks = message.get("content_blocks")
        if not content_blocks:
            return _visible_text(message.get("content"))

        parts: list[str] = []
        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "text":
                parts.append(block.get("text", "") or "")
            elif block_type == "tool_result":
                tool_content = block.get("content", "")
                if isinstance(tool_content, list):
                    inner: list[str] = []
                    for sub in tool_content:
                        if isinstance(sub, Mapping) and sub.get("type") == "text":
                            inner.append(sub.get("text", "") or "")
                        else:
                            kind = (
                                sub.get("type")
                                if isinstance(sub, Mapping)
                                else type(sub).__name__
                            )
                            inner.append(f"[Unsupported {kind}]")
                    tool_content = "\n\n".join(inner)
                parts.append(f"<tool_result>{tool_content}</tool_result>")
            else:
                parts.append(f"[Unsupported {block_type}]")
        return "\n\n".join(parts)

    def _assistant_header_str(self, ctx: RenderContext) -> str:
        # Boundary suffix from the *previous* user/developer message, attributed
        # here as the assistant's own header so masking lines up.
        is_terminal = ctx.last_user_index < 0 or ctx.idx > ctx.last_user_index
        thinking_branch = self.thinking_mode == "thinking" and (
            not self._effective_strip or is_terminal
        )
        return _ASSISTANT_SP + (_THINK_OPEN if thinking_branch else _THINK_CLOSE)

    def _render_assistant(
        self,
        message: Message,
        ctx: RenderContext,
    ) -> RenderedMessage:
        header_str = self._assistant_header_str(ctx)
        emit_thinking_body = header_str.endswith(_THINK_OPEN)

        # The encoder reads ``reasoning_content`` directly from the message;
        # we mirror that, but also accept the cookbook conventions of
        # ``<think>...</think>`` embedded in ``content`` and structured
        # ``[{"type": "thinking", ...}]`` blocks for backwards compat.
        explicit_reasoning = message.get("reasoning_content")
        if explicit_reasoning is not None:
            reasoning = explicit_reasoning
            visible = _visible_text(message.get("content"))
        else:
            reasoning, visible = _extract_reasoning_and_text(message.get("content"))
        body = ""
        if emit_thinking_body:
            # Header already injects ``<think>``; output starts with the
            # reasoning text + the closing ``</think>`` so the model trains
            # to produce that closing tag itself.
            body += reasoning + _THINK_CLOSE
        body += visible

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            body += _format_tool_calls(list(tool_calls))

        body += _EOS_TEXT

        header = tinker.types.EncodedTextChunk(tokens=self._encode(header_str))
        output = [tinker.types.EncodedTextChunk(tokens=self._encode(body))]
        return RenderedMessage(header=header, output=output)

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        del ctx
        if role == "assistant":
            # A generation prompt always targets the *terminal* assistant slot,
            # so the boundary always uses the thinking-start branch in
            # thinking mode and the thinking-end branch in chat mode.
            suffix = _ASSISTANT_SP + (
                _THINK_OPEN if self.thinking_mode == "thinking" else _THINK_CLOSE
            )
            return self._encode(suffix)
        if role == "user":
            return self._encode(_USER_SP)
        # System has no role tag in DSv4; emit nothing.
        if role == "system":
            return []
        raise ValueError(f"DeepseekV4Renderer: cannot generate role {role!r}")


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _deepseek_v4_factory(
    tokenizer: Tokenizer,
    image_processor: Any = None,
) -> DeepseekV4Renderer:
    del image_processor
    return DeepseekV4Renderer(tokenizer)


register_renderer("deepseek_v4", _deepseek_v4_factory)
