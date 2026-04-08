"""Renderer for Google Gemma 4 instruct models.

Built against the official ``chat_template.jinja`` shipped with
``google/gemma-4-E2B-it`` (and the other gemma-4 *-it variants on the
HuggingFace Hub). The earlier version of this file was built against an
unofficial dev fixture and was wrong on multiple fundamental dimensions
(role name, generation suffix, content trimming, tool format) — see
``training/renderer/README.md`` for the postmortem.

Format produced for a basic conversation::

    <bos><|turn>system
    {trimmed system content}<turn|>
    <|turn>user
    {trimmed user content}<turn|>
    <|turn>model
    {trimmed model content}<turn|>
    <|turn>user
    {trimmed user content}<turn|>
    <|turn>model    <- generation prompt

Key facts (all derived from the official Jinja template, not from the
model card or the older dev fixture):

* The assistant role is rendered as ``model``, not ``assistant``. Other
  roles (``system``, ``user``, ``tool``, ``developer``) are emitted
  verbatim.
* Every message's text content is run through Jinja's ``| trim`` filter,
  i.e. ``str.strip()``. Leading and trailing whitespace never reach the
  tokenizer.
* Each turn produces ``<|turn>{role}\\n{trimmed_content}<turn|>\\n``. The
  trailing ``\\n`` after ``<turn|>`` is the inter-turn separator and is
  part of every closed turn.
* The generation prompt appended at the end is ``<|turn>model\\n``.
* The model role's content is additionally passed through the template's
  ``strip_thinking`` macro, which removes ``<|channel>...<channel|>``
  blocks from history. We replicate this for parity.

Tool / thinking support
-----------------------

This renderer reproduces the official template's tool-call and thinking
paths byte-for-byte. The pieces:

* **Tool definitions**: Use
  ``Gemma4Renderer.create_conversation_prefix_with_tools(tools, system_prompt)``
  to obtain a synthetic system message that bundles ``system_prompt`` and
  the serialized tool block (``<|tool>declaration:...<tool|>`` per tool).
  Prepend that to your conversation. The tool serializer mirrors the
  template's ``format_function_declaration`` / ``format_parameters`` /
  ``format_argument`` macros, including the alphabetical key sort
  (``dictsort``) and the ``<|"|>`` string-escape token.
* **Assistant tool calls**: Pass an HF-shaped tool call dict on the
  message: ``message["tool_calls"] = [{"function": {"name": ...,
  "arguments": <dict or json string>}}]``. The renderer emits
  ``<|tool_call>call:name{key:value,...}<tool_call|>`` matching the
  template. ``arguments`` may be a dict (preferred) or a JSON string;
  JSON strings are parsed and re-emitted as a dict so that round-trips
  via tinker's ``ToolCall`` (which always carries a JSON string) match
  HF parity.
* **Tool responses**: Pass ``message["tool_responses"] = [{"name": ...,
  "response": <dict or scalar>}]`` on a regular turn. The renderer
  emits ``<|tool_response>response:name{...}<tool_response|>`` and
  follows the template's quirk of suppressing the trailing ``<turn|>\\n``
  when the message is purely a tool response with no other content.
  After such a message, the generation suffix ``<|turn>model\\n`` is
  also suppressed (the model is expected to keep filling the same turn).
* **Thinking mode**: Construct the renderer with ``enable_thinking=True``
  to inject the ``<|think|>`` system-level marker that the template adds
  at the very start of the first system turn. If the conversation has no
  system message, an empty one is synthesized just to carry the marker —
  matching the template's behavior.

Special-token IDs (Gemma 4 tokenizer)::

    <bos>     = 2
    <|turn>   = 105      <turn|>   = 106
    \\n        = 107
    model token (string ``model``) varies by tokenizer
"""

from __future__ import annotations

import json
import re
from typing import Any

import tinker
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    TextPart,
    ThinkingPart,
    ToolSpec,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

_BOS_TOKEN = "<bos>"
_TURN_OPEN = "<|turn>"
_TURN_CLOSE = "<turn|>"
_CHANNEL_OPEN = "<|channel>"
_CHANNEL_CLOSE = "<channel|>"
_TOOL_OPEN = "<|tool>"
_TOOL_CLOSE = "<tool|>"
_TOOL_CALL_OPEN = "<|tool_call>"
_TOOL_CALL_CLOSE = "<tool_call|>"
_TOOL_RESPONSE_OPEN = "<|tool_response>"
_TOOL_RESPONSE_CLOSE = "<tool_response|>"
_THINK_TOKEN = "<|think|>"
_STRING_DELIM = '<|"|>'
_MODEL_ROLE = "model"

# Pattern matching the template's strip_thinking macro: remove every
# `<|channel>...<channel|>` block from a string and `.strip()` what's left.
_THINKING_BLOCK_RE = re.compile(
    re.escape(_CHANNEL_OPEN) + r".*?" + re.escape(_CHANNEL_CLOSE),
    re.DOTALL,
)

# Tool-call extraction. Brace-balanced via lookahead anchor on the close tag.
_TOOL_CALL_RE = re.compile(
    re.escape(_TOOL_CALL_OPEN)
    + r"call:([^{]+)\{(.*?)\}"
    + re.escape(_TOOL_CALL_CLOSE),
    re.DOTALL,
)

# Property keys consumed structurally by format_parameters and therefore
# never emitted as their own field. Mirrors `standard_keys` in the jinja.
_STANDARD_PARAM_KEYS = frozenset({"description", "type", "properties", "required", "nullable"})


def _strip_thinking(text: str) -> str:
    """Match the template's ``strip_thinking`` macro: drop ``<|channel>...<channel|>``
    blocks and ``.strip()`` the result."""
    return _THINKING_BLOCK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Pure-python ports of the jinja macros: format_argument, format_parameters,
# format_function_declaration. These produce *byte-identical* output to the
# template for the inputs covered by the parity tests.
# ---------------------------------------------------------------------------


def _format_argument(value: Any, escape_keys: bool = True) -> str:
    """Mirror the template's ``format_argument`` macro.

    Strings get the ``<|"|>...<|"|>`` escape; bools render as ``true``/``false``;
    mappings recurse with sorted keys (Jinja's ``dictsort``); sequences recurse
    over items; everything else is rendered with ``str()``.

    ``escape_keys`` controls whether mapping keys are wrapped in the string
    delimiter — argument-passing contexts (tool call args, tool responses) use
    bare keys; the schema-declaration context wraps them.
    """
    # bool must come before int (bool is a subclass of int in Python).
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f"{_STRING_DELIM}{value}{_STRING_DELIM}"
    if isinstance(value, dict):
        parts = []
        for k, v in sorted(value.items()):
            key_str = f"{_STRING_DELIM}{k}{_STRING_DELIM}" if escape_keys else str(k)
            parts.append(f"{key_str}:{_format_argument(v, escape_keys=escape_keys)}")
        return "{" + ",".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_argument(item, escape_keys=escape_keys) for item in value) + "]"
    return str(value)


def _format_parameters(properties: dict, required: list) -> str:
    """Mirror the template's ``format_parameters`` macro.

    The ``required`` arg matches the macro signature but is unused inside this
    function — it only flows through to the ``OBJECT`` recursion path via the
    nested property's own ``required`` field. Kept for fidelity with the macro.
    """
    del required  # unused; carried for signature parity with the jinja macro
    chunks: list[str] = []
    for key, value in sorted(properties.items()):
        if key in _STANDARD_PARAM_KEYS:
            continue
        if not isinstance(value, dict):
            continue
        prop_chunks: list[str] = [f"{key}:{{"]
        add_comma = False
        if value.get("description"):
            prop_chunks.append(f'description:{_STRING_DELIM}{value["description"]}{_STRING_DELIM}')
            add_comma = True
        if value.get("nullable"):
            if add_comma:
                prop_chunks.append(",")
            else:
                add_comma = True
            prop_chunks.append("nullable:true")

        type_upper = (value.get("type") or "").upper()
        if type_upper == "STRING":
            if value.get("enum"):
                if add_comma:
                    prop_chunks.append(",")
                else:
                    add_comma = True
                prop_chunks.append(f"enum:{_format_argument(value['enum'])}")
        elif type_upper == "OBJECT":
            prop_chunks.append(",properties:{")
            inner_props = value.get("properties")
            if isinstance(inner_props, dict):
                prop_chunks.append(_format_parameters(inner_props, value.get("required") or []))
            elif isinstance(value, dict):
                prop_chunks.append(_format_parameters(value, value.get("required") or []))
            prop_chunks.append("}")
            req = value.get("required")
            if req:
                prop_chunks.append(",required:[")
                for i, item in enumerate(req):
                    if i:
                        prop_chunks.append(",")
                    prop_chunks.append(f"{_STRING_DELIM}{item}{_STRING_DELIM}")
                prop_chunks.append("]")
        elif type_upper == "ARRAY":
            items = value.get("items")
            if isinstance(items, dict) and items:
                prop_chunks.append(",items:{")
                items_first = True
                for item_key, item_value in sorted(items.items()):
                    if item_value is None:
                        continue
                    if not items_first:
                        prop_chunks.append(",")
                    items_first = False
                    if item_key == "properties":
                        prop_chunks.append("properties:{")
                        if isinstance(item_value, dict):
                            prop_chunks.append(
                                _format_parameters(item_value, items.get("required") or [])
                            )
                        prop_chunks.append("}")
                    elif item_key == "required":
                        prop_chunks.append("required:[")
                        for i, req_item in enumerate(item_value):
                            if i:
                                prop_chunks.append(",")
                            prop_chunks.append(f"{_STRING_DELIM}{req_item}{_STRING_DELIM}")
                        prop_chunks.append("]")
                    elif item_key == "type":
                        if isinstance(item_value, str):
                            prop_chunks.append(f"type:{_format_argument(item_value.upper())}")
                        else:
                            prop_chunks.append(
                                f"type:{_format_argument([str(s).upper() for s in item_value])}"
                            )
                    else:
                        prop_chunks.append(f"{item_key}:{_format_argument(item_value)}")
                prop_chunks.append("}")

        # Trailing `type:` field. Always emitted; comma logic mirrors the macro.
        if add_comma:
            prop_chunks.append(",")
        prop_chunks.append(f"type:{_STRING_DELIM}{type_upper}{_STRING_DELIM}}}")
        chunks.append("".join(prop_chunks))
    return ",".join(chunks)


def _format_function_declaration(tool: dict) -> str:
    """Mirror the template's ``format_function_declaration`` macro.

    Accepts the OpenAI-style ``{"type": "function", "function": {...}}``
    dict shape that HF's ``apply_chat_template(..., tools=...)`` consumes.
    """
    fn = tool["function"] if "function" in tool else tool
    parts: list[str] = [
        f"declaration:{fn['name']}{{description:{_STRING_DELIM}{fn.get('description', '')}{_STRING_DELIM}"
    ]
    params = fn.get("parameters")
    if params:
        parts.append(",parameters:{")
        if params.get("properties"):
            parts.append("properties:{")
            parts.append(_format_parameters(params["properties"], params.get("required") or []))
            parts.append("},")
        if params.get("required"):
            parts.append("required:[")
            for i, item in enumerate(params["required"]):
                if i:
                    parts.append(",")
                parts.append(f"{_STRING_DELIM}{item}{_STRING_DELIM}")
            parts.append("],")
        if params.get("type"):
            parts.append(f"type:{_STRING_DELIM}{params['type'].upper()}{_STRING_DELIM}}}")
    if "response" in fn:
        # Faithful but minimal port of the response branch — only the cases
        # that the template itself emits (description + OBJECT-typed response).
        resp = fn["response"]
        parts.append(",response:{")
        if resp.get("description"):
            parts.append(f"description:{_STRING_DELIM}{resp['description']}{_STRING_DELIM},")
        if (resp.get("type") or "").upper() == "OBJECT":
            parts.append(f"type:{_STRING_DELIM}{resp['type'].upper()}{_STRING_DELIM}}}")
    parts.append("}")
    return "".join(parts)


def _format_tool_block(tools: list) -> str:
    """Render the contiguous run of ``<|tool>...<tool|>`` blocks the template
    inserts inside the system turn when ``tools`` is non-empty."""
    if not tools:
        return ""
    return "".join(f"{_TOOL_OPEN}{_format_function_declaration(t).strip()}{_TOOL_CLOSE}" for t in tools)


def _coerce_tool_arguments(args: Any) -> Any:
    """Normalize tool-call arguments for parity rendering.

    Tinker's ``ToolCall`` model carries arguments as a JSON string; HF's chat
    template input shape carries them as a dict. We accept both: a dict is
    used as-is, and a string is parsed as JSON if it round-trips to a dict
    (matching what HF would emit). A non-JSON string is preserved verbatim
    so the literal-string branch of the template's ``arguments is string``
    case still works.
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return args
        if isinstance(parsed, dict):
            return parsed
        return args
    return args


def _format_tool_call(tc: Any) -> str:
    """Render a single tool call as ``<|tool_call>call:name{args}<tool_call|>``.

    Accepts both HF-dict and tinker ``ToolCall`` pydantic shapes.
    """
    if hasattr(tc, "function") and not isinstance(tc, dict):
        fn_name = tc.function.name
        raw_args: Any = tc.function.arguments
    else:
        fn = tc["function"]
        fn_name = fn["name"]
        raw_args = fn.get("arguments")

    args = _coerce_tool_arguments(raw_args) if raw_args is not None else {}
    out = [f"{_TOOL_CALL_OPEN}call:{fn_name}{{"]
    if isinstance(args, dict):
        items = []
        for k, v in sorted(args.items()):
            items.append(f"{k}:{_format_argument(v, escape_keys=False)}")
        out.append(",".join(items))
    elif isinstance(args, str):
        out.append(args)
    out.append(f"}}{_TOOL_CALL_CLOSE}")
    return "".join(out)


def _format_tool_response(tr: dict) -> str:
    """Render a single tool response as
    ``<|tool_response>response:name{...}<tool_response|>``.

    Mirrors the template's mapping/non-mapping branches: dict responses are
    laid out key by key (dictsort, bare keys, escape_keys=False); scalar
    responses become ``value:<formatted>``.
    """
    name = tr.get("name", "unknown")
    response = tr.get("response")
    out = [f"{_TOOL_RESPONSE_OPEN}response:{name}{{"]
    if isinstance(response, dict):
        items = []
        for k, v in sorted(response.items()):
            items.append(f"{k}:{_format_argument(v, escape_keys=False)}")
        out.append(",".join(items))
    else:
        out.append(f"value:{_format_argument(response, escape_keys=False)}")
    out.append(f"}}{_TOOL_RESPONSE_CLOSE}")
    return "".join(out)


def _render_text_content(message: Message) -> str:
    """Apply the per-role text transformation to a message's content.

    For ``role=assistant`` (== ``model``), every text chunk is passed through
    ``strip_thinking``. For all other roles every text chunk is plain
    ``str.strip()``-ed. The result is the joined text.
    """
    is_model = message["role"] == "assistant"
    transform = _strip_thinking if is_model else str.strip
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return transform(content)
    parts: list[str] = []
    for part in content:
        if part["type"] == "text":
            parts.append(transform(part["text"]))
    return "".join(parts)


class Gemma4Renderer(Renderer):
    """Renderer matching the official Gemma 4 chat template byte-for-byte
    for the text-only path including tools and thinking.

    Args:
        tokenizer: a Gemma 4 tokenizer (must recognize the special tokens
            ``<|turn>``/``<turn|>``, ``<|tool>``/``<tool|>``,
            ``<|tool_call>``/``<tool_call|>``, etc.).
        enable_thinking: when True, the system block is forced and prefixed
            with the ``<|think|>`` marker that gates thinking-mode generation.
            Mirrors the template's ``enable_thinking`` argument.

    Thread safety: instances are immutable after construction and safe to
    share across threads. The underlying tokenizer must also be thread-safe.
    """

    def __init__(self, tokenizer: Tokenizer, *, enable_thinking: bool = False):
        super().__init__(tokenizer)
        self.enable_thinking = enable_thinking
        self._bos_token_id = self._encode_single_special(_BOS_TOKEN)
        self._turn_close_id = self._encode_single_special(_TURN_CLOSE)
        # `<|tool_response>` (id 50) is the second valid end-of-message signal:
        # the model emits it immediately after `<tool_call|>` to hand off to
        # the tool runner. Both vLLM (`gemma4_utils.has_tool_response_tag`)
        # and SGLang treat it as the natural completion marker for a tool-
        # calling turn, and we have to as well — otherwise `parse_response`
        # silently fails on tool-call outputs and the sampler doesn't know
        # when to stop.
        self._tool_response_open_id = self._encode_single_special(_TOOL_RESPONSE_OPEN)
        # Validated for fail-fast even though the IDs aren't read elsewhere:
        # an unexpected multi-token encoding signals the wrong tokenizer.
        self._encode_single_special(_TURN_OPEN)

    def _encode_single_special(self, token_str: str) -> int:
        token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
        assert len(token_ids) == 1, (
            f"Expected single token for {token_str!r}, got {token_ids}. "
            "This tokenizer does not look like a Gemma 4 tokenizer."
        )
        return token_ids[0]

    @property
    def has_extension_property(self) -> bool:
        # No history rewriting beyond the per-message strip_thinking, which
        # only ever shortens content, so the prefix-extension property holds
        # at the conversation-prefix level used by build_supervised_example.
        return True

    @property
    def _bos_tokens(self) -> list[int]:
        return [self._bos_token_id]

    @property
    def _end_message_token(self) -> int:
        return self._turn_close_id

    @staticmethod
    def _gemma_role_for(message: Message) -> str:
        """Map the message role to the literal string emitted in the turn header."""
        return _MODEL_ROLE if message["role"] == "assistant" else message["role"]

    @staticmethod
    def _is_tool_response_only(message: Message) -> bool:
        """Mirror the template's
        ``message['tool_responses'] and not message['content']`` predicate.

        When True, the trailing ``<turn|>\\n`` is suppressed and the next
        generation prompt is also suppressed.
        """
        return bool(message.get("tool_responses")) and not message.get("content")

    # ------------------------------------------------------------------
    # System-block / preprocessing
    # ------------------------------------------------------------------

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Return a synthetic system message bundling ``system_prompt`` and the
        serialized tool block.

        The returned message can be prepended to a conversation directly. The
        Gemma 4 template puts both the trimmed system text and the tool
        declarations into the same ``<|turn>system\\n...<turn|>`` block, with no
        whitespace between them, so we encode that joined string as the body
        of a single synthetic system message. ``system_prompt`` is trimmed
        first to match the template's ``messages[0]['content'] | trim``.

        ``tools`` accepts the OpenAI-style ``{"type": "function", "function":
        {...}}`` dicts that HF's ``apply_chat_template(..., tools=...)``
        consumes. Plain ``ToolSpec`` dicts (``{"name", "description",
        "parameters"}``) are also accepted and wrapped automatically.
        """
        normalized: list[dict] = []
        for tool in tools or []:
            if isinstance(tool, dict) and "function" in tool:
                normalized.append(tool)
            else:
                normalized.append({"type": "function", "function": tool})
        bundled = (system_prompt or "").strip() + _format_tool_block(normalized)
        return [Message(role="system", content=bundled)]

    def _preprocess_messages(self, messages: list[Message]) -> list[Message]:
        """Inject the ``<|think|>`` marker when ``enable_thinking`` is True.

        Mirrors the template's ``if enable_thinking ... <|think|>`` block at
        the very top of the first system turn. If the conversation has no
        system message, an empty one is synthesized just to carry the marker.
        """
        if not self.enable_thinking:
            return list(messages)

        result: list[Message] = list(messages)
        if result and result[0]["role"] in ("system", "developer"):
            sys_msg = result[0]
            content = sys_msg.get("content", "")
            if isinstance(content, str):
                # Trim FIRST to match the template's `messages[0]['content'] | trim`,
                # then prepend the think marker so the marker hugs the role header.
                new_content: Any = _THINK_TOKEN + content.strip()
            else:
                new_content = [TextPart(type="text", text=_THINK_TOKEN), *content]
            result[0] = {**sys_msg, "content": new_content}
        else:
            result.insert(0, Message(role="system", content=_THINK_TOKEN))
        return result

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        return super().build_generation_prompt(self._preprocess_messages(messages), role=role, prefill=prefill)

    def build_supervised_example(self, messages, train_on_what=None):  # type: ignore[override]
        from tinker_cookbook.renderers.base import TrainOnWhat

        if train_on_what is None:
            train_on_what = TrainOnWhat.LAST_ASSISTANT_MESSAGE
        return super().build_supervised_example(
            self._preprocess_messages(messages), train_on_what=train_on_what
        )

    # ------------------------------------------------------------------
    # render_message: tool_calls + tool_responses + content
    # ------------------------------------------------------------------

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        del ctx
        role = self._gemma_role_for(message)
        header_str = f"{_TURN_OPEN}{role}\n"

        body_parts: list[str] = []

        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                body_parts.append(_format_tool_call(tc))

        tool_responses = message.get("tool_responses")
        if tool_responses:
            for tr in tool_responses:
                body_parts.append(_format_tool_response(tr))

        text = _render_text_content(message)
        if text:
            body_parts.append(text)

        # Per the template, the trailing `<turn|>\n` is omitted when the
        # message is purely a tool response with no content. Otherwise it
        # closes every turn.
        if not self._is_tool_response_only(message):
            body_parts.append(f"{_TURN_CLOSE}\n")
        body_str = "".join(body_parts)

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False),
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(body_str, add_special_tokens=False),
            ),
        ]
        return RenderedMessage(header=header, output=output)

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        # The official template suppresses the generation prompt entirely
        # when the previous message was a tool response, leaving the model to
        # continue inside the same turn the tool call started.
        if ctx.prev_message is not None and self._is_tool_response_only(ctx.prev_message):
            return []
        suffix_str = f"{_TURN_OPEN}{_MODEL_ROLE}\n"
        return self.tokenizer.encode(suffix_str, add_special_tokens=False)

    def get_stop_sequences(self) -> list[int]:
        # Both `<turn|>` (normal close) and `<|tool_response>` (tool-call
        # handoff) are valid completion signals. Sampling without the second
        # would let the model run past the tool call into garbage when the
        # serving stack is supposed to inject the tool result there.
        return [self._turn_close_id, self._tool_response_open_id]

    # ------------------------------------------------------------------
    # parse_response
    # ------------------------------------------------------------------

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """Parse sampled tokens into an assistant Message.

        Decodes up to the first end-of-message signal — either ``<turn|>``
        (normal close) or ``<|tool_response>`` (tool-call handoff signal,
        which the model emits immediately after ``<tool_call|>`` when it
        wants the runner to inject a tool result). Then:

        * Pulls every ``<|tool_call>...<tool_call|>`` block out of the body
          into a structured ``tool_calls`` list on the message (raw arg
          string preserved as the ``arguments`` field — downstream code can
          parse it further with the SGLang/vLLM gemma4 detector if needed).
        * Splits any ``<|channel>...<channel|>`` thinking blocks out of the
          remaining body into ``ThinkingPart`` content parts so the
          render → sample → parse round-trip preserves structured content.
        """
        # Find the earliest occurrence of either terminator. We can't use
        # parse_response_for_stop_token directly because it asserts there's
        # exactly one stop token of a single ID, but a real model output
        # may legitimately contain `<|tool_response>` after `<tool_call|>`.
        end_idx = len(response)
        for terminator in (self._turn_close_id, self._tool_response_open_id):
            try:
                idx = response.index(terminator)
            except ValueError:
                continue
            if idx < end_idx:
                end_idx = idx
        parse_success = end_idx < len(response)
        text = self.tokenizer.decode(response[:end_idx])
        assistant_message: Message = Message(role="assistant", content=text)
        if not parse_success:
            return assistant_message, False

        # Pull out tool calls first so they don't pollute the text path.
        tool_calls_raw: list[dict] = []
        if _TOOL_CALL_OPEN in text:
            def _capture(m: re.Match) -> str:
                tool_calls_raw.append(
                    {"name": m.group(1), "arguments": m.group(2)}
                )
                return ""

            text = _TOOL_CALL_RE.sub(_capture, text)

        if tool_calls_raw:
            # Surface the parsed calls on the message in a forward-compatible
            # shape. We avoid importing tinker's ToolCall pydantic model here
            # to keep the renderer dependency-free; downstream consumers can
            # convert these dicts to their preferred tool-call type.
            assistant_message["tool_calls"] = tool_calls_raw  # type: ignore[typeddict-item]

        if _CHANNEL_OPEN not in text:
            assistant_message["content"] = text
            return assistant_message, True

        # Split into alternating text / thinking parts in order.
        parts: list[ThinkingPart | TextPart] = []
        cursor = 0
        for match in _THINKING_BLOCK_RE.finditer(text):
            before = text[cursor : match.start()]
            if before:
                parts.append(TextPart(type="text", text=before))
            inner = match.group(0)[len(_CHANNEL_OPEN) : -len(_CHANNEL_CLOSE)]
            parts.append(ThinkingPart(type="thinking", thinking=inner))
            cursor = match.end()
        tail = text[cursor:]
        if tail:
            parts.append(TextPart(type="text", text=tail))
        assistant_message["content"] = parts
        return assistant_message, True


def _gemma4_factory(tokenizer: Tokenizer, image_processor=None) -> Gemma4Renderer:
    # ``image_processor`` is part of the register_renderer factory contract
    # but Gemma 4 text-only does not use it.
    return Gemma4Renderer(tokenizer)


register_renderer("gemma4", _gemma4_factory)
