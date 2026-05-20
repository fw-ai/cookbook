"""Renderer for Mistral / Ministral chat templates.

This matches the HuggingFace ``mistralai/Ministral-3-3B-Instruct-2512`` (and other
Mistral / Ministral models that ship the same Tekken-style chat template):

- Sequence starts with the tokenizer's BOS token
- Optional ``[SYSTEM_PROMPT]<content>[/SYSTEM_PROMPT]`` block (default system
  prompt is auto-injected when the conversation has no system message — exactly
  matching the chat template's behavior)
- Optional ``[AVAILABLE_TOOLS]<json>[/AVAILABLE_TOOLS]`` block emitted after the
  system block when tools are supplied (via the renderer's ``_pending_tools``
  hook used by ``create_conversation_prefix_with_tools``)
- User turns: ``[INST]<content>[/INST]``
- Assistant turns: ``<content>[TOOL_CALLS]<name>[ARGS]<args>...</s>``  — tool
  calls (if any) are appended directly with no separator between calls; ``</s>``
  terminates the turn
- Tool turns: ``[TOOL_RESULTS]<content>[/TOOL_RESULTS]``

Generation does not require an explicit suffix — the model is expected to
continue right after ``[/INST]`` (or after a ``[TOOL_RESULTS]`` block).

Image content (``[IMG]`` token) is not yet supported; pass image-bearing
messages via the multimodal SFT path or override ``render_message`` for that.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import tinker
import torch
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
    parse_response_for_stop_token,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

_SYSTEM_PROMPT_OPEN = "[SYSTEM_PROMPT]"
_SYSTEM_PROMPT_CLOSE = "[/SYSTEM_PROMPT]"
_AVAILABLE_TOOLS_OPEN = "[AVAILABLE_TOOLS]"
_AVAILABLE_TOOLS_CLOSE = "[/AVAILABLE_TOOLS]"
_INST_OPEN = "[INST]"
_INST_CLOSE = "[/INST]"
_TOOL_CALLS = "[TOOL_CALLS]"
_TOOL_ARGS = "[ARGS]"
_TOOL_RESULTS_OPEN = "[TOOL_RESULTS]"
_TOOL_RESULTS_CLOSE = "[/TOOL_RESULTS]"

# Sentinel content marking a synthetic system message that should be filled with
# the chat template's default system prompt at render time. We use a sentinel
# rather than expanding eagerly so that any Message-pipeline code that runs
# before the renderer (e.g. ``_message_output_slices`` in tests) sees a stable,
# small message list — and so an explicitly empty user-supplied system message
# (``{"role": "system", "content": ""}``) still renders as
# ``[SYSTEM_PROMPT][/SYSTEM_PROMPT]`` (matching HF chat-template behavior).
_DEFAULT_SYSTEM_SENTINEL = "__MISTRAL_RENDERER_DEFAULT_SYSTEM__"


_TOOL_CALL_RE = re.compile(
    r"\[TOOL_CALLS\](?P<name>[^\[]+)\[ARGS\](?P<args>.+?)(?=\[TOOL_CALLS\]|$)",
    re.DOTALL,
)


def _visible_text(content: Any) -> str:
    """Render visible text from a string or structured content list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                rendered_parts.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                rendered_parts.append(item["text"])
                continue
            if isinstance(item.get("output"), str):
                rendered_parts.append(item["output"])
        return "".join(rendered_parts)
    return str(content)


def _normalize_tool_arguments(raw_arguments: Any) -> str:
    """Render tool-call arguments to the JSON string Mistral's template emits.

    The Jinja chat template runs ``arguments | tojson`` for non-string args and
    leaves string args untouched (treating ``""`` as ``"{}"``). Mirror that here
    so token-for-token parity holds.
    """
    if isinstance(raw_arguments, str):
        return "{}" if raw_arguments == "" else raw_arguments
    return json.dumps(raw_arguments, ensure_ascii=False)


def _format_tool_calls(tool_calls: list[ToolCall]) -> str:
    parts: list[str] = []
    for tool_call in tool_calls:
        rendered_args = _normalize_tool_arguments(tool_call.function.arguments)
        parts.append(
            f"{_TOOL_CALLS}{tool_call.function.name}{_TOOL_ARGS}{rendered_args}"
        )
    return "".join(parts)


def _format_tools_block(tools: list[ToolSpec | Mapping[str, Any]]) -> str:
    """Serialize a tools list into the ``[AVAILABLE_TOOLS]…`` block.

    Mistral's chat template runs ``tools | tojson`` over the raw input. We
    accept either OpenAI ``{"type": "function", "function": {...}}`` envelopes
    or bare-function dicts and emit them with ``json.dumps`` (no fancy
    sort/spacing — Python's default ``", "`` / ``": "`` separators happen to
    match the Jinja ``tojson`` filter for our test cases).
    """
    return _AVAILABLE_TOOLS_OPEN + json.dumps(list(tools)) + _AVAILABLE_TOOLS_CLOSE


class MistralRenderer(Renderer):
    """Renderer for Mistral / Ministral Tekken-style chat models.

    Validated against ``mistralai/Ministral-3-3B-Instruct-2512``; the format is
    shared by other recent Mistral / Ministral instruct checkpoints (Mistral 7B
    Instruct v0.3+, Ministral 8B 2410, Mistral Small 3 Instruct).
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__(tokenizer)
        self.default_system_prompt = self._detect_default_system_prompt()
        # Filled by ``create_conversation_prefix_with_tools`` and consumed by
        # ``render_message`` on the system turn so that the
        # ``[AVAILABLE_TOOLS]`` block lands in the right spot.
        self._pending_tools: list[Any] | None = None

    # ------------------------------------------------------------------
    # Special-token helpers
    # ------------------------------------------------------------------
    @property
    def _bos_tokens(self) -> list[int]:
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_token_id is None:
            return []
        return [int(bos_token_id)]

    @property
    def _eos_token_id(self) -> int:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise RuntimeError(
                "MistralRenderer requires the tokenizer to define an EOS token."
            )
        return int(eos_token_id)

    def _detect_default_system_prompt(self) -> str:
        """Extract the default system prompt baked into the chat template.

        We render a one-message ``user`` probe through ``apply_chat_template`` —
        empty-messages probes raise on Mistral templates — and slice out the
        text between ``[SYSTEM_PROMPT]`` and ``[/SYSTEM_PROMPT]``. This keeps
        the renderer in lockstep with whatever default the model ships without
        having to hardcode the (long, model-specific) string. If the template
        emits no system block when none is provided, we return an empty string
        so ``_render_system_message`` simply emits ``[SYSTEM_PROMPT][/SYSTEM_PROMPT]``.
        """
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_chat_template is None:
            return ""
        try:
            rendered = apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            return ""
        if not isinstance(rendered, str):
            return ""
        open_idx = rendered.find(_SYSTEM_PROMPT_OPEN)
        close_idx = rendered.find(_SYSTEM_PROMPT_CLOSE, open_idx + 1)
        if open_idx == -1 or close_idx == -1:
            return ""
        # Anchor the search before the first user/tool block so a system block
        # appearing later (e.g., a user-supplied probe) does not slip through.
        first_user_idx = rendered.find(_INST_OPEN)
        if first_user_idx != -1 and close_idx > first_user_idx:
            return ""
        return rendered[open_idx + len(_SYSTEM_PROMPT_OPEN) : close_idx]

    # ------------------------------------------------------------------
    # Message preprocessing
    # ------------------------------------------------------------------
    def _ensure_system_message(self, messages: list[Message]) -> list[Message]:
        """Synthesize a default-prompt system message when the dataset omits one.

        Mistral's chat template unconditionally emits a ``[SYSTEM_PROMPT]`` block
        when no system message is present, falling back to the model's baked-in
        default. We mirror that by prepending a synthetic system message tagged
        with ``_DEFAULT_SYSTEM_SENTINEL`` so ``render_message`` can substitute
        the default at render time.
        """
        if messages and messages[0]["role"] == "system":
            return list(messages)
        synthetic: Message = {
            "role": "system",
            "content": _DEFAULT_SYSTEM_SENTINEL,
        }
        return [synthetic, *messages]

    def _preprocess_messages(self, messages: list[Message]) -> list[Message]:
        return self._ensure_system_message(messages)

    # ------------------------------------------------------------------
    # Renderer ABC
    # ------------------------------------------------------------------
    @property
    def has_extension_property(self) -> bool:
        # Mistral has no thinking truncation; rendering message [0..n] is always
        # a prefix of [0..n+1].
        return True

    def get_stop_sequences(self) -> list[int]:
        return [self._eos_token_id]

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _render_system_message(self, message: Message) -> tuple[str, str]:
        """Return (header, output_body) text for a system message.

        Header carries ``[SYSTEM_PROMPT]`` (model sees but does not generate);
        body carries the system content followed by ``[/SYSTEM_PROMPT]`` plus
        the optional ``[AVAILABLE_TOOLS]`` block. Tools live on the system turn
        so all training-loss bookkeeping stays per-message.
        """
        raw = message["content"]
        if isinstance(raw, str) and raw == _DEFAULT_SYSTEM_SENTINEL:
            content = self.default_system_prompt
        else:
            content = _visible_text(raw)
        body = content + _SYSTEM_PROMPT_CLOSE
        if self._pending_tools:
            body += _format_tools_block(self._pending_tools)
            self._pending_tools = None
        return _SYSTEM_PROMPT_OPEN, body

    def _render_user_message(self, message: Message) -> tuple[str, str]:
        content = _visible_text(message["content"])
        return _INST_OPEN, content + _INST_CLOSE

    def _render_assistant_message(self, message: Message) -> tuple[str, str]:
        content = _visible_text(message["content"])
        body = content
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            body += _format_tool_calls(tool_calls)
        body += self.tokenizer.decode([self._eos_token_id], skip_special_tokens=False)
        # Header is empty for assistant turns.
        return "", body

    def _render_tool_message(self, message: Message) -> tuple[str, str]:
        content = _visible_text(message["content"])
        return _TOOL_RESULTS_OPEN, content + _TOOL_RESULTS_CLOSE

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        del ctx  # Mistral does not use context for any per-turn decisions.

        role = message["role"]
        if role == "system":
            header_text, body_text = self._render_system_message(message)
        elif role == "user":
            header_text, body_text = self._render_user_message(message)
        elif role == "assistant":
            header_text, body_text = self._render_assistant_message(message)
        elif role == "tool":
            header_text, body_text = self._render_tool_message(message)
        else:
            raise ValueError(f"Unsupported role for Mistral renderer: {role!r}")

        # Encode output as a single chunk so we end up with one EncodedTextChunk
        # per message body (matching how supervised loss-mask code consumes the
        # rendered output).
        output_tokens = self._encode(body_text)
        output_chunks: list[tinker.ModelInputChunk] = (
            [tinker.types.EncodedTextChunk(tokens=output_tokens)]
            if output_tokens
            else []
        )

        header_chunk: tinker.EncodedTextChunk | None = None
        header_tokens = self._encode(header_text)
        if header_tokens:
            header_chunk = tinker.types.EncodedTextChunk(tokens=header_tokens)

        return RenderedMessage(header=header_chunk, output=output_chunks)

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        del role, ctx
        # Mistral's chat template adds nothing for ``add_generation_prompt=True``;
        # the assistant continues directly after the previous turn's body.
        return []

    # ------------------------------------------------------------------
    # Generation / supervised entry points
    # ------------------------------------------------------------------
    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        return super().build_generation_prompt(
            self._preprocess_messages(messages),
            role=role,
            prefill=prefill,
        )

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        return super().build_supervised_example(
            self._preprocess_messages(messages),
            train_on_what=train_on_what,
        )

    # ------------------------------------------------------------------
    # Tool plumbing
    # ------------------------------------------------------------------
    def create_conversation_prefix_with_tools(
        self,
        tools: list[ToolSpec],
        system_prompt: str = "",
    ) -> list[Message]:
        """Stage tool specs to be emitted on the next system turn.

        Returns a single system message (carrying the user-supplied prompt or a
        marker for the default) and stashes the tool list on ``self`` so that
        ``render_message`` can append the ``[AVAILABLE_TOOLS]`` block at the
        right position. This matches the Jinja template, where tools are
        rendered immediately after the system block and before any user turn.
        """
        self._pending_tools = list(tools)
        if system_prompt:
            return [Message(role="system", content=system_prompt)]
        return [Message(role="system", content=_DEFAULT_SYSTEM_SENTINEL)]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def parse_response(
        self,
        response: list[int],
    ) -> tuple[Message, ParseTermination]:
        message, termination = parse_response_for_stop_token(
            response=response,
            tokenizer=self.tokenizer,
            stop_token=self._eos_token_id,
        )
        if termination == ParseTermination.MALFORMED:
            return message, termination

        assert isinstance(message["content"], str)
        text = message["content"]

        tool_calls: list[ToolCall] = []
        unparsed: list[UnparsedToolCall] = []
        first_call_idx = text.find(_TOOL_CALLS)
        if first_call_idx >= 0:
            visible_text = text[:first_call_idx]
            for match in _TOOL_CALL_RE.finditer(text, first_call_idx):
                name = match.group("name")
                raw_args = match.group("args")
                try:
                    parsed_args = json.loads(raw_args)
                    if not isinstance(parsed_args, dict):
                        raise TypeError(
                            f"Mistral tool args must decode to an object, got {type(parsed_args)!r}"
                        )
                    tool_calls.append(
                        ToolCall(
                            function=ToolCall.FunctionBody(
                                name=name,
                                arguments=json.dumps(parsed_args, ensure_ascii=False),
                            )
                        )
                    )
                except (json.JSONDecodeError, TypeError) as exc:
                    unparsed.append(
                        UnparsedToolCall(raw_text=match.group(0), error=str(exc))
                    )
        else:
            visible_text = text

        message["content"] = visible_text
        if tool_calls:
            message["tool_calls"] = tool_calls
        if unparsed:
            message["unparsed_tool_calls"] = unparsed
        return message, termination

    def to_openai_message(self, message: Message) -> dict[str, Any]:
        result: dict[str, Any] = {"role": message["role"]}
        content = message["content"]
        result["content"] = (
            content if isinstance(content, str) else _visible_text(content)
        )
        if "tool_calls" in message and message["tool_calls"]:
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": getattr(tool_call, "id", None),
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": _normalize_tool_arguments(
                            tool_call.function.arguments
                        ),
                    },
                }
                for tool_call in message["tool_calls"]
            ]
        if message["role"] == "tool":
            if "tool_call_id" in message:
                result["tool_call_id"] = message["tool_call_id"]
            if "name" in message:
                result["name"] = message["name"]
        return result


def _mistral_factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> MistralRenderer:
    del image_processor  # image content is not supported in this renderer yet
    return MistralRenderer(tokenizer)


register_renderer("mistral", _mistral_factory)
