"""Renderer for ZhipuAI GLM-5.1 chat template.

Handles the GLM-5.1 chat format as shipped with ``zai-org/GLM-5.1``
(and its FP8 variant ``zai-org/GLM-5.1-FP8``, which ships an identical
tokenizer and chat template).

Token-level layout follows ``tokenizer.apply_chat_template`` byte-for-byte
(verified by the unit tests in ``test_glm5_renderer.py``), modulo the
intentional training EOS on the terminal assistant message. The registered
renderer uses GLM preserved-thinking semantics by default; opt-in
``strip_thinking_from_history=True`` matches the template's standard
``clear_thinking`` behavior.

Role tag layout (as the shipped Jinja template emits them):

- ``<|system|>{content}``  — no newline after the tag, no newline before the next tag
- ``<|user|>{content}``    — same
- ``<|assistant|>...``     — see below
- ``<|observation|>{content}`` — same

Assistant turn layout:

- **Terminal turn, ``enable_thinking=True`` (default), no reasoning content**::

      <|assistant|><think></think>{content}

- **Terminal turn, reasoning content provided**::

      <|assistant|><think>{reasoning}</think>{content}

- **Terminal turn, ``enable_thinking=False``** (or non-thinking mode) — the
  shipped template emits ``</think>`` alone so the model skips the think
  phase::

      <|assistant|></think>{content}

- **Historical assistant turn** (any turn before the last user message) when
  ``strip_thinking_from_history=False`` (the default) and reasoning content is
  provided::

      <|assistant|><think>{reasoning}</think>{content}

- **Historical assistant turn** when ``strip_thinking_from_history=True``::

      <|assistant|></think>{content}

  This opt-in strip-history mode matches the shipped template's default
  ``clear_thinking`` behavior.

Other invariants:

- ``[gMASK]<sop>`` is emitted once at the very start of the conversation.
- The shipped Jinja template does **not** emit ``<|endoftext|>`` at message
  boundaries. This renderer appends ``<|endoftext|>`` only to the *last*
  message in the conversation so the trained model learns when to stop;
  historical assistant turns still match the Jinja byte-for-byte without
  any EOS adjustment.
- Generation suffix (``add_generation_prompt=True`` in Jinja):
  ``<|assistant|><think>`` for thinking mode (default),
  ``<|assistant|></think>`` for non-thinking mode.

Tool-call layout (assistant turns only — ``role: "tool"`` responses are
rendered as ``<|observation|><tool_response>...</tool_response>`` and never
contribute to loss):

- Each call serialised right after the assistant's visible content with no
  separator: ``<tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>``.
- Multiple calls in one assistant turn are concatenated end-to-end.
- ``arguments`` is the JSON-string form Tinker's ``ToolCall`` schema uses;
  string values are emitted raw, anything else is JSON-encoded with
  ``ensure_ascii=False`` (matches the shipped Jinja's ``v | tojson`` branch).

Multimodal content is left for a future extension.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Mapping
from typing import Any

import tinker
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
    parse_think_blocks,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

_BOS_TEXT = "[gMASK]<sop>"
_USER_TEXT = "<|user|>"
_OBSERVATION_TEXT = "<|observation|>"
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>",
    re.DOTALL,
)


def _visible_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif (
                isinstance(item, Mapping)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def _format_arg_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _format_tool_calls(tool_calls: list[ToolCall]) -> str:
    parts: list[str] = []
    for tc in tool_calls:
        raw_args = tc.function.arguments
        args = json.loads(raw_args) if raw_args else {}
        if not isinstance(args, Mapping):
            raise TypeError(
                f"GLM-5.1 tool arguments must be a JSON object, got {type(args)!r}"
            )
        kv = "".join(
            f"<arg_key>{k}</arg_key><arg_value>{_format_arg_value(v)}</arg_value>"
            for k, v in args.items()
        )
        parts.append(f"<tool_call>{tc.function.name}{kv}</tool_call>")
    return "".join(parts)


def _parse_arg_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_tool_call(
    raw_body: str,
    raw_text: str,
) -> tuple[ToolCall | None, UnparsedToolCall | None]:
    first_arg = raw_body.find("<arg_key>")
    name = raw_body[:first_arg].strip() if first_arg >= 0 else raw_body.strip()
    if not name:
        return None, UnparsedToolCall(raw_text=raw_text, error="No tool name found")

    arguments = {
        match.group(1): _parse_arg_value(match.group(2))
        for match in _TOOL_ARG_RE.finditer(raw_body)
    }
    cleaned_body = _TOOL_ARG_RE.sub("", raw_body).strip()
    trailing = cleaned_body.removeprefix(name).strip()
    unparsed = (
        UnparsedToolCall(
            raw_text=raw_text,
            error=f"Unexpected content inside <tool_call>: {trailing!r}",
        )
        if trailing
        else None
    )
    return (
        ToolCall(
            function=ToolCall.FunctionBody(
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            )
        ),
        unparsed,
    )


def _extract_tool_calls_from_content(
    content: str,
) -> tuple[str, list[ToolCall], list[UnparsedToolCall]]:
    cleaned_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    unparsed_tool_calls: list[UnparsedToolCall] = []
    pos = 0
    for match in _TOOL_CALL_RE.finditer(content):
        cleaned_parts.append(content[pos : match.start()])
        tool_call, unparsed = _parse_tool_call(match.group(1), match.group(0))
        if tool_call is not None:
            tool_calls.append(tool_call)
        if unparsed is not None:
            unparsed_tool_calls.append(unparsed)
        pos = match.end()
    cleaned_parts.append(content[pos:])
    return "".join(cleaned_parts), tool_calls, unparsed_tool_calls


def _extract_reasoning_and_text(content: Any) -> tuple[str, str]:
    """Return ``(reasoning, visible_text)`` for an assistant message."""
    if isinstance(content, str):
        if "</think>" not in content:
            return "", content
        reasoning = content.split("</think>")[0].split("<think>")[-1].strip("\n")
        visible = content.split("</think>")[-1].lstrip("\n")
        return reasoning, visible

    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "thinking" and isinstance(part.get("thinking"), str):
                reasoning_parts.append(part["thinking"])
            elif part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
    return "".join(reasoning_parts), "".join(text_parts)


class GLM5Renderer(Renderer):
    """Renderer for ZhipuAI GLM-5.1 instruct models."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        strip_thinking_from_history: bool = False,
    ) -> None:
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history

    @property
    def has_extension_property(self) -> bool:
        return not self.strip_thinking_from_history

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode(_BOS_TEXT, add_special_tokens=False)

    @property
    def _end_message_token(self) -> int:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            raise RuntimeError(
                "GLM5Renderer requires tokenizer.eos_token_id to be set "
                "(expected <|endoftext|>)."
            )
        return int(eos)

    def _encode_single_special(self, token_str: str) -> int:
        token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(
                f"GLM5Renderer expected {token_str!r} to encode as one token, "
                f"got {token_ids}."
            )
        return int(token_ids[0])

    @property
    def _user_token(self) -> int:
        return self._encode_single_special(_USER_TEXT)

    @property
    def _observation_token(self) -> int:
        return self._encode_single_special(_OBSERVATION_TEXT)

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token, self._user_token, self._observation_token]

    def build_supervised_examples(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_TURN,
    ):
        """Build extension-safe supervised examples for multi-turn GLM data.

        The registered GLM renderer preserves historical thinking by default,
        so it can train all assistant messages in one datum. If callers opt
        into strip-history mode, split by user turns and train each assistant
        suffix in the same position it would occupy during generation.
        """
        if self.has_extension_property:
            return [
                self.build_supervised_example(
                    messages,
                    train_on_what=train_on_what,
                )
            ]

        if train_on_what in (
            TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            TrainOnWhat.LAST_ASSISTANT_TURN,
        ):
            return [
                self.build_supervised_example(
                    messages,
                    train_on_what=train_on_what,
                )
            ]

        user_message_idxs = [
            idx for idx, message in enumerate(messages) if message["role"] == "user"
        ]

        if train_on_what != TrainOnWhat.ALL_ASSISTANT_MESSAGES:
            warnings.warn(
                "WARNING: Using train_on_what=ALL_MESSAGES/ALL_TOKENS/"
                "ALL_USER_AND_SYSTEM_MESSAGES/CUSTOMIZED with a renderer that "
                "does not satisfy the extension property "
                "(has_extension_property=False). The same train_on_what mode is "
                "applied to each user-turn prefix.",
                UserWarning,
                stacklevel=2,
            )

        supervised_examples = []
        for user_message_idx in [*user_message_idxs[1:], len(messages)]:
            current_messages = messages[:user_message_idx]
            if train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                supervised_examples.append(
                    self.build_supervised_example(
                        current_messages,
                        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
                    )
                )
            else:
                supervised_examples.append(
                    self.build_supervised_example(
                        current_messages,
                        train_on_what=train_on_what,
                    )
                )
        return supervised_examples

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]
        if role == "assistant":
            return self._render_assistant(message, ctx)
        # GLM-5.1 role tags do not have a trailing newline; content is
        # concatenated directly (e.g. ``<|user|>hello``).
        if role == "user":
            header_str = "<|user|>"
            output_str = _visible_text(message["content"])
        elif role == "system":
            header_str = "<|system|>"
            output_str = _visible_text(message["content"])
        elif role == "tool":
            prev_is_tool = (
                ctx.prev_message is not None
                and ctx.prev_message.get("role") == "tool"
            )
            header_str = "" if prev_is_tool else "<|observation|>"
            output_str = (
                f"<tool_response>{_visible_text(message['content'])}</tool_response>"
            )
        else:
            raise ValueError(f"GLM5Renderer: unsupported role {role!r}")

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False),
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False),
            )
        ]
        return RenderedMessage(header=header, output=output)

    def _render_assistant(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        # The role tag ``<|assistant|>`` is the header; the ``<think>...
        # </think>`` block lives in ``output`` so it is part of the training
        # target for this assistant turn.
        header_str = "<|assistant|>"

        before_last_user = (
            ctx.last_user_index >= 0
            and ctx.idx < ctx.last_user_index
        )

        reasoning, visible = _extract_reasoning_and_text(message["content"])

        # Match the shipped HF chat template thinking-block logic:
        #
        # 1. Historical turn with strip_thinking=True:
        #    always ``</think>`` (drops any reasoning).
        # 2. Historical turn with strip_thinking=False AND reasoning
        #    exists: ``<think>{reasoning}</think>`` (keep it).
        # 3. Historical turn with strip_thinking=False AND no reasoning:
        #    ``</think>`` — the template leaves ``reasoning_content``
        #    undefined so it falls to the else branch.
        # 4. Terminal turn with reasoning: ``<think>{reasoning}</think>``.
        # 5. Terminal turn without reasoning (thinking-mode default):
        #    ``<think></think>``.
        if before_last_user and self.strip_thinking_from_history:
            think_block = "</think>"
        elif before_last_user and not reasoning:
            think_block = "</think>"
        elif reasoning:
            think_block = f"<think>{reasoning.strip()}</think>"
        else:
            think_block = "<think></think>"

        visible_stripped = visible.strip()
        output_str = think_block + visible_stripped

        tool_calls = message.get("tool_calls")
        if tool_calls:
            output_str += _format_tool_calls(tool_calls)

        output_tokens = self.tokenizer.encode(output_str, add_special_tokens=False)
        # Append <|endoftext|> only on the final message in the conversation.
        # Historical assistant turns are delimited only by the next role tag
        # in the GLM Jinja template, so emitting EOS there would add a token
        # the template never produces. The final turn still gets EOS so the
        # trained model learns when to stop.
        if ctx.is_last:
            output_tokens.append(self._end_message_token)

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False),
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(tokens=output_tokens),
        ]
        return RenderedMessage(header=header, output=output)

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        del ctx
        # For the assistant role, match the shipped template's
        # ``add_generation_prompt=True`` thinking-mode output:
        # ``<|assistant|><think>``. The model produces the rest of the think
        # block + ``</think>`` + visible content itself.
        if role == "assistant":
            suffix_str = "<|assistant|><think>"
        else:
            suffix_str = f"<|{role}|>"
        return self.tokenizer.encode(suffix_str, add_special_tokens=False)

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        end_idx = len(response)
        for stop_token in self.get_stop_sequences():
            try:
                idx = response.index(stop_token)
            except ValueError:
                continue
            end_idx = min(end_idx, idx)
        ok = end_idx < len(response)
        assistant_message = Message(
            role="assistant",
            content=str(self.tokenizer.decode(response[:end_idx])),
        )
        if not ok:
            return assistant_message, False
        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"].lstrip("\n")
        content, tool_calls, unparsed_tool_calls = _extract_tool_calls_from_content(
            content
        )
        parts = parse_think_blocks(content)
        assistant_message["content"] = parts if parts is not None else content
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if unparsed_tool_calls:
            assistant_message["unparsed_tool_calls"] = unparsed_tool_calls
        return assistant_message, True


def _glm5_factory(tokenizer: Tokenizer, image_processor=None) -> GLM5Renderer:
    del image_processor
    return GLM5Renderer(tokenizer)


register_renderer("glm5", _glm5_factory)
