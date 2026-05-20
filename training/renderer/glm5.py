"""Renderer for ZhipuAI GLM-5.1 chat template.

Handles the GLM-5.1 chat format as shipped with ``zai-org/GLM-5.1``
(and its FP8 variant ``zai-org/GLM-5.1-FP8``, which ships an identical
tokenizer and chat template).

Token-level layout follows ``tokenizer.apply_chat_template`` byte-for-byte
(verified by the unit tests in ``test_glm5_renderer.py``), modulo a synthetic
terminal role sentinel used only for supervised examples that end on an
assistant message. Historical assistant ``<think>`` blocks are always
stripped, matching the shipped chat template's default ``clear_thinking``
behavior (the same rendering every standard inference stack feeds the
model). Multi-turn ``ALL_ASSISTANT_MESSAGES`` SFT is handled by
disaggregating per user turn — see
:class:`training.renderer._disaggregate_mixin.DisaggregateMultiTurnMixin`.

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

- **Historical assistant turn** (any turn before the last user message;
  matches the shipped template's ``clear_thinking`` default)::

      <|assistant|></think>{content}

Other invariants:

- ``[gMASK]<sop>`` is emitted once at the very start of the conversation.
- The shipped Jinja template does **not** emit ``<|endoftext|>`` at message
  boundaries. Assistant turns stop by generating the next role sentinel:
  ``<|user|>`` for a normal assistant answer or ``<|observation|>`` for a
  tool-call handoff. For supervised examples, this renderer gives those role
  sentinels loss weight after trainable assistant turns. If a supervised row
  ends on an assistant message, it appends the appropriate sentinel as
  ``stop_overlap`` so the trained model still learns where to stop.
- Generation suffix (``add_generation_prompt=True`` in Jinja):
  ``<|assistant|><think>`` for thinking mode (default),
  ``<|assistant|></think>`` for non-thinking mode.
- In supervised examples, the opening ``<think>`` token is kept in the
  rendered sequence for template parity but masked out of the loss because it
  is already injected by the generation suffix.

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
from collections.abc import Mapping
from typing import Any

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
    parse_think_blocks,
)

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin
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


class GLM5Renderer(DisaggregateMultiTurnMixin, Renderer):
    """Renderer for ZhipuAI GLM-5.1 instruct models.

    Thinking is always stripped from historical assistant turns (matching
    the shipped chat template's default ``clear_thinking`` behavior, i.e.
    what every standard inference stack feeds the model). Multi-turn
    ``ALL_ASSISTANT_MESSAGES`` SFT is handled by
    :class:`DisaggregateMultiTurnMixin`, which splits the conversation
    per user turn so each datum's prompt context byte-equals what
    ``apply_chat_template`` produces for the same prefix.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__(tokenizer)

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode(_BOS_TEXT, add_special_tokens=False)

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

    @property
    def _think_open_token(self) -> int:
        return self._encode_single_special("<think>")

    def get_stop_sequences(self) -> list[int]:
        return [self._user_token, self._observation_token]

    def _assistant_stop_token(self, message: Message) -> int:
        return (
            self._observation_token
            if message.get("tool_calls")
            else self._user_token
        )

    def _assistant_stop_overlap(self, message: Message) -> tinker.EncodedTextChunk:
        return tinker.types.EncodedTextChunk(
            tokens=[self._assistant_stop_token(message)]
        )

    @staticmethod
    def _output_has_weight(
        message: Message,
        *,
        idx: int,
        is_last_message: bool,
        last_user_idx: int,
        train_on_what: TrainOnWhat,
    ) -> bool:
        is_assistant = message["role"] == "assistant"
        is_user_or_system = message["role"] in ["user", "system"]
        is_after_last_user = last_user_idx == -1 or idx > last_user_idx

        match train_on_what:
            case TrainOnWhat.LAST_ASSISTANT_MESSAGE:
                return is_last_message and is_assistant
            case TrainOnWhat.LAST_ASSISTANT_TURN:
                return is_assistant and is_after_last_user
            case TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                return is_assistant
            case TrainOnWhat.ALL_MESSAGES:
                return True
            case TrainOnWhat.ALL_TOKENS:
                return True
            case TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES:
                return is_user_or_system
            case TrainOnWhat.CUSTOMIZED:
                return bool(message.get("trainable", False))
            case _:
                raise ValueError(f"Unknown train_on_what: {train_on_what}")

    def _header_is_stop_for_previous_assistant(
        self,
        messages: list[Message],
        *,
        idx: int,
        last_user_idx: int,
        train_on_what: TrainOnWhat,
    ) -> bool:
        if idx == 0:
            return False

        prev_message = messages[idx - 1]
        if prev_message["role"] != "assistant":
            return False

        prev_has_weight = self._output_has_weight(
            prev_message,
            idx=idx - 1,
            is_last_message=(idx - 1 == len(messages) - 1),
            last_user_idx=last_user_idx,
            train_on_what=train_on_what,
        )
        if not prev_has_weight:
            return False

        current_role = messages[idx]["role"]
        expected_role = "tool" if prev_message.get("tool_calls") else "user"
        return current_role == expected_role

    def _append_output_chunks_with_weights(
        self,
        model_input_chunks_weights: list[tuple[tinker.ModelInputChunk, float]],
        *,
        message: Message,
        output_parts: list[tinker.ModelInputChunk],
        output_has_weight: bool,
        train_on_what: TrainOnWhat,
    ) -> None:
        for output_part in output_parts:
            if not output_part:
                continue

            if (
                message["role"] == "assistant"
                and output_has_weight
                and train_on_what != TrainOnWhat.ALL_TOKENS
                and isinstance(output_part, tinker.types.EncodedTextChunk)
                and output_part.tokens
                and int(output_part.tokens[0]) == self._think_open_token
            ):
                # ``add_generation_prompt=True`` injects
                # ``<|assistant|><think>``. Keep the token in the rendered
                # sequence for template parity, but mask it because the model
                # starts generating after that prefix.
                model_input_chunks_weights.append(
                    (tinker.types.EncodedTextChunk(tokens=[output_part.tokens[0]]), 0.0)
                )
                if len(output_part.tokens) > 1:
                    model_input_chunks_weights.append(
                        (
                            tinker.types.EncodedTextChunk(
                                tokens=list(output_part.tokens[1:])
                            ),
                            1.0,
                        )
                    )
            else:
                model_input_chunks_weights.append(
                    (output_part, int(output_has_weight))
                )

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """Build a GLM supervised example with role-sentinel stop weights.

        GLM-5.1 uses the next role tag, not EOS, as the assistant stop marker.
        The base renderer masks headers, so it would not train ``<|user|>`` or
        ``<|observation|>`` after historical assistant turns. This override
        preserves the base token layout while assigning loss to those sentinels
        when they close a trainable assistant turn.
        """
        model_input_chunks_weights: list[tuple[tinker.ModelInputChunk, float]] = []
        if self._bos_tokens:
            model_input_chunks_weights.append(
                (tinker.types.EncodedTextChunk(tokens=self._bos_tokens), 0.0)
            )

        last_user_idx = max(
            (idx for idx, message in enumerate(messages) if message["role"] == "user"),
            default=-1,
        )

        for idx, message in enumerate(messages):
            if train_on_what == TrainOnWhat.CUSTOMIZED:
                assert "trainable" in message, (
                    "When using CUSTOMIZED train_on_what, each message must have "
                    "a trainable field."
                )
            else:
                assert "trainable" not in message, (
                    "When using non-CUSTOMIZED train_on_what, each message must "
                    "not have a trainable field."
                )

            is_last_message = idx == len(messages) - 1
            ctx = RenderContext(
                idx=idx,
                is_last=is_last_message,
                prev_message=messages[idx - 1] if idx > 0 else None,
                last_user_index=last_user_idx,
            )
            rendered_message = self.render_message(message, ctx)

            output_has_weight = self._output_has_weight(
                message,
                idx=idx,
                is_last_message=is_last_message,
                last_user_idx=last_user_idx,
                train_on_what=train_on_what,
            )

            header_part = rendered_message.header
            if header_part:
                header_weight = int(train_on_what == TrainOnWhat.ALL_TOKENS)
                if self._header_is_stop_for_previous_assistant(
                    messages,
                    idx=idx,
                    last_user_idx=last_user_idx,
                    train_on_what=train_on_what,
                ):
                    header_weight = 1
                model_input_chunks_weights.append((header_part, header_weight))

            self._append_output_chunks_with_weights(
                model_input_chunks_weights,
                message=message,
                output_parts=rendered_message.output,
                output_has_weight=output_has_weight,
                train_on_what=train_on_what,
            )

            if is_last_message and rendered_message.stop_overlap:
                model_input_chunks_weights.append(
                    (rendered_message.stop_overlap, int(output_has_weight))
                )

        weights_data = [
            w for chunk, w in model_input_chunks_weights for _ in range(chunk.length)
        ]
        weights_tensor = torch.tensor(weights_data)
        model_input_chunks = [chunk for chunk, _ in model_input_chunks_weights]
        return tinker.ModelInput(chunks=model_input_chunks), weights_tensor

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
        # The role tag ``<|assistant|>`` is the header. Thinking-mode
        # assistants keep the template's opening ``<think>`` in ``output`` for
        # token parity, but supervised rendering masks that prefix because
        # ``add_generation_prompt=True`` injects it before model generation.
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
        # 2. Historical turn (always stripped to match HF
        #    ``apply_chat_template`` default): ``</think>``.
        # 3. Terminal turn with reasoning: ``<think>{reasoning}</think>``.
        # 4. Terminal turn without reasoning (thinking-mode default):
        #    ``<think></think>``.
        if before_last_user:
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

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False),
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False),
            ),
        ]
        return RenderedMessage(
            header=header,
            output=output,
            stop_overlap=(
                self._assistant_stop_overlap(message) if ctx.is_last else None
            ),
        )

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
