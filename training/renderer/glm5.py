"""Renderers for ZhipuAI GLM-5.x chat templates.

Handles the GLM-5.1 chat format as shipped with ``zai-org/GLM-5.1``
(and its FP8 variant ``zai-org/GLM-5.1-FP8``, which ships an identical
tokenizer and chat template).

Token-level layout follows ``tokenizer.apply_chat_template`` byte-for-byte
(verified by the unit tests in ``test_glm5_renderer.py``), modulo a synthetic
terminal role sentinel used only for supervised examples that end on an
assistant message. Historical assistant ``<think>`` blocks are stripped by
default, matching the shipped chat template's default ``clear_thinking``
behavior. The registered ``*_preserve_thinking`` variants instead match
``apply_chat_template(clear_thinking=False)`` and keep reasoning across user
turns. Multi-turn ``ALL_ASSISTANT_MESSAGES`` SFT in the default strip mode is
handled by disaggregating per user turn — see
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

GLM-5.2 reuses the same role tags, stop tags, and tool-call layout, but its
upstream template injects a default ``Reasoning Effort: Max`` system line and
uses ``<think></think>`` for stripped historical reasoning. The
``glm_moe_dsa`` renderer id selects that template variant while sharing the
same renderer implementation.

GLM-5.3 preserves historical reasoning by default and canonicalizes a
well-formed consecutive tool-result block into the preceding assistant turn's
tool-call order. The ``glm53`` renderer keeps that behavior separate from the
legacy GLM-5.2 contract. GLM-5.3-Flash shares that text contract and adds real
multimodal branches: its dedicated renderer preserves interleaved images in
user and tool messages as Tinker ``ImageChunk`` objects. The training backend
expands each chunk to GLM's begin/image/end token envelope after pixel
preprocessing.

Video content is left for a future extension because the public training
request schema does not yet carry a self-contained video or sampled-frame
unit.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

import tinker
import torch
from transformers.image_processing_utils import BaseImageProcessor
from training.renderer import RendererError, register_renderer
from training._vendor.tinker_cookbook_0_4_3.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
    image_to_chunk,
    parse_think_blocks,
)

from training.renderer._disaggregate_mixin import DisaggregateMultiTurnMixin
from training.renderer.reasoning_fields import original_reasoning_content
from training.renderer.tokenizer import Tokenizer

_BOS_TEXT = "[gMASK]<sop>"
_USER_TEXT = "<|user|>"
_OBSERVATION_TEXT = "<|observation|>"
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>",
    re.DOTALL,
)
# Mirrors the tools branch in zai-org/GLM-5.1's tokenizer chat_template.
_TOOL_DECLARATION_PREFIX = """\

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
"""
_TOOL_DECLARATION_SUFFIX = """\
</tools>

For each function call, output the function name and arguments within the following XML format:
""" + (
    "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>"
    "<arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>"
    "<arg_value>{arg-value-2}</arg_value>...</tool_call>"
)


class Glm53FlashImageTokenCounter:
    """Count GLM-5.3-Flash image tokens from the released processor config.

    Transformers 5.5.4 can read the nested config but does not yet register
    ``Glm5NextImageProcessor``. FireTitan owns pixel preprocessing; rendering
    only needs the exact patch count declared on each wire ``ImageChunk``.
    """

    def __init__(
        self,
        *,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_image_tokens: int = 16,
        max_image_tokens: int = 8000,
    ) -> None:
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.merge_size = int(merge_size)
        self.min_image_tokens = int(min_image_tokens)
        self.max_image_tokens = int(max_image_tokens)
        if min(
            self.patch_size,
            self.temporal_patch_size,
            self.merge_size,
            self.min_image_tokens,
        ) <= 0:
            raise ValueError(
                "GLM-5.3-Flash image processor dimensions must be positive"
            )
        if self.max_image_tokens < self.min_image_tokens:
            raise ValueError(
                "GLM-5.3-Flash max_image_tokens must be at least min_image_tokens"
            )

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Glm53FlashImageTokenCounter":
        config, _ = BaseImageProcessor.get_image_processor_dict(model_name)
        return cls(
            patch_size=config.get("patch_size", 14),
            temporal_patch_size=config.get("temporal_patch_size", 2),
            merge_size=config.get("merge_size", 2),
            min_image_tokens=config.get("min_image_tokens", 16),
            max_image_tokens=config.get("max_image_tokens", 8000),
        )

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
        images_kwargs: Mapping[str, Any] | None = None,
    ) -> int:
        """Match the official processor's aligned-canvas patch count."""
        if height <= 0 or width <= 0:
            raise ValueError(
                f"Image dimensions must be positive, got {height}x{width}"
            )
        images_kwargs = images_kwargs or {}
        patch_size = int(images_kwargs.get("patch_size", self.patch_size))
        temporal_factor = int(
            images_kwargs.get("temporal_patch_size", self.temporal_patch_size)
        )
        merge_size = int(images_kwargs.get("merge_size", self.merge_size))
        min_tokens = int(
            images_kwargs.get("min_image_tokens", self.min_image_tokens)
        )
        max_tokens = int(
            images_kwargs.get("max_image_tokens", self.max_image_tokens)
        )
        factor = patch_size * merge_size
        min_pixels = min_tokens * temporal_factor * factor**2
        max_pixels = max_tokens * temporal_factor * factor**2

        def align(value: int) -> int:
            return math.ceil(value / factor) * factor

        target_height, target_width = align(height), align(width)
        budget = temporal_factor * target_height * target_width
        if budget < min_pixels:
            scale = math.sqrt(min_pixels / (temporal_factor * height * width))
            target_height = align(max(1, math.ceil(height * scale)))
            target_width = align(max(1, math.ceil(width * scale)))
            budget = temporal_factor * target_height * target_width

        if budget > max_pixels:
            minimum_pixels = temporal_factor * factor**2
            if max_pixels < minimum_pixels:
                raise ValueError(
                    f"max_image_tokens permits {max_pixels} pixels, but "
                    f"at least {minimum_pixels} are required"
                )
            low, high = 1, height
            target_height = target_width = factor
            while low <= high:
                content_height = (low + high) // 2
                content_width = max(1, math.floor(width * content_height / height))
                candidate_height = align(content_height)
                candidate_width = align(content_width)
                if temporal_factor * candidate_height * candidate_width <= max_pixels:
                    target_height, target_width = candidate_height, candidate_width
                    low = content_height + 1
                else:
                    high = content_height - 1

        return (target_height // patch_size) * (target_width // patch_size)


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


def _format_tool_declarations(tools: list[ToolSpec]) -> str:
    rendered_tools: list[str] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            raise TypeError(f"GLM5Renderer expected tool mapping, got {type(tool)!r}")
        tool_spec = (
            tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
        )
        if tool_spec.get("defer_loading"):
            continue
        filtered = {
            key: value
            for key, value in tool_spec.items()
            if key not in {"defer_loading", "strict"}
        }
        rendered_tools.append(json.dumps(filtered, ensure_ascii=False))
    return "\n".join(rendered_tools)


def _tool_result_id(message: Mapping[str, Any]) -> str:
    """Return the GLM-5.3 identity for one ordinary tool-result message."""
    value = message.get("tool_call_id") or message.get("id")
    return str(value) if value else ""


def _tool_call_id(tool_call: Any) -> str:
    """Return the GLM-5.3 identity for one assistant tool call."""
    if isinstance(tool_call, Mapping):
        value = tool_call.get("tool_call_id") or tool_call.get("id")
    else:
        value = getattr(tool_call, "tool_call_id", None) or getattr(
            tool_call, "id", None
        )
    return str(value) if value else ""


def _canonicalize_glm53_tool_result_blocks(messages: list[Message]) -> list[Message]:
    """Match GLM-5.3's ID-aware ordering for supported tool-result messages.

    The official template sorts a consecutive tool-result block only when the
    immediately preceding assistant has tool calls and every call/result ID is
    present, unique, and mutually consistent. Otherwise it renders the block in
    source order. Managed training's normalized schema represents one ordinary
    result per tool message, so this helper implements that exact supported
    subset without changing the messages themselves.
    """
    ordered = list(messages)
    index = 0
    while index < len(ordered):
        if ordered[index].get("role") != "tool":
            index += 1
            continue

        block_start = index
        while index < len(ordered) and ordered[index].get("role") == "tool":
            index += 1
        block_end = index

        if block_start == 0:
            continue
        assistant = ordered[block_start - 1]
        if assistant.get("role") != "assistant":
            continue
        tool_calls = assistant.get("tool_calls") or []
        if not tool_calls:
            continue

        call_ids = [_tool_call_id(tool_call) for tool_call in tool_calls]
        result_ids = [
            _tool_result_id(message) for message in ordered[block_start:block_end]
        ]
        can_sort = (
            all(call_ids)
            and len(set(call_ids)) == len(call_ids)
            and all(result_ids)
            and len(set(result_ids)) == len(result_ids)
            and all(result_id in set(call_ids) for result_id in result_ids)
        )
        if not can_sort:
            continue

        result_by_id = dict(zip(result_ids, ordered[block_start:block_end]))
        ordered[block_start:block_end] = [
            result_by_id[call_id] for call_id in call_ids if call_id in result_by_id
        ]

    return ordered


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


def _message_has_explicit_reasoning_field(message: Message) -> bool:
    """Whether GLM-5.1 considers this assistant explicitly thinking.

    The upstream GLM-5.1 template builds ``thinking_indices`` from
    ``m.reasoning_content is string``.  A reasoning field retained by
    cookbook's normalized representation becomes a structured ``thinking``
    part, so both forms count here.  Presence matters, not truthiness: an empty
    string still marks the whole user trajectory as having thinking.  A
    ``<think>`` block embedded in plain string content deliberately does *not*
    mark the trajectory, matching the template's first pass.
    """
    has_reasoning_content, _reasoning_content = original_reasoning_content(message)
    if has_reasoning_content:
        return True

    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(part, Mapping)
        and part.get("type") == "thinking"
        and isinstance(part.get("thinking"), str)
        for part in content
    )


def _user_indices_with_explicit_reasoning(
    messages: list[Message],
) -> set[int]:
    """Mirror GLM-5.1's ``thinking_indices`` pre-pass.

    Each explicit assistant reasoning field marks the most recent user index.
    Assistants before the first user do not affect ``has_thinking`` because the
    template only updates that state while rendering a user message.
    """
    current_user_idx: int | None = None
    thinking_user_indices: set[int] = set()
    for idx, message in enumerate(messages):
        if message["role"] == "user":
            current_user_idx = idx
        elif (
            message["role"] == "assistant"
            and current_user_idx is not None
            and _message_has_explicit_reasoning_field(message)
        ):
            thinking_user_indices.add(current_user_idx)
    return thinking_user_indices


def _extract_assistant_reasoning_and_text(message: Message) -> tuple[str, str]:
    """Return assistant reasoning/text with HF field precedence.

    Direct renderer callers may still supply the OpenAI-style top-level
    ``reasoning_content`` field.  Honor it before inspecting embedded
    ``<think>`` text, just as the official template does.  Production's
    normalized structured ``thinking`` parts continue through the existing
    content parser.
    """
    has_reasoning_content, reasoning_content = original_reasoning_content(message)
    if has_reasoning_content:
        return reasoning_content, _visible_text(message.get("content"))
    return _extract_reasoning_and_text(message.get("content"))


class GLM5Renderer(DisaggregateMultiTurnMixin, Renderer):
    """Renderer for ZhipuAI GLM-5.1 instruct models.

    Thinking is stripped from historical assistant turns by default (matching
    the shipped chat template's default ``clear_thinking`` behavior). When
    ``clear_thinking=False``, historical reasoning is kept, matching the
    official template flag directly. Default-mode multi-turn
    ``ALL_ASSISTANT_MESSAGES`` SFT is handled by
    :class:`DisaggregateMultiTurnMixin`, which splits the conversation per user
    turn so each datum's prompt context byte-equals what ``apply_chat_template``
    produces for the same prefix. GLM-5.1 PRESERVED mode also disaggregates: an
    assistant with no explicit reasoning can change from ``<think></think>``
    to ``</think>`` when its user turn becomes historical, so the renderer does
    not satisfy the extension property even though explicit reasoning is kept.
    """

    _initial_prompt_text = ""
    _historical_stripped_think_block = "</think>"
    _preserve_has_extension_property = False

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        clear_thinking: bool = True,
        honor_source_reasoning_fields: bool = False,
    ) -> None:
        super().__init__(tokenizer)
        self._clear_thinking = clear_thinking
        self._honor_source_reasoning_fields = honor_source_reasoning_fields

    @property
    def has_extension_property(self) -> bool:
        """Whether PRESERVED mode is prefix-stable for this GLM template family."""
        return not self._clear_thinking and self._preserve_has_extension_property

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

    @property
    def _initial_prompt_tokens(self) -> list[int]:
        if not self._initial_prompt_text:
            return []
        return self.tokenizer.encode(
            self._initial_prompt_text,
            add_special_tokens=False,
        )

    def get_stop_sequences(self) -> list[int]:
        return [self._user_token, self._observation_token]

    def _assistant_stop_token(self, message: Message) -> int:
        return (
            self._observation_token if message.get("tool_calls") else self._user_token
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

            if isinstance(output_part, tinker.types.ImageChunk):
                # Image embeddings are context, never token targets, even for
                # ALL_TOKENS or a trainable tool message.
                model_input_chunks_weights.append((output_part, 0.0))
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
                model_input_chunks_weights.append((output_part, int(output_has_weight)))

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
        if self._initial_prompt_tokens:
            model_input_chunks_weights.append(
                (
                    tinker.types.EncodedTextChunk(tokens=self._initial_prompt_tokens),
                    0.0,
                )
            )

        last_user_idx = max(
            (idx for idx, message in enumerate(messages) if message["role"] == "user"),
            default=-1,
        )
        thinking_user_indices = _user_indices_with_explicit_reasoning(messages)
        current_user_idx: int | None = None

        for idx, message in enumerate(messages):
            if message["role"] == "user":
                current_user_idx = idx
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
            rendered_message = self.render_message(
                message,
                ctx,
                user_turn_has_explicit_reasoning=(
                    current_user_idx is not None
                    and current_user_idx in thinking_user_indices
                ),
            )

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

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        chunks: list[tinker.ModelInputChunk] = []
        if self._bos_tokens:
            chunks.append(tinker.types.EncodedTextChunk(tokens=self._bos_tokens))
        if self._initial_prompt_tokens:
            chunks.append(
                tinker.types.EncodedTextChunk(tokens=self._initial_prompt_tokens)
            )

        last_user_idx = max(
            (idx for idx, message in enumerate(messages) if message["role"] == "user"),
            default=-1,
        )
        thinking_user_indices = _user_indices_with_explicit_reasoning(messages)
        current_user_idx: int | None = None

        for idx, message in enumerate(messages):
            if message["role"] == "user":
                current_user_idx = idx
            ctx = RenderContext(
                idx=idx,
                is_last=(idx == len(messages) - 1),
                prev_message=messages[idx - 1] if idx > 0 else None,
                last_user_index=last_user_idx,
            )
            rendered_message = self.render_message(
                message,
                ctx,
                user_turn_has_explicit_reasoning=(
                    current_user_idx is not None
                    and current_user_idx in thinking_user_indices
                ),
            )
            if rendered_message.header:
                chunks.append(rendered_message.header)
            chunks.extend(
                [
                    output
                    for output in rendered_message.output
                    if not isinstance(output, tinker.EncodedTextChunk) or output.tokens
                ]
            )

        suffix_ctx = RenderContext(
            idx=len(messages),
            is_last=True,
            prev_message=messages[-1] if messages else None,
            last_user_index=last_user_idx,
        )
        suffix_tokens = self._get_generation_suffix(role, suffix_ctx)
        if suffix_tokens:
            chunks.append(tinker.types.EncodedTextChunk(tokens=suffix_tokens))

        if prefill:
            chunks.append(
                tinker.types.EncodedTextChunk(
                    tokens=self.tokenizer.encode(prefill, add_special_tokens=False)
                )
            )
        return tinker.ModelInput(chunks=chunks)

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
        *,
        user_turn_has_explicit_reasoning: bool | None = None,
    ) -> RenderedMessage:
        role = message["role"]
        if role == "assistant":
            if user_turn_has_explicit_reasoning is None:
                # ``render_message`` is also a public low-level hook. With no
                # conversation available, the current message is the strongest
                # faithful fallback; full prompt/example builders always pass
                # the precomputed per-user trajectory state.
                user_turn_has_explicit_reasoning = (
                    _message_has_explicit_reasoning_field(message)
                )
            return self._render_assistant(
                message,
                ctx,
                user_turn_has_explicit_reasoning=user_turn_has_explicit_reasoning,
            )
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
                ctx.prev_message is not None and ctx.prev_message.get("role") == "tool"
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

    def _render_assistant(
        self,
        message: Message,
        ctx: RenderContext,
        *,
        user_turn_has_explicit_reasoning: bool,
    ) -> RenderedMessage:
        # The role tag ``<|assistant|>`` is the header. Thinking-mode
        # assistants keep the template's opening ``<think>`` in ``output`` for
        # token parity, but supervised rendering masks that prefix because
        # ``add_generation_prompt=True`` injects it before model generation.
        header_str = "<|assistant|>"

        before_last_user = ctx.last_user_index >= 0 and ctx.idx < ctx.last_user_index

        reasoning, visible = (
            _extract_assistant_reasoning_and_text(message)
            if self._honor_source_reasoning_fields
            else _extract_reasoning_and_text(message.get("content"))
        )

        think_block = self._assistant_think_block(
            before_last_user=before_last_user,
            reasoning=reasoning,
            user_turn_has_explicit_reasoning=user_turn_has_explicit_reasoning,
        )

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

    def _assistant_think_block(
        self,
        *,
        before_last_user: bool,
        reasoning: str,
        user_turn_has_explicit_reasoning: bool,
    ) -> str:
        # Historical turns strip reasoning using the renderer family's collapse
        # marker. In GLM-5.1 PRESERVED mode, a no-reasoning assistant uses an
        # empty block only when another explicit reasoning field marked the same
        # user trajectory via ``thinking_indices / has_thinking``. GLM-5.2's
        # collapse marker is already the empty block, so the same branch also
        # reproduces its simpler behavior.
        if before_last_user:
            if self._clear_thinking:
                return self._historical_stripped_think_block
            if reasoning:
                return f"<think>{reasoning.strip()}</think>"
            if user_turn_has_explicit_reasoning:
                return "<think></think>"
            return self._historical_stripped_think_block
        if reasoning:
            return f"<think>{reasoning.strip()}</think>"
        return "<think></think>"

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

    def create_conversation_prefix_with_tools(
        self,
        tools: list[ToolSpec],
        system_prompt: str = "",
    ) -> list[Message]:
        """Render top-level OpenAI tool schemas in GLM's system tool block."""
        prefix_messages = [
            Message(
                role="system",
                content=(
                    _TOOL_DECLARATION_PREFIX
                    + _format_tool_declarations(tools)
                    + "\n"
                    + _TOOL_DECLARATION_SUFFIX
                ),
            )
        ]
        if system_prompt:
            prefix_messages.append(Message(role="system", content=system_prompt))
        return prefix_messages

    def _normalize_response_tokens(self, response: list[int]) -> list[int]:
        """Restore the prefilled ``<think>`` opener before parsing.

        ``_get_generation_suffix`` prefills ``<|assistant|><think>``, so sampled
        tokens start INSIDE the think block: they contain ``</think>`` but no
        opening ``<think>``. Without restoring the opener, ``parse_think_blocks``
        can't split the block and the reasoning leaks into the graded content.
        Mirrors ``training._vendor.tinker_cookbook_0_4_3.renderers.qwen3_5.Qwen3_5Renderer``.
        """
        think_prefix = self.tokenizer.encode("<think>", add_special_tokens=False)
        if response[: len(think_prefix)] == think_prefix:
            return response
        if "</think>" in str(self.tokenizer.decode(response)):
            return think_prefix + response
        return response

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        response = self._normalize_response_tokens(response)
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


class GLMMoeDsaRenderer(GLM5Renderer):
    """Renderer for GLM-5.2 / ``glm_moe_dsa`` chat-template differences."""

    _initial_prompt_text = "<|system|>Reasoning Effort: Max"
    _historical_stripped_think_block = "<think></think>"
    _preserve_has_extension_property = True


class GLM53Renderer(GLMMoeDsaRenderer):
    """Renderer for the pinned ``zai-org/GLM-5.3`` chat-template contract.

    GLM-5.3 shares GLM-5.2's role/tool wire format and default Max reasoning
    prefix, but defaults ``clear_thinking`` to false and sorts well-formed
    consecutive tool results by their preceding assistant tool-call IDs.
    """

    # Tool-result ordering depends on the surrounding consecutive block, so a
    # verifier cannot reconstruct the whole render by calling render_message
    # independently for each message.
    supports_per_message_rendering = False

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        clear_thinking: bool = False,
        honor_source_reasoning_fields: bool = True,
    ) -> None:
        super().__init__(
            tokenizer,
            clear_thinking=clear_thinking,
            honor_source_reasoning_fields=honor_source_reasoning_fields,
        )

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        return super().build_supervised_example(
            _canonicalize_glm53_tool_result_blocks(messages),
            train_on_what=train_on_what,
        )

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        return super().build_generation_prompt(
            _canonicalize_glm53_tool_result_blocks(messages),
            role=role,
            prefill=prefill,
        )

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
        *,
        user_turn_has_explicit_reasoning: bool | None = None,
    ) -> RenderedMessage:
        # The official GLM-5.3 template has no developer-role branch, so such
        # messages contribute no bytes. Keep this behavior version-local; the
        # earlier GLM renderers intentionally continue to reject the role.
        if message["role"] == "developer":
            return RenderedMessage(header=None, output=[])
        return super().render_message(
            message,
            ctx,
            user_turn_has_explicit_reasoning=user_turn_has_explicit_reasoning,
        )


class GLM53FlashRenderer(GLM53Renderer):
    """GLM-5.3 text semantics plus the Flash multimodal template branches."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        image_processor: Any | None = None,
        clear_thinking: bool = False,
        honor_source_reasoning_fields: bool = True,
    ) -> None:
        super().__init__(
            tokenizer,
            clear_thinking=clear_thinking,
            honor_source_reasoning_fields=honor_source_reasoning_fields,
        )
        self.image_processor = image_processor

    @staticmethod
    def _has_image_content(content: Any) -> bool:
        return isinstance(content, list) and any(
            isinstance(part, Mapping) and part.get("type") == "image"
            for part in content
        )

    def _render_image_content(
        self,
        content: Any,
        *,
        prefix: str = "",
        suffix: str = "",
    ) -> list[tinker.ModelInputChunk]:
        if self.image_processor is None:
            raise RendererError(
                "GLM-5.3-Flash image content requires an image processor; "
                "build the renderer with image loading enabled."
            )

        chunks: list[tinker.ModelInputChunk] = []
        pending_text = prefix

        def flush_text() -> None:
            nonlocal pending_text
            if pending_text:
                chunks.append(
                    tinker.types.EncodedTextChunk(
                        tokens=self.tokenizer.encode(
                            pending_text,
                            add_special_tokens=False,
                        )
                    )
                )
                pending_text = ""

        for part in content:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "text":
                pending_text += str(part.get("text", ""))
            elif part.get("type") == "image":
                flush_text()
                image = part.get("image")
                if image is None:
                    raise RendererError(
                        "GLM-5.3-Flash image content is missing the 'image' payload."
                    )
                chunks.append(image_to_chunk(image, self.image_processor))
            # Match the template by dropping unknown content-part types.
        pending_text += suffix
        flush_text()
        return chunks

    def render_message(
        self,
        message: Message,
        ctx: RenderContext,
        *,
        user_turn_has_explicit_reasoning: bool | None = None,
    ) -> RenderedMessage:
        content = message.get("content")
        if self._has_image_content(content):
            if message["role"] not in {"user", "tool"}:
                raise RendererError(
                    "GLM-5.3-Flash images are supported only in user or tool messages"
                )
            if message["role"] == "user":
                return RenderedMessage(
                    header=tinker.types.EncodedTextChunk(
                        tokens=self.tokenizer.encode(
                            "<|user|>", add_special_tokens=False
                        )
                    ),
                    output=self._render_image_content(content),
                )

            prev_is_tool = (
                ctx.prev_message is not None
                and ctx.prev_message.get("role") == "tool"
            )
            return RenderedMessage(
                header=tinker.types.EncodedTextChunk(
                    tokens=self.tokenizer.encode(
                        "" if prev_is_tool else "<|observation|>",
                        add_special_tokens=False,
                    )
                ),
                output=self._render_image_content(
                    content,
                    prefix="<tool_response>",
                    suffix="</tool_response>",
                ),
            )

        return super().render_message(
            message,
            ctx,
            user_turn_has_explicit_reasoning=user_turn_has_explicit_reasoning,
        )


def _glm5_factory(tokenizer: Tokenizer, image_processor=None) -> GLM5Renderer:
    del image_processor
    return GLM5Renderer(tokenizer)


def _glm5_interleaved_factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> GLM5Renderer:
    del image_processor
    return GLM5Renderer(tokenizer, honor_source_reasoning_fields=True)


def _glm5_preserve_thinking_factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> GLM5Renderer:
    del image_processor
    # Public PRESERVED maps to the official inverted flag.
    return GLM5Renderer(
        tokenizer,
        clear_thinking=False,
        honor_source_reasoning_fields=True,
    )


def _glm_moe_dsa_factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> GLMMoeDsaRenderer:
    del image_processor
    return GLMMoeDsaRenderer(tokenizer)


def _glm_moe_dsa_interleaved_factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> GLMMoeDsaRenderer:
    del image_processor
    return GLMMoeDsaRenderer(tokenizer, honor_source_reasoning_fields=True)


def _glm_moe_dsa_preserve_thinking_factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> GLMMoeDsaRenderer:
    del image_processor
    # Public PRESERVED maps to the official inverted flag.
    return GLMMoeDsaRenderer(
        tokenizer,
        clear_thinking=False,
        honor_source_reasoning_fields=True,
    )


def _glm53_factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> GLM53Renderer:
    del image_processor
    return GLM53Renderer(tokenizer)


def _glm53_interleaved_factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> GLM53Renderer:
    del image_processor
    return GLM53Renderer(tokenizer, clear_thinking=True)


def _glm53_preserve_thinking_factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> GLM53Renderer:
    del image_processor
    return GLM53Renderer(tokenizer, clear_thinking=False)


def _glm53_flash_factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> GLM53FlashRenderer:
    return GLM53FlashRenderer(tokenizer, image_processor=image_processor)


def _glm53_flash_interleaved_factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> GLM53FlashRenderer:
    return GLM53FlashRenderer(
        tokenizer,
        image_processor=image_processor,
        clear_thinking=True,
    )


def _glm53_flash_preserve_thinking_factory(
    tokenizer: Tokenizer,
    image_processor=None,
) -> GLM53FlashRenderer:
    return GLM53FlashRenderer(
        tokenizer,
        image_processor=image_processor,
        clear_thinking=False,
    )


register_renderer("glm5", _glm5_factory)
# Keep the existing concrete names above available for legacy direct callers.
# Managed semantic modes materialize the new names below, so future
# correctness work cannot silently reinterpret an already persisted name.
register_renderer("glm5_interleaved", _glm5_interleaved_factory)
register_renderer("glm5_preserve_thinking", _glm5_preserve_thinking_factory)
register_renderer("glm_moe_dsa", _glm_moe_dsa_factory)
register_renderer("glm_moe_dsa_interleaved", _glm_moe_dsa_interleaved_factory)
register_renderer(
    "glm_moe_dsa_preserve_thinking",
    _glm_moe_dsa_preserve_thinking_factory,
)
register_renderer("glm53", _glm53_factory)
register_renderer("glm53_interleaved", _glm53_interleaved_factory)
register_renderer("glm53_preserve_thinking", _glm53_preserve_thinking_factory)
register_renderer("glm53_flash", _glm53_flash_factory)
register_renderer("glm53_flash_interleaved", _glm53_flash_interleaved_factory)
register_renderer(
    "glm53_flash_preserve_thinking",
    _glm53_flash_preserve_thinking_factory,
)
