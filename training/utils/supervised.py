"""Shared supervised-rendering helpers.

This module gives cookbook training code one token-level representation for
supervised data:

- text/tool conversations rendered via a Tinker renderer
- eval-protocol-style token trajectories with a per-token mask

Both paths end in the same ``tinker.Datum`` schema with ``target_tokens`` and
token-level ``weights`` so training uses the same spans that the UI shows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import tinker
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import (
    Message,
    Renderer,
    ToolCall,
    TrainOnWhat,
    get_renderer,
)

from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.supervised.common import datum_from_model_input_weights
import training.renderer.minimax_m2 as _minimax_m2_renderer  # noqa: F401 — triggers register_renderer
import training.renderer.minimax_m3 as _minimax_m3_renderer  # noqa: F401 — triggers register_renderer
import training.renderer.gemma4 as _gemma4_renderer  # noqa: F401 — triggers register_renderer
import training.renderer._gemma4_split as _gemma4_split_renderer  # noqa: F401 — split override
import training.renderer.deepseek_v4 as _deepseek_v4_renderer  # noqa: F401 — triggers register_renderer
import training.renderer.mistral as _mistral_renderer  # noqa: F401 — triggers register_renderer
import training.renderer.kimi_k27_code as _kimi_k27_code_renderer  # noqa: F401 — triggers register_renderer
from training.renderer.thinking_trace import (
    ResolvedThinkingTraceRendererPlan,
    ThinkingTraceHistoryMode,
    get_thinking_trace_model_capability,
    normalize_thinking_trace_history_mode,
    resolve_thinking_trace_renderer_plan,
)
from training.renderer.reasoning_fields import (
    ORIGINAL_REASONING,
    ORIGINAL_REASONING_CONTENT,
)
from training.utils.tokenizers import load_tokenizer


@dataclass(frozen=True)
class RenderedSupervisedDatum:
    """Rendered sequence ids, token-level weights, and the final training datum."""

    token_ids: list[int]
    token_weights: list[float]
    datum: tinker.Datum


@dataclass(frozen=True)
class RenderedPreferencePair:
    """Rendered chosen/rejected preference pair with a shared response boundary."""

    chosen_tokens: list[int]
    rejected_tokens: list[int]
    response_start: int
    chosen_datum: tinker.Datum
    rejected_datum: tinker.Datum


def parse_train_on_what(value: str | TrainOnWhat) -> TrainOnWhat:
    """Normalize ``train_on_what`` config into Tinker's enum."""
    if isinstance(value, TrainOnWhat):
        return value
    return TrainOnWhat(value.lower())


def _tool_prefix_builder(renderer: Renderer):
    """Return the renderer's tool-prefix hook if it is actually implemented."""
    prefix_builder = getattr(renderer, "create_conversation_prefix_with_tools", None)
    if prefix_builder is None:
        return None
    class_builder = getattr(
        type(renderer), "create_conversation_prefix_with_tools", None
    )
    if class_builder is Renderer.create_conversation_prefix_with_tools:
        return None
    return prefix_builder


def renderer_declares_tools(renderer: Renderer) -> bool:
    """Whether ``renderer`` actually renders tool declarations.

    True only when the renderer OVERRIDES
    ``create_conversation_prefix_with_tools`` (not the base :class:`Renderer`
    stub). Both production (:func:`render_messages_to_datums`) and the renderer
    QA harness gate tool-declaration rendering on this, so a renderer that can't
    declare tools is never asked to.
    """
    return _tool_prefix_builder(renderer) is not None


def build_tool_prefixed_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    renderer: Renderer,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> list[Message]:
    """Normalize a chat row and prepend the renderer's tool-declaration prefix.

    This is the **single** message-assembly path shared by production SFT
    (:func:`render_messages_to_datums`) and the renderer QA harness, so tests
    render tool rows byte-for-byte the way training does. When ``tools`` is
    provided and the renderer implements ``create_conversation_prefix_with_tools``
    (see :func:`renderer_declares_tools`), the tool definitions are encoded as
    renderer-specific prefix messages and prepended, folding a leading system
    message's content into the prefix's ``system_prompt``. Renderers without
    tool-prefix support drop the field (they do NOT raise), matching production.
    """
    normalized_messages = list(normalize_messages(messages))
    if not tools:
        return normalized_messages
    prefix_builder = _tool_prefix_builder(renderer)
    if prefix_builder is None:
        return normalized_messages
    tool_specs = [
        t["function"]
        for t in tools
        if isinstance(t, Mapping) and isinstance(t.get("function"), Mapping)
    ]
    if not tool_specs:
        return normalized_messages
    system_prompt = ""
    has_explicit_empty_system_message = False
    if normalized_messages and normalized_messages[0].get("role") == "system":
        sys_content = normalized_messages.pop(0).get("content")
        if isinstance(sys_content, str):
            system_prompt = sys_content
            has_explicit_empty_system_message = sys_content == ""
        elif isinstance(sys_content, list):
            system_prompt = "\n".join(
                part.get("text", "")
                for part in sys_content
                if isinstance(part, Mapping) and part.get("type") == "text"
            )
    prefix_messages = list(prefix_builder(tool_specs, system_prompt=system_prompt))
    if has_explicit_empty_system_message and getattr(
        renderer,
        "_preserves_explicit_empty_system_with_tools",
        False,
    ):
        prefix_messages.append(Message(role="system", content=""))
    if any("trainable" in m for m in normalized_messages):
        for prefix_msg in prefix_messages:
            prefix_msg.setdefault("trainable", False)
    return prefix_messages + normalized_messages


def resolve_renderer_name(
    tokenizer_model: str,
    renderer_name: str = "",
) -> str:
    """Choose the existing/default renderer for message -> token rendering."""
    if renderer_name:
        return renderer_name
    normalized_model_name = tokenizer_model.lower()
    if (
        "moonshotai/kimi-k2.5" in normalized_model_name
        or "kimi-k2p5" in normalized_model_name
    ):
        return "kimi_k25"
    # Preserve the legacy direct/default alias. Explicit semantic mode
    # resolution uses the capability registry's dedicated K2.6 names instead.
    if (
        "moonshotai/kimi-k2.6" in normalized_model_name
        or "kimi-k2p6" in normalized_model_name
    ):
        return "kimi_k25"
    # Kimi-K2.7-Code keeps the K2.6 tokenizer/vocab/tool format, but flips the
    # HF chat-template default to preserve historical thinking and no longer
    # injects Kimi's default system prompt when none is provided.
    if (
        "moonshotai/kimi-k2.7-code" in normalized_model_name
        or "kimi-k2p7-code" in normalized_model_name
    ):
        return "kimi_k27_code"
    if "nemotron" in normalized_model_name:
        # Route the Nemotron family to tinker_cookbook's upstream Nemotron-3
        # renderer ("nemotron3"), which restores the prompt-prefilled <think>
        # in parse_response via _normalize_response_tokens. The retired
        # cookbook "nemotron" renderer inherited Qwen3Renderer's identity
        # normalize and left the prefilled think unrestored, so sampled
        # completions (which start inside the think block: </think> with no
        # opening <think>) were graded with the reasoning still attached.
        return "nemotron3"
    if "minimax-m2" in normalized_model_name or "minimax_m2" in normalized_model_name:
        return "minimax_m2"
    # The released MiniMax-M3 tokenizer uses its own BOD/BOS role protocol,
    # thinking tags, and XML tool-call format; it is not M2-compatible.
    if "minimax-m3" in normalized_model_name or "minimax_m3" in normalized_model_name:
        return "minimax_m3"
    if "qwen3-vl" in normalized_model_name:
        return "qwen3_vl_instruct"
    # Qwen3.6 reuses Qwen3.5's vocab + special tokens; the chat template only
    # adds an opt-in `preserve_thinking` flag (renders historical thinking
    # for ALL assistant turns when true). Default invocation produces output
    # byte-identical to Qwen3.5's template, so the qwen3_5 renderer family
    # is correct for INTERLEAVED-history workflows on Qwen3.6 checkpoints.
    # Same alias pattern as the kimi-k25 → Kimi-K2.6 case above.
    if (
        "qwen3.6" in normalized_model_name
        or "qwen3_6" in normalized_model_name
        or "qwen3p6" in normalized_model_name
    ):
        return "qwen3_6"
    if (
        "qwen3.5" in normalized_model_name
        or "qwen3_5" in normalized_model_name
        or "qwen3p5" in normalized_model_name
    ):
        return "qwen3_5"
    if "gemma-4" in normalized_model_name or "gemma4" in normalized_model_name:
        return "gemma4"
    # DeepSeek-V4 ships a custom non-Jinja encoder (see encoding_dsv4.py upstream)
    # with thinking blocks and DSML tool calls. Match the V4 family explicitly so
    # we don't accidentally claim V3 (which routes through tinker_cookbook's
    # built-in deepseekv3_thinking renderer via get_recommended_renderer_name).
    if (
        "deepseek-v4" in normalized_model_name
        or "deepseek_v4" in normalized_model_name
        or "deepseekv4" in normalized_model_name
    ):
        return "deepseek_v4"
    # ZhipuAI GLM-5.2 keeps GLM-5 role/tool tags, but its shipped template
    # adds a default Reasoning Effort system prefix and slightly different
    # stripped-thinking blocks. Route it to the dedicated variant.
    if "glm-5p2" in normalized_model_name or "glm-5.2" in normalized_model_name:
        return "glm_moe_dsa"
    # ZhipuAI GLM-5.1 chat template (`[gMASK]<sop>`, `<|user|>`,
    # `<|assistant|>`, `<think>...</think>`, `<|endoftext|>`).
    if (
        "glm-5p1" in normalized_model_name
        or "glm-5.1" in normalized_model_name
        or "glm5" in normalized_model_name
    ):
        return "glm5"
    # Mistral / Ministral Tekken-style chat (`[SYSTEM_PROMPT]`, `[INST]…[/INST]`,
    # `[TOOL_CALLS]…[ARGS]…`, `[TOOL_RESULTS]…[/TOOL_RESULTS]`). Covers
    # Ministral 3 / 8B, Mistral 7B Instruct v0.3+, Mistral Small 3 Instruct, and
    # any other ``mistralai/...`` checkpoint that ships the same template.
    # Validated end-to-end via SFT training on Ministral-3-3B-Instruct-2512.
    if (
        normalized_model_name.startswith("mistralai/")
        or "ministral-" in normalized_model_name
        or "mistral-7b-instruct-v0.3" in normalized_model_name
        or "mistral-7b-instruct-v0.4" in normalized_model_name
        or "mistral-small" in normalized_model_name
    ):
        return "mistral"
    try:
        return get_recommended_renderer_name(tokenizer_model)
    except Exception as exc:  # pragma: no cover - message only
        raise ValueError(
            f"Could not infer a renderer for tokenizer_model={tokenizer_model!r}. "
            "Set Config.renderer_name explicitly."
        ) from exc


def resolve_renderer_plan(
    tokenizer_model: str,
    renderer_name: str = "",
    *,
    thinking_trace_history_mode: (
        str | ThinkingTraceHistoryMode | None
    ) = ThinkingTraceHistoryMode.UNSPECIFIED,
) -> ResolvedThinkingTraceRendererPlan:
    """Resolve a semantic history mode to a concrete registered renderer.

    This is deliberately separate from :func:`resolve_renderer_name`: legacy
    callers retain their byte-for-byte default path, while Managed Training can
    opt into the explicit model capability contract.  The registry performs
    model validation; no shared ``preserve_thinking`` boolean is forwarded to
    arbitrary renderers.
    """

    requested_mode = normalize_thinking_trace_history_mode(thinking_trace_history_mode)
    default_renderer_name = resolve_renderer_name(tokenizer_model, renderer_name)

    # Preserve the legacy explicit renderer override when the new field is
    # absent. In particular, an existing `qwen3_6_disable_thinking` config must
    # not silently become the registry's thinking-enabled INTERLEAVED renderer.
    if renderer_name and requested_mode is ThinkingTraceHistoryMode.UNSPECIFIED:
        capability = get_thinking_trace_model_capability(tokenizer_model)
        matched_plan = (
            next(
                (
                    plan
                    for plan in capability.plans
                    if plan.renderer_name == renderer_name
                ),
                None,
            )
            if capability is not None
            else None
        )
        return ResolvedThinkingTraceRendererPlan(
            requested_mode=requested_mode,
            effective_mode=(
                matched_plan.mode
                if matched_plan is not None
                else ThinkingTraceHistoryMode.UNSPECIFIED
            ),
            renderer_name=renderer_name,
            unrolls_multi_turn=(
                matched_plan.unrolls_multi_turn if matched_plan is not None else None
            ),
            canonical_family=(
                capability.canonical_family if capability is not None else None
            ),
        )

    resolved = resolve_thinking_trace_renderer_plan(
        tokenizer_model,
        requested_mode=requested_mode,
        default_renderer_name=default_renderer_name,
    )
    if renderer_name and renderer_name != resolved.renderer_name:
        raise ValueError(
            f"renderer_name={renderer_name!r} conflicts with "
            f"thinking_trace_history_mode={requested_mode.value!r}, which "
            f"requires renderer {resolved.renderer_name!r}."
        )
    return resolved


def build_renderer(
    tokenizer: Any,
    tokenizer_model: str,
    renderer_name: str = "",
    *,
    thinking_trace_history_mode: (
        str | ThinkingTraceHistoryMode | None
    ) = ThinkingTraceHistoryMode.UNSPECIFIED,
    load_image_processor: bool = True,
) -> Renderer:
    """Construct the Tinker renderer used for supervised formatting."""
    resolved_name = resolve_renderer_plan(
        tokenizer_model,
        renderer_name,
        thinking_trace_history_mode=thinking_trace_history_mode,
    ).renderer_name
    return build_renderer_from_resolved_name(
        tokenizer,
        tokenizer_model,
        resolved_name,
        load_image_processor=load_image_processor,
    )


def build_renderer_from_resolved_name(
    tokenizer: Any,
    tokenizer_model: str,
    renderer_name: str,
    *,
    load_image_processor: bool = True,
) -> Renderer:
    """Construct exactly ``renderer_name`` without consulting capabilities."""

    if (
        load_image_processor
        and get_image_processor is not None
        and _renderer_uses_images(renderer_name)
    ):
        _ensure_trust_remote_code_for_image_processor(tokenizer_model)
        return get_renderer(
            renderer_name,
            tokenizer,
            image_processor=get_image_processor(tokenizer_model),
        )
    return get_renderer(renderer_name, tokenizer)


def resolve_renderer_snapshot(
    *,
    tokenizer_model: str,
    renderer_name: str,
    thinking_trace_history_mode: str,
    renderer_name_is_resolved: bool = False,
) -> str:
    """Resolve a direct request or reuse a concrete managed renderer.

    ``renderer_name`` alone remains a direct cookbook override and is validated
    against any explicit semantic mode. Managed Training additionally sets
    ``renderer_name_is_resolved`` after materializing the renderer; only that
    provenance makes the concrete name authoritative across retries/resumes.
    Renderer registrations must stay backward compatible with persisted names.
    """

    if renderer_name_is_resolved:
        if not renderer_name:
            raise ValueError(
                "renderer_name_is_resolved=True requires a non-empty renderer_name."
            )
        return renderer_name

    normalized_mode = normalize_thinking_trace_history_mode(thinking_trace_history_mode)
    if normalized_mode is not ThinkingTraceHistoryMode.UNSPECIFIED:
        return resolve_renderer_plan(
            tokenizer_model,
            renderer_name,
            thinking_trace_history_mode=normalized_mode,
        ).renderer_name
    return resolve_renderer_name(tokenizer_model, renderer_name)


def populate_render_worker_state(
    state: dict,
    *,
    tokenizer_model: str,
    tokenizer_revision: str | None = None,
    tokenizer_trust_remote_code: bool | None = None,
    renderer_name: str,
    max_seq_len: int,
    thinking_trace_history_mode: (
        str | ThinkingTraceHistoryMode | None
    ) = ThinkingTraceHistoryMode.UNSPECIFIED,
    renderer_name_is_resolved: bool = False,
    **extras: Any,
) -> None:
    """Build a tokenizer + renderer and populate ``state`` for DataLoader workers.

    Centralises the per-worker setup that every streaming render recipe needs
    (SFT, DPO, ORPO, ...). Call from your recipe's module-level
    ``_init_<recipe>_worker(_worker_id)`` shim, which DataLoader spawns with
    ``worker_init_fn``. ``state`` must be a module-level dict so the recipe's
    top-level render function (also picklable for spawn) can read from it.

    Stores ``tokenizer``, ``renderer``, ``max_seq_len`` plus any keyword
    ``extras`` (e.g. ``train_on_what`` for SFT). A ``None`` remote-code policy
    preserves the legacy permissive load; reviewed plans pass an explicit value.
    """
    tokenizer = load_tokenizer(
        tokenizer_model,
        tokenizer_revision,
        tokenizer_trust_remote_code,
    )
    renderer = (
        build_renderer_from_resolved_name(
            tokenizer,
            tokenizer_model,
            renderer_name,
        )
        if renderer_name_is_resolved
        else build_renderer(
            tokenizer,
            tokenizer_model,
            renderer_name,
            thinking_trace_history_mode=thinking_trace_history_mode,
        )
    )
    state.update(
        tokenizer=tokenizer,
        renderer=renderer,
        max_seq_len=max_seq_len,
        **extras,
    )


# tinker_cookbook.image_processing_utils.get_image_processor has a hard-coded
# trust_remote_code=True branch only for moonshotai/Kimi-K2.5. Other Kimi-family
# checkpoints that ship a custom image processor (e.g. moonshotai/Kimi-K2.6)
# also require trust_remote_code, otherwise AutoImageProcessor.from_pretrained
# raises ValueError in non-interactive environments (CI, service mode). The
# same module honors HF_TRUST_REMOTE_CODE as an opt-in hook, so set it for
# those checkpoints before the first (cached) call.
_MODELS_REQUIRING_TRUST_REMOTE_CODE_FOR_IMAGE_PROCESSOR: tuple[str, ...] = (
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k2.7-code",
)


def _ensure_trust_remote_code_for_image_processor(tokenizer_model: str) -> None:
    normalized = tokenizer_model.lower()
    if any(
        marker in normalized
        for marker in _MODELS_REQUIRING_TRUST_REMOTE_CODE_FOR_IMAGE_PROCESSOR
    ):
        os.environ.setdefault("HF_TRUST_REMOTE_CODE", "1")


def _renderer_uses_images(renderer_name: str) -> bool:
    return any(
        marker in renderer_name
        for marker in (
            "_vl",
            "qwen3_5",
            "qwen3_6",
            "kimi_k25",
            "kimi_k26",
            "kimi_k27",
        )
    )


logger = logging.getLogger(__name__)


def _truncate_model_input(model_input: tinker.ModelInput) -> tinker.ModelInput:
    """Return a copy of *model_input* with the last text token removed.

    This is the multimodal equivalent of ``tokens[:-1]`` used for the standard
    next-token prediction shift.
    """
    chunks = list(model_input.chunks)
    for i in range(len(chunks) - 1, -1, -1):
        chunk = chunks[i]
        if isinstance(chunk, tinker.types.EncodedTextChunk) and len(chunk.tokens) > 0:
            remaining = list(chunk.tokens)[:-1]
            if remaining:
                chunks[i] = tinker.types.EncodedTextChunk(tokens=remaining)
            else:
                chunks.pop(i)
            result = tinker.ModelInput.empty()
            for c in chunks:
                result = result.append(c)
            return result
    raise ValueError("ModelInput has no text tokens to truncate")


def _normalize_tool_calls(tool_calls: Any) -> list[ToolCall]:
    """Normalize common tool-call shapes into Tinker's structured ToolCall form."""
    normalized: list[ToolCall] = []
    for tool_call in tool_calls or []:
        if not isinstance(tool_call, Mapping):
            raise TypeError(f"Unsupported tool call type: {type(tool_call)!r}")

        if isinstance(tool_call.get("name"), str) and isinstance(
            tool_call.get("args"), Mapping
        ):
            normalized.append(
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name=tool_call["name"],
                        arguments=json.dumps(dict(tool_call["args"])),
                    ),
                    id=tool_call.get("id"),
                )
            )
            continue

        function = tool_call.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            raw_args = function.get("arguments", {})
            if isinstance(raw_args, str):
                parsed_args = json.loads(raw_args) if raw_args else {}
            elif isinstance(raw_args, Mapping):
                parsed_args = dict(raw_args)
            else:
                raise TypeError(
                    f"Unsupported tool call arguments type: {type(raw_args)!r}"
                )
            normalized.append(
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name=function["name"],
                        arguments=json.dumps(parsed_args),
                    ),
                    id=tool_call.get("id"),
                )
            )
            continue

        raise ValueError(f"Unsupported tool call shape: {tool_call}")
    return normalized


def _normalize_image_part(part: Mapping[str, Any]) -> dict[str, Any]:
    image_value = part.get("image")
    if image_value is not None:
        return {"type": "image", "image": image_value}

    image_url = part.get("image_url")
    if isinstance(image_url, str):
        return {"type": "image", "image": image_url}
    if isinstance(image_url, Mapping) and isinstance(image_url.get("url"), str):
        return {"type": "image", "image": image_url["url"]}
    raise TypeError(f"Unsupported image content part: {part}")


def _normalize_content(content: Any) -> str | list[dict[str, Any]]:
    """Convert OpenAI-style message content into Tinker's text or structured format."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        part_type = content.get("type")
        if part_type in {"image", "image_url"}:
            return [_normalize_image_part(content)]
        if part_type == "thinking" and isinstance(content.get("thinking"), str):
            return [{"type": "thinking", "thinking": content["thinking"]}]
        if isinstance(content.get("text"), str):
            return content["text"]
        raise TypeError(f"Unsupported message content mapping: {content}")
    if isinstance(content, Sequence):
        normalized_parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                normalized_parts.append({"type": "text", "text": part})
                continue
            if not isinstance(part, Mapping):
                raise TypeError(f"Unsupported message content part: {part!r}")
            part_type = part.get("type")
            if part_type == "text" and isinstance(part.get("text"), str):
                normalized_parts.append({"type": "text", "text": part["text"]})
                continue
            if part_type in {"image", "image_url"}:
                normalized_parts.append(_normalize_image_part(part))
                continue
            if part_type == "thinking" and isinstance(part.get("thinking"), str):
                normalized_parts.append(
                    {"type": "thinking", "thinking": part["thinking"]}
                )
                continue
            raise TypeError(f"Unsupported message content part: {part!r}")
        # Preserve the content parts instead of collapsing an all-text list into
        # a single joined string. Joining is lossy: it discards per-part
        # boundaries, and each model's chat template trims parts differently
        # (gemma-4 trims EACH part before joining, e.g. ["a ", "b"] -> "ab";
        # every other cookbook renderer concatenates raw, e.g. -> "a b"). By
        # keeping the list, each renderer applies its own per-part policy so the
        # training tokens match the template. Renderers already handle list
        # content (mixed text / thinking / image), and a single-element text
        # list renders identically to the equivalent string.
        return normalized_parts
    raise TypeError(f"Unsupported message content type: {type(content)!r}")


def _ensure_content_parts(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def _any_message_has_per_message_training_flag(
    messages: Iterable[Mapping[str, Any]],
) -> bool:
    """Return True if any message carries an explicit per-message training flag.

    The Fireworks SFT dataset schema uses ``weight`` (int, 0 or 1) on
    individual messages to mark which assistant turns should contribute loss.
    Tinker's schema uses ``trainable`` (bool) for the same purpose. Either
    field enables the per-message ``CUSTOMIZED`` training path.
    """
    return any(("weight" in m) or ("trainable" in m) for m in messages)


def _resolve_trainable(
    message: Mapping[str, Any],
    *,
    assistant_default: bool,
) -> bool:
    """Derive the per-message ``trainable`` flag from ``trainable`` or ``weight``.

    ``trainable`` (bool) wins if present. Otherwise, a legacy ``weight`` (int
    or float) maps to ``bool(weight)`` — matching the V1 SFT trainer
    convention where ``weight=0`` marks an assistant message as context only
    (no loss) and ``weight=1`` (or the field being absent) marks it as a
    trainable target. When neither field is present, falls back to
    ``assistant_default`` (True for assistants, False otherwise).
    """
    trainable = message.get("trainable")
    if trainable is not None:
        return bool(trainable)

    weight = message.get("weight")
    if weight is not None:
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise TypeError(
                f"Unsupported weight value type: {type(weight)!r} (expected int or float)"
            )
        return bool(weight)

    return assistant_default


def normalize_messages(
    messages: Iterable[Mapping[str, Any]],
) -> list[Message]:
    """Normalize cookbook/eval-style messages into Tinker's message schema.

    When any message in the conversation carries an explicit ``weight`` or
    ``trainable`` field, every returned message gets a ``trainable`` flag so
    that renderers invoked with ``train_on_what=CUSTOMIZED`` can honor the
    per-message selection. Assistant messages default to trainable=True,
    non-assistant messages to False. The legacy Fireworks ``weight`` field
    (0 or 1) is translated to ``trainable = bool(weight)``, matching the V1
    SFT trainer semantics.
    """
    messages_list = list(messages)
    use_per_message_trainable = _any_message_has_per_message_training_flag(
        messages_list
    )

    normalized: list[Message] = []
    for message in messages_list:
        role = message.get("role")
        if not isinstance(role, str):
            raise ValueError(f"Message is missing a string role: {message}")

        normalized_message: Message = {
            "role": role,
            "content": _normalize_content(message.get("content")),
        }

        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            normalized_message["tool_calls"] = _normalize_tool_calls(tool_calls)

        thinking = message.get("thinking")
        if thinking is not None:
            if not isinstance(thinking, str):
                raise TypeError(f"Unsupported thinking value type: {type(thinking)!r}")
            normalized_message["content"] = [
                {"type": "thinking", "thinking": thinking},
                *_ensure_content_parts(normalized_message["content"]),
            ]

        # OpenAI/Fireworks chat convention: assistant turns carry their
        # chain-of-thought in top-level ``reasoning`` / ``reasoning_content``
        # strings (rather than Tinker's ``thinking`` field). Without these
        # branches, reasoning traces in training datasets are silently dropped
        # here before they ever reach a renderer, so thought blocks show up
        # empty in the training tokens and the fine-tuned model never learns
        # to emit a CoT. Promote both into ``ThinkingPart`` for renderers that
        # understand structured thinking (qwen3*, kimi_k2*, kimi_k25*,
        # kimi_k26*, deepseekv3_thinking, gemma4, gemma4_thinking, nemotron3*,
        # gpt_oss_*).
        # Generic precedence uses Jinja truthiness: ``thinking`` > non-empty
        # ``reasoning`` > non-empty ``reasoning_content``. Kimi is different:
        # its official template uses field presence for ``reasoning``. Preserve
        # both original source strings on private metadata keys so Kimi's
        # concrete adapters can apply that rule without making Qwen/Gemma drop
        # a non-empty ``reasoning_content`` merely because ``reasoning=""``.
        reasoning = message.get("reasoning")
        reasoning_content = message.get("reasoning_content")
        if thinking is None:
            if reasoning is not None and not isinstance(reasoning, str):
                raise TypeError(
                    f"Unsupported reasoning value type: {type(reasoning)!r}"
                )

            if reasoning_content is not None and not isinstance(
                reasoning_content, str
            ):
                raise TypeError(
                    "Unsupported reasoning_content value type: "
                    f"{type(reasoning_content)!r}"
                )

            # ``Message`` is a TypedDict owned by tinker_cookbook. These
            # private keys are intentionally runtime-only metadata consumed by
            # cookbook-local renderer adapters and ignored by upstream ones.
            if isinstance(reasoning, str):
                normalized_message[ORIGINAL_REASONING] = reasoning  # type: ignore[typeddict-unknown-key]
            if isinstance(reasoning_content, str):
                normalized_message[ORIGINAL_REASONING_CONTENT] = reasoning_content  # type: ignore[typeddict-unknown-key]

            promoted_reasoning = (
                reasoning
                if isinstance(reasoning, str) and reasoning
                else reasoning_content
                if isinstance(reasoning_content, str) and reasoning_content
                else None
            )
            if promoted_reasoning is not None:
                normalized_message["content"] = [
                    {"type": "thinking", "thinking": promoted_reasoning},
                    *_ensure_content_parts(normalized_message["content"]),
                ]

        if use_per_message_trainable:
            normalized_message["trainable"] = _resolve_trainable(
                message,
                assistant_default=(role == "assistant"),
            )

        tool_call_id = message.get("tool_call_id")
        if tool_call_id is not None:
            normalized_message["tool_call_id"] = str(tool_call_id)

        name = message.get("name")
        if name is not None:
            normalized_message["name"] = str(name)

        normalized.append(normalized_message)

    return normalized


def _stable_chunk_sentinel(chunk: Any) -> int:
    if isinstance(chunk, tinker.types.ImageAssetPointerChunk):
        payload = f"{chunk.type}:{chunk.location}:{chunk.format}:{chunk.expected_tokens}".encode()
    elif isinstance(chunk, tinker.types.ImageChunk):
        payload = b"|".join(
            [
                chunk.type.encode(),
                chunk.format.encode(),
                str(chunk.expected_tokens).encode(),
                bytes(chunk.data),
            ]
        )
    else:  # pragma: no cover - defensive branch for future chunk types
        payload = repr(chunk).encode()

    digest = hashlib.sha1(payload).digest()
    return -(int.from_bytes(digest[:8], "big") + 1)


def _flatten_model_input_sequence_ids(model_input: tinker.ModelInput) -> list[int]:
    sequence_ids: list[int] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            sequence_ids.extend(int(token) for token in chunk.tokens)
            continue

        sentinel = _stable_chunk_sentinel(chunk)
        sequence_ids.extend([sentinel] * int(chunk.length))
    return sequence_ids


def _rendered_sequence_ids_from_datum(datum: tinker.Datum) -> list[int]:
    sequence_ids = _flatten_model_input_sequence_ids(datum.model_input)
    target_tokens = [int(x) for x in datum.loss_fn_inputs["target_tokens"].data]
    if not target_tokens:
        raise ValueError(
            "Need at least one target token to reconstruct the rendered sequence."
        )
    return sequence_ids + [target_tokens[-1]]


def build_datum_from_tokens_and_weights(
    token_ids: Sequence[int],
    token_weights: Sequence[float],
    *,
    max_seq_len: int | None = None,
    include_loss_mask: bool = False,
) -> RenderedSupervisedDatum:
    """Build a weighted ``tinker.Datum`` from full tokens and per-token weights."""
    tokens = [int(x) for x in token_ids]
    weights = [float(x) for x in token_weights]
    if len(tokens) != len(weights):
        raise ValueError(
            f"tokens/weights length mismatch: {len(tokens)} != {len(weights)}"
        )
    if len(tokens) < 2:
        raise ValueError("Need at least 2 tokens to build a supervised datum.")

    if max_seq_len is not None:
        tokens = tokens[:max_seq_len]
        weights = weights[:max_seq_len]
        if len(tokens) < 2:
            raise ValueError("Truncation left fewer than 2 tokens.")

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    datum = datum_from_model_input_weights(
        tinker.ModelInput.from_ints(tokens),
        weight_tensor,
        max_length=max_seq_len,
    )

    if include_loss_mask:
        shifted_weights = [float(x) for x in datum.loss_fn_inputs["weights"].data]
        datum.loss_fn_inputs["loss_mask"] = tinker.TensorData(
            data=shifted_weights,
            dtype="float32",
            shape=[len(shifted_weights)],
        )

    return RenderedSupervisedDatum(
        token_ids=[int(x) for x in datum.model_input.to_ints()]
        + [int(datum.loss_fn_inputs["target_tokens"].data[-1])],
        token_weights=[0.0] + [float(x) for x in datum.loss_fn_inputs["weights"].data],
        datum=datum,
    )


def _extract_token_ids(model_input: tinker.ModelInput) -> list[int]:
    """Extract token IDs from a ModelInput, handling multimodal chunks.

    Text chunks contribute their token IDs directly.  Non-text chunks
    (e.g. ``ImageAssetPointerChunk``) contribute placeholder zeros
    matching their ``expected_tokens`` count.
    """
    ids: list[int] = []
    for chunk in model_input.chunks:
        if hasattr(chunk, "tokens"):
            ids.extend(chunk.tokens)
        elif hasattr(chunk, "expected_tokens"):
            ids.extend([0] * chunk.expected_tokens)
    return ids


def _extract_text_only_token_ids(model_input: tinker.ModelInput) -> list[int]:
    """Extract only text token IDs from a ModelInput, skipping image chunks.

    Non-text chunks are silently skipped.  The returned list contains only
    actual text token IDs in sequence order, with no placeholders for images.
    This is required because the server corrupts logprobs when target_tokens
    contains zeros at image chunk positions.
    """
    ids: list[int] = []
    for chunk in model_input.chunks:
        if hasattr(chunk, "tokens"):
            ids.extend(chunk.tokens)
    return ids


def has_non_text_chunks(model_input: tinker.ModelInput) -> bool:
    """Return True when *model_input* carries image (or other non-text) chunks."""
    return any(
        not isinstance(c, tinker.types.EncodedTextChunk) for c in model_input.chunks
    )


def _has_non_text_chunks(model_input: tinker.ModelInput) -> bool:
    return has_non_text_chunks(model_input)


def _chunk_weight_slots(chunk: Any) -> int:
    if isinstance(chunk, tinker.types.EncodedTextChunk):
        return len(chunk.tokens)
    if hasattr(chunk, "length"):
        return int(chunk.length)
    if hasattr(chunk, "expected_tokens"):
        return int(chunk.expected_tokens)
    return 0


def build_multimodal_rollout_weights(
    prompt_model_input: tinker.ModelInput,
    completion_token_count: int,
) -> list[float]:
    """Per-position training weights for a single-turn RL trajectory."""
    weights: list[float] = []
    for chunk in prompt_model_input.chunks:
        weights.extend([0.0] * _chunk_weight_slots(chunk))
    weights.extend([1.0] * completion_token_count)
    return weights


def build_multimodal_policy_datum(
    prompt_model_input: tinker.ModelInput,
    completion_tokens: Sequence[int],
    *,
    max_seq_len: int | None = None,
) -> tinker.Datum:
    """Build a policy :class:`tinker.Datum` that preserves image chunks in ``model_input``."""
    completion = [int(t) for t in completion_tokens]
    chunks = list(prompt_model_input.chunks) + [
        tinker.types.EncodedTextChunk(tokens=completion)
    ]
    full_input = tinker.ModelInput(chunks=chunks)
    weights = build_multimodal_rollout_weights(prompt_model_input, len(completion))
    if len(weights) != full_input.length:
        raise ValueError(
            f"multimodal rollout weights length {len(weights)} != "
            f"model_input.length {full_input.length}"
        )
    return _build_multimodal_datum(full_input, weights, max_seq_len)


def _build_multimodal_datum(
    model_input: tinker.ModelInput,
    weights: list[float],
    max_seq_len: int | None = None,
) -> tinker.Datum:
    """Build a next-token-prediction datum that preserves image chunks.

    ``target_tokens`` contains only text token IDs (no image placeholders).
    The server uses these for logprob gathering; including zeros at image
    positions corrupts the logprob computation.
    """
    token_ids = _extract_token_ids(model_input)

    if max_seq_len is not None and len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]
        weights = weights[:max_seq_len]

    if len(token_ids) < 2:
        raise ValueError("Need at least 2 tokens to build a supervised datum.")

    input_mi = _truncate_model_input(model_input)
    shifted_weights = weights[1:]

    text_target_tokens = _extract_text_only_token_ids(model_input)[1:]

    return tinker.Datum(
        model_input=input_mi,
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=shifted_weights,
                dtype="float32",
                shape=[len(shifted_weights)],
            ),
            "target_tokens": tinker.TensorData(
                data=[int(x) for x in text_target_tokens],
                dtype="int64",
                shape=[len(text_target_tokens)],
            ),
        },
    )


def build_datum_from_model_input_and_weights(
    model_input: tinker.ModelInput,
    token_weights: Sequence[int | float],
    *,
    max_seq_len: int | None = None,
    include_loss_mask: bool = False,
) -> RenderedSupervisedDatum:
    """Build a weighted datum from a multimodal-capable ``ModelInput``."""
    weights = [float(x) for x in token_weights]

    if _has_non_text_chunks(model_input):
        # Multimodal path: use our datum builder which produces text-only
        # target_tokens.  The upstream datum_from_model_input_weights includes
        # image-position zeros in target_tokens that corrupt the server's
        # logprob computation.
        datum = _build_multimodal_datum(model_input, weights, max_seq_len)
    elif datum_from_model_input_weights is not None:
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        if weight_tensor.numel() != model_input.length:
            raise ValueError(
                f"model_input/weights length mismatch: {model_input.length} != {weight_tensor.numel()}"
            )
        datum = datum_from_model_input_weights(
            model_input, weight_tensor, max_length=max_seq_len
        )
    else:
        token_ids = _extract_token_ids(model_input)
        return build_datum_from_tokens_and_weights(
            token_ids,
            token_weights,
            max_seq_len=max_seq_len,
            include_loss_mask=include_loss_mask,
        )

    if include_loss_mask:
        shifted_weights = [float(x) for x in datum.loss_fn_inputs["weights"].data]
        datum.loss_fn_inputs["loss_mask"] = tinker.TensorData(
            data=shifted_weights,
            dtype="float32",
            shape=[len(shifted_weights)],
        )

    return RenderedSupervisedDatum(
        token_ids=_rendered_sequence_ids_from_datum(datum),
        token_weights=[0.0] + [float(x) for x in datum.loss_fn_inputs["weights"].data],
        datum=datum,
    )


def build_next_token_datum(token_ids: Sequence[int]) -> tinker.Datum:
    """Build a standard next-token prediction datum without token weights."""
    tokens = [int(x) for x in token_ids]
    if len(tokens) < 2:
        raise ValueError("Need at least 2 tokens to build a next-token datum.")
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=tokens[1:],
                dtype="int64",
                shape=[len(tokens) - 1],
            )
        },
    )


def build_datum_from_token_mask(
    token_ids: Sequence[int],
    token_mask: Sequence[int | float],
    *,
    max_seq_len: int | None = None,
    include_loss_mask: bool = False,
) -> RenderedSupervisedDatum:
    """Build a weighted datum from an eval-protocol-style per-token mask."""
    token_weights = [1.0 if float(mask_value) > 0 else 0.0 for mask_value in token_mask]
    return build_datum_from_tokens_and_weights(
        token_ids,
        token_weights,
        max_seq_len=max_seq_len,
        include_loss_mask=include_loss_mask,
    )


def render_messages_to_datum(
    messages: Sequence[Mapping[str, Any]],
    *,
    renderer: Renderer,
    train_on_what: str | TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    max_seq_len: int | None = None,
    include_loss_mask: bool = False,
) -> RenderedSupervisedDatum:
    """Render a multi-turn conversation into the shared weighted datum format.

    When any message in the conversation carries an explicit ``weight`` or
    ``trainable`` field, ``train_on_what`` is overridden to ``CUSTOMIZED`` so
    that the renderer honors per-message training flags. This matches the V1
    SFT trainer contract (``weight=0`` means "context-only, no loss") and
    prevents the cookbook from silently training on assistant turns the
    dataset author explicitly excluded.
    """
    normalized_messages = normalize_messages(messages)
    effective_train_on_what = parse_train_on_what(train_on_what)
    if any("trainable" in m for m in normalized_messages):
        effective_train_on_what = TrainOnWhat.CUSTOMIZED
    rendered_input, weights = renderer.build_supervised_example(
        normalized_messages,
        train_on_what=_equivalent_single_example_train_on_what(
            renderer,
            normalized_messages,
            effective_train_on_what,
        ),
    )
    return _build_rendered_supervised_datum(
        rendered_input,
        weights,
        max_seq_len=max_seq_len,
        include_loss_mask=include_loss_mask,
    )


def _build_rendered_supervised_datum(
    rendered_input: Any,
    weights: Any,
    *,
    max_seq_len: int | None,
    include_loss_mask: bool,
) -> RenderedSupervisedDatum:
    weight_values = weights.tolist() if hasattr(weights, "tolist") else list(weights)
    if isinstance(rendered_input, tinker.ModelInput):
        return build_datum_from_model_input_and_weights(
            rendered_input,
            weight_values,
            max_seq_len=max_seq_len,
            include_loss_mask=include_loss_mask,
        )
    token_values = (
        rendered_input.tolist()
        if hasattr(rendered_input, "tolist")
        else list(rendered_input)
    )
    return build_datum_from_tokens_and_weights(
        token_values,
        weight_values,
        max_seq_len=max_seq_len,
        include_loss_mask=include_loss_mask,
    )


_SPLIT_REQUIRED_TRAINING_MODES = {
    TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    TrainOnWhat.ALL_MESSAGES,
    TrainOnWhat.ALL_TOKENS,
    TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES,
    TrainOnWhat.CUSTOMIZED,
}


def _requires_renderer_supervised_examples(
    renderer: Renderer,
    messages: list[Message],
    train_on_what: TrainOnWhat,
) -> bool:
    """Whether rendering must use the renderer's multi-example dispatcher."""
    if getattr(renderer, "has_extension_property", False):
        return False
    if train_on_what not in _SPLIT_REQUIRED_TRAINING_MODES:
        return False
    return sum(1 for message in messages if message["role"] == "user") > 1


def _equivalent_single_example_train_on_what(
    renderer: Renderer,
    messages: list[Message],
    train_on_what: TrainOnWhat,
) -> TrainOnWhat:
    """Use an equivalent mode for a singular render when possible.

    Upstream Tinker warns on ALL_ASSISTANT_MESSAGES for every non-extension
    renderer, but for a singular render where every assistant message is after
    the last user message, ALL_ASSISTANT_MESSAGES and LAST_ASSISTANT_TURN
    assign identical token weights. Passing the narrower equivalent mode avoids
    the false-positive warning while preserving true warnings for non-equivalent
    conversations.
    """
    if train_on_what != TrainOnWhat.ALL_ASSISTANT_MESSAGES or getattr(
        renderer, "has_extension_property", False
    ):
        return train_on_what

    last_user_idx = max(
        (idx for idx, message in enumerate(messages) if message["role"] == "user"),
        default=-1,
    )
    assistant_idxs = [
        idx for idx, message in enumerate(messages) if message["role"] == "assistant"
    ]
    if assistant_idxs and all(idx > last_user_idx for idx in assistant_idxs):
        return TrainOnWhat.LAST_ASSISTANT_TURN

    return train_on_what


def _build_renderer_supervised_examples(
    renderer: Renderer,
    messages: list[Message],
    train_on_what: TrainOnWhat,
) -> list[tuple[Any, Any]]:
    if _requires_renderer_supervised_examples(renderer, messages, train_on_what):
        build_supervised_examples = getattr(renderer, "build_supervised_examples", None)
        if build_supervised_examples is None:
            raise TypeError(
                f"{type(renderer).__name__} has has_extension_property=False and "
                f"cannot safely render train_on_what={train_on_what.value!r} without "
                "build_supervised_examples."
            )
        return list(
            build_supervised_examples(
                messages,
                train_on_what=train_on_what,
            )
        )

    return [
        renderer.build_supervised_example(
            messages,
            train_on_what=_equivalent_single_example_train_on_what(
                renderer,
                messages,
                train_on_what,
            ),
        )
    ]


def render_messages_to_datums(
    messages: Sequence[Mapping[str, Any]],
    *,
    renderer: Renderer,
    train_on_what: str | TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    max_seq_len: int | None = None,
    include_loss_mask: bool = False,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> list[RenderedSupervisedDatum]:
    """Render a chat row, splitting multi-target rows when required.

    When ``tools`` is provided (top-level OpenAI function-calling array
    with ``{"type": "function", "function": {...}}`` items) and the
    renderer implements ``create_conversation_prefix_with_tools``, the
    tool definitions are encoded as renderer-specific prefix messages and
    prepended to the conversation. An existing leading system message's
    content is preserved as the system prompt passed to the renderer.
    Renderers without tool-prefix support silently drop the field.
    """
    normalized_messages = build_tool_prefixed_messages(
        messages, renderer=renderer, tools=tools
    )

    effective_train_on_what = parse_train_on_what(train_on_what)
    if any("trainable" in m for m in normalized_messages):
        effective_train_on_what = TrainOnWhat.CUSTOMIZED

    examples = _build_renderer_supervised_examples(
        renderer,
        normalized_messages,
        effective_train_on_what,
    )

    return [
        _build_rendered_supervised_datum(
            rendered_input,
            weights,
            max_seq_len=max_seq_len,
            include_loss_mask=include_loss_mask,
        )
        for rendered_input, weights in examples
    ]


def _common_prefix_length(tokens_a: Sequence[int], tokens_b: Sequence[int]) -> int:
    min_len = min(len(tokens_a), len(tokens_b))
    for idx in range(min_len):
        if int(tokens_a[idx]) != int(tokens_b[idx]):
            return idx
    return min_len


def _render_preference_item_tokens(
    item: Mapping[str, Any],
    *,
    renderer: Renderer,
    tokenizer: Any,
) -> tuple[list[int], tinker.Datum] | None:
    if "messages" in item:
        messages = item.get("messages") or []
        if not messages:
            return None
        rendered = render_messages_to_datum(messages, renderer=renderer)
        return rendered.token_ids, rendered.datum
    if isinstance(item.get("text"), str):
        token_ids = [int(x) for x in tokenizer.encode(item["text"])]
        return token_ids, build_next_token_datum(token_ids)
    return None


def render_preference_pair(
    chosen: Mapping[str, Any],
    rejected: Mapping[str, Any],
    *,
    renderer: Renderer,
    tokenizer: Any,
    max_seq_len: int | None = None,
) -> RenderedPreferencePair | None:
    """Render a chosen/rejected pair through the shared tokenizer path."""
    chosen_rendered = _render_preference_item_tokens(
        chosen, renderer=renderer, tokenizer=tokenizer
    )
    rejected_rendered = _render_preference_item_tokens(
        rejected, renderer=renderer, tokenizer=tokenizer
    )
    if chosen_rendered is None or rejected_rendered is None:
        return None
    chosen_tokens, chosen_datum = chosen_rendered
    rejected_tokens, rejected_datum = rejected_rendered
    if len(chosen_tokens) < 2 or len(rejected_tokens) < 2:
        return None
    if max_seq_len is not None and (
        len(chosen_tokens) > max_seq_len or len(rejected_tokens) > max_seq_len
    ):
        return None

    return RenderedPreferencePair(
        chosen_tokens=chosen_tokens,
        rejected_tokens=rejected_tokens,
        response_start=_common_prefix_length(chosen_tokens, rejected_tokens),
        chosen_datum=chosen_datum,
        rejected_datum=rejected_datum,
    )
