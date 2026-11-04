"""Pinned Muse Spark 1.2 open TITO model-format primitive.

Contract (verified against the pinned ``muse_spark_1_2_oss.jinja`` chat template
carried by the certified tokenizer bundle): the generation prompt ends at
``<|start|>assistant``; a completion opens a channel with `` to=<recipient>``
followed by ``<|message|>``. ``self`` is the private reasoning channel,
``commentary`` is a user-visible non-terminal channel, ``user`` is the visible
terminal channel, and a tool name introduces an ``<atem:function_calls>`` block.
Channels continue the same sampled response across ``<|eom|><|start|>assistant``
junctions and the turn terminates at ``<|eot|>``.

This is the TITO counterpart of ``training.renderer.muse_spark``. It shares that
renderer's protocol but not its code: TITO delegates all prompt construction to
the tokenizer's authoritative chat template and owns only request validation and
the output parser.

Two deliberate divergences from the training renderer's defaults, both required
by TITO's exact-token contract:

* ``current_date`` is omitted rather than defaulted to today. The template's
  default calls ``strftime_now``, so an unpinned render changes at UTC midnight;
  mid-trajectory that rewrites the system block and invalidates every stored
  checkpoint. Training runs that mix with these trajectories should render with
  ``MuseSparkOptions(include_current_date=False)``.
* ``reasoning_effort`` is fixed per certification instead of being read from the
  request, so the sampled prompt cannot drift between calls in one trajectory.

Muse Spark is certified for full-history rendering only. The template flips a
terminal assistant message from ``<|eot|>`` to ``<|eom|>`` when another
assistant message follows, so a stored checkpoint is never a prefix of the
extended render and incremental prompt construction is not claimed.

Images are not supported here. The checkpoint renders each image as ``<|patch|>``
and relies on an out-of-band pixel payload, which an exact-token rollout does not
carry; image requests are rejected rather than silently rendered as text.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from fireworks.training.sdk import (
    TITOParseError,
    TITOChatRequest,
    TITOError,
    TITOIncrementalPrompt,
    TITOParsedAssistant,
)

from training.renderer.tito.shared import (
    TITORendererCertification,
    _ensure_tool_call_ids,
    _normalize_template_messages,
    _normalize_template_tools,
    validate_no_dynamic_template_fields,
)

MUSE_SPARK_RENDERER_NAME = "muse_spark"

# Vendor numeric reasoning budgets, mirroring the training renderer.
REASONING_EFFORTS = ("low", "medium", "high", "xhigh")
_DEFAULT_REASONING_EFFORT = "xhigh"

# Pinned to the open checkpoint's tokenizer bundle at revision
# 112a20b71946a6c662da04adb623530861245016. Regenerate with:
#   python -m training.renderer.tito.muse_spark <tokenizer-dir>
MUSE_SPARK_TOKENIZER_FINGERPRINT = "41b0094585d144e919ed1bd5534cd00df9086dfc2ac8f2b21c39e58cd606a271"

_CERTIFICATION_PREFIX = "muse-spark-1p2-open"
_CERTIFICATION_REVISION = "v2"

_EOT = "<|eot|>"
_EOM = "<|eom|>"
_START = "<|start|>"
_MESSAGE = "<|message|>"
_SEGMENT_SEPARATOR = _EOM + _START + "assistant"
_RESERVED_RECIPIENTS = frozenset({"self", "commentary", "user"})

# Attributes may precede ``name`` and the block carries no required wrapper
# beyond ``<atem:invoke>``; match the training renderer's tolerance.
_ATEM_CALL_RE = re.compile(
    r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">(?P<body>.*?)</atem:invoke>',
    re.DOTALL,
)
_ATEM_ARG_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<name>[^"]+)"[^>]*?>' r"(?P<value>.*?)</atem:parameter>",
    re.DOTALL,
)

_IMAGE_PART_TYPES = frozenset(
    {
        "image",
        "input_image",
        "image_url",
        "video",
        "input_video",
        "video_url",
        "audio",
        "input_audio",
        "audio_url",
    }
)


class MuseSparkTITORenderer:
    """Pinned Muse Spark 1.2 open TITO model-format primitive."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        certification: TITORendererCertification,
        reasoning_effort: str = _DEFAULT_REASONING_EFFORT,
    ) -> None:
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                "Muse Spark reasoning_effort must be one of "
                + ", ".join(REASONING_EFFORTS)
                + f"; got {reasoning_effort!r}"
            )
        self.tokenizer = tokenizer
        self.renderer_id = MUSE_SPARK_RENDERER_NAME
        self.certification_id = certification.certification_id
        self.tokenizer_fingerprint = certification.tokenizer_fingerprint
        self.reasoning_effort = reasoning_effort
        self._eot_token = self._single_token(_EOT)
        self._stop = (self._eot_token,)

    def _single_token(self, text: str) -> int:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Muse Spark expected {text!r} to encode as one token")
        return int(tokens[0])

    def _validate_request(self, request: TITOChatRequest) -> None:
        validate_no_dynamic_template_fields(request)
        for tool in request.tools:
            name = str((tool.get("function") or {}).get("name") or "")
            if name in _RESERVED_RECIPIENTS:
                # The recipient slot is the only thing distinguishing a tool
                # channel from a reasoning or visible one, so a tool with a
                # reserved name would be parsed back as text and silently
                # dropped rather than executed.
                raise TITOError(
                    "tito_invalid_request",
                    400,
                    f"Muse Spark reserves the {name!r} recipient; a tool cannot " "use it as its function name",
                )
        for message in request.messages:
            content = message.get("content")
            if isinstance(content, str) or content is None:
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if str(part.get("type", "")) in _IMAGE_PART_TYPES:
                    raise TITOError(
                        "tito_invalid_request",
                        400,
                        "Muse Spark TITO rollouts are text-only; media content "
                        "requires an out-of-band payload that an exact-token "
                        "rollout does not carry",
                    )

    def _template_inputs(
        self,
        request: TITOChatRequest,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        wire = request.wire_value()
        if wire is not None:
            messages = wire.get("messages") or ()
            tools = wire.get("tools") or ()
        else:
            messages = request.messages
            tools = request.tools
        return _normalize_template_messages(messages), _normalize_template_tools(tools)

    def _render_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any:
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            reasoning_effort=self.reasoning_effort,
            # Defined-but-falsy suppresses the template's ``strftime_now``
            # default and omits the "Current date" line entirely.
            current_date=None,
        )

    def _render(self, request: TITOChatRequest) -> tuple[int, ...]:
        self._validate_request(request)
        messages, tools = self._template_inputs(request)
        rendered = self._render_template(
            messages,
            tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(rendered, Mapping):
            rendered = rendered["input_ids"]
        return tuple(int(token) for token in rendered)

    def render_conversation_tokens(self, request: TITOChatRequest) -> Sequence[int]:
        return self._render(request)

    def prepare_incremental_prompt(
        self,
        request: TITOChatRequest,
        stored_messages: Sequence[Mapping[str, Any]],
        appended_messages: Sequence[Mapping[str, Any]],
        exact_checkpoint_ids: Sequence[int],
    ) -> TITOIncrementalPrompt | None:
        """Muse Spark is certified for full-history rendering only.

        The template rewrites a terminal ``<|eot|>`` into ``<|eom|>`` when a
        consecutive assistant message is appended, so an exact stored checkpoint
        is never a prefix of the extended render.
        """
        del request, stored_messages, appended_messages, exact_checkpoint_ids
        return None

    @staticmethod
    def _parse_argument(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _parse_calls(self, payload: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for match in _ATEM_CALL_RE.finditer(payload):
            name = html.unescape(match.group("name"))
            if not name:
                raise TITOParseError("Muse Spark tool call is missing a name")
            arguments = {
                html.unescape(item.group("name")): self._parse_argument(item.group("value"))
                for item in _ATEM_ARG_RE.finditer(match.group("body"))
            }
            residue = _ATEM_ARG_RE.sub("", match.group("body")).strip()
            if residue:
                raise TITOParseError(f"unparsed Muse Spark tool-call content: {residue!r}")
            calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
        return calls

    @staticmethod
    def _reject_unrepresentable_shape(
        kinds: Sequence[str],
        visible_recipient: str | None,
    ) -> None:
        """Reject turns a single assistant message cannot re-render verbatim.

        The template emits at most one ``to=self`` block, then either one
        ``to=commentary`` body followed by tool calls, or exactly one visible
        block. A sampled turn is free to leave that shape -- ``commentary``
        then ``user`` is legal for the model, since commentary is explicitly a
        pre-tool/intermediate channel. One OpenAI message carries a single
        ``content`` and a single ``recipient``, so those turns would silently
        re-render as one merged block and break the exact-token lineage that
        full-history TITO depends on. Fail closed instead of collapsing them.
        """
        if kinds.count("reasoning") > 1:
            raise TITOParseError("Muse Spark emitted multiple reasoning segments")
        if "reasoning" in kinds and kinds[0] != "reasoning":
            raise TITOParseError("Muse Spark emitted reasoning after a visible segment")
        if kinds.count("visible") > 1:
            raise TITOParseError("Muse Spark emitted multiple visible segments")
        if "tool" not in kinds:
            return
        if "visible" not in kinds:
            return
        if visible_recipient != "commentary":
            # A tool-call turn re-renders its body on the commentary channel,
            # so a ``to=user`` body alongside tool calls cannot round-trip.
            raise TITOParseError("Muse Spark emitted a non-commentary body with tool calls")
        if kinds.index("visible") > kinds.index("tool"):
            raise TITOParseError("Muse Spark emitted a commentary body after tool calls")

    def _parse_message(
        self,
        request: TITOChatRequest,
        completion_ids: Sequence[int],
    ) -> tuple[dict[str, Any], bool]:
        text = str(self.tokenizer.decode([int(token) for token in completion_ids]))
        stop_at = text.find(_EOT)
        clean = stop_at >= 0
        if clean:
            text = text[:stop_at]
        # A complete nonterminal message ends with EOM. That is framing rather
        # than payload, but it does not terminate the turn: keep ``clean`` false.
        text = text.removesuffix(_EOM)
        # Sampling begins right after the prompt-owned ``<|start|>assistant``.
        # Be liberal when a caller feeds a full assistant segment back in.
        text = text.removeprefix(_START + "assistant")

        kinds: list[str] = []
        reasoning_parts: list[str] = []
        visible_parts: list[str] = []
        calls: list[dict[str, Any]] = []
        visible_recipient: str | None = None
        for segment in text.split(_SEGMENT_SEPARATOR):
            if _MESSAGE not in segment:
                if segment.strip():
                    raise TITOParseError("Muse Spark segment has no message boundary")
                continue
            recipient, payload = segment.split(_MESSAGE, 1)
            if not recipient.startswith(" to="):
                raise TITOParseError("Muse Spark segment is missing its recipient")
            recipient = recipient[len(" to=") :]
            if recipient == "self":
                kinds.append("reasoning")
                reasoning_parts.append(payload)
                continue
            if recipient in {"commentary", "user"}:
                kinds.append("visible")
                visible_parts.append(payload)
                visible_recipient = recipient
                continue
            segment_calls = self._parse_calls(payload)
            if not segment_calls:
                raise TITOParseError(f"unknown Muse Spark recipient {recipient!r}")
            kinds.append("tool")
            calls.extend(segment_calls)
        self._reject_unrepresentable_shape(kinds, visible_recipient)

        allowed_tool_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in request.tools
            if (tool.get("function") or {}).get("name")
        }
        if calls and (
            not allowed_tool_names or any(str(call["function"]["name"]) not in allowed_tool_names for call in calls)
        ):
            raise TITOParseError("Muse Spark tool call names are absent from the request")

        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(visible_parts),
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if calls:
            # The template already re-renders content on a tool-call turn as a
            # commentary segment, so the recipient must not also be pinned here.
            message["tool_calls"] = calls
        elif visible_recipient == "commentary":
            # Preserve the non-terminal channel so re-rendering this turn
            # reproduces ``to=commentary`` and ``<|eom|>`` rather than
            # ``to=user`` and ``<|eot|>``.
            message["recipient"] = "commentary"
        return message, clean

    def parse_assistant(
        self,
        request: TITOChatRequest,
        completion_ids: Sequence[int],
        completion_text: str,
        finish_reason: str,
    ) -> TITOParsedAssistant:
        del completion_text
        message, clean = self._parse_message(request, completion_ids)
        if not clean and finish_reason != "length":
            # Servers strip the stop token from the returned completion; retry
            # once with it appended before declaring the parse unclean.
            message, clean = self._parse_message(
                request,
                [*completion_ids, self._eot_token],
            )
        if not clean and finish_reason != "length":
            raise TITOParseError("unclean Muse Spark renderer parse")
        message = _ensure_tool_call_ids(message, completion_ids)
        tool_calls = message.get("tool_calls") or []
        if finish_reason == "length" and tool_calls:
            raise TITOParseError("truncated structured tool call is not protocol-safe")
        output_kind = (
            "tool_calls" if tool_calls else "reasoning" if message.get("reasoning_content") is not None else "text"
        )
        return TITOParsedAssistant(message=message, output_kind=output_kind)

    def render_contract_id(self, request: TITOChatRequest) -> str:
        contract = {
            "renderer_id": self.renderer_id,
            "certification_id": self.certification_id,
            "model": request.model,
            "tools": [dict(tool) for tool in request.tools],
            "renderer_class": type(self).__qualname__,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "reasoning_effort": self.reasoning_effort,
        }
        return hashlib.sha256(
            json.dumps(
                contract,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()

    def stop_sequences(self, request: TITOChatRequest) -> Sequence[str]:
        del request
        return tuple(str(self.tokenizer.decode([token])) for token in self._stop)


def _build_renderer(reasoning_effort: str):
    def factory(
        tokenizer: Any,
        certification: TITORendererCertification,
    ) -> MuseSparkTITORenderer:
        return MuseSparkTITORenderer(
            tokenizer,
            certification=certification,
            reasoning_effort=reasoning_effort,
        )

    return factory


def muse_spark_certifications() -> tuple[TITORendererCertification, ...]:
    """Return one certification per pinned reasoning-effort contract.

    Effort is part of the rendered system block, so each effort is a distinct
    prompt contract and gets its own certification and renderer name rather than
    a per-request option.
    """
    certifications = [
        TITORendererCertification(
            certification_id=(
                f"{_CERTIFICATION_PREFIX}-default-{_DEFAULT_REASONING_EFFORT}" f"@{_CERTIFICATION_REVISION}"
            ),
            renderer_names=frozenset({MUSE_SPARK_RENDERER_NAME}),
            tokenizer_fingerprint=MUSE_SPARK_TOKENIZER_FINGERPRINT,
            renderer_factory=_build_renderer(_DEFAULT_REASONING_EFFORT),
        )
    ]
    certifications.extend(
        TITORendererCertification(
            certification_id=(f"{_CERTIFICATION_PREFIX}-{effort}@{_CERTIFICATION_REVISION}"),
            renderer_names=frozenset({f"{MUSE_SPARK_RENDERER_NAME}_{effort}"}),
            tokenizer_fingerprint=MUSE_SPARK_TOKENIZER_FINGERPRINT,
            renderer_factory=_build_renderer(effort),
        )
        for effort in REASONING_EFFORTS
    )
    return tuple(certifications)


def _main(argv: Sequence[str]) -> int:
    """Print the certification fingerprint for a pinned tokenizer directory."""
    from transformers.tokenization_utils_tokenizers import TokenizersBackend

    from training.renderer.tito.shared import _tokenizer_fingerprint

    if len(argv) != 1:
        print("usage: python -m training.renderer.tito.muse_spark <tokenizer-dir>")
        return 2
    tokenizer = TokenizersBackend.from_pretrained(
        argv[0],
        local_files_only=True,
        trust_remote_code=False,
    )
    print(_tokenizer_fingerprint(tokenizer))
    return 0


__all__ = [
    "MUSE_SPARK_RENDERER_NAME",
    "MUSE_SPARK_TOKENIZER_FINGERPRINT",
    "MuseSparkTITORenderer",
    "REASONING_EFFORTS",
    "muse_spark_certifications",
]


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(_main(sys.argv[1:]))
