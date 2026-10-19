"""Shared TITO renderer primitives: tokenizer pinning and template normalization.

Prompt construction delegates to the pinned chat template or native encoder.
This module owns only protocol normalization shared by the
per-model renderers; importing it must not load Tinker or Torch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fireworks.training.sdk import (
    TITOError,
    normalize_openai_tool_arguments,
)

_DYNAMIC_TEMPLATE_FIELDS = frozenset(
    {
        "chat_template_kwargs",
        "clear_thinking",
        "drop_thinking",
        "enable_thinking",
        "preserve_thinking",
        "reasoning_effort",
        "response_format",
        "thinking",
    }
)


@dataclass(frozen=True)
class TITORendererCertification:
    """Reviewed full-history capability for one renderer/tokenizer contract.

    This base certification does not certify the experimental incremental
    method; renderer authors own that additional model-specific contract.
    """

    certification_id: str
    renderer_names: frozenset[str]
    tokenizer_fingerprint: str
    renderer_factory: Callable[[Any, "TITORendererCertification"], Any]
    artifact_repository: str | None = None
    artifact_revision: str | None = None
    artifact_files: tuple[tuple[str, str], ...] = ()


def renderer_artifacts(tokenizer: Any, certification: TITORendererCertification) -> dict[str, Path]:
    """Resolve only reviewed, hash-pinned native formatter artifacts.

    Local bundles never fetch missing files. Downloads are allowed only from
    the certification's exact repository/revision, not tokenizer remote code.
    """
    paths: dict[str, Path] = {}
    root = Path(tokenizer.name_or_path)
    for filename, expected_hash in certification.artifact_files:
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid certified renderer artifact path")
        if root.is_dir():
            path = root / relative
        else:
            if tokenizer.name_or_path != certification.artifact_repository or not certification.artifact_revision:
                raise ValueError("native renderer requires its pinned artifact source")
            # lazy: HF is needed only by host-side bundle construction, not the
            # offline container runtime.
            from huggingface_hub import hf_hub_download

            path = Path(
                hf_hub_download(
                    repo_id=certification.artifact_repository,
                    revision=certification.artifact_revision,
                    filename=filename,
                )
            )
        if not path.is_file():
            raise ValueError("TITO bundle is missing a certified renderer artifact")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("native renderer artifact does not match certification")
        paths[filename] = path
    return paths


def load_sidecar_tokenizer(path: str | Path, *, renderer_name: str | None = None) -> Any:
    """Load the pinned tokenizer and verify its chat template or native artifacts.

    The bundle is written by ``save_pretrained`` from an already-certified
    tokenizer, so ``tokenizer.json`` is always present (certification requires
    a serializable backend) and already carries that tokenizer's live
    post-processor. Load it verbatim rather than through ``AutoTokenizer``:
    Transformers >= 5.9 honors a declared ``tokenizer_class`` (Qwen3.8 declares
    ``Qwen2Tokenizer``) and would rebuild the tokenizer from that class's
    hardcoded pre-tokenizer, silently invalidating the certified contract.

    Fail closed here, unlike ``load_tokenizer``: a sidecar that cannot load its
    own pinned artifact has no certified token contract to fall back to.
    """
    from transformers.tokenization_utils_tokenizers import TokenizersBackend

    root = Path(path)
    if not (root / "tokenizer.json").is_file():
        raise ValueError(f"TITO sidecar bundle is missing tokenizer.json: {root}")
    tokenizer = TokenizersBackend.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    artifacts = {}
    if renderer_name is not None:
        # lazy: registry imports this module; resolve after it has initialized.
        from training.renderer.tito.registry import get_tito_renderer_certification

        certification = get_tito_renderer_certification(renderer_name, tokenizer)
        artifacts = renderer_artifacts(tokenizer, certification)
    if not chat_template and not artifacts:
        raise ValueError("TITO sidecar tokenizer has no bundled chat template")
    return tokenizer


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        raise ValueError(
            "production TITO certification requires a fast tokenizer with a "
            "serializable backend"
        )
    contract = {
        "backend": json.loads(backend.to_str()),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_no_dynamic_template_fields(request: Any) -> None:
    """Reject per-request template options; the certified contract fixes them."""
    fields = sorted(_DYNAMIC_TEMPLATE_FIELDS.intersection(request.sampling_fields))
    if fields:
        raise TITOError(
            "tito_invalid_request",
            400,
            "TITO renderer/template options are fixed by the certified "
            "renderer contract; unsupported per-request fields: "
            + ", ".join(fields),
        )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def parse_xml_tool_argument(
    value: str,
    *,
    parameter_name: str,
    tool_name: str,
    tools: Sequence[Mapping[str, Any]],
) -> Any:
    """Preserve XML text for explicitly string-typed tool parameters."""
    for tool in tools:
        function = tool.get("function") or {}
        if function.get("name") != tool_name:
            continue
        parameters = function.get("parameters") or {}
        schema = (parameters.get("properties") or {}).get(parameter_name) or {}
        if isinstance(schema, Mapping) and schema.get("type") == "string":
            return value
        break
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_template_tools(
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_tool in tools:
        tool = _plain(raw_tool)
        function = tool.get("function")
        if isinstance(function, dict):
            # Match ChatCompletionTool.model_dump() field order at Fireworks
            # chat admission. Templates render this function object directly,
            # so its envelope order is prompt-visible too.
            parameters = function.get("parameters") or {}
            tool["function"] = {
                "name": function.get("name"),
                "description": function.get("description"),
                # ChatCompletionFunction treats parameters as an opaque
                # mapping, so request admission preserves its key order.
                "parameters": _plain(parameters),
            }
            normalized.append(tool)
            continue
        normalized.append(tool)
    return normalized


def _normalize_template_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = [_plain(message) for message in messages]
    for message in normalized:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                # Templates iterate historical tool arguments as a mapping;
                # string arguments must round-trip to one.
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise TITOError(
                        "tito_invalid_request",
                        400,
                        "historical assistant tool arguments are not valid JSON",
                    ) from exc
    return normalized


def _ensure_tool_call_ids(
    message: Mapping[str, Any],
    completion_ids: Sequence[int],
) -> dict[str, Any]:
    calls = message.get("tool_calls") or []
    if not calls:
        return dict(message)
    normalized_calls: list[dict[str, Any]] = []
    for index, raw_call in enumerate(calls):
        call = dict(raw_call)
        function = dict(call.get("function") or {})
        if not call.get("id"):
            identity_function = {
                **function,
                "arguments": normalize_openai_tool_arguments(
                    function.get("arguments", "")
                ),
            }
            identity = {
                "completion_ids": [int(token) for token in completion_ids],
                "index": index,
                "function": identity_function,
            }
            digest = hashlib.sha256(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            call["id"] = f"call_{digest[:24]}"
        normalized_calls.append(call)
    return {**message, "tool_calls": normalized_calls}
