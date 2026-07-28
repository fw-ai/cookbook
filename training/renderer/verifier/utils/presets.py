"""Validated JSON preset catalogs for the renderer verifier UI."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from tinker_cookbook.renderers.base import TrainOnWhat

PRESET_SCHEMA_VERSION = 1

_CATALOG_KEYS = {"schema_version", "profiles"}
_PROFILE_KEYS = {"id", "label", "description", "defaults", "examples"}
_EXAMPLE_META_KEYS = {"id", "label", "description"}
_PROBE_KEYS = {
    "renderer",
    "tokenizer_model",
    "image_processor_model",
    "model",
    "deployment_id",
    "messages",
    "tools",
    "max_tokens",
    "temperature",
    "train_on_what",
    "extra_completion_kwargs",
}
_CREDENTIAL_KEYS = {
    "apikey",
    "authorization",
    "bearertoken",
    "credentials",
    "fireworksapikey",
    "hftoken",
    "secret",
    "xapikey",
}
_RESERVED_COMPLETION_KEYS = {
    "echo",
    "max_tokens",
    "messages",
    "model",
    "raw_output",
    "return_token_ids",
    "temperature",
    "tools",
}


def _require_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _credential_paths(value: Any, *, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            normalized_key = "".join(
                char for char in key_text.lower() if char.isalnum()
            )
            if normalized_key in _CREDENTIAL_KEYS:
                found.append(child_path)
            found.extend(_credential_paths(item, path=child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            found.extend(_credential_paths(item, path=child_path))
    return found


def _validate_completion_kwargs(value: Any, *, context: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    credential_paths = sorted(_credential_paths(value))
    if credential_paths:
        raise ValueError(
            f"{context} must not contain credential keys: {', '.join(credential_paths)}"
        )
    reserved = sorted(set(value) & _RESERVED_COMPLETION_KEYS)
    if reserved:
        raise ValueError(
            f"{context} must not override verifier-controlled keys: "
            f"{', '.join(reserved)}"
        )


def _validate_probe_values(
    value: Any,
    *,
    context: str,
    require_complete: bool,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    unknown = sorted(set(value) - _PROBE_KEYS)
    if unknown:
        raise ValueError(f"{context} has unsupported keys: {', '.join(unknown)}")

    for key in (
        "renderer",
        "tokenizer_model",
        "image_processor_model",
        "model",
        "deployment_id",
        "train_on_what",
    ):
        if key in value and value[key] is not None and not isinstance(value[key], str):
            raise ValueError(f"{context}.{key} must be a string or null")
    if value.get("train_on_what"):
        try:
            TrainOnWhat(value["train_on_what"])
        except ValueError as exc:
            raise ValueError(
                f"{context}.train_on_what is not a supported training mode"
            ) from exc

    if value.get("model") and value.get("deployment_id"):
        raise ValueError(f"{context} cannot set both model and deployment_id")

    if "messages" in value and not isinstance(value["messages"], list):
        raise ValueError(f"{context}.messages must be a JSON array")
    if "tools" in value and not isinstance(value["tools"], list):
        raise ValueError(f"{context}.tools must be a JSON array")
    if "extra_completion_kwargs" in value:
        _validate_completion_kwargs(
            value["extra_completion_kwargs"],
            context=f"{context}.extra_completion_kwargs",
        )
    if "max_tokens" in value and (
        type(value["max_tokens"]) is not int or value["max_tokens"] < 1
    ):
        raise ValueError(f"{context}.max_tokens must be a positive integer")
    if "temperature" in value and (
        isinstance(value["temperature"], bool)
        or not isinstance(value["temperature"], (int, float))
    ):
        raise ValueError(f"{context}.temperature must be a number")

    if require_complete:
        for key in ("renderer", "tokenizer_model", "messages"):
            if not value.get(key):
                raise ValueError(f"{context}.{key} is required after applying defaults")
        if not value.get("model") and not value.get("deployment_id"):
            raise ValueError(
                f"{context} requires either model or deployment_id after applying defaults"
            )


def load_preset_catalog(path: str | Path) -> dict[str, Any]:
    """Load and validate a browser-facing preset catalog."""
    preset_path = Path(path)
    with preset_path.open("r", encoding="utf-8") as file:
        catalog = json.load(file)
    if not isinstance(catalog, Mapping):
        raise ValueError("preset catalog must be a JSON object")
    unknown = sorted(set(catalog) - _CATALOG_KEYS)
    if unknown:
        raise ValueError(f"preset catalog has unsupported keys: {', '.join(unknown)}")
    if catalog.get("schema_version") != PRESET_SCHEMA_VERSION:
        raise ValueError(
            f"preset catalog schema_version must be {PRESET_SCHEMA_VERSION}"
        )

    profiles = catalog.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("preset catalog profiles must be a JSON array")

    profile_ids: set[str] = set()
    for profile_index, profile in enumerate(profiles):
        context = f"profiles[{profile_index}]"
        if not isinstance(profile, Mapping):
            raise ValueError(f"{context} must be a JSON object")
        unknown = sorted(set(profile) - _PROFILE_KEYS)
        if unknown:
            raise ValueError(f"{context} has unsupported keys: {', '.join(unknown)}")

        profile_id = _require_string(profile.get("id"), context=f"{context}.id")
        _require_string(profile.get("label"), context=f"{context}.label")
        if "description" in profile and not isinstance(profile["description"], str):
            raise ValueError(f"{context}.description must be a string")
        if profile_id in profile_ids:
            raise ValueError(f"duplicate preset profile id: {profile_id}")
        profile_ids.add(profile_id)

        defaults = profile.get("defaults")
        _validate_probe_values(
            defaults,
            context=f"{context}.defaults",
            require_complete=False,
        )
        if not defaults.get("renderer") or not defaults.get("tokenizer_model"):
            raise ValueError(
                f"{context}.defaults must set renderer and tokenizer_model"
            )

        examples = profile.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"{context}.examples must be a non-empty JSON array")

        example_ids: set[str] = set()
        for example_index, example in enumerate(examples):
            example_context = f"{context}.examples[{example_index}]"
            if not isinstance(example, Mapping):
                raise ValueError(f"{example_context} must be a JSON object")
            unknown = sorted(set(example) - _EXAMPLE_META_KEYS - _PROBE_KEYS)
            if unknown:
                raise ValueError(
                    f"{example_context} has unsupported keys: {', '.join(unknown)}"
                )
            example_id = _require_string(
                example.get("id"),
                context=f"{example_context}.id",
            )
            _require_string(
                example.get("label"),
                context=f"{example_context}.label",
            )
            if "description" in example and not isinstance(example["description"], str):
                raise ValueError(f"{example_context}.description must be a string")
            if example_id in example_ids:
                raise ValueError(
                    f"duplicate example id {example_id!r} in profile {profile_id!r}"
                )
            example_ids.add(example_id)

            overrides = {
                key: item
                for key, item in example.items()
                if key not in _EXAMPLE_META_KEYS
            }
            _validate_probe_values(
                overrides,
                context=example_context,
                require_complete=False,
            )
            merged = {**defaults, **overrides}
            _validate_probe_values(
                merged,
                context=f"{example_context} merged values",
                require_complete=True,
            )

    return dict(catalog)


def resolve_preset_case(
    catalog: Mapping[str, Any],
    *,
    profile_id: str,
    example_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve one runnable request exclusively from a validated input catalog.

    Returns deep copies of ``(profile, example, request)``. The request contains
    only probe fields: profile defaults merged with the selected example's
    overrides. Saved probe artifacts are not accepted by this schema.
    """
    profile = next(
        (
            candidate
            for candidate in catalog.get("profiles", [])
            if candidate.get("id") == profile_id
        ),
        None,
    )
    if profile is None:
        raise ValueError(f"unknown input profile: {profile_id!r}")

    example = next(
        (
            candidate
            for candidate in profile.get("examples", [])
            if candidate.get("id") == example_id
        ),
        None,
    )
    if example is None:
        raise ValueError(
            f"unknown input example {example_id!r} in profile {profile_id!r}"
        )

    overrides = {
        key: value for key, value in example.items() if key not in _EXAMPLE_META_KEYS
    }
    request = {**profile["defaults"], **overrides}
    _validate_probe_values(
        request,
        context=f"input case {profile_id!r}/{example_id!r}",
        require_complete=True,
    )
    return (
        copy.deepcopy(dict(profile)),
        copy.deepcopy(dict(example)),
        copy.deepcopy(request),
    )
