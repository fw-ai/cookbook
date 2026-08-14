"""Private managed-training error carrier and ErrorInfo validation.

This module lives at the lightweight ``training`` package root so managed
failure handling can import it without executing the dependency-heavy
``training.utils`` package initializer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

ERROR_INFO_TYPE_URL = "type.googleapis.com/google.rpc.ErrorInfo"
ERROR_INFO_DOMAIN = "training.fireworks.ai"
ERROR_INFO_VERSION = "1"
ERROR_INFO_METADATA_VERSION = "version"
ERROR_INFO_METADATA_SOURCE = "source"
ERROR_INFO_METADATA_CATEGORY = "category"
ERROR_INFO_METADATA_QUOTA_REQUIRED = "quota_required"
ERROR_INFO_METADATA_QUOTA_AVAILABLE = "quota_available"
MAX_METADATA_VALUE_LENGTH = 128

SOURCE_MANAGED = "managed"
SOURCE_TINKER = "tinker"
SOURCE_SERVERLESS_GATEWAY = "serverless_gateway"
SOURCE_LIFECYCLE = "lifecycle"

REASON_DATASET_INVALID = "DATASET_INVALID"
REASON_INVALID_INPUT = "INVALID_INPUT"
REASON_RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
REASON_INSUFFICIENT_CAPACITY = "INSUFFICIENT_CAPACITY"
REASON_QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
REASON_TIER_REQUIRED = "TIER_REQUIRED"
REASON_RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
REASON_CANCELLED = "CANCELLED"
REASON_PERMISSION_DENIED = "PERMISSION_DENIED"
REASON_PREREQUISITE_NOT_MET = "PREREQUISITE_NOT_MET"
REASON_RESOURCE_INVALID = "RESOURCE_INVALID"
REASON_TIMEOUT = "TIMEOUT"
REASON_BACKEND_ERROR = "BACKEND_ERROR"
REASON_INTERNAL_ERROR = "INTERNAL_ERROR"

GRPC_CANCELLED = 1
GRPC_INVALID_ARGUMENT = 3
GRPC_NOT_FOUND = 5
GRPC_RESOURCE_EXHAUSTED = 8
GRPC_ABORTED = 10
GRPC_INTERNAL = 13

INTERNAL_ERROR_MESSAGE = "Internal error"

_REASON_SOURCES: dict[str, frozenset[str]] = {
    REASON_DATASET_INVALID: frozenset({SOURCE_MANAGED}),
    REASON_INVALID_INPUT: frozenset({SOURCE_MANAGED, SOURCE_TINKER, SOURCE_SERVERLESS_GATEWAY}),
    REASON_RESOURCE_NOT_FOUND: frozenset({SOURCE_MANAGED, SOURCE_TINKER, SOURCE_SERVERLESS_GATEWAY}),
    REASON_INSUFFICIENT_CAPACITY: frozenset({SOURCE_MANAGED, SOURCE_TINKER}),
    REASON_QUOTA_EXCEEDED: frozenset({SOURCE_LIFECYCLE, SOURCE_MANAGED}),
    REASON_TIER_REQUIRED: frozenset({SOURCE_LIFECYCLE, SOURCE_MANAGED}),
    REASON_RATE_LIMIT_EXCEEDED: frozenset({SOURCE_SERVERLESS_GATEWAY, SOURCE_MANAGED}),
    REASON_CANCELLED: frozenset({SOURCE_MANAGED, SOURCE_TINKER}),
    REASON_PERMISSION_DENIED: frozenset({SOURCE_LIFECYCLE, SOURCE_MANAGED, SOURCE_SERVERLESS_GATEWAY}),
    REASON_PREREQUISITE_NOT_MET: frozenset({SOURCE_MANAGED}),
    REASON_RESOURCE_INVALID: frozenset({SOURCE_MANAGED}),
    REASON_TIMEOUT: frozenset({SOURCE_MANAGED, SOURCE_TINKER}),
    REASON_BACKEND_ERROR: frozenset({SOURCE_MANAGED, SOURCE_TINKER, SOURCE_SERVERLESS_GATEWAY}),
    REASON_INTERNAL_ERROR: frozenset({SOURCE_MANAGED, SOURCE_TINKER, SOURCE_SERVERLESS_GATEWAY}),
}

_REASON_METADATA_KEYS: dict[str, frozenset[str]] = {
    REASON_INVALID_INPUT: frozenset({ERROR_INFO_METADATA_CATEGORY}),
    REASON_RESOURCE_NOT_FOUND: frozenset({ERROR_INFO_METADATA_CATEGORY}),
    REASON_INSUFFICIENT_CAPACITY: frozenset({ERROR_INFO_METADATA_CATEGORY}),
    REASON_QUOTA_EXCEEDED: frozenset(
        {
            ERROR_INFO_METADATA_QUOTA_REQUIRED,
            ERROR_INFO_METADATA_QUOTA_AVAILABLE,
        }
    ),
    REASON_TIER_REQUIRED: frozenset(
        {
            ERROR_INFO_METADATA_QUOTA_REQUIRED,
            ERROR_INFO_METADATA_QUOTA_AVAILABLE,
        }
    ),
    REASON_CANCELLED: frozenset({ERROR_INFO_METADATA_CATEGORY}),
    REASON_TIMEOUT: frozenset({ERROR_INFO_METADATA_CATEGORY}),
    REASON_BACKEND_ERROR: frozenset({ERROR_INFO_METADATA_CATEGORY}),
    REASON_INTERNAL_ERROR: frozenset({ERROR_INFO_METADATA_CATEGORY}),
}


@dataclass(frozen=True)
class _TrainingErrorStatus:
    """Private status carried between trusted training components."""

    grpc_code: int
    public_message: str
    reason: str
    domain: str = ERROR_INFO_DOMAIN
    metadata: Mapping[str, object] = field(default_factory=dict)
    source: str = SOURCE_MANAGED


def build_error_info_detail(
    value: _TrainingErrorStatus | Mapping[str, Any] | object,
) -> dict[str, Any]:
    """Build one canonical protojson ErrorInfo detail."""

    status = _coerce_training_error_status(value)
    metadata = {
        ERROR_INFO_METADATA_VERSION: ERROR_INFO_VERSION,
        ERROR_INFO_METADATA_SOURCE: status.source,
        **dict(status.metadata),
    }
    return {
        "@type": ERROR_INFO_TYPE_URL,
        "reason": status.reason,
        "domain": status.domain,
        "metadata": metadata,
    }


def extract_training_error_status(
    exc: BaseException,
) -> _TrainingErrorStatus | None:
    """Return only the explicit private carrier attached to an exception."""

    try:
        value = exc._fireworks_training_error_status  # type: ignore[attr-defined]
    except AttributeError:
        return None
    return _coerce_training_error_status(value)


def _coerce_training_error_status(value: Any) -> _TrainingErrorStatus:
    if isinstance(value, _TrainingErrorStatus):
        raw = value
    elif isinstance(value, Mapping):
        raw = _TrainingErrorStatus(
            grpc_code=value.get("grpc_code"),
            public_message=value.get("public_message"),
            reason=value.get("reason"),
            domain=value.get("domain"),
            metadata=value.get("metadata") or {},
            source=value.get("source"),
        )
    else:
        raw = _TrainingErrorStatus(
            grpc_code=getattr(value, "grpc_code", None),
            public_message=getattr(value, "public_message", None),
            reason=getattr(value, "reason", None),
            domain=getattr(value, "domain", None),
            metadata=getattr(value, "metadata", None) or {},
            source=getattr(value, "source", None),
        )

    if not isinstance(raw.grpc_code, int) or isinstance(raw.grpc_code, bool) or not 0 <= raw.grpc_code <= 16:
        raise ValueError("training error status grpc_code must be a gRPC code")
    if not isinstance(raw.public_message, str):
        raise ValueError("training error status public_message must be a string")
    if not isinstance(raw.reason, str) or not raw.reason:
        raise ValueError("training error status reason must be a non-empty string")
    if raw.reason.strip() != raw.reason:
        raise ValueError("training error status reason must not contain whitespace")
    if raw.domain != ERROR_INFO_DOMAIN:
        raise ValueError("training ErrorInfo has an unexpected domain")
    if not isinstance(raw.source, str) or not raw.source:
        raise ValueError("training ErrorInfo source must be a non-empty string")
    if not isinstance(raw.metadata, Mapping):
        raise ValueError("training ErrorInfo metadata must be a mapping")

    metadata = _validated_metadata(
        reason=raw.reason,
        source=raw.source,
        metadata=raw.metadata,
    )
    return _TrainingErrorStatus(
        grpc_code=raw.grpc_code,
        public_message=raw.public_message,
        reason=raw.reason,
        domain=raw.domain,
        metadata=metadata,
        source=raw.source,
    )


def _validated_metadata(
    *,
    reason: str,
    source: str,
    metadata: Mapping[str, object],
) -> dict[str, str]:
    allowed_sources = _REASON_SOURCES.get(reason)
    if allowed_sources is None:
        raise ValueError(f"unregistered training ErrorInfo reason {reason!r}")
    if source not in allowed_sources:
        raise ValueError(f"source {source!r} cannot emit training ErrorInfo reason {reason!r}")

    allowed_metadata = _REASON_METADATA_KEYS.get(reason, frozenset())
    result: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            continue
        if key == ERROR_INFO_METADATA_VERSION:
            if value != ERROR_INFO_VERSION:
                raise ValueError("training ErrorInfo has an unexpected version")
            continue
        if key == ERROR_INFO_METADATA_SOURCE:
            if value != source:
                raise ValueError("training ErrorInfo has a conflicting source")
            continue
        if key not in allowed_metadata:
            continue
        try:
            rendered = str(value)
        except Exception as exc:
            raise ValueError(f"metadata value for {key!r} cannot be stringified") from exc
        encoded = rendered.encode("utf-8")
        if len(encoded) > MAX_METADATA_VALUE_LENGTH:
            rendered = encoded[:MAX_METADATA_VALUE_LENGTH].decode(
                "utf-8",
                errors="ignore",
            )
        result[key] = rendered
    return result
