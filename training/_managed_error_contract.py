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
    REASON_INVALID_INPUT: frozenset(
        {SOURCE_MANAGED, SOURCE_TINKER, SOURCE_SERVERLESS_GATEWAY}
    ),
    REASON_RESOURCE_NOT_FOUND: frozenset(
        {SOURCE_MANAGED, SOURCE_TINKER, SOURCE_SERVERLESS_GATEWAY}
    ),
    REASON_INSUFFICIENT_CAPACITY: frozenset({SOURCE_MANAGED, SOURCE_TINKER}),
    REASON_QUOTA_EXCEEDED: frozenset({SOURCE_LIFECYCLE, SOURCE_MANAGED}),
    REASON_TIER_REQUIRED: frozenset({SOURCE_LIFECYCLE, SOURCE_MANAGED}),
    REASON_RATE_LIMIT_EXCEEDED: frozenset({SOURCE_SERVERLESS_GATEWAY, SOURCE_MANAGED}),
    REASON_CANCELLED: frozenset({SOURCE_MANAGED, SOURCE_TINKER}),
    REASON_PERMISSION_DENIED: frozenset(
        {SOURCE_LIFECYCLE, SOURCE_MANAGED, SOURCE_SERVERLESS_GATEWAY}
    ),
    REASON_PREREQUISITE_NOT_MET: frozenset({SOURCE_MANAGED}),
    REASON_RESOURCE_INVALID: frozenset({SOURCE_MANAGED}),
    REASON_TIMEOUT: frozenset({SOURCE_MANAGED, SOURCE_TINKER}),
    REASON_BACKEND_ERROR: frozenset(
        {SOURCE_MANAGED, SOURCE_TINKER, SOURCE_SERVERLESS_GATEWAY}
    ),
    REASON_INTERNAL_ERROR: frozenset(
        {SOURCE_MANAGED, SOURCE_TINKER, SOURCE_SERVERLESS_GATEWAY}
    ),
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


@dataclass(frozen=True)
class _SourceErrorSpec:
    reason: str
    grpc_code: int
    public_message: str


_TINKER_ERROR_CLASS_SPECS = {
    "validation": _SourceErrorSpec(
        reason=REASON_INVALID_INPUT,
        grpc_code=GRPC_INVALID_ARGUMENT,
        public_message="The training request is invalid. Review the request and try again.",
    ),
    "not_found": _SourceErrorSpec(
        reason=REASON_RESOURCE_NOT_FOUND,
        grpc_code=GRPC_NOT_FOUND,
        public_message="A required training resource was not found. Verify the referenced resources and try again.",
    ),
    "capacity_exhausted": _SourceErrorSpec(
        reason=REASON_INSUFFICIENT_CAPACITY,
        grpc_code=GRPC_RESOURCE_EXHAUSTED,
        public_message="Training capacity is temporarily unavailable. Please try again later.",
    ),
    "cancelled": _SourceErrorSpec(
        reason=REASON_CANCELLED,
        grpc_code=GRPC_ABORTED,
        public_message="Training was cancelled.",
    ),
    "timeout": _SourceErrorSpec(
        reason=REASON_TIMEOUT,
        grpc_code=GRPC_INTERNAL,
        public_message=INTERNAL_ERROR_MESSAGE,
    ),
    "backend": _SourceErrorSpec(
        reason=REASON_BACKEND_ERROR,
        grpc_code=GRPC_INTERNAL,
        public_message=INTERNAL_ERROR_MESSAGE,
    ),
    "internal": _SourceErrorSpec(
        reason=REASON_INTERNAL_ERROR,
        grpc_code=GRPC_INTERNAL,
        public_message=INTERNAL_ERROR_MESSAGE,
    ),
    "unknown": _SourceErrorSpec(
        reason=REASON_INTERNAL_ERROR,
        grpc_code=GRPC_INTERNAL,
        public_message=INTERNAL_ERROR_MESSAGE,
    ),
}

_GATEWAY_CODE_SPECS = {
    "BAD_REQUEST": _TINKER_ERROR_CLASS_SPECS["validation"],
    "NOT_FOUND": _TINKER_ERROR_CLASS_SPECS["not_found"],
    "RATE_LIMIT_EXCEEDED": _SourceErrorSpec(
        reason=REASON_RATE_LIMIT_EXCEEDED,
        grpc_code=GRPC_RESOURCE_EXHAUSTED,
        public_message="Too many training requests. Wait and try again.",
    ),
    "SERVICE_UNAVAILABLE": _TINKER_ERROR_CLASS_SPECS["backend"],
    "INTERNAL_SERVER_ERROR": _TINKER_ERROR_CLASS_SPECS["internal"],
}

_MISSING = object()


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
    """Resolve one explicit private carrier, rejecting ambiguous context."""

    try:
        status = exc._fireworks_training_error_status  # type: ignore[attr-defined]
    except AttributeError:
        status = _MISSING
    try:
        source = exc._fireworks_training_error_source  # type: ignore[attr-defined]
    except AttributeError:
        source = _MISSING

    if status is not _MISSING and source is not _MISSING:
        raise ValueError("conflicting structured training error carriers")
    if status is not _MISSING:
        return _coerce_training_error_status(status)
    if source is not _MISSING:
        return _status_from_source_error(source)
    return None


def _status_from_source_error(value: Any) -> _TrainingErrorStatus | None:
    source = getattr(value, "source", None)
    if source == SOURCE_TINKER:
        if _has_any_source_field(value, ("code", "type")):
            raise ValueError("Tinker source carrier contains gateway fields")
        error = _validated_source_field(value, "error")
        category = _validated_source_field(value, "category")
        error_class = _validated_source_field(value, "error_class")
        if error is None and category is None and error_class is None:
            raise ValueError("empty Tinker source carrier")
        if error_class is None:
            return None
        spec = _TINKER_ERROR_CLASS_SPECS.get(error_class)
        if spec is None:
            return None
        metadata = (
            {ERROR_INFO_METADATA_CATEGORY: category} if category is not None else {}
        )
    elif source == SOURCE_SERVERLESS_GATEWAY:
        if _has_any_source_field(value, ("error", "category", "error_class")):
            raise ValueError("gateway source carrier contains Tinker fields")
        code = _validated_source_field(value, "code")
        error_type = _validated_source_field(value, "type")
        if code is None and error_type is None:
            raise ValueError("empty gateway source carrier")
        if code is None:
            return None
        spec = _GATEWAY_CODE_SPECS.get(code)
        if spec is None:
            return None
        metadata = {}
    else:
        raise ValueError("structured training error carrier has an unexpected source")

    return _coerce_training_error_status(
        _TrainingErrorStatus(
            grpc_code=spec.grpc_code,
            public_message=spec.public_message,
            reason=spec.reason,
            metadata=metadata,
            source=source,
        )
    )


def _has_any_source_field(value: Any, names: tuple[str, ...]) -> bool:
    return any(getattr(value, name, _MISSING) is not _MISSING for name in names)


def _validated_source_field(
    value: Any,
    name: str,
) -> str | None:
    raw = getattr(value, name, _MISSING)
    if raw is _MISSING or raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"structured source field {name!r} must be a non-empty string")
    try:
        encoded = raw.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(
            f"structured source field {name!r} is not valid UTF-8"
        ) from exc
    if len(encoded) > MAX_METADATA_VALUE_LENGTH:
        raise ValueError(f"structured source field {name!r} exceeds the size limit")
    return raw


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

    if (
        not isinstance(raw.grpc_code, int)
        or isinstance(raw.grpc_code, bool)
        or not 0 <= raw.grpc_code <= 16
    ):
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
        raise ValueError(
            f"source {source!r} cannot emit training ErrorInfo reason {reason!r}"
        )

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
            raise ValueError(
                f"metadata value for {key!r} cannot be stringified"
            ) from exc
        encoded = rendered.encode("utf-8")
        if len(encoded) > MAX_METADATA_VALUE_LENGTH:
            rendered = encoded[:MAX_METADATA_VALUE_LENGTH].decode(
                "utf-8",
                errors="ignore",
            )
        result[key] = rendered
    return result
