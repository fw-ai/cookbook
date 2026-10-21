"""Explicit, bounded failure policy for asynchronous rollouts."""

from __future__ import annotations

import asyncio
import errno
import re
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Callable, TypeAlias

import aiohttp
import httpx
import requests
from fireworks.training.sdk.sampling_observability import SamplingRequestError, extract_request_id

if TYPE_CHECKING:
    from fireworks.training.sdk import TITOTrajectoryArtifact


class ErrorDisposition(str, Enum):
    DISCARD = "discard"
    RECOVERABLE = "recoverable"
    FATAL = "fatal"


class RecoverableRolloutError(RuntimeError):
    """Explicitly mark a rollout-infrastructure failure as recoverable.

    ``reason`` is an optional stable slug for failure telemetry. Exception
    attributes, including retained trace pointers, survive worker serialization.
    """

    def __init__(
        self,
        *args: object,
        reason: str | None = None,
        trace_paths: tuple[str, ...] = (),
        diagnostics: dict | None = None,
    ) -> None:
        super().__init__(*args)
        self.reason = reason
        self.trace_paths = trace_paths
        self.diagnostics = _safe_diagnostics(diagnostics or {})


class DiscardTrajectoryError(RuntimeError):
    """A diagnosed trajectory-local failure; never retry it as infrastructure."""

    def __init__(
        self,
        message: str,
        code: str = "tito_cookbook_parse_error",
        trace_path: str | None = None,
        *,
        diagnostics: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.trace_path = trace_path
        self.diagnostics = _safe_diagnostics(diagnostics or {})

    def __reduce__(self) -> tuple:
        return (
            type(self),
            (str(self), self.code, self.trace_path),
            {"diagnostics": self.diagnostics},
        )


_TITO_DISCARD_CODES = frozenset(
    {
        "tito_sdk_parse_error",
        "tito_sdk_parser_internal_error",
        "tito_sdk_sampling_parse_error",
        "tito_sdk_internal_error",
        "tito_cookbook_parse_error",
        "tito_model_malformed_output",
    }
)


def _tito_failure_details(error: BaseException) -> dict:
    # SDK exceptions, OpenAI SDK exceptions and raw HTTP failures all preserve
    # the same structured code. Never classify by free-form message matching.
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        response = getattr(error, "response", None)
        try:
            body = response.json() if response is not None else None
        except (ValueError, TypeError, RuntimeError):
            body = None
    if isinstance(body, dict):
        details = body.get("error", body)
        if isinstance(details, dict):
            return details
    context = getattr(error, "context", None)
    if isinstance(context, dict) or isinstance(getattr(error, "code", None), str):
        details = dict(context) if isinstance(context, dict) else {}
        for key in ("code", "origin"):
            value = getattr(error, key, None)
            if value is not None:
                details[key] = value
        return details
    return {}


def _safe_tag(value: object, fallback: str = "other") -> str:
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value)
        else fallback
    )


def _safe_diagnostics(details: dict) -> dict:
    selected = {}
    for key in (
        "code",
        "origin",
        "parser_reason_code",
        "finish_reason",
        "renderer_id",
        "renderer_certification_id",
        "logical_request_id",
        "request_id",
        "upstream_request_id",
        "upstream_response_id",
        "upstream_error_code",
        "upstream_error_type",
        "upstream_finish_reason",
        "final_error_kind",
        "wire_status",
        "final_status",
        "upstream_status",
        "attempts",
        "retry_budget_exhausted",
        "should_retry",
        "discard_trajectory",
        "retry_scope",
        "phase",
        "transport_hop",
        "prompt_tokens",
        "completion_tokens",
        "requested_max_tokens",
        "max_output_tokens",
        "termination_phase",
        "terminal_action",
        "requested_terminal_status",
        "terminal_status",
        "terminal_reason",
        "terminal_started_at",
        "terminal_at",
        "cause_error_code",
        "cause_origin",
        "trajectory_id",
        "call_id",
    ):
        value = details.get(key)
        if isinstance(value, (bool, int, float)):
            selected[key] = value
        elif (
            isinstance(value, str)
            and len(value) <= 256
            and all(ord(character) >= 32 for character in value)
        ):
            selected[key] = value
    return selected


def trajectory_discard_error(
    artifact: TITOTrajectoryArtifact,
) -> DiscardTrajectoryError | None:
    """A failed trajectory may never contribute its successful prefix to training."""
    failure = artifact.metadata.get("tito_failure") or {}
    code = failure.get("code")
    if code not in _TITO_DISCARD_CODES:
        code = next(
            (
                call.error_code
                for call in artifact.calls
                if call.error_code in _TITO_DISCARD_CODES
                and not (call.error_code == "tito_model_malformed_output" and artifact.status == "completed")
            ),
            None,
        )
    if code is None:
        return None
    return DiscardTrajectoryError(
        f"Discarding TITO trajectory {artifact.trajectory_id}: {code}",
        code,
        failure.get("trace_path"),
        diagnostics=failure,
    )


@dataclass(frozen=True, slots=True)
class ErrorClassification:
    disposition: ErrorDisposition
    reason: str
    status_code: int | None = None
    trace_path: str | None = None
    trace_paths: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    @property
    def recoverable(self) -> bool:
        return self.disposition is ErrorDisposition.RECOVERABLE

    def as_log_record(self) -> dict:
        return {
            **_safe_diagnostics(self.diagnostics),
            "disposition": self.disposition.value,
            "reason": _safe_tag(self.reason),
            "status_code": self.status_code,
            "trace_paths": self.trace_paths,
        }


RolloutErrorClassifier: TypeAlias = Callable[[BaseException], ErrorClassification]

_RETRYABLE_HTTP_STATUSES = frozenset({408, 429})
_RETRYABLE_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.EPIPE,
        errno.ETIMEDOUT,
    }
)
_RECOVERABLE_CLIENT_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "BadGatewayError",
        "DeploymentSamplerTimeoutError",
        "InternalServerError",
        "RateLimitError",
        "SamplingRequestError",
        "ServiceUnavailableError",
    }
)
_KNOWN_CLIENT_MODULES = ("fireworks", "tinker", "openai")


def classify_rollout_error(error: BaseException) -> ErrorClassification:
    """Return a narrow disposition; unknown and contract errors are fatal."""

    details = _tito_failure_details(error)
    if isinstance(error, SamplingRequestError):
        details.update(error.as_error_record())
        details["origin"] = "server" if error.final_status is not None or error.final_error_kind == "upstream_stream" else "transport"
    classification = _classify_rollout_error(error, details)
    diagnostics = getattr(error, "diagnostics", None)
    details = {
        **(diagnostics if isinstance(diagnostics, dict) else {}),
        **details,
    }
    response = getattr(error, "response", None)
    if details.get("wire_status") is None:
        wire_status = getattr(error, "_fireworks_wire_status", None)
        if wire_status is None:
            wire_status = getattr(error, "_fireworks_http_status", None)
        if wire_status is None:
            wire_status = getattr(response, "status_code", None)
        if wire_status is not None:
            details["wire_status"] = wire_status
    if details.get("request_id") is None:
        request_id = getattr(error, "_fireworks_response_request_id", None) or extract_request_id(
            getattr(response, "headers", None)
        )
        if request_id is not None:
            details["request_id"] = request_id
    trace_paths = getattr(error, "trace_paths", ())
    trace_paths = trace_paths[:16] if isinstance(trace_paths, (list, tuple)) else ()
    paths = tuple(
        dict.fromkeys(
            path
            for path in (classification.trace_path, details.get("trace_path"), *trace_paths)
            if isinstance(path, str) and 0 < len(path) <= 4096
        )
    )[:16]
    return replace(
        classification,
        status_code=classification.status_code if classification.status_code is not None else _http_status_code(error),
        trace_path=paths[0] if paths else None,
        trace_paths=paths,
        diagnostics=_safe_diagnostics(details),
    )


def _classify_rollout_error(error: BaseException, details: dict) -> ErrorClassification:

    if isinstance(error, asyncio.CancelledError):
        return ErrorClassification(ErrorDisposition.FATAL, "unexpected_cancellation")
    if isinstance(error, DiscardTrajectoryError):
        return ErrorClassification(ErrorDisposition.DISCARD, error.code, trace_path=error.trace_path)
    if isinstance(details.get("code"), str) and details["code"] in _TITO_DISCARD_CODES:
        return ErrorClassification(
            ErrorDisposition.DISCARD,
            details["code"],
            trace_path=details.get("trace_path"),
        )
    if isinstance(error, RecoverableRolloutError):
        return ErrorClassification(
            ErrorDisposition.RECOVERABLE,
            _safe_tag(error.reason, "explicit_rollout_infrastructure"),
        )
    code = details.get("code")
    origin = details.get("origin")
    stream_error = code == "tito_server_stream_error" or (
        isinstance(error, SamplingRequestError) and error.final_error_kind == "upstream_stream"
    )
    if stream_error and origin in {None, "server", "unknown"}:
        semantic_status = details.get("upstream_status") or details.get("final_status")
        if semantic_status is None:
            outward_status = _http_status_code(error)
            if outward_status is not None and 400 <= outward_status < 500:
                semantic_status = outward_status
        if semantic_status is None:
            return ErrorClassification(ErrorDisposition.FATAL, "upstream_stream_rejected")
        if semantic_status in {408, 429, 500, 502, 503, 504}:
            return ErrorClassification(ErrorDisposition.RECOVERABLE, "upstream_stream_transient", semantic_status)
        return ErrorClassification(ErrorDisposition.FATAL, "upstream_stream_rejected", semantic_status)
    if origin in {"sdk", "sdk_parser", "adapter", "environment"} or (
        origin == "unknown" and details.get("transport_hop") != "nemo_proxy_to_tito"
    ):
        return ErrorClassification(ErrorDisposition.FATAL, _safe_tag(code, "local_rollout_failure"))
    if origin == "transport":
        return ErrorClassification(ErrorDisposition.RECOVERABLE, "network_transport")
    if isinstance(error, (AssertionError, TypeError, ValueError)):
        return ErrorClassification(
            ErrorDisposition.FATAL,
            "programmer_or_data_contract",
        )

    status_code = _http_status_code(error)
    if status_code is not None:
        if status_code in _RETRYABLE_HTTP_STATUSES or 500 <= status_code <= 599:
            return ErrorClassification(
                ErrorDisposition.RECOVERABLE,
                "retryable_http_status",
                status_code,
            )
        return ErrorClassification(
            ErrorDisposition.FATAL,
            "non_retryable_http_status",
            status_code,
        )

    if isinstance(
        error,
        (
            TimeoutError,
            ConnectionError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            aiohttp.ClientConnectionError,
            aiohttp.ServerTimeoutError,
        ),
    ):
        return ErrorClassification(ErrorDisposition.RECOVERABLE, "network_transport")
    if isinstance(error, OSError) and error.errno in _RETRYABLE_ERRNOS:
        return ErrorClassification(ErrorDisposition.RECOVERABLE, "network_os_error")
    if _is_known_recoverable_client_error(error):
        return ErrorClassification(ErrorDisposition.RECOVERABLE, "client_transport")
    return ErrorClassification(ErrorDisposition.FATAL, "unknown_exception")


def _http_status_code(error: BaseException) -> int | None:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        status_code = getattr(error, "status", None)
    if status_code is None:
        status_code = getattr(error, "final_status", None)
    if status_code is None:
        if isinstance(error, httpx.RequestError):
            return None
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            status_code = getattr(response, "status", None)
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        return None
    return status_code if 100 <= status_code <= 599 else None


def _is_known_recoverable_client_error(error: BaseException) -> bool:
    return any(
        cls.__module__.startswith(_KNOWN_CLIENT_MODULES)
        and cls.__name__ in _RECOVERABLE_CLIENT_ERROR_NAMES
        for cls in type(error).__mro__
    )


@dataclass(frozen=True, slots=True)
class CircuitBreakerConfig:
    max_consecutive_failures: int = 5
    rolling_window_size: int = 20
    rolling_min_observations: int = 10
    max_failure_rate: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "max_consecutive_failures",
            "rolling_window_size",
            "rolling_min_observations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.rolling_min_observations > self.rolling_window_size:
            raise ValueError("rolling_min_observations must be <= rolling_window_size")
        if isinstance(self.max_failure_rate, bool) or not isinstance(
            self.max_failure_rate,
            (int, float),
        ):
            raise TypeError("max_failure_rate must be a float")
        if not 0.0 < float(self.max_failure_rate) <= 1.0:
            raise ValueError("max_failure_rate must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class CircuitBreakerSnapshot:
    consecutive_failures: int
    observations_in_window: int
    failures_in_window: int
    failure_rate: float
    total_successes: int
    total_recoverable_failures: int
    tripped: bool
    reasons_in_window: dict[str, int] = field(default_factory=dict)
    codes_in_window: dict[str, int] = field(default_factory=dict)


class CircuitBreakerTripped(RuntimeError):
    def __init__(
        self,
        *,
        reason: str,
        snapshot: CircuitBreakerSnapshot,
        last_error: BaseException,
    ) -> None:
        self.reason = reason
        self.snapshot = snapshot
        self.last_error = last_error
        super().__init__(
            "rollout circuit breaker tripped "
            f"({reason}; consecutive={snapshot.consecutive_failures}, "
            f"rolling={snapshot.failures_in_window}/{snapshot.observations_in_window}, "
            f"reasons={snapshot.reasons_in_window}, codes={snapshot.codes_in_window}, "
            f"last_type={type(last_error).__name__})"
        )


class RecoverableCircuitBreaker:
    """Turn a sustained transient rollout outage into a fatal run error."""

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self.config = config or CircuitBreakerConfig()
        self._observations: deque[tuple[str, str | None] | None] = deque(
            maxlen=self.config.rolling_window_size
        )
        self._consecutive_failures = 0
        self._total_successes = 0
        self._total_recoverable_failures = 0
        self._tripped = False
        self._trip_reason: str | None = None
        self._last_error: BaseException | None = None

    @property
    def snapshot(self) -> CircuitBreakerSnapshot:
        failures = sum(observation is not None for observation in self._observations)
        observations = len(self._observations)
        return CircuitBreakerSnapshot(
            consecutive_failures=self._consecutive_failures,
            observations_in_window=observations,
            failures_in_window=failures,
            failure_rate=failures / observations if observations else 0.0,
            total_successes=self._total_successes,
            total_recoverable_failures=self._total_recoverable_failures,
            tripped=self._tripped,
            reasons_in_window=dict(
                Counter(item[0] for item in self._observations if item is not None)
            ),
            codes_in_window=dict(
                Counter(
                    item[1]
                    for item in self._observations
                    if item is not None and item[1] is not None
                )
            ),
        )

    def record_success(self) -> None:
        self._raise_if_tripped()
        self._observations.append(None)
        self._consecutive_failures = 0
        self._total_successes += 1

    def record_failure(
        self,
        error: BaseException,
        classification: ErrorClassification,
    ) -> None:
        self._raise_if_tripped()
        if not classification.recoverable:
            raise ValueError("fatal errors cannot enter recoverable-failure accounting")
        code = classification.diagnostics.get("code")
        self._observations.append(
            (
                _safe_tag(classification.reason),
                _safe_tag(code) if code is not None else None,
            )
        )
        self._consecutive_failures += 1
        self._total_recoverable_failures += 1
        self._last_error = error
        if self._consecutive_failures >= self.config.max_consecutive_failures:
            self._trip("consecutive_failures")
        snapshot = self.snapshot
        if (
            snapshot.observations_in_window >= self.config.rolling_min_observations
            and snapshot.failure_rate >= self.config.max_failure_rate
        ):
            self._trip("rolling_failure_rate")

    def _trip(self, reason: str) -> None:
        self._tripped = True
        self._trip_reason = reason
        self._raise_if_tripped()

    def _raise_if_tripped(self) -> None:
        if not self._tripped:
            return
        if self._trip_reason is None or self._last_error is None:
            raise RuntimeError("circuit breaker entered an invalid tripped state")
        raise CircuitBreakerTripped(
            reason=self._trip_reason,
            snapshot=self.snapshot,
            last_error=self._last_error,
        ) from self._last_error


__all__ = [
    "CircuitBreakerConfig",
    "CircuitBreakerSnapshot",
    "CircuitBreakerTripped",
    "ErrorClassification",
    "ErrorDisposition",
    "DiscardTrajectoryError",
    "RecoverableCircuitBreaker",
    "RecoverableRolloutError",
    "RolloutErrorClassifier",
    "classify_rollout_error",
]
