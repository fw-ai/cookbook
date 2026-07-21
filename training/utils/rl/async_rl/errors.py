"""Explicit, bounded failure policy for asynchronous rollouts."""

from __future__ import annotations

import asyncio
import errno
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeAlias

import aiohttp
import httpx
import requests


class ErrorDisposition(str, Enum):
    RECOVERABLE = "recoverable"
    FATAL = "fatal"


class RecoverableRolloutError(RuntimeError):
    """Explicitly mark a rollout-infrastructure failure as recoverable."""


@dataclass(frozen=True, slots=True)
class ErrorClassification:
    disposition: ErrorDisposition
    reason: str
    status_code: int | None = None

    @property
    def recoverable(self) -> bool:
        return self.disposition is ErrorDisposition.RECOVERABLE


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

    if isinstance(error, asyncio.CancelledError):
        return ErrorClassification(ErrorDisposition.FATAL, "unexpected_cancellation")
    if isinstance(error, RecoverableRolloutError):
        return ErrorClassification(
            ErrorDisposition.RECOVERABLE,
            "explicit_rollout_infrastructure",
        )
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
            f"last={type(last_error).__name__}: {last_error})"
        )


class RecoverableCircuitBreaker:
    """Turn a sustained transient rollout outage into a fatal run error."""

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self.config = config or CircuitBreakerConfig()
        self._observations: deque[bool] = deque(maxlen=self.config.rolling_window_size)
        self._consecutive_failures = 0
        self._total_successes = 0
        self._total_recoverable_failures = 0
        self._tripped = False
        self._trip_reason: str | None = None
        self._last_error: BaseException | None = None

    @property
    def snapshot(self) -> CircuitBreakerSnapshot:
        failures = sum(self._observations)
        observations = len(self._observations)
        return CircuitBreakerSnapshot(
            consecutive_failures=self._consecutive_failures,
            observations_in_window=observations,
            failures_in_window=failures,
            failure_rate=failures / observations if observations else 0.0,
            total_successes=self._total_successes,
            total_recoverable_failures=self._total_recoverable_failures,
            tripped=self._tripped,
        )

    def record_success(self) -> None:
        self._raise_if_tripped()
        self._observations.append(False)
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
        self._observations.append(True)
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
    "RecoverableCircuitBreaker",
    "RecoverableRolloutError",
    "RolloutErrorClassifier",
    "classify_rollout_error",
]
