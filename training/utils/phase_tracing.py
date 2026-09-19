"""Opt-in client phase spans for Perfetto and OpenTelemetry.

Set ``COOKBOOK_TRACE_FILE`` to write a Perfetto-compatible Chrome trace. Set
``COOKBOOK_OTEL_ENABLED=1`` to also mirror spans through the process's configured
OpenTelemetry tracer when the optional package is installed.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import math
import os
import threading
import time
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

logger = logging.getLogger(__name__)

TRACE_FILE_ENV = "COOKBOOK_TRACE_FILE"
OTEL_ENABLED_ENV = "COOKBOOK_OTEL_ENABLED"
DEFAULT_MAX_EVENTS = 100_000
DEFAULT_MAX_LANES = 1_024
T = TypeVar("T")

_SPAN_STACK: ContextVar[tuple[int, ...]] = ContextVar(
    "cookbook_phase_span_stack",
    default=(),
)
_RECORDER_LOCK = threading.Lock()
_RECORDER: PhaseTraceRecorder | None = None
_RECORDER_INITIALIZED = False


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _load_otel_tracer() -> Any | None:
    if not _env_enabled(OTEL_ENABLED_ENV):
        return None
    try:
        # lazy: OpenTelemetry is an optional customer-managed integration.
        from opentelemetry import trace
    except ImportError:
        logger.warning(
            "%s is set, but opentelemetry-api is not installed; "
            "continuing with Perfetto tracing only",
            OTEL_ENABLED_ENV,
        )
        return None
    try:
        return trace.get_tracer("fireworks.training.cookbook")
    except Exception:
        logger.warning(
            "OpenTelemetry tracer initialization failed; "
            "continuing with Perfetto tracing only",
            exc_info=True,
        )
        return None


@dataclass
class PhaseSpan:
    """A running client phase span."""

    span_id: int
    attributes: dict[str, Any] = field(default_factory=dict)
    _otel_span: Any | None = field(default=None, repr=False)

    def set_attribute(self, name: str, value: Any) -> None:
        normalized = _json_value(value)
        self.attributes[name] = normalized
        if self._otel_span is not None:
            try:
                self._otel_span.set_attribute(name, normalized)
            except Exception:
                logger.debug("OpenTelemetry set_attribute failed", exc_info=True)


class PhaseTraceRecorder:
    """Thread-safe span recorder with an atomic Perfetto JSON export."""

    def __init__(
        self,
        trace_file: str | os.PathLike[str] | None,
        *,
        otel_tracer: Any | None = None,
        max_events: int = DEFAULT_MAX_EVENTS,
        max_lanes: int = DEFAULT_MAX_LANES,
    ) -> None:
        self.trace_file = Path(trace_file).expanduser() if trace_file else None
        self.otel_tracer = otel_tracer
        self.max_events = max(1, int(max_events))
        self.max_lanes = max(1, min(int(max_lanes), self.max_events))
        self._origin_ns = time.perf_counter_ns()
        self._origin_unix_ns = time.time_ns()
        self._pid = os.getpid()
        self._events: list[dict[str, Any]] = []
        self._task_lane_ids: weakref.WeakKeyDictionary[Any, int] = (
            weakref.WeakKeyDictionary()
        )
        self._thread_lane_ids: weakref.WeakKeyDictionary[Any, int] = (
            weakref.WeakKeyDictionary()
        )
        self._lane_names: dict[int, str] = {}
        self._next_span_id = 1
        self._next_lane_id = 1
        self._dropped_events = 0
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()

    def _lane(self) -> int:
        thread = threading.current_thread()
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        task_name = task.get_name() if task is not None else ""
        lane_name = thread.name if not task_name else f"{thread.name}/{task_name}"

        with self._lock:
            lane_ids = (
                self._task_lane_ids if task is not None else self._thread_lane_ids
            )
            lane_key = task if task is not None else thread
            lane_id = lane_ids.get(lane_key)
            if lane_id is None:
                if len(self._lane_names) >= self.max_lanes:
                    return 0
                lane_id = self._next_lane_id
                self._next_lane_id += 1
                lane_ids[lane_key] = lane_id
                self._lane_names[lane_id] = lane_name
        return lane_id

    def start_span(self) -> tuple[int, int | None, int, int]:
        stack = _SPAN_STACK.get()
        parent_id = stack[-1] if stack else None
        lane_id = self._lane()
        with self._lock:
            span_id = self._next_span_id
            self._next_span_id += 1
        return span_id, parent_id, lane_id, time.perf_counter_ns()

    def finish_span(
        self,
        *,
        span_id: int,
        parent_id: int | None,
        lane_id: int,
        started_ns: int,
        name: str,
        category: str,
        attributes: dict[str, Any],
    ) -> None:
        ended_ns = time.perf_counter_ns()
        args = {
            "span_id": span_id,
            **({"parent_span_id": parent_id} if parent_id is not None else {}),
            **{str(key): _json_value(value) for key, value in attributes.items()},
        }
        event = {
            "name": name,
            "cat": category,
            "ph": "X",
            "ts": max(0, (started_ns - self._origin_ns) // 1_000),
            "dur": max(0, (ended_ns - started_ns) // 1_000),
            "pid": self._pid,
            "tid": lane_id,
            "args": args,
        }
        with self._lock:
            if len(self._events) >= self.max_events:
                self._dropped_events += 1
                return
            self._events.append(event)

    def payload(self) -> dict[str, Any]:
        with self._lock:
            lane_names = dict(self._lane_names)
            events = [dict(event) for event in self._events]
            dropped_events = self._dropped_events

        metadata = [
            {
                "name": "thread_name",
                "ph": "M",
                "pid": self._pid,
                "tid": lane_id,
                "args": {"name": lane_name},
            }
            for lane_id, lane_name in sorted(lane_names.items())
        ]
        events.sort(key=lambda event: (event["ts"], -event["dur"], event["name"]))
        return {
            "schema_version": 1,
            "displayTimeUnit": "ms",
            "trace_origin_unix_ns": self._origin_unix_ns,
            "dropped_events": dropped_events,
            "traceEvents": [*metadata, *events],
        }

    def flush(self) -> str | None:
        if self.trace_file is None:
            return None
        with self._flush_lock:
            path = self.trace_file
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(
                f".{path.name}.{self._pid}.{threading.get_ident()}.tmp"
            )
            try:
                with temp.open("w", encoding="utf-8") as output:
                    json.dump(self.payload(), output, separators=(",", ":"))
                    output.write("\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temp, path)
            except Exception:
                temp.unlink(missing_ok=True)
                raise
        return str(path)


def configure_phase_tracing(
    trace_file: str | os.PathLike[str] | None = None,
    *,
    otel_tracer: Any | None = None,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> PhaseTraceRecorder | None:
    """Configure the process-wide recorder.

    An explicit ``trace_file`` overrides ``COOKBOOK_TRACE_FILE``. Calling this
    without a trace file and without enabled/configured OpenTelemetry disables
    tracing.
    """

    global _RECORDER, _RECORDER_INITIALIZED
    resolved_file = trace_file or os.environ.get(TRACE_FILE_ENV)
    resolved_otel = otel_tracer if otel_tracer is not None else _load_otel_tracer()
    recorder = (
        PhaseTraceRecorder(
            resolved_file,
            otel_tracer=resolved_otel,
            max_events=max_events,
        )
        if resolved_file or resolved_otel is not None
        else None
    )
    with _RECORDER_LOCK:
        _RECORDER = recorder
        _RECORDER_INITIALIZED = True
    return recorder


def get_phase_trace_recorder() -> PhaseTraceRecorder | None:
    global _RECORDER, _RECORDER_INITIALIZED
    if _RECORDER_INITIALIZED:
        return _RECORDER
    if not os.environ.get(TRACE_FILE_ENV) and not _env_enabled(OTEL_ENABLED_ENV):
        with _RECORDER_LOCK:
            _RECORDER_INITIALIZED = True
        return None
    with _RECORDER_LOCK:
        if _RECORDER is None:
            resolved_file = os.environ.get(TRACE_FILE_ENV)
            resolved_otel = _load_otel_tracer()
            if resolved_file or resolved_otel is not None:
                _RECORDER = PhaseTraceRecorder(
                    resolved_file,
                    otel_tracer=resolved_otel,
                )
            _RECORDER_INITIALIZED = True
        return _RECORDER


def bind_phase_trace_context(function: Callable[[], T]) -> Callable[[], T]:
    """Bind only tracing state for a call that will run in another thread."""

    span_stack = _SPAN_STACK.get()
    if not span_stack:
        return function

    otel_context_api = None
    otel_parent_context = None
    recorder = get_phase_trace_recorder()
    if recorder is not None and recorder.otel_tracer is not None:
        try:
            # lazy: OpenTelemetry remains an optional customer dependency.
            from opentelemetry import context as otel_context_api

            otel_parent_context = otel_context_api.get_current()
        except Exception:
            logger.debug("OpenTelemetry context capture failed", exc_info=True)

    def run() -> T:
        stack_token = _SPAN_STACK.set(span_stack)
        otel_token = None
        try:
            if otel_context_api is not None and otel_parent_context is not None:
                try:
                    otel_token = otel_context_api.attach(otel_parent_context)
                except Exception:
                    logger.debug("OpenTelemetry context attach failed", exc_info=True)
            return function()
        finally:
            if otel_token is not None:
                try:
                    otel_context_api.detach(otel_token)
                except Exception:
                    logger.debug("OpenTelemetry context detach failed", exc_info=True)
            _SPAN_STACK.reset(stack_token)

    return run


@contextmanager
def phase_span(
    name: str,
    *,
    category: str = "client",
    attributes: dict[str, Any] | None = None,
) -> Iterator[PhaseSpan | None]:
    """Record one nested client phase and optionally mirror it to OpenTelemetry."""

    recorder = get_phase_trace_recorder()
    if recorder is None:
        yield None
        return

    span_id, parent_id, lane_id, started_ns = recorder.start_span()
    stack_token = _SPAN_STACK.set((*_SPAN_STACK.get(), span_id))
    normalized_attributes = {
        str(key): _json_value(value) for key, value in (attributes or {}).items()
    }

    otel_manager = None
    otel_span = None
    if recorder.otel_tracer is not None:
        try:
            otel_manager = recorder.otel_tracer.start_as_current_span(
                name,
                attributes=dict(normalized_attributes),
            )
            otel_span = otel_manager.__enter__()
        except Exception:
            logger.warning(
                "OpenTelemetry span start failed; continuing with Perfetto tracing",
                exc_info=True,
            )
            otel_manager = None
            otel_span = None

    span = PhaseSpan(
        span_id=span_id,
        attributes=normalized_attributes,
        _otel_span=otel_span,
    )
    failure: BaseException | None = None
    try:
        yield span
    except BaseException as error:
        failure = error
        raise
    finally:
        if failure is not None:
            span.set_attribute("error.type", type(failure).__name__)
        if otel_manager is not None:
            try:
                otel_manager.__exit__(
                    type(failure) if failure is not None else None,
                    failure,
                    failure.__traceback__ if failure is not None else None,
                )
            except Exception:
                logger.warning(
                    "OpenTelemetry span finish failed; Perfetto tracing continued",
                    exc_info=True,
                )
        recorder.finish_span(
            span_id=span_id,
            parent_id=parent_id,
            lane_id=lane_id,
            started_ns=started_ns,
            name=name,
            category=category,
            attributes=span.attributes,
        )
        _SPAN_STACK.reset(stack_token)


def flush_phase_trace() -> str | None:
    """Atomically write the current Perfetto trace, if configured."""

    recorder = _RECORDER
    return recorder.flush() if recorder is not None else None


def _reset_phase_tracing_for_tests() -> None:
    global _RECORDER, _RECORDER_INITIALIZED
    with _RECORDER_LOCK:
        _RECORDER = None
        _RECORDER_INITIALIZED = False
    _SPAN_STACK.set(())


def _flush_at_exit() -> None:
    try:
        flush_phase_trace()
    except Exception:
        logger.warning("Failed to flush client phase trace at exit", exc_info=True)


atexit.register(_flush_at_exit)
