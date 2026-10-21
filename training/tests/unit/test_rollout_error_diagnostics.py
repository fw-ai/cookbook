from __future__ import annotations

import multiprocessing
import pickle
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx
import pytest
from fireworks.training.sdk import TITOError
from fireworks.training.sdk.sampling_observability import SamplingRequestError
from training.utils.rl.async_rl.errors import (
    CircuitBreakerConfig,
    CircuitBreakerTripped,
    DiscardTrajectoryError,
    ErrorDisposition,
    RecoverableCircuitBreaker,
    RecoverableRolloutError,
    classify_rollout_error,
)


def _worker_failure(destination: str) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "failure.jsonl"
        source.write_text("retained private evidence")
        shutil.copy2(source, destination)
    raise RecoverableRolloutError(
        "private sandbox payload",
        reason="whole_trial_bound",
        trace_paths=(destination,),
        diagnostics={
            "code": "tito_transport_error",
            "wire_status": 200,
            "request_id": "request-1",
            "message": "private",
        },
    )


def test_spawn_worker_preserves_reason_diagnostics_and_retained_evidence(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "retained.jsonl"
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        with pytest.raises(RecoverableRolloutError) as failure:
            executor.submit(_worker_failure, str(destination)).result(timeout=30)
    classification = classify_rollout_error(failure.value)
    assert classification.reason == "whole_trial_bound"
    assert classification.recoverable
    assert classification.trace_paths == (str(destination),)
    assert classification.trace_path == str(destination)
    assert classification.diagnostics == {
        "code": "tito_transport_error",
        "wire_status": 200,
        "request_id": "request-1",
    }
    assert "private" not in repr(classification.as_log_record())
    assert destination.read_text() == "retained private evidence"


@pytest.mark.parametrize("reason", [None, "private reason\n/tmp/x", "x" * 129])
def test_invalid_or_missing_reason_uses_legacy_fallback(reason: str | None) -> None:
    classification = classify_rollout_error(
        RecoverableRolloutError("private", reason=reason)
    )
    assert classification.reason == "explicit_rollout_infrastructure"


def test_http_error_diagnostics_preserve_policy_and_drop_payload() -> None:
    response = httpx.Response(
        422,
        request=httpx.Request("POST", "http://localhost/chat/completions"),
        json={
            "error": {
                "code": "tito_sdk_parse_error",
                "parser_reason_code": "tool_unparsed_boundary",
                "completion_tokens": 12,
                "trace_path": "/local/trace.jsonl",
                "message": "private",
                "raw_output": [1, 2],
            }
        },
    )
    classification = classify_rollout_error(
        httpx.HTTPStatusError("private", request=response.request, response=response)
    )
    assert classification.disposition is ErrorDisposition.DISCARD
    assert classification.status_code == 422
    assert classification.diagnostics["parser_reason_code"] == "tool_unparsed_boundary"
    assert classification.diagnostics["completion_tokens"] == 12
    assert classification.trace_paths == ("/local/trace.jsonl",)
    assert "private" not in repr(classification)
    assert "raw_output" not in classification.diagnostics


def test_discard_diagnostics_survive_worker_pickle() -> None:
    error = DiscardTrajectoryError(
        "private",
        trace_path="/local/trace",
        diagnostics={"parser_reason_code": "native_parse_rejected", "origin": "sdk_parser"},
    )
    classification = classify_rollout_error(pickle.loads(pickle.dumps(error)))
    assert classification.diagnostics["parser_reason_code"] == "native_parse_rejected"
    assert classification.diagnostics["origin"] == "sdk_parser"
    assert classification.trace_paths == ("/local/trace",)


@pytest.mark.parametrize("evidence", ["response", "legacy_stamp", "wire_stamp"])
def test_response_headers_do_not_turn_a_read_failure_into_a_fatal_http_200(evidence: str) -> None:
    request = httpx.Request("POST", "http://localhost/completions")
    error = httpx.ReadError("private stream failure", request=request)
    if evidence == "response":
        error.response = httpx.Response(200, request=request, headers={"x-request-id": "read-failure-request"})
    else:
        attribute = "_fireworks_http_status" if evidence == "legacy_stamp" else "_fireworks_wire_status"
        setattr(error, attribute, 200)
        error._fireworks_response_request_id = "read-failure-request"
    classification = classify_rollout_error(error)
    assert classification.recoverable
    assert classification.reason == "network_transport"
    assert classification.status_code is None
    assert classification.diagnostics["wire_status"] == 200
    assert classification.diagnostics["request_id"] == "read-failure-request"


def test_server_error_keeps_private_trace_pointer_and_retry_facts() -> None:
    response = httpx.Response(
        502,
        request=httpx.Request("POST", "http://localhost/chat/completions"),
        json={
            "error": {
                "code": "tito_server_stream_error",
                "trace_path": "/local/server.jsonl",
                "should_retry": False,
                "wire_status": 200,
                "retry_budget_exhausted": False,
            }
        },
    )
    classification = classify_rollout_error(
        httpx.HTTPStatusError("private", request=response.request, response=response)
    )
    assert classification.disposition is ErrorDisposition.FATAL
    assert classification.trace_paths == ("/local/server.jsonl",)
    assert classification.diagnostics["should_retry"] is False
    assert classification.diagnostics["wire_status"] == 200
    assert classification.diagnostics["retry_budget_exhausted"] is False


@pytest.mark.parametrize("code,finish", [(None, "error"), ("invalid_completion_event", None)])
def test_permanent_sampler_stream_error_is_not_retried_as_transport(code: str | None, finish: str | None) -> None:
    error = SamplingRequestError(
        final_error_kind="upstream_stream", upstream_error_code=code, upstream_finish_reason=finish,
        wire_status=200, attempts=1, retry_budget_exhausted=False,
    )
    classification = classify_rollout_error(error)
    assert classification.disposition is ErrorDisposition.FATAL
    assert classification.reason == "upstream_stream_rejected"
    assert classification.diagnostics["origin"] == "server"


@pytest.mark.parametrize("code,status,origin,semantic,disposition,reason", [
    ("tito_debug_storage_error", 507, "sdk", None, ErrorDisposition.FATAL, "tito_debug_storage_error"),
    ("tito_sidecar_closed", 503, "sdk", None, ErrorDisposition.FATAL, "tito_sidecar_closed"),
    ("tito_server_stream_error", 502, "server", None, ErrorDisposition.FATAL, "upstream_stream_rejected"),
    ("tito_server_stream_error", 502, "server", 503, ErrorDisposition.RECOVERABLE, "upstream_stream_transient"),
    ("tito_server_stream_error", 400, "server", 400, ErrorDisposition.FATAL, "upstream_stream_rejected"),
    ("tito_transport_error", 502, "transport", None, ErrorDisposition.RECOVERABLE, "network_transport"),
])
def test_structured_origins_take_precedence_over_proxy_status(
    code: str, status: int, origin: str, semantic: int | None, disposition: ErrorDisposition, reason: str,
) -> None:
    error = TITOError(code, status, "private", origin=origin, context={"upstream_status": semantic})
    classification = classify_rollout_error(error)
    assert classification.disposition is disposition
    assert classification.reason == reason
    assert classification.diagnostics["origin"] == origin


@pytest.mark.parametrize("status,hop,recoverable", [
    (503, "nemo_proxy_to_tito", True),
    (400, "nemo_proxy_to_tito", False),
    (503, None, False),
    (503, "another_hop", False),
])
def test_unknown_origin_http_recovery_requires_a_known_sidecar_hop(
    status: int, hop: str | None, recoverable: bool,
) -> None:
    error = TITOError(
        "unidentified_sidecar_error", status, "private", origin="unknown",
        context={"transport_hop": hop, "wire_status": status},
    )
    assert classify_rollout_error(error).recoverable is recoverable


def test_statusless_stream_rejection_stays_fatal_with_missing_origin() -> None:
    error = TITOError(
        "tito_server_stream_error", 502, "private", origin="unknown",
        context={"transport_hop": "nemo_proxy_to_tito", "wire_status": 200},
    )
    classification = classify_rollout_error(error)
    assert classification.disposition is ErrorDisposition.FATAL
    assert classification.reason == "upstream_stream_rejected"


def test_window_counts_expire_without_changing_aggregate_policy() -> None:
    breaker = RecoverableCircuitBreaker(
        CircuitBreakerConfig(
            max_consecutive_failures=100,
            rolling_window_size=4,
            rolling_min_observations=4,
            max_failure_rate=1,
        )
    )
    first = RecoverableRolloutError(
        reason="whole_trial_bound", diagnostics={"code": "tito_transport_error"}
    )
    second = RecoverableRolloutError(diagnostics={"code": "tito_server_stream_error"})
    breaker.record_failure(first, classify_rollout_error(first))
    breaker.record_failure(second, classify_rollout_error(second))
    breaker.record_success()
    breaker.record_failure(first, classify_rollout_error(first))
    original = breaker.snapshot
    assert original.reasons_in_window == {
        "whole_trial_bound": 2,
        "explicit_rollout_infrastructure": 1,
    }
    assert original.failures_in_window == 3
    breaker.record_success()
    assert breaker.snapshot.codes_in_window == {
        "tito_transport_error": 1,
        "tito_server_stream_error": 1,
    }
    breaker.record_success()
    assert breaker.snapshot.reasons_in_window == {"whole_trial_bound": 1}
    breaker.record_success()
    breaker.record_success()
    assert breaker.snapshot.reasons_in_window == breaker.snapshot.codes_in_window == {}
    assert breaker.snapshot.failures_in_window == 0
    assert original.failures_in_window == 3


def test_direct_decoder_carrier_is_not_a_caller_contract_failure() -> None:
    error = ValueError("private undecodable completion")
    error.code = "tito_sdk_sampling_parse_error"
    error.origin = "sdk_parser"
    error._fireworks_wire_status = 200
    error._fireworks_response_request_id = "decoder-request"
    classification = classify_rollout_error(error)
    assert classification.disposition is ErrorDisposition.DISCARD
    assert classification.reason == "tito_sdk_sampling_parse_error"
    assert classification.diagnostics["origin"] == "sdk_parser"
    assert classification.diagnostics["wire_status"] == 200
    assert classification.diagnostics["request_id"] == "decoder-request"


def test_breaker_trip_message_contains_bounded_counts_not_exception_payload() -> None:
    breaker = RecoverableCircuitBreaker(
        CircuitBreakerConfig(max_consecutive_failures=1)
    )
    error = RecoverableRolloutError("private tool payload", reason="whole_trial_bound")
    with pytest.raises(CircuitBreakerTripped) as failure:
        breaker.record_failure(error, classify_rollout_error(error))
    assert failure.value.snapshot.reasons_in_window == {"whole_trial_bound": 1}
    assert "whole_trial_bound" in str(failure.value)
    assert "private" not in str(failure.value)
    assert failure.value.last_error is error
