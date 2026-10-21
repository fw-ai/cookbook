from __future__ import annotations

import asyncio
import pickle
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from fireworks.training.sdk import TITOSidecar, TITOParsedAssistant, TITOParseError
from fireworks.training.sdk.sampling import SampledRequestResult

from training.examples.rl.nemo_gym import train as nemo_train
from training.utils.rl.async_rl.errors import DiscardTrajectoryError, classify_rollout_error
from training.tests.unit.test_nemo_gym_proxy import StubRenderer, StubSampler, body, make_proxy


def test_missing_scoped_retention_api_fails_before_proxy_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(TITOSidecar, "dump_failure_trace")
    proxy = make_proxy(monkeypatch, StubSampler())
    with pytest.raises(RuntimeError, match="requires an SDK with TITOSidecar.dump_failure_trace"):
        asyncio.run(proxy.start())
    assert proxy.port == 0
    assert proxy._http is None
    assert proxy._sidecar is None


@pytest.mark.parametrize("mode", ["parser", "upstream", "parallel_policy"])
@pytest.mark.parametrize("trace_failure", [False, True])
def test_native_adapter_preserves_model_error_over_agent_http(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, mode: str, trace_failure: bool,
) -> None:
    class ErrorSampler(StubSampler):
        async def sample_with_prompt_tokens_result(self, prompt_ids: list[int], **kwargs: Any) -> SampledRequestResult:
            if mode == "upstream":
                response = httpx.Response(
                    503, headers={"x-request-id": "native-upstream-request"}, request=httpx.Request("POST", "http://upstream")
                )
                raise httpx.HTTPStatusError("synthetic upstream unavailable", request=response.request, response=response)
            return await super().sample_with_prompt_tokens_result(prompt_ids, **kwargs)

    def reject(*args: Any) -> TITOParsedAssistant:
        raise TITOParseError("synthetic private payload", reason_code="tool_unparsed_boundary")

    if mode == "parser":
        monkeypatch.setattr(StubRenderer, "parse_assistant", reject)
    elif mode == "parallel_policy":
        def parallel_calls(*args: Any) -> TITOParsedAssistant:
            return TITOParsedAssistant(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"type": "function", "id": f"call-{index}",
                         "function": {"name": "lookup", "arguments": "{}"}}
                        for index in range(2)
                    ],
                },
                output_kind="tool_calls",
            )

        monkeypatch.setattr(StubRenderer, "parse_assistant", parallel_calls)
    trace_root = tmp_path
    if trace_failure:
        trace_root = tmp_path / "unavailable"
        trace_root.write_text("not a directory")

        def fail_dump(*args: Any, **kwargs: Any) -> str | None:
            raise OSError("private dump payload")

        monkeypatch.setattr(TITOSidecar, "dump_failure_trace", fail_dump)
    proxy = make_proxy(monkeypatch, ErrorSampler(), failure_trace_dir=trace_root)
    proxy._initial_sampler = None
    thread = nemo_train.ProxyThread(proxy)
    thread.start(0)

    async def run() -> None:
        async def handler(request: web.Request) -> web.Response:
            payload = await request.json()
            async with aiohttp.ClientSession() as http:
                response = await http.post(
                    f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
                    json=body(user=payload["_ng_rollout_id"], parallel_tool_calls=False),
                )
                assert response.status == (503 if mode == "upstream" else 422)
            return web.json_response({"error": "native agent lost inner diagnostics"}, status=500)

        app = web.Application()
        app.router.add_post("/run", handler)
        async with TestServer(app) as server:
            rollout = nemo_train.build_rollout_fn_factory(
                proxy_thread=thread, agent_url=str(server.make_url("")), run_timeout_s=10
            )(SimpleNamespace(sampler=ErrorSampler(), sample_kwargs={}))
            with pytest.raises(Exception) as caught:
                await rollout({"id": 0}, cursor_index=0, rollout_idx=0)
            classification = classify_rollout_error(caught.value)
            restored = pickle.loads(pickle.dumps(caught.value))
            assert classify_rollout_error(restored) == classification
            if mode == "parser":
                assert isinstance(caught.value, DiscardTrajectoryError)
                assert classification.disposition.value == "discard"
                assert classification.reason == "tito_sdk_parse_error"
                assert classification.diagnostics["parser_reason_code"] == "tool_unparsed_boundary"
                assert classification.diagnostics["origin"] == "sdk_parser"
            elif mode == "parallel_policy":
                assert isinstance(caught.value, DiscardTrajectoryError)
                assert classification.disposition.value == "discard"
                assert classification.reason == "nemo_parallel_tool_calls"
                assert classification.diagnostics["origin"] == "adapter"
            else:
                assert classification.status_code == 503
                assert classification.diagnostics["origin"] == "server"
            if trace_failure:
                assert classification.trace_path is None
                assert "NeMo failure trace unavailable" in caplog.text
                assert "private dump payload" not in caplog.text
            else:
                assert classification.trace_path and Path(classification.trace_path).is_file()
                assert Path(classification.trace_path).is_relative_to(tmp_path)
                assert classification.trace_path in caplog.text
            assert proxy.active_rollout_ids() == []

    try:
        asyncio.run(run())
    finally:
        thread.stop()


@pytest.mark.parametrize("body_failure", ["payload", "content_type", "invalid_json"])
@pytest.mark.parametrize("close_session", [False, True])
def test_response_body_failure_is_closed_only_after_abandon(
    monkeypatch: pytest.MonkeyPatch, body_failure: str, close_session: bool,
) -> None:
    proxy = make_proxy(monkeypatch, StubSampler())

    async def run() -> None:
        headers_sent = asyncio.Event()
        release_body = asyncio.Event()

        async def incomplete_response(request: web.Request) -> web.StreamResponse:
            headers = {
                "Content-Type": "text/plain" if body_failure == "content_type" else "application/json",
                "x-request-id": "sidecar-response-id",
            }
            if body_failure == "payload":
                headers["Content-Length"] = "1000"
            response = web.StreamResponse(headers=headers)
            await response.prepare(request)
            await response.write(b"bad")
            headers_sent.set()
            await release_body.wait()
            if body_failure == "payload":
                request.transport.close()
            else:
                await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/v1/chat/completions", incomplete_response)
        await proxy.start()
        try:
            async with TestServer(app) as server:
                session = await proxy.open_session("body-race")
                session.endpoint = replace(session.endpoint, openai_base_url=str(server.make_url("/v1")))
                async with aiohttp.ClientSession() as http:
                    pending = asyncio.create_task(http.post(
                        f"http://127.0.0.1:{proxy.port}/v1/chat/completions", json=body(user="body-race")))
                    await asyncio.wait_for(headers_sent.wait(), timeout=2)
                    if close_session:
                        await proxy.finish_session("body-race", abandon=True)
                    release_body.set()
                    response = await asyncio.wait_for(pending, timeout=2)
                    if close_session:
                        assert response.status == 410
                        assert (await response.json())["error"]["terminal_action"] == "abandon"
                        assert proxy.active_rollout_ids() == []
                    else:
                        assert response.status == 502
                        error = (await response.json())["error"]
                        assert error["origin"] == ("transport" if body_failure == "payload" else "adapter")
                        assert error["transport_hop"] == "nemo_proxy_to_tito"
                        assert error["wire_status"] == 200
                        assert error["request_id"] == "sidecar-response-id"
                        assert proxy.active_rollout_ids() == ["body-race"]
                        finished = await proxy.finish_session("body-race", abandon=True)
                        assert Path(finished.failure.context["trace_path"]).is_file()
        finally:
            release_body.set()
            await proxy.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["disconnect", "missing_origin", "missing_origin_400"])
def test_sidecar_hop_preserves_origin_and_trace(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, mode: str,
) -> None:
    proxy = make_proxy(monkeypatch, StubSampler())

    async def run() -> None:
        async def handler(request: web.Request) -> web.Response:
            if mode == "disconnect":
                request.transport.close()
                return web.Response()
            return web.json_response(
                {"error": {"code": "unidentified_sidecar_error"}}, status=400 if mode == "missing_origin_400" else 503,
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        await proxy.start()
        try:
            async with TestServer(app) as server:
                session = await proxy.open_session("sidecar-hop")
                session.endpoint = replace(session.endpoint, openai_base_url=str(server.make_url("/v1")))
                async with aiohttp.ClientSession() as http:
                    response = await http.post(
                        f"http://127.0.0.1:{proxy.port}/v1/chat/completions", json=body(user="sidecar-hop")
                    )
                    assert response.status == (502 if mode == "disconnect" else 400 if mode == "missing_origin_400" else 503)
                    error = (await response.json())["error"]
                    assert error["origin"] == ("transport" if mode == "disconnect" else "unknown")
                finished = await proxy.finish_session("sidecar-hop", abandon=True)
                assert classify_rollout_error(finished.failure).recoverable is (mode != "missing_origin_400")
                trace_path = finished.failure.context["trace_path"]
                assert Path(trace_path).is_file()
                assert trace_path in caplog.text
                if mode == "disconnect":
                    assert finished.failure.context["transport_hop"] == "nemo_proxy_to_tito"
                    assert finished.failure.context["phase"] == "sidecar_request"
        finally:
            await proxy.close()

    asyncio.run(run())


@pytest.mark.parametrize("status", [400, 503])
@pytest.mark.parametrize("content_type", ["text/plain", "application/json"])
def test_non_json_sidecar_http_error_is_not_a_parser_discard(
    monkeypatch: pytest.MonkeyPatch, status: int, content_type: str,
) -> None:
    proxy = make_proxy(monkeypatch, StubSampler())

    async def run() -> None:
        async def handler(request: web.Request) -> web.Response:
            return web.Response(
                status=status, text="unavailable", content_type=content_type,
                headers={"x-request-id": "sidecar-http-id"},
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        await proxy.start()
        try:
            async with TestServer(app) as server:
                session = await proxy.open_session("http-hop")
                session.endpoint = replace(session.endpoint, openai_base_url=str(server.make_url("/v1")))
                async with aiohttp.ClientSession() as http:
                    response = await http.post(
                        f"http://127.0.0.1:{proxy.port}/v1/chat/completions", json=body(user="http-hop")
                    )
                    assert response.status == status
                    error = (await response.json())["error"]
                    assert error["code"] == "nemo_sidecar_http_error"
                    assert error["origin"] == "server"
                    assert error["wire_status"] == status
                    assert error["transport_hop"] == "nemo_proxy_to_tito"
                    assert error["request_id"] == "sidecar-http-id"
                    assert error["discard_trajectory"] is False
                    assert error["should_retry"] is (status == 503)
                assert classify_rollout_error(session.failure).recoverable is (status == 503)
                finished = await proxy.finish_session("http-hop")
                assert finished.artifact.status == "completed"
                assert Path(finished.failure.context["trace_path"]).is_file()
        finally:
            await proxy.close()

    asyncio.run(run())


@pytest.mark.parametrize("reward_mode", ["valid", "missing", "nonnumeric", "nan", "infinity", "json", "content_type"])
@pytest.mark.parametrize("trace_failure", [False, True])
def test_reward_validation_after_a_non_discard_model_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, reward_mode: str, trace_failure: bool,
) -> None:
    if trace_failure:
        def fail_dump(*args: Any, **kwargs: Any) -> str | None:
            raise OSError("private dump payload")

        monkeypatch.setattr(TITOSidecar, "dump_failure_trace", fail_dump)
    class RecoveringSampler(StubSampler):
        async def sample_with_prompt_tokens_result(self, prompt_ids: list[int], **kwargs: Any) -> SampledRequestResult:
            if self.calls:
                response = httpx.Response(503, request=httpx.Request("POST", "http://upstream"))
                raise httpx.HTTPStatusError("synthetic unavailable", request=response.request, response=response)
            return await super().sample_with_prompt_tokens_result(prompt_ids, **kwargs)

    sampler = RecoveringSampler()
    proxy = make_proxy(monkeypatch, sampler, failure_trace_dir=tmp_path)
    proxy._initial_sampler = None
    thread = nemo_train.ProxyThread(proxy)
    thread.start(0)

    async def run() -> None:
        async def handler(request: web.Request) -> web.Response:
            payload = await request.json()
            rollout_id = payload["_ng_rollout_id"]
            async with aiohttp.ClientSession() as http:
                endpoint = f"http://127.0.0.1:{proxy.port}/v1/chat/completions"
                first = await http.post(endpoint, json=body(user=rollout_id))
                assert first.status == 200
                followup = body(user=rollout_id)
                followup["messages"].append({"role": "user", "content": "followup"})
                failed = await http.post(endpoint, json=followup)
                assert failed.status == 503
            if reward_mode == "json":
                return web.Response(text="{", content_type="application/json")
            if reward_mode == "content_type":
                return web.Response(text='{"reward":1.0}')
            rewards = {"valid": 1.0, "nonnumeric": "bad", "nan": "nan", "infinity": "inf"}
            return web.json_response({} if reward_mode == "missing" else {"reward": rewards[reward_mode]})

        app = web.Application()
        app.router.add_post("/run", handler)
        async with TestServer(app) as server:
            factory = nemo_train.build_rollout_fn_factory(
                proxy_thread=thread, agent_url=str(server.make_url("")), run_timeout_s=2
            )
            rollout = factory(SimpleNamespace(sampler=sampler, sample_kwargs={}))
            if reward_mode == "valid":
                result = await rollout({"id": 0}, cursor_index=0, rollout_idx=0)
                assert len(result.segments) == 1
                assert result.segments[0].reward == 1.0
            else:
                with pytest.raises((KeyError, ValueError, aiohttp.ContentTypeError)) as caught:
                    await rollout({"id": 0}, cursor_index=0, rollout_idx=0)
                classification = classify_rollout_error(caught.value)
                assert classification.disposition.value == "fatal"
                assert classification.diagnostics["code"] == "nemo_reward_invalid"
                assert classification.diagnostics["origin"] == "environment"
                assert classification.diagnostics["phase"] == "reward"
                if trace_failure:
                    assert classification.trace_path is None
                    assert "NeMo failure trace unavailable" in caplog.text
                    assert "private dump payload" not in caplog.text
                else:
                    assert classification.trace_path is not None
                    trace_path = Path(classification.trace_path)
                    assert trace_path.is_file() and trace_path.is_relative_to(tmp_path)
                    assert str(trace_path) in caplog.text
            assert proxy.active_rollout_ids() == []

    try:
        asyncio.run(run())
    finally:
        thread.stop()


def test_upstream_closed_trajectory_discards_only_the_native_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    sampler = StubSampler()
    proxy = make_proxy(monkeypatch, sampler)
    proxy._initial_sampler = None
    original_open = proxy.open_session

    async def open_closed(rollout_id: str) -> Any:
        session = await original_open(rollout_id)
        await proxy._sidecar.finish_trajectory(rollout_id)
        return session

    monkeypatch.setattr(proxy, "open_session", open_closed)
    thread = nemo_train.ProxyThread(proxy)
    thread.start(0)

    async def run() -> None:
        async def handler(request: web.Request) -> web.Response:
            payload = await request.json()
            async with aiohttp.ClientSession() as http:
                response = await http.post(
                    f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
                    json=body(user=payload["_ng_rollout_id"]),
                )
                assert response.status == 410
            return web.json_response({"error": "model call closed"}, status=500)

        app = web.Application()
        app.router.add_post("/run", handler)
        async with TestServer(app) as server:
            factory = nemo_train.build_rollout_fn_factory(
                proxy_thread=thread, agent_url=str(server.make_url("")), run_timeout_s=2
            )
            rollout = factory(SimpleNamespace(sampler=sampler, sample_kwargs={}))
            with pytest.raises(DiscardTrajectoryError) as failure:
                await rollout({"id": 0}, cursor_index=0, rollout_idx=0)
            assert classify_rollout_error(failure.value).reason == "tito_trajectory_closed"
            assert proxy.active_rollout_ids() == []

    try:
        asyncio.run(run())
    finally:
        thread.stop()


def test_run_preserves_existing_head_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "env.yaml"
    existing_config = nemo_train._GENERATED_HEADER + "\npolicy_base_url: http://existing-run/v1\n"
    config_path.write_text(existing_config)
    monkeypatch.setenv("FIREWORKS_API_KEY", "unused-test-credential")
    monkeypatch.setattr(nemo_train, "NEMO_GYM_DIR", tmp_path)
    monkeypatch.setattr(nemo_train, "parse_args", lambda: SimpleNamespace())
    monkeypatch.setattr(nemo_train, "_http_up", lambda _: True)

    with pytest.raises(RuntimeError, match="refusing to reuse another run"):
        nemo_train.run()

    assert config_path.read_text() == existing_config
    assert sorted(path.name for path in tmp_path.iterdir()) == ["env.yaml"]


def test_startup_refuses_existing_head_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(nemo_train, "_http_up", lambda *args: True)

    def refuse_launch(*args: Any) -> None:
        raise AssertionError("must not launch against another run's head server")

    monkeypatch.setattr(nemo_train, "_launch_gym_env", refuse_launch)
    with pytest.raises(RuntimeError, match="refusing to reuse"):
        nemo_train.start_gym_env(resources_server="test", agent_name="test", timeout_s=1, log_dir=tmp_path)


def test_shutdown_reaps_group_even_when_leader_exits_first(monkeypatch: pytest.MonkeyPatch) -> None:
    signals = []
    process = SimpleNamespace(pid=12345, wait=lambda **kwargs: 0)
    monkeypatch.setattr(nemo_train.os, "killpg", lambda group, action: signals.append((group, action)))
    nemo_train.stop_gym_env(process)
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]


def test_shutdown_timeout_preserves_startup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    wait_timeouts = []
    signals = []
    startup_error = RuntimeError("startup unavailable")

    def wait(*, timeout: float) -> None:
        wait_timeouts.append(timeout)
        raise subprocess.TimeoutExpired("gym", timeout)

    def fail_startup(*_args: Any) -> None:
        raise startup_error

    process = SimpleNamespace(pid=12345, wait=wait)
    monkeypatch.setattr(nemo_train, "_check_gym_head_available", lambda: None)
    monkeypatch.setattr(nemo_train, "_launch_gym_env", lambda *_args: process)
    monkeypatch.setattr(nemo_train, "_wait_for_gym_env", fail_startup)
    monkeypatch.setattr(nemo_train.os, "killpg", lambda group, action: signals.append((group, action)))
    with pytest.raises(RuntimeError, match="startup unavailable") as failure:
        nemo_train.start_gym_env(resources_server="test", agent_name="test", timeout_s=1, log_dir=tmp_path)
    assert failure.value is startup_error
    assert wait_timeouts == [30, 10]
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert "did not exit after SIGKILL pid=12345" in caplog.text


def test_shutdown_terminates_owned_server_descendants(tmp_path: Path) -> None:
    marker = tmp_path / "child.pid"
    script = (
        "import signal,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "def stop(*args):\n    child.wait(timeout=5)\n    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"Path({str(marker)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        child_pid = int(marker.read_text())
        nemo_train.stop_gym_env(process)
        assert process.poll() == 0
        assert not Path(f"/proc/{child_pid}").exists()
    finally:
        if process.poll() is None:
            nemo_train.stop_gym_env(process)
