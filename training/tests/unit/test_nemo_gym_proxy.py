"""CPU HTTP regressions for the NeMo adapter and real SDK TiTo runtime."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import threading
from types import SimpleNamespace
from typing import Any

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest
from fireworks.training.sdk import TITOParsedAssistant
from fireworks.training.sdk.sampling import SampledCompletion, SampledRequestResult

from training.examples.rl.nemo_gym.proxy import RecordingChatProxy, _is_user_simulator_call
from training.examples.rl.nemo_gym import train as nemo_train
from training.utils.rl.rollout.types import Rollout, RolloutRun, rollout_to_prompt_group


class StubRenderer:
    renderer_id = "synthetic-nemo-test"

    def render_conversation_tokens(self, request: Any) -> list[int]:
        return [1, 2, 3] + [len(request.messages)]

    def parse_assistant(
        self, request: Any, completion_ids: Any, completion_text: str, finish_reason: str
    ) -> TITOParsedAssistant:
        return TITOParsedAssistant(message={"role": "assistant", "content": "ok"})

    def fallback_assistant_text(self, *args: Any) -> str:
        return "ok"

    def render_contract_id(self, request: Any) -> str:
        return self.renderer_id

    def stop_sequences(self, request: Any) -> None:
        return None


class StubSampler:
    def __init__(self, routes: Any = None, block: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.routes = routes
        self.block = block
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def sample_with_prompt_tokens_result(self, prompt_ids: Any, **kwargs: Any) -> SampledRequestResult:
        self.calls.append(kwargs)
        self.started.set()
        if self.block:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        completion = SampledCompletion(
            text="ok",
            full_tokens=[*prompt_ids, 7, 8],
            prompt_len=len(prompt_ids),
            completion_len=2,
            sampling_logprobs=[-0.1, -0.2],
            finish_reason="stop",
            routing_matrices=self.routes,
        )
        return SampledRequestResult(
            completions=[completion], server_metrics=None, logical_request_id="test", attempts=1, wall_seconds=0.01
        )


def make_proxy(monkeypatch: Any, sampler: Any, **kwargs: Any) -> RecordingChatProxy:
    monkeypatch.setattr(
        "training.examples.rl.nemo_gym.proxy.build_sidecar_tito_renderer", lambda *args: StubRenderer()
    )
    return RecordingChatProxy(
        tokenizer=object(), renderer_name="synthetic", sampler=sampler, max_sample_tokens=64, temperature=0.7, **kwargs
    )


def body(**extra: Any) -> dict[str, Any]:
    return {"model": "policy", "messages": [{"role": "user", "content": "hi"}], **extra}


def test_admission_defaults_and_segments(monkeypatch: Any) -> None:
    sampler = StubSampler()
    proxy = make_proxy(monkeypatch, sampler)

    async def run() -> None:
        await proxy.start()
        try:
            async with aiohttp.ClientSession() as http:
                url = f"http://127.0.0.1:{proxy.port}/v1/chat/completions"
                assert (await http.post(url, json=body())).status == 400
                assert (await http.post(url, json=body(user="unknown"))).status == 410
                await proxy.open_session("rollout")
                response = await http.post(url, json=body(user="rollout", temperature=0, max_tokens=10000))
                assert response.status == 200, await response.text()
                assert (await response.json())["choices"][0]["message"]["content"] == "ok"
                # Changed history creates another segment rather than silently
                # assuming that a correlation id proves token ancestry.
                second = body(user="rollout")
                second["messages"].append({"role": "user", "content": "changed history"})
                response = await http.post(url, json=second)
                assert response.status == 200, await response.text()
                session = await proxy.finish_session("rollout")
                assert session is not None and session.turn_count == 2
                samples = session.to_run(reward=1.0).segments
                assert len(samples) == 2 and all(any(s.loss_mask) and s.reward == 1 for s in samples)
                assert (await http.post(url, json=body(user="rollout"))).status == 410
                assert proxy.active_rollout_ids() == []
                assert await proxy.finish_session("rollout") is None
                assert sampler.calls[0]["temperature"] == 0.7
                assert sampler.calls[0]["max_tokens"] == 64
                assert sampler.calls[0]["top_p"] == 1 and sampler.calls[0]["top_k"] == 0
        finally:
            await proxy.close()

    asyncio.run(run())


@pytest.mark.parametrize("routes", [["route-7", "route-8"], ["bad-route-count"], None])
def test_moe_routing_and_group_admission(monkeypatch: Any, routes: Any) -> None:
    sampler = StubSampler(routes)
    proxy = make_proxy(monkeypatch, sampler)
    proxy._initial_sampler = None

    async def run() -> None:
        await proxy.start()
        await proxy.set_sampler(sampler, {"include_routing_matrix": True, "echo": False})
        try:
            await proxy.open_session("moe")
            async with aiohttp.ClientSession() as http:
                response = await http.post(f"http://127.0.0.1:{proxy.port}/v1/chat/completions", json=body(user="moe"))
                if routes is None or len(routes) != 2:
                    assert response.status != 200
                    await proxy.finish_session("moe", abandon=True)
                    return
                assert response.status == 200, await response.text()
            session = await proxy.finish_session("moe")
            samples = session.to_run(reward=1.0).segments
            group = rollout_to_prompt_group(
                Rollout(runs=[RolloutRun(segments=samples)]),
                advantage_fn=lambda _: [1.0],
                router_replay_completion_only=True,
            )
            assert group is not None
            assert group.data[0].model_input.routing_matrices[-2:] == routes
        finally:
            await proxy.close()

    asyncio.run(run())


def test_abandon_cancels_sampling_and_rejects_late_requests(monkeypatch: Any) -> None:
    sampler = StubSampler(block=True)
    proxy = make_proxy(monkeypatch, sampler)

    async def run() -> None:
        await proxy.start()
        try:
            await proxy.open_session("timeout")
            async with aiohttp.ClientSession() as http:
                url = f"http://127.0.0.1:{proxy.port}/v1/chat/completions"
                pending = asyncio.create_task(http.post(url, json=body(user="timeout")))
                await asyncio.wait_for(sampler.started.wait(), timeout=2)
                await asyncio.wait_for(proxy.finish_session("timeout", abandon=True), timeout=2)
                await asyncio.wait_for(sampler.cancelled.wait(), timeout=2)
                assert (await http.post(url, json=body(user="timeout"))).status == 410
                await asyncio.gather(pending, return_exceptions=True)
                assert proxy.active_rollout_ids() == []
        finally:
            await proxy.close()

    asyncio.run(run())


def test_user_simulator_filter_is_opt_in(monkeypatch: Any) -> None:
    async def run(exclude: bool) -> list[Any]:
        proxy = make_proxy(monkeypatch, StubSampler(), exclude_user_simulator=exclude)
        await proxy.start()
        try:
            await proxy.open_session("sim")
            async with aiohttp.ClientSession() as http:
                response = await http.post(f"http://127.0.0.1:{proxy.port}/v1/chat/completions", json=body(user="sim"))
                assert response.status == 200, await response.text()
            session = await proxy.finish_session("sim")
            return session.to_run(reward=1.0).segments
        finally:
            await proxy.close()

    assert len(asyncio.run(run(False))) == 1
    assert asyncio.run(run(True)) == []
    assert _is_user_simulator_call([])
    assert _is_user_simulator_call([{"function": {"name": "end_conversation"}}])
    assert not _is_user_simulator_call([{"function": {"name": "a"}}, {"function": {"name": "b"}}])


@pytest.mark.parametrize("failure", [False, True])
def test_rollout_factory_closes_session_on_gym_failure(failure: bool) -> None:
    events: list[Any] = []

    async def opened(rollout_id: str) -> None:
        events.append(("open", rollout_id))

    async def finished(rollout_id: str, *, abandon: bool) -> Any:
        events.append(("finish", rollout_id, abandon))
        return SimpleNamespace(to_run=lambda **kw: RolloutRun(segments=[]), turn_count=0)

    async def run() -> None:
        app = web.Application()

        async def handler(request: Any) -> web.Response:
            payload = await request.json()
            assert payload["_ng_rollout_id"] == events[-1][1]
            return web.json_response({} if failure else {"reward": 1.0})

        app.router.add_post("/run", handler)
        async with TestServer(app) as server:
            thread = SimpleNamespace(set_sampler=lambda *args: None, open_session=opened, finish_session=finished)
            factory = nemo_train.build_rollout_fn_factory(
                proxy_thread=thread, agent_url=str(server.make_url("")), run_timeout_s=1
            )
            rollout = factory(SimpleNamespace(sampler=object(), sample_kwargs={}))
            if failure:
                with pytest.raises(KeyError):
                    await rollout({"id": 0}, cursor_index=1, rollout_idx=0)
            else:
                assert await rollout({"id": 0}, cursor_index=1, rollout_idx=0) is None
            assert events[-1][0] == "finish" and events[-1][2] is failure

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["config", "gym", "stop", None])
def test_recipe_startup_and_main_contract(monkeypatch, failure):
    events = []
    args = {
        "dataset_path": "rows.jsonl",
        "max_rows": 1,
        "tokenizer_model": "fake",
        "renderer_name": "fake",
        "max_completion_tokens": 2,
        "temperature": 1.0,
        "exclude_user_simulator": False,
        "proxy_port": 0,
        "resources_server": "example",
        "agent_name": "agent",
        "gym_start_timeout_s": 1,
        "log_path": ".",
        "wandb_run_name": "test",
        "base_model": "accounts/fireworks/models/test",
        "learning_rate": 1e-5,
        "completions_per_prompt": 2,
        "epochs": 1,
        "lora_rank": 8,
        "prompt_groups_per_step": 1,
        "max_concurrency_rollout_sample": 1,
        "output_model_id": "test",
        "training_shape_id": "shape",
        "replica_count": 1,
        "wandb_entity": "",
        "wandb_project": "test",
        "run_timeout_s": 1,
    }
    monkeypatch.setenv("FIREWORKS_API_KEY", "unused-test-credential")
    monkeypatch.setattr(nemo_train, "parse_args", lambda: argparse.Namespace(**args))
    monkeypatch.setattr(nemo_train, "_load_rows", lambda _: [{"id": "row"}])
    monkeypatch.setattr(nemo_train, "get_tokenizer", lambda _: object())
    monkeypatch.setattr(nemo_train, "RecordingChatProxy", lambda **kw: object())
    thread = SimpleNamespace(start=lambda _: 1234, stop=lambda: events.append("proxy-stop"))
    monkeypatch.setattr(nemo_train, "ProxyThread", lambda _: thread)

    def configure(**kwargs):
        if failure == "config":
            raise SystemExit("config refused")

    def start(**kwargs):
        if failure == "gym":
            raise RuntimeError("gym refused")
        return object(), "http://127.0.0.1"

    def stop(proc):
        events.append("gym-stop")
        if failure == "stop":
            raise RuntimeError("stop refused")

    signature = inspect.signature(nemo_train.main)

    def main(*args, **kwargs):
        signature.bind(*args, **kwargs)
        events.append("main")

    monkeypatch.setattr(nemo_train, "write_nemo_gym_config", configure)
    monkeypatch.setattr(nemo_train, "start_gym_env", start)
    monkeypatch.setattr(nemo_train, "stop_gym_env", stop)
    monkeypatch.setattr(nemo_train, "main", main)
    monkeypatch.setattr(nemo_train, "build_rollout_fn_factory", lambda **kw: object())
    if failure:
        with pytest.raises((SystemExit, RuntimeError)):
            nemo_train.run()
    else:
        nemo_train.run()
        assert "main" in events
    assert events[-1] == "proxy-stop"
    assert ("gym-stop" in events) == (failure not in {"config", "gym"})


@pytest.mark.parametrize("mode", ["ok", "http_error", "timeout", "no_reward"])
def test_real_proxy_thread_rollout_cleanup(monkeypatch: Any, mode: str) -> None:
    proxy = make_proxy(monkeypatch, StubSampler())
    proxy._initial_sampler = None
    thread = nemo_train.ProxyThread(proxy)
    thread.start(0)

    async def run() -> None:
        bodies: list[dict[str, Any]] = []

        async def handler(request: Any) -> web.Response:
            payload = await request.json()
            bodies.append(payload)
            if mode == "http_error":
                return web.Response(status=500)
            if mode == "timeout":
                await asyncio.sleep(0.2)
                return web.json_response({"reward": 1.0})
            async with aiohttp.ClientSession() as http:
                response = await http.post(
                    f"http://127.0.0.1:{proxy.port}/v1/chat/completions", json=body(user=payload["_ng_rollout_id"])
                )
                assert response.status == 200, await response.text()
            return web.json_response({} if mode == "no_reward" else {"reward": 1.0})

        app = web.Application()
        app.router.add_post("/run", handler)
        async with TestServer(app) as server:
            factory = nemo_train.build_rollout_fn_factory(
                proxy_thread=thread, agent_url=str(server.make_url("")), run_timeout_s=0.05 if mode == "timeout" else 2
            )
            rollout = factory(SimpleNamespace(sampler=StubSampler(), sample_kwargs={}))
            if mode == "ok":
                result = await rollout({"id": 0}, cursor_index=0, rollout_idx=0)
                assert len(result.segments) == 1 and result.segments[0].reward == 1.0
                assert result.run_id == bodies[0]["_ng_rollout_id"]
                assert result.metadata["tito_metrics"]["tito/coverage/sampled_completion_tokens"] == 2.0
                assert result.metadata["tito_metrics"]["tito/coverage/trained_tokens"] == 2.0
                assert result.metadata["tito_segment_count"] == 1
            else:
                expected = {"http_error": aiohttp.ClientResponseError, "timeout": TimeoutError, "no_reward": KeyError}[
                    mode
                ]
                with pytest.raises(expected):
                    await rollout({"id": 0}, cursor_index=0, rollout_idx=0)
            async with aiohttp.ClientSession() as http:
                response = await http.post(
                    f"http://127.0.0.1:{proxy.port}/v1/chat/completions", json=body(user=bodies[0]["_ng_rollout_id"])
                )
                assert response.status == 410

    try:
        asyncio.run(run())
    finally:
        thread.stop()


def test_single_tool_policy_is_enforced(monkeypatch: Any) -> None:
    class ToolRenderer(StubRenderer):
        def parse_assistant(self, *args: Any) -> TITOParsedAssistant:
            return TITOParsedAssistant(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"type": "function", "id": f"call-{i}", "function": {"name": "f", "arguments": "{}"}}
                        for i in range(2)
                    ],
                },
                output_kind="tool_calls",
            )

    proxy = make_proxy(monkeypatch, StubSampler())
    proxy._renderer = ToolRenderer()

    async def run() -> None:
        await proxy.start()
        try:
            await proxy.open_session("single-tool")
            async with aiohttp.ClientSession() as http:
                response = await http.post(
                    f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
                    json=body(user="single-tool", parallel_tool_calls=False),
                )
                assert response.status == 502
            await proxy.finish_session("single-tool", abandon=True)
        finally:
            await proxy.close()

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["before_creation", "after_creation", "after_registration"])
@pytest.mark.parametrize("repeat_cancel", [False, True])
def test_cancelled_admission_abandons_trajectory(monkeypatch: Any, stage: str, repeat_cancel: bool) -> None:
    proxy = make_proxy(monkeypatch, StubSampler())
    proxy._initial_sampler = None
    thread = nemo_train.ProxyThread(proxy)
    thread.start(0)
    started = threading.Event()
    finished = threading.Event()
    gate = asyncio.Event()
    artifacts: list[Any] = []
    original_finish = proxy.finish_session

    async def finish(rollout_id: str, *, abandon: bool = False) -> Any:
        session = await original_finish(rollout_id, abandon=abandon)
        if session is not None:
            artifacts.append(session.artifact)
        finished.set()
        return session

    monkeypatch.setattr(proxy, "finish_session", finish)

    async def run() -> None:
        app = web.Application()

        async def handler(request: Any) -> web.Response:
            pytest.fail("cancelled admission must not call Gym")

        app.router.add_post("/run", handler)
        async with TestServer(app) as server:
            factory = nemo_train.build_rollout_fn_factory(
                proxy_thread=thread, agent_url=str(server.make_url("")), run_timeout_s=2
            )
            rollout = factory(SimpleNamespace(sampler=StubSampler(), sample_kwargs={}))
            original_create = proxy._sidecar.create_trajectory_async
            original_open = proxy.open_session

            async def create(**kwargs: Any) -> Any:
                if stage == "before_creation":
                    started.set()
                    await gate.wait()
                endpoint = await original_create(**kwargs)
                if stage == "after_creation":
                    started.set()
                    await gate.wait()
                return endpoint

            async def opened(rollout_id: str) -> Any:
                session = await original_open(rollout_id)
                started.set()
                await gate.wait()
                return session

            if stage == "after_registration":
                monkeypatch.setattr(proxy, "open_session", opened)
            else:
                monkeypatch.setattr(proxy._sidecar, "create_trajectory_async", create)
            task = asyncio.create_task(rollout({"id": 0}, cursor_index=0, rollout_idx=0))
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            if repeat_cancel:
                task.cancel()
            thread._loop.call_soon_threadsafe(gate.set)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
            assert await asyncio.to_thread(finished.wait, 2)
            assert proxy.active_rollout_ids() == []
            assert thread._admissions == {}
            assert len(artifacts) == 1 and artifacts[0].status == "abandoned"
            assert proxy._sidecar._engines == {}

    try:
        asyncio.run(run())
    finally:
        thread._loop.call_soon_threadsafe(gate.set)
        thread.stop()
