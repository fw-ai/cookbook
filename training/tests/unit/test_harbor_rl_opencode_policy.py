"""Mechanism tests for Harbor/OpenCode history recording."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from training.examples.rl.harbor import openai_policy
from training.examples.rl.harbor.openai_policy import (
    OpenCodePolicyServer,
    OpenCodePolicySession,
)
from training.renderer.reasoning_fields import ORIGINAL_REASONING_CONTENT
from training.utils.rl.rollout import (
    Rollout,
    RolloutRun,
    RolloutSample,
    rollout_to_prompt_group,
)
from training.utils.rl.agent.openai import CookbookTurnRenderer
from training.utils.rl.agent.sampling import token_segment_to_sample
from training.utils.rl.agent.trajectory import TurnRecord


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "run a command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


class _FakeRenderer:
    prompts = {
        "task": [1, 2],
        "obs1": [1, 2, 10, 20],
        "title": [7],
        "summary1": [9, 9],
        "obs2": [9, 9, 30, 40],
        "summary2": [8, 8],
    }

    def prompt_tokens(self, *, messages, tools, system_prompt):
        del tools, system_prompt
        return list(self.prompts[messages[-1]["content"]])

    def stop_sequences(self):
        return ["<stop>"]

    def parse_completion(self, tokens):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call-{tokens[0]}",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"true"}',
                    },
                }
            ],
        }


class _FakeSampler:
    def __init__(self):
        self.outputs = iter(([10], [11], [99], [30], [31], [50]))
        self.calls: list[dict[str, Any]] = []

    async def sample_with_prompt_tokens(self, prompt, **kwargs):
        self.calls.append(dict(kwargs))
        output = list(next(self.outputs))
        token = output[0]
        return [
            SimpleNamespace(
                prompt_len=len(prompt),
                full_tokens=[*prompt, *output],
                sampling_logprobs=[-0.1 - token / 1000],
                inference_logprobs=[-0.2 - token / 1000],
                routing_matrices=[f"route-{token}"],
                finish_reason="stop",
                text=f"output-{token}",
                logprobs_echoed=False,
            )
        ]


class _FakeTokenizer:
    @staticmethod
    def decode(tokens, **kwargs):
        del kwargs
        return " ".join(map(str, tokens))


def _body(content: str | list[dict], *, tools=_TOOLS):
    messages = (
        content if isinstance(content, list) else [{"role": "user", "content": content}]
    )
    return {
        "messages": messages,
        "tools": tools,
        "max_tokens": 4,
    }


class _Request:
    def __init__(self, key: str, body: dict[str, Any]) -> None:
        self.headers = {"Authorization": f"Bearer {key}"}
        self._body = body

    async def json(self) -> dict[str, Any]:
        return self._body


def _assistant_tool_call(token: int) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "index": 0,
                "id": f"call-{token}",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"true"}',
                },
            }
        ],
    }


def _assistant_tool_call_without_stream_index(token: int) -> dict[str, Any]:
    message = _assistant_tool_call(token)
    del message["tool_calls"][0]["index"]
    return message


def _server(
    monkeypatch,
    *,
    capture_request_traces: bool = True,
) -> OpenCodePolicyServer:
    monkeypatch.setattr(
        openai_policy,
        "build_turn_renderer",
        lambda tokenizer, renderer_name: _FakeRenderer(),
    )
    return OpenCodePolicyServer(
        sampler=_FakeSampler(),
        tokenizer=_FakeTokenizer(),
        sample_kwargs={"include_routing_matrix": True},
        renderer_name="fake",
        max_seq_len=128,
        max_sample_tokens=4,
        capture_request_traces=capture_request_traces,
    )


async def _sample_and_record(
    server: OpenCodePolicyServer,
    session: OpenCodePolicySession,
    body: dict[str, Any],
) -> None:
    message, turn, prepared = await server._sample_turn(session, body)
    server._record_turn(
        session,
        prepared=prepared,
        message=message,
        turn=turn,
    )


def test_history_matching_ignores_stream_only_tool_call_index(monkeypatch):
    server = _server(monkeypatch)
    session = OpenCodePolicySession(run_id="run-1", max_context_tokens=128)
    task = [{"role": "user", "content": "task"}]
    appended = [
        *task,
        _assistant_tool_call_without_stream_index(10),
        {"role": "tool", "tool_call_id": "call-10", "content": "obs1"},
    ]

    async def record():
        await _sample_and_record(server, session, _body(task))
        await _sample_and_record(server, session, _body(appended))

    asyncio.run(record())

    assert [event["kind"] for event in session.match_events] == ["new", "append"]
    [segment] = session.drain()
    assert segment.loss_mask == [1, 0, 1]


def test_history_wipe_splits_token_chains_but_keeps_one_rollout(monkeypatch):
    server = _server(monkeypatch)
    session = OpenCodePolicySession(run_id="run-1", max_context_tokens=128)
    task = [{"role": "user", "content": "task"}]
    appended_task = [
        *task,
        _assistant_tool_call(10),
        {"role": "tool", "tool_call_id": "call-10", "content": "obs1"},
    ]
    compacted = [{"role": "user", "content": "summary1"}]
    appended_compacted = [
        *compacted,
        _assistant_tool_call(30),
        {"role": "tool", "tool_call_id": "call-30", "content": "obs2"},
    ]

    async def record():
        await _sample_and_record(server, session, _body(task))
        await _sample_and_record(server, session, _body(appended_task))
        # Title/summary-style auxiliary calls are sampled for the agent but do
        # not enter the task trajectory or advance its matching fingerprint.
        await _sample_and_record(server, session, _body("title", tools=[]))
        await _sample_and_record(server, session, _body(compacted))
        await _sample_and_record(server, session, _body(appended_compacted))
        await _sample_and_record(server, session, _body("summary2"))

    asyncio.run(record())
    segments = session.drain()

    assert [event["kind"] for event in session.match_events] == [
        "new",
        "append",
        "wipe",
        "append",
        "wipe",
    ]
    assert session.history_wipes == 2
    assert session.auxiliary_turns == 1
    assert len(session.request_traces) == 6
    assert sum(trace["trainable"] for trace in session.request_traces) == 5
    assert session.request_traces[0]["messages"] == task
    assert session.request_traces[0]["tools"] == _TOOLS
    assert session.request_traces[0]["prompt_ids"] == [1, 2]
    assert session.request_traces[0]["completion_ids"] == [10]
    assert session.request_traces[0]["parser_fallback"] is False
    assert session.request_traces[2]["turn_kind"] == "auxiliary"
    assert [segment.metadata["segment_kind"] for segment in segments] == [
        "history_wipe",
        "history_wipe",
        "final",
    ]

    # Generated task tokens occur exactly once.  Tool/context suffixes remain
    # present for conditioning but carry zero loss, logprob, raw logprob, and R3.
    assert [
        [token for token, mask in zip(segment.response_ids, segment.loss_mask) if mask]
        for segment in segments
    ] == [[10, 11], [30, 31], [50]]
    assert 99 not in [token for segment in segments for token in segment.response_ids]
    assert segments[0].response_ids == [10, 20, 11]
    assert segments[0].loss_mask == [1, 0, 1]
    assert segments[0].routing_matrices == ["route-10", "", "route-11"]
    assert segments[0].rollout_raw_log_probs[1] == 0.0

    samples = [token_segment_to_sample(segment, reward=1.0) for segment in segments]
    run = RolloutRun(
        segments=samples,
        run_id="run-1",
        metadata={"history_wipes": 2},
    )
    other = RolloutRun(
        segments=[
            RolloutSample(
                tokens=[70, 71],
                logprobs=[0.0, -0.5],
                loss_mask=[0, 1],
                reward=0.0,
            )
        ],
        run_id="run-2",
    )
    group = rollout_to_prompt_group(
        Rollout(runs=[run, other]),
        advantage_fn=lambda rewards: [1.0, -1.0],
        router_replay_completion_only=True,
    )
    assert group is not None
    # Advantage is computed once per rollout and broadcast to all three split
    # segments.  The second rollout remains one group member, not a fourth
    # completion of the first.
    assert group.rewards == [1.0, 0.0]
    assert group.run_metadata == [
        {"history_wipes": 2, "trainable_tokens": 5},
        {"trainable_tokens": 1},
    ]
    assert group.advantages == [1.0, 1.0, 1.0, -1.0]
    assert len(group.data) == 4
    assert group.data[0].model_input.routing_matrices == [
        "",
        "route-10",
        "",
        "route-11",
    ]
    assert group.raw_inf_logprobs[0] == pytest.approx([0.0, -0.21, 0.0, -0.211])


def test_history_retry_branches_from_shared_training_parent(monkeypatch):
    server = _server(monkeypatch)
    session = OpenCodePolicySession(run_id="run-1", max_context_tokens=128)
    task = [{"role": "user", "content": "task"}]
    continued = [
        *task,
        _assistant_tool_call(10),
        {"role": "tool", "tool_call_id": "call-10", "content": "obs1"},
    ]

    async def record():
        await _sample_and_record(server, session, _body(task))
        await _sample_and_record(server, session, _body(continued))
        await _sample_and_record(server, session, _body(continued))

    asyncio.run(record())
    segments = session.drain()

    assert [event["kind"] for event in session.match_events] == [
        "new",
        "append",
        "wipe",
    ]
    assert session.training_session.tree.leaf_ids == [1, 2]
    assert [segment.loss_mask for segment in segments] == [
        [1, 0, 1],
        [0, 0, 1],
    ]
    assert [
        [token for token, mask in zip(segment.response_ids, segment.loss_mask) if mask]
        for segment in segments
    ] == [[10, 11], [99]]


def test_policy_uses_attempt_affinity_instead_of_logical_run_id(monkeypatch):
    server = _server(monkeypatch)
    session = OpenCodePolicySession(run_id="logical-run", max_context_tokens=128)

    async def sample():
        await _sample_and_record(server, session, _body("task"))
        await _sample_and_record(server, session, _body("summary1"))

    asyncio.run(sample())

    users = [call["user"] for call in server._sampler.calls]
    assert users == [session.training_session.affinity_key] * 2
    assert users[0] != session.run_id


def test_policy_server_start_failure_resets_lifecycle_state(monkeypatch):
    server = _server(monkeypatch)
    fail_start = True
    runners = []

    class FakeRunner:
        def __init__(self, _app, *, access_log):
            assert access_log is None
            self.cleaned = False
            runners.append(self)

        async def setup(self):
            return None

        async def cleanup(self):
            self.cleaned = True

    class FakeSocket:
        @staticmethod
        def getsockname():
            return ("0.0.0.0", 9123)

    class FakeSite:
        def __init__(self, _runner, _host, _port):
            self._server = SimpleNamespace(sockets=[FakeSocket()])

        async def start(self):
            nonlocal fail_start
            if fail_start:
                fail_start = False
                raise RuntimeError("bind failed")

    monkeypatch.setattr(openai_policy.web, "AppRunner", FakeRunner)
    monkeypatch.setattr(openai_policy.web, "TCPSite", FakeSite)

    with pytest.raises(RuntimeError, match="bind failed"):
        asyncio.run(server.start())

    assert runners[0].cleaned
    assert server._runner is None
    assert server._site is None
    assert server.port == 0

    asyncio.run(server.start())
    assert server.port == 9123
    asyncio.run(server.close())


def test_stream_chunks_are_well_formed_openai_sse():
    payload = {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "policy",
        "choices": [
            {
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
        },
    }
    chunks = OpenCodePolicyServer._stream_chunks(payload)
    assert chunks[-1] == "data: [DONE]\n\n"
    assert '"object": "chat.completion.chunk"' in chunks[0]
    assert '"finish_reason": "stop"' in chunks[1]


def test_request_traces_are_disabled_by_default(monkeypatch):
    server = _server(monkeypatch, capture_request_traces=False)
    session = OpenCodePolicySession(run_id="run-1", max_context_tokens=128)

    asyncio.run(_sample_and_record(server, session, _body("task")))

    assert session.request_traces == []
    assert len(session.match_events) == 1


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"prompt_len": 1}, "prompt tokens do not match"),
        ({"full_tokens": [1, 2]}, "empty OpenCode completion"),
        ({"sampling_logprobs": None}, "missing sampling_logprobs"),
        ({"sampling_logprobs": []}, "sampling_logprobs are misaligned"),
        ({"inference_logprobs": []}, "inference_logprobs are misaligned"),
        ({"routing_matrices": None}, "missing requested routing matrices"),
        ({"routing_matrices": []}, "routing matrices are misaligned"),
    ],
)
def test_completion_alignment_failures_are_trace_integrity_errors(
    monkeypatch,
    updates,
    message,
):
    server = _server(monkeypatch)
    completion = SimpleNamespace(
        prompt_len=2,
        full_tokens=[1, 2, 10],
        sampling_logprobs=[-0.1],
        inference_logprobs=[-0.2],
        routing_matrices=["route-10"],
        finish_reason="stop",
        text="output-10",
        logprobs_echoed=False,
    )
    for name, value in updates.items():
        setattr(completion, name, value)

    with pytest.raises(openai_policy.TraceIntegrityError, match=message):
        server._turn_from_completion(completion, prompt_tokens=[1, 2])


def test_unparseable_completion_falls_back_to_raw_text(monkeypatch):
    server = _server(monkeypatch)
    completion = SimpleNamespace(
        prompt_len=2,
        full_tokens=[1, 2, 10],
        sampling_logprobs=[-0.1],
        inference_logprobs=[-0.2],
        routing_matrices=["route-10"],
        finish_reason="stop",
        text="output-10",
        logprobs_echoed=False,
    )

    def fail_parse(_tokens):
        raise ValueError("malformed tool call")

    monkeypatch.setattr(server._renderer, "parse_completion", fail_parse)

    message, turn = server._turn_from_completion(
        completion,
        prompt_tokens=[1, 2],
    )

    assert message == {"role": "assistant", "content": "10"}
    assert turn.output_ids == [10]
    assert turn.output_log_probs == [-0.1]
    assert turn.metadata["parser_fallback"] is True


def test_policy_request_preserves_trace_integrity_failure(monkeypatch):
    server = _server(monkeypatch)

    class MisalignedSampler:
        async def sample_with_prompt_tokens(self, prompt, **_kwargs):
            return [
                SimpleNamespace(
                    prompt_len=len(prompt),
                    full_tokens=[*prompt, 10],
                    sampling_logprobs=[],
                    inference_logprobs=[-0.2],
                    routing_matrices=["route-10"],
                    finish_reason="stop",
                    text="output-10",
                    logprobs_echoed=False,
                )
            ]

    server._sampler = MisalignedSampler()
    key = server.register_session("run-1")
    session = server._sessions[key]

    async def request():
        with pytest.raises(openai_policy.web.HTTPServiceUnavailable):
            await server._chat_completions(_Request(key, _body("task")))

    asyncio.run(request())

    assert session.sampling_failures == 1
    assert session.trace_integrity_failures == 1
    assert "sampling_logprobs are misaligned" in session.trace_integrity_error


def test_transient_policy_request_failure_is_not_trace_integrity(monkeypatch):
    server = _server(monkeypatch)

    class UnavailableSampler:
        async def sample_with_prompt_tokens(self, _prompt, **_kwargs):
            raise TimeoutError("sampler unavailable")

    server._sampler = UnavailableSampler()
    key = server.register_session("run-1")
    session = server._sessions[key]

    async def request():
        with pytest.raises(openai_policy.web.HTTPServiceUnavailable):
            await server._chat_completions(_Request(key, _body("task")))

    asyncio.run(request())

    assert session.sampling_failures == 1
    assert session.trace_integrity_failures == 0
    assert session.trace_integrity_error is None


def test_trace_recording_failure_is_preserved(monkeypatch):
    server = _server(monkeypatch)
    key = server.register_session("run-1")
    session = server._sessions[key]

    def fail_record(*_args, **_kwargs):
        raise ValueError("history changed")

    monkeypatch.setattr(server, "_record_turn", fail_record)

    async def request():
        with pytest.raises(openai_policy.web.HTTPServiceUnavailable):
            await server._chat_completions(_Request(key, _body("task")))

    asyncio.run(request())

    assert session.trace_integrity_failures == 1
    assert "trace recording failed" in session.trace_integrity_error


def test_stream_disconnect_does_not_record_undelivered_turn(monkeypatch):
    server = _server(monkeypatch)
    key = server.register_session("run-1")
    session = server._sessions[key]

    class Request:
        headers = {"Authorization": f"Bearer {key}"}

        @staticmethod
        async def json():
            return {**_body("task"), "stream": True}

    async def fail_prepare(response, request):
        del response, request
        raise ConnectionResetError("client disconnected")

    monkeypatch.setattr(openai_policy.web.StreamResponse, "prepare", fail_prepare)

    async def request():
        with pytest.raises(ConnectionResetError, match="client disconnected"):
            await server._chat_completions(Request())

    asyncio.run(request())

    assert session.chain.turns == []
    assert session.selected_leaves == []
    assert session.match_events == []


def test_stream_disconnect_after_content_records_delivered_turn(monkeypatch):
    server = _server(monkeypatch)
    key = server.register_session("run-1")
    session = server._sessions[key]

    class Request:
        headers = {"Authorization": f"Bearer {key}"}

        @staticmethod
        async def json():
            return {**_body("task"), "stream": True}

    async def prepare(response, request):
        del response, request

    writes = 0

    async def disconnect_after_content(response, data):
        del response, data
        nonlocal writes
        writes += 1
        if writes == 2:
            raise ConnectionResetError("client disconnected")

    monkeypatch.setattr(openai_policy.web.StreamResponse, "prepare", prepare)
    monkeypatch.setattr(
        openai_policy.web.StreamResponse,
        "write",
        disconnect_after_content,
    )

    async def request():
        with pytest.raises(ConnectionResetError, match="client disconnected"):
            await server._chat_completions(Request())

    asyncio.run(request())

    assert writes == 2
    assert len(session.chain.turns) == 1
    assert session.selected_leaves == []
    assert len(session.match_events) == 1


def test_session_retirement_rejects_late_request_before_sampling(monkeypatch):
    server = _server(monkeypatch)
    key = server.register_session("run-1")
    session = server._sessions[key]

    class Request:
        headers = {"Authorization": f"Bearer {key}"}

        @staticmethod
        async def json():
            return {**_body("task"), "stream": True}

    async def retire_with_waiting_request():
        await session.lock.acquire()
        pop_task = asyncio.create_task(server.pop_session(key))
        await asyncio.sleep(0)
        request_task = asyncio.create_task(server._chat_completions(Request()))
        await asyncio.sleep(0)
        session.lock.release()

        popped = await pop_task
        with pytest.raises(openai_policy.web.HTTPUnauthorized) as exc_info:
            await request_task
        assert exc_info.value.text == "trial session is closed"
        return popped

    assert asyncio.run(retire_with_waiting_request()) is session
    assert session.closed
    assert session.chain.turns == []
    assert session.match_events == []


def test_recorded_reasoning_aligns_from_the_latest_turn():
    first = {"role": "assistant", "content": "first answer"}
    latest = {"role": "assistant", "content": "latest answer"}
    turns = [
        TurnRecord(
            prompt_ids=[1],
            output_ids=[2],
            finish_reason="stop",
            metadata={
                "reasoning_content": "first reasoning",
                "assistant_identity": openai_policy._assistant_identity(first),
            },
        ),
        TurnRecord(
            prompt_ids=[3],
            output_ids=[4],
            finish_reason="stop",
            metadata={
                "reasoning_content": "latest reasoning",
                "assistant_identity": openai_policy._assistant_identity(latest),
            },
        ),
    ]
    messages = [
        {"role": "user", "content": "compacted history"},
        latest,
        {"role": "tool", "content": "observation"},
    ]

    restored = openai_policy._with_recorded_reasoning(messages, turns)

    assert restored[1]["reasoning_content"] == "latest reasoning"
    assert "reasoning_content" not in messages[1]


def test_history_summary_does_not_steal_retained_reasoning():
    retained = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{ "command": "true" }',
                },
            }
        ],
    }
    summary = {
        "role": "assistant",
        "content": "Summary of the work so far",
    }
    turn = TurnRecord(
        prompt_ids=[1],
        output_ids=[2],
        finish_reason="stop",
        metadata={
            "reasoning_content": "retained task reasoning",
            "assistant_identity": openai_policy._assistant_identity(
                {
                    **retained,
                    "tool_calls": [
                        {
                            **retained["tool_calls"][0],
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"true"}',
                            },
                        }
                    ],
                }
            ),
        },
    )
    messages = [
        {"role": "user", "content": "retained task"},
        retained,
        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
        summary,
        {"role": "user", "content": "Continue"},
    ]

    restored = openai_policy._with_recorded_reasoning(messages, [turn])

    assert restored[1]["reasoning_content"] == "retained task reasoning"
    assert "reasoning_content" not in restored[3]


def test_openai_renderer_preserves_reasoning_as_thinking_content():
    class CapturingRenderer:
        def __init__(self):
            self.messages = []

        @staticmethod
        def get_stop_sequences():
            return []

        def build_generation_prompt(self, messages):
            self.messages = messages
            return SimpleNamespace(to_ints=lambda: [1, 2, 3])

    renderer = CapturingRenderer()
    adapter = CookbookTurnRenderer(renderer)

    assert adapter.prompt_tokens(
        messages=[
            {
                "role": "assistant",
                "content": "visible answer",
                "reasoning_content": "private reasoning",
                "recipient": "self",
                "end_turn": False,
            }
        ],
        tools=[],
        system_prompt="",
    ) == [1, 2, 3]
    assert renderer.messages[0]["content"] == [
        {"type": "thinking", "thinking": "private reasoning"},
        {"type": "text", "text": "visible answer"},
    ]
    assert ORIGINAL_REASONING_CONTENT not in renderer.messages[0]
    assert "recipient" not in renderer.messages[0]
    assert "end_turn" not in renderer.messages[0]


def test_opencode_config_uses_absolute_xdg_path_when_shell_env_is_absent():
    pytest.importorskip("harbor")
    # lazy: ConfigurableOpenCode subclasses Harbor's optional OpenCode agent.
    from training.examples.rl.harbor.opencode import (
        ConfigurableOpenCode,
    )

    agent = SimpleNamespace(_policy_config=lambda: {"provider": {}})

    command = ConfigurableOpenCode._write_config_command(agent)

    assert '"/logs/agent/opencode/xdg-config/opencode/opencode.json"' in command
    assert "$XDG_CONFIG_HOME" not in command
    assert '"$HOME/.config/opencode' not in command


@pytest.mark.parametrize(
    ("installed_version", "raises"),
    [("1.18.8\n", False), ("1.18.10\n", True)],
)
def test_baked_opencode_version_must_match_the_requested_pin(
    tmp_path,
    installed_version,
    raises,
):
    pytest.importorskip("harbor")
    from training.examples.rl.harbor.opencode import (
        ConfigurableOpenCode,
    )

    agent = ConfigurableOpenCode(
        logs_dir=tmp_path,
        policy_base_url="http://host/v1",
        policy_api_key="test-key",
        context_limit=128,
        output_limit=32,
        version="1.18.8",
    )

    class Environment:
        async def exec(self, *, command):
            assert "opencode --version" in command
            return SimpleNamespace(return_code=0, stdout=installed_version)

    install = agent.install(Environment())
    if raises:
        with pytest.raises(RuntimeError, match="baked OpenCode version mismatch"):
            asyncio.run(install)
    else:
        asyncio.run(install)
