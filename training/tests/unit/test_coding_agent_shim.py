"""Unit tests for the black-box coding-agent shim + trajectory merge.

Lightweight: imports only ``shim`` + ``trajectory`` (no async_rl_loop / SDK /
renderer), drives the per-turn pipeline with fakes, and asserts the token-level
trajectory is stitched correctly (assistant tokens trained, scaffolding/tool
masked) and that TITO logprob slicing is correct.
"""

from __future__ import annotations

import asyncio
import json
import types

from training.examples.rl.coding_agent import shim
from training.examples.rl.coding_agent.trajectory import (
    TurnRecord,
    make_turn_segment,
    merge_turn_segments,
    merge_turns,
)


# --------------------------------------------------------------------------- #
# trajectory.merge_turns                                                       #
# --------------------------------------------------------------------------- #
def test_merge_turns_append_only_mask_and_logprobs():
    turns = [
        TurnRecord(prompt_ids=[1, 2, 3], output_ids=[10, 11], finish_reason="stop", output_log_probs=[-0.1, -0.2]),
        # turn2 prompt = base prompt + turn1 output + a 2-token gap (tool result)
        TurnRecord(prompt_ids=[1, 2, 3, 10, 11, 20, 21], output_ids=[30], finish_reason="stop", output_log_probs=[-0.3]),
    ]
    seg = merge_turns(turns)
    assert seg is not None
    assert seg.prompt_ids == [1, 2, 3]
    assert seg.response_ids == [10, 11, 20, 21, 30]
    assert seg.loss_mask == [1, 1, 0, 0, 1]            # gap masked, assistant trained
    assert seg.rollout_log_probs == [-0.1, -0.2, 0.0, 0.0, -0.3]
    # all three parallel lists align once packed into the full sequence
    full = seg.prompt_ids + seg.response_ids
    assert len(full) == len(seg.prompt_ids) + len(seg.loss_mask) == len(seg.prompt_ids) + len(seg.rollout_log_probs)


def test_merge_turns_masks_drifted_output():
    # turn2 prompt diverges *inside* turn1's output (10 ok, 11 -> 99): the
    # partially-matched turn-1 output must be masked, not trained.
    turns = [
        TurnRecord(prompt_ids=[1, 2], output_ids=[10, 11], finish_reason="stop", output_log_probs=[-0.1, -0.2]),
        TurnRecord(prompt_ids=[1, 2, 10, 99], output_ids=[30], finish_reason="stop", output_log_probs=[-0.3]),
    ]
    seg = merge_turns(turns)
    assert seg is not None
    # turn1 output dropped/masked; only turn2 output is trained
    assert seg.loss_mask[-1] == 1
    assert sum(seg.loss_mask) == 1
    assert len(seg.loss_mask) == len(seg.response_ids) == len(seg.rollout_log_probs)


def test_pop_via_make_turn_segment_single_chain():
    turns = [TurnRecord(prompt_ids=[1], output_ids=[5, 6], finish_reason="stop", output_log_probs=[-0.1, -0.2])]
    ts = make_turn_segment(turns, kind="final")
    assert ts.metadata["segment_kind"] == "final"


def test_merge_turn_segments_drops_oversized_segment():
    turns = [TurnRecord(prompt_ids=[1, 2], output_ids=[3, 4], finish_reason="stop")]
    segments = merge_turn_segments([make_turn_segment(turns, kind="final")], max_context_tokens=3)
    assert segments == []


def test_turn_stats_suffix_growth():
    # turn2 prompt = turn1 (prompt+output) + 1 new context token -> suffix_len 1
    turns = [
        TurnRecord(prompt_ids=[1, 2, 3], output_ids=[10, 11], finish_reason="stop"),
        TurnRecord(prompt_ids=[1, 2, 3, 10, 11, 20], output_ids=[30], finish_reason="stop"),
    ]
    stats = shim._turn_stats(turns)
    assert [s["suffix_len"] for s in stats] == [3, 1]
    assert [s["prompt_len"] for s in stats] == [3, 6]
    assert [s["out_len"] for s in stats] == [2, 1]


def test_pop_session_split_carries_turn_metadata():
    turns = [
        TurnRecord(prompt_ids=[1, 2, 3], output_ids=[10, 11], finish_reason="stop", output_log_probs=[-0.1, -0.2]),
        TurnRecord(prompt_ids=[1, 2, 3, 10, 11, 20], output_ids=[30], finish_reason="stop", output_log_probs=[-0.3]),
    ]
    s = shim.Session()
    s.main.turns = turns
    store = {"sid-1": s}
    segs = shim.pop_session_split(store, "sid-1")
    assert "sid-1" not in store  # popped
    assert len(segs) == 1
    md = segs[0].metadata
    assert md["num_turns"] == 2
    assert len(md["turns"]) == 2
    assert md["segment_kind"] == "final"


# --------------------------------------------------------------------------- #
# shim: Anthropic <-> chat translation                                         #
# --------------------------------------------------------------------------- #
def test_translate_anthropic_tool_use_and_result():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "fix it"}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "looking"},
            {"type": "tool_use", "name": "Read", "input": {"path": "a.py"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "src"}]},
    ]
    out = shim._translate_anthropic(msgs, "SYS")
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[2]["tool_calls"][0]["function"]["name"] == "Read"
    assert out[3]["content"] == "src"


def test_translate_anthropic_hoists_late_system_messages():
    msgs = [
        {"role": "user", "content": "fix it"},
        {"role": "system", "content": "late instruction"},
        {"role": "user", "content": "continue"},
    ]

    out = shim._translate_anthropic(msgs, "top instruction")

    assert [m["role"] for m in out] == ["system", "user", "user"]
    assert out[0]["content"] == "top instruction\n\nlate instruction"


def test_strip_cache_control_drops_ephemeral_markers():
    stripped = shim._strip_cache_control({"a": 1, "cache_control": {"type": "ephemeral"}})
    assert stripped == {"a": 1}


def test_anthropic_tools_to_chat_tools():
    ts = shim._anthropic_tools_to_chat_tools([{"name": "Bash", "description": "run", "input_schema": {"type": "object"}}])
    assert ts[0]["function"]["name"] == "Bash"
    assert ts[0]["type"] == "function"


# --------------------------------------------------------------------------- #
# shim: per-turn pipeline (routing + new/append/wipe + sub-agent segments)     #
# --------------------------------------------------------------------------- #
class _FakeTok:
    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=True):
        # deterministic, additive in message count (so later prompts extend
        # earlier ones token-wise, matching the append contract)
        return list(range(len(messages) * 4))


class _QueueRenderer:
    """Fake renderer: returns the next queued parsed assistant message."""

    def __init__(self, parsed_messages):
        self._queue = list(parsed_messages)

    def parse_response(self, output_ids):
        message = self._queue.pop(0) if self._queue else {"content": "", "tool_calls": []}
        return message, "stop"

    def get_stop_sequences(self):
        return []


def _fake_app(parsed_messages, *, strategy="message_hash"):
    """Minimal aiohttp-app-shaped dict for driving ``_run_turn`` with fakes."""
    comp = _FakeCompletion(
        full_tokens=[0, 1, 2, 10, 11], prompt_len=3, sampling_logprobs=[-0.1, -0.2], logprobs_echoed=False,
    )
    return {
        "tokenizer": _FakeTok(),
        "renderer": _QueueRenderer(parsed_messages),
        "sampler": _FakeSampler(comp),
        "sample_kwargs": {"max_tokens": 64},
        "stop": [],
        "fingerprinter": shim.make_fingerprinter(strategy),
    }


def _plain(text="ok"):
    return {"content": text, "tool_calls": []}


def test_run_turn_new_then_append_keeps_one_chain():
    """A second request that extends the first's message prefix appends to the
    same chain: one growing main chain, no frozen segments."""
    app = _fake_app([_plain(), _plain()])
    s = shim.Session()
    m_a = {"role": "user", "content": "a"}
    m_b = {"role": "assistant", "content": "b"}
    _run(shim._run_turn(s, {"system": "S", "messages": [m_a]}, app, "sid"))
    _run(shim._run_turn(s, {"system": "S", "messages": [m_a, m_b]}, app, "sid"))
    assert len(s.main.turns) == 2
    assert s.segments == []


def test_run_turn_wipe_snapshots_segment_instead_of_discarding():
    """A prefix break (compaction) freezes the current chain into a ``wipe``
    segment rather than discarding the dropped turns."""
    app = _fake_app([_plain(), _plain()])
    s = shim.Session()
    _run(shim._run_turn(s, {"system": "S", "messages": [{"role": "user", "content": "a"}]}, app, "sid"))
    assert len(s.main.turns) == 1
    # Compaction: messages no longer continue the stored prefix -> wipe.
    _run(shim._run_turn(s, {"system": "S", "messages": [{"role": "user", "content": "compacted"}]}, app, "sid"))
    assert len(s.segments) == 1
    assert s.segments[0].metadata["segment_kind"] == "wipe"


def test_run_turn_logs_non_append_decisions(tmp_path, monkeypatch):
    mismatch_log = tmp_path / "prefix-mismatch.jsonl"
    monkeypatch.setenv("SWE_PREFIX_MISMATCH_LOG", str(mismatch_log))
    app = _fake_app([_plain(), _plain()])
    s = shim.Session()

    _run(shim._run_turn(s, {"system": "S", "messages": [{"role": "user", "content": "a"}]}, app, "sid"))
    _run(shim._run_turn(s, {"system": "S", "messages": [{"role": "user", "content": "compacted"}]}, app, "sid"))

    rows = [json.loads(line) for line in mismatch_log.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["new", "wipe"]
    assert {row["strategy"] for row in rows} == {"message_hash"}
    assert all(row["session_id"] == "sid" for row in rows)


def test_run_turn_subagent_excursion_yields_subagent_and_final_segments():
    """A Task/Agent dispatch opens a sub-chain; when main shows the dispatch's
    tool_result the sub-chain freezes into a ``subagent`` segment.  Draining
    yields both the subagent and the final main segment (one run -> 2 segs)."""
    task_dispatch = {"content": [], "tool_calls": [_toolcall("Task", "{}")]}
    app = _fake_app([task_dispatch, _plain("subtask"), _plain("done")])
    s = shim.Session()

    # main turn 1 dispatches a sub-agent (random dispatch id captured below).
    _run(shim._run_turn(s, {"system": "S", "messages": [{"role": "user", "content": "go"}]}, app, "sid"))
    dispatch_id = s.pending_dispatch_id
    assert dispatch_id and s.active_sub is not None

    # the sub-agent's own call has a different prefix -> routes to the sub-chain.
    _run(shim._run_turn(s, {"system": "SUB", "messages": [{"role": "user", "content": "subtask"}]}, app, "sid"))
    assert len(s.active_sub.turns) == 1

    # main resumes with the dispatch's tool_result -> sub frozen as 'subagent'.
    b3 = {"system": "S", "messages": [
        {"role": "user", "content": "go"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": dispatch_id, "content": "ok"}]},
    ]}
    _run(shim._run_turn(s, b3, app, "sid"))
    assert s.active_sub is None
    assert [seg.metadata["segment_kind"] for seg in s.segments] == ["subagent"]

    segs = shim.pop_session_split({"sid": s}, "sid")
    assert [seg.metadata.get("segment_kind") for seg in segs] == ["subagent", "final"]
    # Disjoint segments: the subagent's tokens are NOT concatenated into main.
    assert all(seg.response_ids for seg in segs)


# --------------------------------------------------------------------------- #
# shim: parsed-output -> Anthropic blocks                                      #
# --------------------------------------------------------------------------- #
def _toolcall(name, arguments):
    fn = types.SimpleNamespace(name=name, arguments=arguments)
    return types.SimpleNamespace(function=fn)


def test_anthropic_blocks_text_and_tool():
    msg = {"content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "done"}],
           "tool_calls": [_toolcall("Edit", '{"path":"a.py"}')]}
    blocks, stop, dispatch_id = shim._anthropic_blocks(msg, finish="stop")
    types_ = [b["type"] for b in blocks]
    assert types_ == ["thinking", "text", "tool_use"]
    assert blocks[2]["name"] == "Edit"
    assert blocks[2]["input"] == {"path": "a.py"}
    assert stop == "tool_use"
    assert dispatch_id == ""  # Edit is not a sub-agent tool


def test_anthropic_blocks_text_only_end_turn():
    blocks, stop, dispatch_id = shim._anthropic_blocks({"content": "hi", "tool_calls": []}, finish="stop")
    assert blocks == [{"type": "text", "text": "hi"}]
    assert stop == "end_turn"
    assert dispatch_id == ""


def test_anthropic_blocks_subagent_tool_sets_dispatch_id():
    """A ``Task``/``Agent`` tool_use returns its id as dispatch_id so the
    handler can open a sub-chain."""
    msg = {"content": [], "tool_calls": [_toolcall("Task", '{"prompt":"sub"}')]}
    blocks, stop, dispatch_id = shim._anthropic_blocks(msg, finish="stop")
    assert stop == "tool_use"
    tool_block = next(b for b in blocks if b["type"] == "tool_use")
    assert tool_block["name"] == "Task"
    assert dispatch_id == tool_block["id"]


# --------------------------------------------------------------------------- #
# shim: TITO logprob slicing in _generate                                      #
# --------------------------------------------------------------------------- #
class _FakeCompletion:
    def __init__(
        self,
        full_tokens,
        prompt_len,
        sampling_logprobs,
        logprobs_echoed,
        finish_reason="stop",
        inference_logprobs=None,
    ):
        self.full_tokens = full_tokens
        self.prompt_len = prompt_len
        self.sampling_logprobs = sampling_logprobs
        self.inference_logprobs = inference_logprobs
        self.logprobs_echoed = logprobs_echoed
        self.finish_reason = finish_reason
        self.text = ""


class _FakeSampler:
    def __init__(self, completion):
        self._c = completion
        self.calls = []

    async def sample_with_prompt_tokens(self, prompt_token_ids, n=1, max_tokens=1024, stop=None, **kwargs):
        self.calls.append({"prompt": list(prompt_token_ids), "max_tokens": max_tokens, "stop": stop, "kwargs": kwargs})
        return [self._c]


def _run(coro):
    return asyncio.run(coro)


def test_generate_completion_only_logprobs():
    prompt = [1, 2, 3]
    comp = _FakeCompletion(full_tokens=prompt + [10, 11], prompt_len=3, sampling_logprobs=[-0.5, -0.6], logprobs_echoed=False)
    sampler = _FakeSampler(comp)
    app = {"sampler": sampler, "sample_kwargs": {"max_tokens": 100, "logprobs": True}, "stop": []}
    s = shim.Session()
    s.user_key = "cagent-1-2-abcd"  # sticky-routing key
    turn = _run(shim._generate(prompt, s, {}, app))
    assert turn.output_ids == [10, 11]
    assert turn.output_log_probs == [-0.5, -0.6]
    # the trajectory's session id is forwarded as the completions `user` field
    # so the gateway sticky-routes all turns to one replica (KV reuse).
    assert sampler.calls[-1]["kwargs"].get("user") == "cagent-1-2-abcd"


def test_generate_echoed_logprobs_sliced():
    prompt = [1, 2, 3]
    # echoed: logprobs cover prompt+output (prompt_len + 2)
    comp = _FakeCompletion(full_tokens=prompt + [10, 11], prompt_len=3,
                           sampling_logprobs=[-9, -9, -9, -0.5, -0.6], logprobs_echoed=True)
    app = {"sampler": _FakeSampler(comp), "sample_kwargs": {"max_tokens": 100}, "stop": []}
    turn = _run(shim._generate(prompt, shim.Session(), {}, app))
    assert turn.output_ids == [10, 11]
    assert turn.output_log_probs == [-0.5, -0.6]


def test_generate_mismatch_zeroes_logprobs():
    prompt = [1, 2, 3]
    comp = _FakeCompletion(full_tokens=prompt + [10, 11, 12], prompt_len=3, sampling_logprobs=[-0.5], logprobs_echoed=False)
    app = {"sampler": _FakeSampler(comp), "sample_kwargs": {"max_tokens": 100}, "stop": []}
    turn = _run(shim._generate(prompt, shim.Session(), {}, app))
    assert turn.output_ids == [10, 11, 12]
    assert turn.output_log_probs == [0.0, 0.0, 0.0]


def test_generate_context_budget_length_stop():
    prompt = list(range(50))
    comp = _FakeCompletion(full_tokens=prompt + [99], prompt_len=50, sampling_logprobs=[-0.1], logprobs_echoed=False)
    app = {"sampler": _FakeSampler(comp), "sample_kwargs": {"max_tokens": 100}, "stop": []}
    s = shim.Session()
    s.max_context_tokens = 40  # prompt already over budget
    turn = _run(shim._generate(prompt, s, {}, app))
    assert turn.output_ids == []
    assert turn.finish_reason == "length"


class _MetricsSampler(_FakeSampler):
    """Fake sampler that also exposes drain_metrics (KV-reuse signal)."""

    def __init__(self, completion, metrics):
        super().__init__(completion)
        self._metrics = metrics
        self.drained = False

    def drain_metrics(self):
        self.drained = True
        return self._metrics


def test_generate_kv_metrics_drained_when_enabled(monkeypatch=None):
    import os
    prompt = [1, 2, 3]
    comp = _FakeCompletion(full_tokens=prompt + [10], prompt_len=3, sampling_logprobs=[-0.5], logprobs_echoed=False)
    metric = types.SimpleNamespace(prompt_tokens=3, cached_prompt_tokens=2)
    sampler = _MetricsSampler(comp, [metric])
    app = {"sampler": sampler, "sample_kwargs": {"max_tokens": 100}, "stop": []}
    s = shim.Session()
    s.user_key = "cagent-0-0-dead"
    os.environ["SWE_SHIM_METRICS"] = "1"
    try:
        _run(shim._generate(prompt, s, {}, app))
    finally:
        os.environ.pop("SWE_SHIM_METRICS", None)
    assert sampler.drained is True  # env-gated drain ran and matched prompt_tokens


def test_generate_kv_metrics_not_drained_by_default():
    prompt = [1, 2, 3]
    comp = _FakeCompletion(full_tokens=prompt + [10], prompt_len=3, sampling_logprobs=[-0.5], logprobs_echoed=False)
    sampler = _MetricsSampler(comp, [types.SimpleNamespace(prompt_tokens=3, cached_prompt_tokens=2)])
    app = {"sampler": sampler, "sample_kwargs": {"max_tokens": 100}, "stop": []}
    _run(shim._generate(prompt, shim.Session(), {}, app))
    assert sampler.drained is False  # not drained unless SWE_SHIM_METRICS is set
