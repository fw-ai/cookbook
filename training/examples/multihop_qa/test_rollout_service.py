"""Smoke test for the MultiHopQA ``RolloutService`` adapter.

The adapter lives next to ``MultiHopQARolloutProcessor`` in
``multihop_qa_rollout.py`` (already EP-aware), so the AC-6 boundary is
preserved (no NEW eval_protocol-importing files).

Asserts:
* token-native ``RolloutPayload`` emission with correct loss-mask;
* ``messages`` are forwarded into ``EvaluationRow.messages``;
* ``context`` / ``ground_truth`` / ``question`` survive into
  ``input_metadata.dataset_info`` (the multihop_qa processor reads them
  there);
* the question is derived from the user message when not supplied
  explicitly.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List

import pytest

from training.examples.multihop_qa.multihop_qa_rollout import MultiHopQARolloutService
from training.utils.rl.text_rollout import pack_payload_to_sample


class _StubProcessor:
    def __init__(self, traces, step_rewards):
        self.traces = list(traces)
        self.step_rewards = list(step_rewards)
        self.received_rows: List[Any] = []

    def __call__(self, rows, config):
        self.received_rows.extend(rows)

        async def _make(row):
            row.execution_metadata = SimpleNamespace(
                extra={
                    "token_turn_traces": list(self.traces),
                    "step_rewards": list(self.step_rewards),
                }
            )
            return row
        return [asyncio.create_task(_make(r)) for r in rows]


class _Ctx:
    tokenizer_id = "stub-tok"
    completions_per_prompt = 1
    sample_kwargs: dict = {}

    def current_version(self) -> int:
        return 1


def test_adapter_emits_token_native_payload():
    traces = [
        {
            "prompt_ids": [1, 2, 3],
            "completion_ids": [10, 11],
            "completion_logprobs": [-0.1, -0.2],
            "finish_reason": "stop",
        },
        {
            "prompt_ids": [1, 2, 3, 10, 11, 50],
            "completion_ids": [20, 21],
            "completion_logprobs": [-0.3, -0.4],
            "finish_reason": "stop",
        },
    ]
    processor = _StubProcessor(traces, step_rewards=[0.0, 0.7])
    service = MultiHopQARolloutService(
        processor=processor,
        rollout_config=None,
        tokenizer_id="stub-tok",
    )

    payloads = asyncio.run(service.rollout(
        messages=[{"role": "user", "content": "Who wrote Hamlet?"}],
        n=1,
        sample_kwargs={},
        row={
            "context": {"docs": ["wikipedia entry"]},
            "ground_truth": "Shakespeare",
            "question": "Who wrote Hamlet?",
        },
    ))
    assert len(payloads) == 1
    payload = payloads[0]
    assert getattr(payload, "_assembled", False) is True
    assert payload.tokenizer_id == "stub-tok"

    sample = asyncio.run(pack_payload_to_sample(payload, ctx=_Ctx(), version=1))
    expected_tokens = [1, 2, 3, 10, 11, 50, 20, 21]
    assert sample.tokens == expected_tokens
    expected_mask = [0, 0, 0] + [1, 1] + [0] + [1, 1]
    assert sample.loss_mask == expected_mask
    assert sample.reward == pytest.approx(0.7)


def test_adapter_forwards_messages_and_metadata():
    """The multihop_qa processor reads ``row.messages`` and
    ``input_metadata.dataset_info[{context, ground_truth, question}]``.
    The adapter must forward all of them so domain logic keeps working."""
    processor = _StubProcessor(traces=[], step_rewards=[])
    service = MultiHopQARolloutService(
        processor=processor,
        rollout_config=None,
        tokenizer_id="stub-tok",
    )
    messages = [
        {"role": "system", "content": "research helper"},
        {"role": "user", "content": "When was Python 3 released?"},
    ]
    asyncio.run(service.rollout(
        messages=messages,
        n=2,
        sample_kwargs={},
        row={
            "context": {"docs": ["py history"]},
            "ground_truth": "2008",
            "question": "When was Python 3 released?",
        },
    ))
    assert len(processor.received_rows) == 2
    for ep_row in processor.received_rows:
        assert len(ep_row.messages) == len(messages)
        for sent, recv in zip(messages, ep_row.messages):
            assert recv.role == sent["role"]
            assert recv.content == sent["content"]
        info = ep_row.input_metadata.dataset_info
        assert info["context"] == {"docs": ["py history"]}
        assert info["ground_truth"] == "2008"
        assert info["question"] == "When was Python 3 released?"


def test_prepare_row_ids_matches_emitted_ids():
    """The entrypoint pre-registers IGPO scorer state keyed by the same
    row_ids the service will assign.  ``prepare_row_ids`` is the contract
    that exposes those IDs ahead of the rollout."""
    processor = _StubProcessor(traces=[], step_rewards=[])
    service = MultiHopQARolloutService(
        processor=processor,
        rollout_config=None,
        tokenizer_id="stub",
    )
    row = {
        "context": {},
        "ground_truth": "X",
        "question": "Q?",
    }
    expected = MultiHopQARolloutService.prepare_row_ids(n=3, row=row)
    asyncio.run(service.rollout(messages=[], n=3, sample_kwargs={}, row=row))
    actual = [r.input_metadata.row_id for r in processor.received_rows]
    assert actual == expected


def test_turn_callback_threads_through_to_processor_config():
    """When the service is constructed with ``turn_callback=...``, every
    call to ``service.rollout`` builds a ``RolloutProcessorConfig`` whose
    ``kwargs["turn_callback"]`` is that same callable.  The legacy IGPO
    path consumes this hook to drive ``scorer.on_turn_complete``."""

    class _CallbackCapturingProcessor:
        def __init__(self):
            self.last_config = None
            self.received_rows: list = []

        def __call__(self, rows, config):
            self.last_config = config
            self.received_rows.extend(rows)

            async def _make(row):
                row.execution_metadata = SimpleNamespace(extra={})
                return row
            return [asyncio.create_task(_make(r)) for r in rows]

    processor = _CallbackCapturingProcessor()

    captured_calls = []

    def _fake_callback(*args, **kwargs):
        captured_calls.append((args, kwargs))

    service = MultiHopQARolloutService(
        processor=processor,
        rollout_config=SimpleNamespace(
            completion_params={}, mcp_config_path="", steps=4, semaphore=None,
        ),
        tokenizer_id="stub",
        turn_callback=_fake_callback,
    )

    asyncio.run(service.rollout(
        messages=[{"role": "user", "content": "test"}],
        n=1,
        sample_kwargs={},
        row={"context": {}, "ground_truth": "x", "question": "test"},
    ))

    cfg = processor.last_config
    assert cfg is not None
    assert cfg.kwargs is not None
    assert cfg.kwargs.get("turn_callback") is _fake_callback


def test_payload_extras_carry_row_id():
    """The service stamps each payload with the same row_id passed to the
    processor (and therefore the same row_id the scorer's turn_callback
    received).  Without this, the entrypoint can't correlate per-payload
    metadata back to scorer state."""
    traces = [
        {
            "prompt_ids": [1, 2],
            "completion_ids": [3, 4],
            "completion_logprobs": [-0.1, -0.2],
            "finish_reason": "stop",
        },
    ]
    processor = _StubProcessor(traces, step_rewards=[0.5])
    service = MultiHopQARolloutService(
        processor=processor,
        rollout_config=None,
        tokenizer_id="stub",
    )
    row = {"context": {}, "ground_truth": "X", "question": "Q?"}
    payloads = asyncio.run(service.rollout(
        messages=[{"role": "user", "content": "Q?"}],
        n=2,
        sample_kwargs={},
        row=row,
    ))
    expected_ids = MultiHopQARolloutService.prepare_row_ids(n=2, row=row)
    assert [p.extras.get("row_id") for p in payloads] == expected_ids


def test_adapter_derives_question_from_user_message_when_absent():
    """When the row dict doesn't carry ``question`` explicitly, it should be
    derived from the first user message — the multihop_qa processor and
    grader expect a non-empty question string."""
    processor = _StubProcessor(traces=[], step_rewards=[])
    service = MultiHopQARolloutService(
        processor=processor,
        rollout_config=None,
        tokenizer_id="stub-tok",
    )
    asyncio.run(service.rollout(
        messages=[
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "What city is the Eiffel Tower in?"},
        ],
        n=1,
        sample_kwargs={},
        row={"context": {}, "ground_truth": "Paris"},
    ))
    info = processor.received_rows[0].input_metadata.dataset_info
    assert info["question"] == "What city is the Eiffel Tower in?"


def test_prepare_row_ids_unique_for_distinct_rows_with_same_question():
    """Two rows with the same question text but different
    ground_truth must produce distinct base IDs.  The previous
    fallback (``q_{hash(question) % 100000}``) collided whenever two
    distinct rows shared the question, so the IGPO scorer's
    ``_turn_futs`` / baseline dicts (keyed on row_id) overwrote each
    other in flight."""
    row_a = {"question": "What is X?", "ground_truth": "A"}
    row_b = {"question": "What is X?", "ground_truth": "B"}
    ids_a = MultiHopQARolloutService.prepare_row_ids(n=1, row=row_a)
    ids_b = MultiHopQARolloutService.prepare_row_ids(n=1, row=row_b)
    assert ids_a != ids_b


def test_prepare_row_ids_stable_across_calls():
    """Identical rows produce identical IDs (deterministic, JSON-stable
    hash).  This is required for the train script's pre-registration to
    line up with the service's emitted IDs."""
    row = {"question": "Q", "ground_truth": "GT", "messages": []}
    ids_a = MultiHopQARolloutService.prepare_row_ids(n=2, row=row)
    ids_b = MultiHopQARolloutService.prepare_row_ids(n=2, row=row)
    assert ids_a == ids_b


def test_prepare_row_ids_explicit_id_wins_over_hash():
    row = {
        "id": "user-supplied-id",
        "question": "Q",
        "ground_truth": "GT",
    }
    ids = MultiHopQARolloutService.prepare_row_ids(n=2, row=row)
    assert ids == ["user-supplied-id_0", "user-supplied-id_1"]
