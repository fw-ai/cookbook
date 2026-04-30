"""Smoke test for the FrozenLake ``RolloutService`` adapter.

The adapter lives next to ``FrozenLakeToolRolloutProcessor`` in
``frozen_lake_rollout.py`` (already EP-aware), so the AC-6 boundary is
preserved (no NEW eval_protocol-importing files).

Asserts:
* the adapter constructs;
* per-turn token traces stitch into a token-native ``RolloutPayload``;
* loss-mask is correct (1 only on assistant token spans);
* ``messages`` are forwarded into ``EvaluationRow.messages``;
* domain metadata (``env_context``, ``user_prompt_template``,
  ``visual_prompt_template``) survives into ``dataset_info``;
* ``rollout_error`` rows are dropped.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List

import pytest

from training.examples.rl.frozen_lake.frozen_lake_rollout import FrozenLakeRolloutService
from training.utils.rl.rollout import pack_payload_to_sample


class _StubProcessor:
    """Records the EvaluationRows it was handed and returns them populated
    with the configured per-turn token traces + step rewards."""

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
            "prompt_ids": [1, 2, 3, 10, 11, 40],
            "completion_ids": [20, 21, 22],
            "completion_logprobs": [-0.3, -0.4, -0.5],
            "finish_reason": "stop",
        },
    ]
    processor = _StubProcessor(traces, step_rewards=[0.0, 1.0])
    service = FrozenLakeRolloutService(
        processor=processor,
        rollout_config=None,
        tokenizer_id="stub-tok",
    )

    payloads = asyncio.run(service.rollout(
        messages=[],
        n=1,
        sample_kwargs={},
        row={"env_context": {"seed": 7}},
    ))
    assert len(payloads) == 1
    payload = payloads[0]
    assert getattr(payload, "_assembled", False) is True
    assert payload.tokenizer_id == "stub-tok"
    assert payload.total_reward == pytest.approx(1.0)

    sample = asyncio.run(pack_payload_to_sample(payload, ctx=_Ctx(), version=1))
    expected_tokens = [1, 2, 3, 10, 11, 40, 20, 21, 22]
    assert sample.tokens == expected_tokens
    expected_mask = [0, 0, 0] + [1, 1] + [0] + [1, 1, 1]
    assert sample.loss_mask == expected_mask


def test_adapter_forwards_messages_into_evaluation_row():
    """The helper contract is ``service.rollout(messages, n=..., row=...)``.
    The frozen_lake processor reads ``row.messages`` as the seed
    conversation; dropping that argument silently breaks rollouts."""
    processor = _StubProcessor(traces=[], step_rewards=[])
    service = FrozenLakeRolloutService(
        processor=processor,
        rollout_config=None,
        tokenizer_id="stub-tok",
    )

    messages = [
        {"role": "system", "content": "play frozen lake"},
        {"role": "user", "content": "first move"},
    ]

    asyncio.run(service.rollout(
        messages=messages,
        n=2,
        sample_kwargs={},
        row={
            "env_context": {"seed": 9},
            "user_prompt_template": "USER: {observation}",
            "visual_prompt_template": "VISUAL: {observation}",
        },
    ))

    assert len(processor.received_rows) == 2
    for ep_row in processor.received_rows:
        assert len(ep_row.messages) == len(messages)
        for sent, recv in zip(messages, ep_row.messages):
            assert recv.role == sent["role"]
            assert recv.content == sent["content"]
        # Dataset metadata is carried through input_metadata.dataset_info.
        info = ep_row.input_metadata.dataset_info
        assert info["environment_context"] == {"seed": 9}
        assert info["user_prompt_template"] == "USER: {observation}"
        assert info["visual_prompt_template"] == "VISUAL: {observation}"


def test_helper_with_allow_empty_messages_invokes_service():
    """Round-4 regression: train_frozen_lake.py wires
    ``make_remote_rollout_fn(service, allow_empty_messages=True)`` and
    passes ``row["messages"] = []`` because the env builds the first
    observation inside the processor.  Without ``allow_empty_messages``
    the helper drops every row before the service is called.

    This test exercises the helper-plus-service stack and verifies the
    service.rollout(...) IS reached (Codex's Round-3 reproduction
    showed it was not, before this fix).
    """
    from training.utils.rl.rollout import make_remote_rollout_fn

    traces = [
        {
            "prompt_ids": [1, 2, 3],
            "completion_ids": [10, 11],
            "completion_logprobs": [-0.1, -0.2],
            "finish_reason": "stop",
        },
    ]
    processor = _StubProcessor(traces, step_rewards=[1.0])
    service = FrozenLakeRolloutService(
        processor=processor,
        rollout_config=None,
        tokenizer_id="stub-tok",
    )
    rollout_fn = make_remote_rollout_fn(service, allow_empty_messages=True)

    rollout = asyncio.run(rollout_fn(
        {"messages": [], "env_context": {"seed": 1}},
        _Ctx(),
    ))
    assert rollout is not None
    # Service WAS invoked (the StubProcessor records every call).
    assert len(processor.received_rows) == _Ctx.completions_per_prompt
    # Sanity: token-native sample emitted.
    assert rollout.samples
    assert sum(rollout.samples[0].loss_mask) == 2  # the two assistant tokens


def test_adapter_drops_rows_with_rollout_error():
    class _Bad(_StubProcessor):
        def __call__(self, rows, config):
            self.received_rows.extend(rows)

            async def _make(row):
                row.execution_metadata = SimpleNamespace(
                    extra={"rollout_error": "tool failed"}
                )
                return row
            return [asyncio.create_task(_make(r)) for r in rows]

    service = FrozenLakeRolloutService(
        processor=_Bad([], []),
        rollout_config=None,
        tokenizer_id="stub",
    )
    payloads = asyncio.run(service.rollout(
        messages=[],
        n=1,
        sample_kwargs={},
        row={"env_context": {"seed": 0}},
    ))
    assert payloads == []
