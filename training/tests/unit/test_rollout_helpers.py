"""Unit tests for ``training.utils.rl.rollout.extract_completion``.

Locks in the token / logprob alignment contract:

  * ``token_ids`` is required; missing or empty raises.
  * ``token_ids`` and structured ``sampling_logprob`` values are filtered IN LOCKSTEP
    when the provider emits a null placeholder, so a leading or
    middle ``None`` token doesn't shift every remaining logprob
    onto the wrong token (which would silently corrupt PPO/GRPO
    ratio + KL for the affected completion).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from training.utils.rl.rollout import (
    Rollout,
    RolloutRun,
    extract_completion,
    rollout_to_prompt_group,
)
from training.utils.rl.agent.sampling import completion_routes, token_segment_to_sample
from training.utils.rl.agent.trajectory import TokenSegment


def _choice(token_ids, sampling_logprobs=None, finish_reason="stop"):
    return {
        "token_ids": token_ids,
        "logprobs": (
            {"content": [{"sampling_logprob": lp} for lp in sampling_logprobs]}
            if sampling_logprobs is not None
            else None
        ),
        "finish_reason": finish_reason,
    }


def test_basic_extract():
    call = extract_completion(
        _choice([10, 11, 12], [-0.1, -0.2, -0.3]),
        input_tokens=[1, 2, 3],
    )
    assert call.input_tokens == [1, 2, 3]
    assert call.output_tokens == [10, 11, 12]
    assert call.output_logprobs == [-0.1, -0.2, -0.3]


def test_missing_token_ids_raises():
    with pytest.raises(ValueError, match="missing 'token_ids'"):
        extract_completion(_choice(None), input_tokens=[1])


def test_empty_token_ids_raises():
    with pytest.raises(ValueError, match="missing 'token_ids'"):
        extract_completion(_choice([]), input_tokens=[1])


def test_null_token_at_start_filters_logprobs_in_lockstep():
    """A provider that emits ``None`` at index 0 of ``token_ids`` used
    to drop the entry from tokens but keep ALL logprobs, then
    tail-trim the resulting length mismatch.  Every remaining
    logprob shifted onto the wrong token, silently corrupting
    PPO/GRPO ratio + KL for that completion.  Filtering both lists
    in lockstep keeps surviving (token, logprob) pairs aligned."""
    call = extract_completion(
        _choice(
            token_ids=[None, 11, 12, 13],
            sampling_logprobs=[-9.0, -0.1, -0.2, -0.3],
        ),
        input_tokens=[1],
    )
    assert call.output_tokens == [11, 12, 13]
    # The logprob paired with the null token (-9.0) is dropped, NOT
    # the trailing logprob (-0.3) — which would have shifted every
    # remaining logprob one slot to the left.
    assert call.output_logprobs == [-0.1, -0.2, -0.3]


def test_null_token_in_middle_filters_logprobs_in_lockstep():
    call = extract_completion(
        _choice(
            token_ids=[10, None, 12, 13],
            sampling_logprobs=[-0.1, -9.0, -0.2, -0.3],
        ),
        input_tokens=[1],
    )
    assert call.output_tokens == [10, 12, 13]
    assert call.output_logprobs == [-0.1, -0.2, -0.3]


def test_no_null_tokens_unchanged():
    call = extract_completion(
        _choice(
            token_ids=[10, 11, 12],
            sampling_logprobs=[-0.1, -0.2, -0.3],
        ),
        input_tokens=[1],
    )
    assert call.output_tokens == [10, 11, 12]
    assert call.output_logprobs == [-0.1, -0.2, -0.3]


def test_logprob_list_padded_by_one_truncates_to_token_count():
    """Some providers emit ``len(logprobs) == len(tokens) + 1``;
    the helper trims to ``len(tokens)`` (back-compat with the
    pre-lockstep behavior for non-null token cases)."""
    call = extract_completion(
        _choice(
            token_ids=[10, 11, 12],
            sampling_logprobs=[-0.1, -0.2, -0.3, -0.4],  # 1 too many
        ),
        input_tokens=[1],
    )
    assert call.output_tokens == [10, 11, 12]
    assert call.output_logprobs == [-0.1, -0.2, -0.3]


def test_logprob_list_too_short_raises():
    with pytest.raises(ValueError, match="aligned with token_ids"):
        extract_completion(
            _choice(
                token_ids=[10, 11, 12],
                sampling_logprobs=[-0.1],  # 2 too few
            ),
            input_tokens=[1],
        )


def test_agent_completion_preserves_echoed_parquet_ranges():
    from fireworks.training.sdk.routing import RoutingReferences

    routes = RoutingReferences(
        4,
        ({"format": "parquet_v1", "row_count": 4, "artifact_id": "capture"},),
        ({"input_token_start": 0, "count": 4, "file_index": 0, "file_row_start": 0},),
    )
    completion = SimpleNamespace(
        routing_matrices=routes,
        prompt_len=3,
        full_tokens=[1, 2, 3, 4, 5],
        logprobs_echoed=True,
    )

    result = completion_routes(completion, output_len=2)

    assert isinstance(result, RoutingReferences)
    assert result == routes[2:]


def test_agent_routes_use_full_model_input_after_leading_response_is_masked(
    caplog,
):
    segment = TokenSegment(
        prompt_ids=[1, 2],
        response_ids=[10, 20, 30],
        loss_mask=[0, 0, 1],
        rollout_log_probs=[0.0, 0.0, -0.3],
        rollout_raw_log_probs=[0.0, 0.0, -0.4],
        routing_matrices=["", "", "route-30"],
    )
    sample = token_segment_to_sample(segment, reward=1.0)

    assert sample.routing_matrices == ["", "", "", "route-30"]

    group = rollout_to_prompt_group(
        Rollout(runs=[RolloutRun(segments=[sample])]),
        advantage_fn=lambda _rewards: [1.0],
        router_replay_completion_only=True,
    )

    assert group is not None
    assert group.data[0].model_input.routing_matrices == [
        "",
        "",
        "",
        "route-30",
    ]
    assert "R3: routing_matrices length" not in caplog.text


def _top_sampling_refs(length):
    from fireworks.training.sdk.routing import RoutingReferences

    return RoutingReferences(
        length,
        ({"store_id": "s", "file_id": "f", "format": "parquet_v1", "row_count": length},),
        ({"input_token_start": 0, "count": length, "file_index": 0, "file_row_start": 0},),
    )


def _covered_positions(references):
    return [
        position
        for span in references.spans
        if span.get("file_index") is not None
        for position in range(span["input_token_start"], span["input_token_start"] + span["count"])
    ]


def test_agent_top_sampling_references_follow_trained_outputs_in_model_input_coordinates():
    from training.utils.rl.agent.trajectory import TurnRecord, TurnSegment, merge_turn_segments

    def turn(prompt, output):
        return TurnRecord(
            prompt_ids=prompt,
            output_ids=output,
            finish_reason="stop",
            output_log_probs=[-0.5] * len(output),
            output_top_sampling_references=_top_sampling_refs(len(output)),
        )

    first = turn([1, 2], [10, 11])
    second = turn([1, 2, 10, 11, 5], [20])
    (segment,) = merge_turn_segments([TurnSegment(turns=[first, second], train_outputs=[False, True])])
    assert _covered_positions(segment.top_sampling_references) == [3]
    sample = token_segment_to_sample(segment, reward=1.0)
    assert sample.tokens == [1, 2, 10, 11, 5, 20]
    assert len(sample.top_sampling_references) == len(sample.tokens) - 1
    assert _covered_positions(sample.top_sampling_references) == [4]
    group = rollout_to_prompt_group(
        Rollout(runs=[RolloutRun(segments=[sample])]),
        advantage_fn=lambda _rewards: [1.0],
    )
    assert group.top_sampling_references == [sample.top_sampling_references]
    assert group.data[0].loss_fn_inputs["weights"].data[4] == 1


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("missing_first", [False, True])
@pytest.mark.parametrize("train_missing", [False, True])
def test_agent_preserves_support_around_missing_turns(
    split: bool, missing_first: bool, train_missing: bool
) -> None:
    from training.utils.rl.agent.trajectory import TurnRecord, TurnSegment, merge_turn_segments

    turns = [
        TurnRecord(
            prompt_ids=[1], output_ids=[10], finish_reason="stop", output_log_probs=[-0.5]
        ),
        TurnRecord(
            prompt_ids=[2] if split else [1, 10, 5],
            output_ids=[20], finish_reason="stop", output_log_probs=[-0.5],
        ),
    ]
    supported = int(missing_first)
    turns[supported] = dataclasses.replace(
        turns[supported], output_top_sampling_references=_top_sampling_refs(1)
    )
    masks = [train_missing, train_missing]
    masks[supported] = True
    segments = merge_turn_segments([TurnSegment(turns=turns, train_outputs=masks)])
    if split:
        assert segments[1 - supported].top_sampling_references is None
        assert _covered_positions(segments[supported].top_sampling_references) == [0]
    else:
        (segment,) = segments
        assert _covered_positions(segment.top_sampling_references) == [2 if supported else 0]
        assert len(segment.top_sampling_references) == len(segment.response_ids)
        assert segment.loss_mask == [int(masks[0]), 0, int(masks[1])]


def test_single_turn_completion_carries_model_input_top_sampling_references():
    from training.utils.rl.rollout.renderer import sampled_completion_to_rollout_run
    from training.utils.rl.rollout.types import RolloutSample

    completion = SimpleNamespace(
        prompt_len=3,
        full_tokens=[1, 2, 3, 7, 8],
        sampling_logprobs=[-0.1, -0.2],
        inference_logprobs=None,
        logprobs_echoed=False,
        top_sampling_references=_top_sampling_refs(2),
    )
    run = sampled_completion_to_rollout_run(completion, reward=1.0)
    references = run.segments[0].top_sampling_references
    assert len(references) == 4 and _covered_positions(references) == [2, 3]

    misaligned = RolloutSample(
        tokens=[1, 2, 3], logprobs=[0.0, -0.1, -0.2], loss_mask=[0, 1, 1], reward=1.0,
        top_sampling_references=_top_sampling_refs(3),
    )
    with pytest.raises(ValueError, match="model-input positions"):
        rollout_to_prompt_group(Rollout(runs=[RolloutRun(segments=[misaligned])]), advantage_fn=lambda r: r)
