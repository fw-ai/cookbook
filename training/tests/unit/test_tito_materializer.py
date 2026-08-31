from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from fireworks.training.sdk import (
    TITOChatRequest,
    TITOCallRecord,
    TITODistribution,
    TITOMetricSummary,
    TITOParsedAssistant,
    TITOResponseAttempt,
    TITOSegmentResult,
    TITOTrajectoryArtifact,
    TITOTurn,
)

from training.utils.rl.rollout.tito import materialize_tito_trajectory
from training.utils.rl.metrics import compute_step_metrics
from training.utils.rl.rollout.types import Rollout, rollout_to_prompt_group


def _distribution(values: list[float]) -> TITODistribution:
    return TITODistribution(
        count=len(values),
        sum=sum(values),
        min=min(values) if values else None,
        max=max(values) if values else None,
    )


def _turn(
    turn_id: str,
    prompt: tuple[int, ...],
    completion: tuple[int, ...],
    *,
    routes: tuple[str, ...] | None = None,
    finish_reason: str = "stop",
    messages: tuple[dict[str, Any], ...] | None = None,
    disposition: str = "new_segment",
    prefix_match_tokens: int | None = None,
    realign_from_token: int | None = None,
    realigned_masked_tokens: int = 0,
    incremental_checkpoint_trim_tokens: int = 0,
) -> TITOTurn:
    return TITOTurn(
        turn_id=turn_id,
        request=TITOChatRequest(
            messages=messages or ({"role": "user", "content": "q"},)
        ),
        assistant=TITOParsedAssistant(
            message={"role": "assistant", "content": turn_id}
        ),
        exact_prompt_ids=prompt,
        exact_completion_ids=completion,
        inference_logprobs=tuple(-0.1 for _ in completion),
        sampling_logprobs=tuple(-0.2 for _ in completion),
        routing_matrices=routes,
        response_id=f"response-{turn_id}",
        finish_reason=finish_reason,
        prompt_disposition=disposition,  # type: ignore[arg-type]
        prefix_match_tokens=prefix_match_tokens,
        realign_from_token=realign_from_token,
        realigned_masked_tokens=realigned_masked_tokens,
        prompt_mode="full_history",
        incremental_contract_id=None,
        incremental_junction_kind=None,
        incremental_checkpoint_trim_tokens=incremental_checkpoint_trim_tokens,
        incremental_fallback_reason=None,
        requested_output_tokens=8,
        effective_output_tokens=8,
        context_remaining_tokens=100,
        server_metrics=None,
        sampler_wall_seconds=0.1,
        logical_request_id=f"logical-{turn_id}",
        upstream_response_id=f"upstream-{turn_id}",
        upstream_attempts=1,
    )


def _result(
    turns: tuple[TITOTurn, ...],
    attempts: tuple[TITOResponseAttempt, ...],
    *,
    classification_sources: dict[str, str] | None = None,
    segment_turns: tuple[tuple[TITOTurn, ...], ...] | None = None,
    status: str = "completed",
    metrics: dict[str, int] | None = None,
) -> TITOTrajectoryArtifact:
    now = time.time()
    grouped_turns = segment_turns or (turns,)
    counters = {
        "calls/total": len(turns),
        "lineage/new_segment": len(grouped_turns),
        "lineage/prefix_check": max(len(turns) - 1, 0),
        "lineage/realign": sum(turn.prompt_disposition == "realign" for turn in turns),
        **(metrics or {}),
    }
    return TITOTrajectoryArtifact(
        trajectory_id="trajectory-1",
        serving_affinity_key_hash="hash",
        metadata={},
        status=status,  # type: ignore[arg-type]
        terminal_reason=None,
        segments=tuple(
            TITOSegmentResult(
                segment_id=f"segment-{index}",
                start_reason="initial" if index == 1 else "token_drift",
                render_contract_id="contract",
                turns=items,
                closed_reason="trajectory_completed",
            )
            for index, items in enumerate(grouped_turns, start=1)
        ),
        calls=tuple(
            TITOCallRecord(
                call_id=f"call-{turn.turn_id}",
                kind="policy",
                classification_source=(classification_sources or {}).get(
                    turn.turn_id,
                    "default",
                ),
                outcome="succeeded",
                started_at=now,
                ended_at=now + 0.1,
                turn_id=turn.turn_id,
            )
            for turn in turns
        ),
        response_attempts=attempts,
        metrics=TITOMetricSummary(
            counters=counters,
            distributions={
                "turn/prompt_tokens": _distribution(
                    [float(len(turn.exact_prompt_ids)) for turn in turns]
                ),
                "turn/completion_tokens": _distribution(
                    [float(len(turn.exact_completion_ids)) for turn in turns]
                ),
                "turn/request_wall_seconds": _distribution([0.1 for _ in turns]),
                "trajectory/policy_turns": _distribution([float(len(turns))]),
            },
        ),
        started_at=now,
        finished_at=now + 1,
    )


def _attempt(turn_id: str, emission: str = "completed") -> TITOResponseAttempt:
    return TITOResponseAttempt(
        attempt_id=f"attempt-{turn_id}",
        turn_id=turn_id,
        emission=emission,  # type: ignore[arg-type]
        created_at=time.time(),
    )


def test_materializes_exact_append_and_pads_completion_only_r3() -> None:
    first = _turn("one", (1, 2), (3, 4), routes=("r3", "r4"))
    second = _turn(
        "two",
        (1, 2, 3, 4, 5),
        (6, 7),
        routes=("r6", "r7"),
        disposition="append",
        prefix_match_tokens=4,
    )
    run = materialize_tito_trajectory(
        _result((first, second), (_attempt("one"), _attempt("two"))),
        reward=1.5,
    )

    assert len(run.segments) == 1
    sample = run.segments[0]
    assert sample.tokens == [1, 2, 3, 4, 5, 6, 7]
    assert sample.loss_mask == [0, 0, 1, 1, 0, 1, 1]
    assert sample.logprobs == [0.0, 0.0, -0.2, -0.2, 0.0, -0.2, -0.2]
    assert sample.routing_matrices == ["", "r3", "r4", "", "r6", "r7"]
    assert sample.reward == 1.5


@pytest.mark.parametrize(
    ("status", "expected_suffix"),
    (("completed", "followed_by_completed"), ("failed", "followed_by_terminal")),
)
def test_model_malformed_metric_is_an_outcome_association(
    status: str,
    expected_suffix: str,
) -> None:
    turn = _turn("one", (1, 2), (3, 4))
    run = materialize_tito_trajectory(
        _result(
            (turn,),
            (_attempt("one"),),
            status=status,
            metrics={
                "calls/total": 2,
                "calls/succeeded": 1,
                "calls/model_malformed": 1,
                "parser/model_malformed": 1,
            },
        ),
        reward=0.0,
    )

    assert run.metadata["tito_metrics"]["tito/parser/model_malformed"] == 1
    assert run.metadata["tito_metrics"]["tito/calls/model_malformed"] == 1
    assert run.metadata["tito_metrics"]["tito/trajectory/model_malformed_observed"] == 1
    assert (
        run.metadata["tito_metrics"][
            f"tito/trajectory/model_malformed_{expected_suffix}"
        ]
        == 1
    )
    assert not any(
        key.endswith("/failed")
        for key in run.metadata["tito_metrics"]
        if "model_malformed" in key
    )


def test_new_segment_boundary_preserves_both_training_examples() -> None:
    first = _turn("one", (1, 2), (3, 4))
    second = _turn("two", (9, 10), (11, 12))
    run = materialize_tito_trajectory(
        _result(
            (first, second),
            (_attempt("one"), _attempt("two")),
            segment_turns=((first,), (second,)),
        ),
        reward=1.0,
    )
    assert [sample.tokens for sample in run.segments] == [
        [1, 2, 3, 4],
        [9, 10, 11, 12],
    ]
    assert all(sum(sample.loss_mask) == 2 for sample in run.segments)


def test_bounded_realign_masks_reconstructed_prior_response_in_one_example() -> None:
    first = _turn("one", (1, 2), (3, 4), routes=("r3", "r4"))
    second = _turn(
        "two",
        (1, 2, 3, 9, 5),
        (6, 7),
        routes=("r6", "r7"),
        disposition="realign",
        prefix_match_tokens=3,
        realign_from_token=2,
        realigned_masked_tokens=3,
    )

    run = materialize_tito_trajectory(
        _result((first, second), (_attempt("one"), _attempt("two"))),
        reward=1.0,
    )

    assert len(run.segments) == 1
    sample = run.segments[0]
    assert sample.tokens == [1, 2, 3, 9, 5, 6, 7]
    assert sample.loss_mask == [0, 0, 0, 0, 0, 1, 1]
    assert sample.logprobs == [0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2]
    assert sample.routing_matrices == ["", "", "", "", "r6", "r7"]
    assert (
        run.metadata["tito_materialized_segments"][0]["tito_realigned_masked_tokens"]
        == 3
    )


def test_incremental_junction_replaces_only_declared_checkpoint_tail() -> None:
    first = _turn("one", (1, 2), (3, 4), routes=("r3", "r4"))
    second = _turn(
        "two",
        (1, 2, 3, 9, 5),
        (6, 7),
        routes=("r6", "r7"),
        disposition="append",
        prefix_match_tokens=3,
        incremental_checkpoint_trim_tokens=1,
    )

    run = materialize_tito_trajectory(
        _result((first, second), (_attempt("one"), _attempt("two"))),
        reward=1.0,
    )

    assert len(run.segments) == 1
    sample = run.segments[0]
    assert sample.tokens == [1, 2, 3, 9, 5, 6, 7]
    assert sample.loss_mask == [0, 0, 1, 0, 0, 1, 1]
    assert sample.logprobs == [0.0, 0.0, -0.2, 0.0, 0.0, -0.2, -0.2]
    assert sample.routing_matrices == ["", "r3", "", "", "r6", "r7"]
    assert (
        run.metadata["tito_materialized_segments"][0][
            "tito_incremental_checkpoint_trimmed_tokens"
        ]
        == 1
    )


def test_later_continuation_proves_prior_ambiguous_turn_visible() -> None:
    first = _turn("one", (1, 2), (3, 4))
    second = _turn(
        "two",
        (1, 2, 3, 4, 5),
        (6, 7),
        messages=(
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "next"},
        ),
        disposition="append",
        prefix_match_tokens=4,
    )
    run = materialize_tito_trajectory(
        _result((first, second), (_attempt("one", "ambiguous"), _attempt("two"))),
        reward=1.0,
    )
    assert len(run.segments) == 1
    assert sum(run.segments[0].loss_mask) == 4


def test_cross_segment_continuation_proves_prior_ambiguous_turn_visible() -> None:
    first = _turn("one", (1, 2), (3, 4))
    second = _turn(
        "two",
        (9, 10),
        (11, 12),
        messages=(
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "next"},
        ),
    )
    run = materialize_tito_trajectory(
        _result(
            (first, second),
            (_attempt("one", "ambiguous"), _attempt("two")),
            segment_turns=((first,), (second,)),
        ),
        reward=1.0,
    )

    assert [sample.tokens for sample in run.segments] == [
        [1, 2, 3, 4],
        [9, 10, 11, 12],
    ]


def test_unrelated_later_segment_does_not_resolve_ambiguous_action() -> None:
    first = _turn("one", (1, 2), (3, 4))
    second = _turn("two", (9, 10), (11, 12))
    run = materialize_tito_trajectory(
        _result(
            (first, second),
            (_attempt("one", "ambiguous"), _attempt("two")),
            segment_turns=((first,), (second,)),
        ),
        reward=1.0,
    )

    assert [sample.tokens for sample in run.segments] == [[9, 10, 11, 12]]


def test_unresolved_terminal_ambiguous_action_is_omitted() -> None:
    turn = _turn("one", (1, 2), (3, 4))
    run = materialize_tito_trajectory(
        _result((turn,), (_attempt("one", "ambiguous"),)),
        reward=1.0,
    )
    assert run.segments == []


def test_harness_abandoned_length_turn_overrides_completed_emission() -> None:
    turn = _turn("one", (1, 2), (3, 4), finish_reason="length")
    run = materialize_tito_trajectory(
        _result((turn,), (_attempt("one"),)),
        reward=1.0,
        harness_abandoned_turn_ids={"one"},
    )

    assert run.segments == []
    assert run.metadata["tito_abandoned_turn_count"] == 1


def test_harness_cannot_abandon_non_length_turn() -> None:
    turn = _turn("one", (1, 2), (3, 4))
    with pytest.raises(ValueError, match="only length-truncated"):
        materialize_tito_trajectory(
            _result((turn,), (_attempt("one"),)),
            reward=1.0,
            harness_abandoned_turn_ids={"one"},
        )


def test_fail_closed_turn_is_context_only_when_later_policy_turn_is_trainable() -> None:
    summary = _turn("summary", (1, 2), (3, 4))
    policy = _turn(
        "policy",
        (1, 2, 3, 4, 5),
        (6, 7),
        disposition="append",
        prefix_match_tokens=4,
    )
    run = materialize_tito_trajectory(
        _result(
            (summary, policy),
            (_attempt("summary"), _attempt("policy")),
            classification_sources={"summary": "fail_closed"},
        ),
        reward=1.0,
    )

    assert len(run.segments) == 1
    assert run.segments[0].tokens == [1, 2, 3, 4, 5, 6, 7]
    assert run.segments[0].loss_mask == [0, 0, 0, 0, 0, 1, 1]
    assert run.metadata["tito_masked_fail_closed_turn_count"] == 1
    assert (
        run.metadata["tito_materialized_segments"][0][
            "tito_masked_fail_closed_turn_count"
        ]
        == 1
    )


def test_fail_closed_only_sample_is_omitted_from_training() -> None:
    summary = _turn("summary", (1, 2), (3, 4))
    run = materialize_tito_trajectory(
        _result(
            (summary,),
            (_attempt("summary"),),
            classification_sources={"summary": "fail_closed"},
        ),
        reward=1.0,
    )

    assert run.segments == []
    assert run.metadata["tito_masked_fail_closed_turn_count"] == 1


def test_training_sequence_limit_omits_whole_turn_without_truncation() -> None:
    first = _turn("one", (1, 2), (3, 4))
    second = _turn(
        "two",
        (1, 2, 3, 4, 5),
        (6, 7),
        disposition="append",
        prefix_match_tokens=4,
    )
    run = materialize_tito_trajectory(
        _result((first, second), (_attempt("one"), _attempt("two"))),
        reward=1.0,
        max_context_tokens=4,
    )

    assert [sample.tokens for sample in run.segments] == [[1, 2, 3, 4]]
    assert run.metadata["tito_max_context_tokens"] == 4
    assert run.metadata["tito_retention_dropped_turn_count"] == 1
    assert run.metadata["tito_retention_dropped_trainable_tokens"] == 2


def test_training_limit_resumes_from_a_later_exact_prompt() -> None:
    first = _turn("one", (1, 2), (3, 4))
    second = _turn(
        "two",
        (1, 2, 3, 4, 5),
        (6, 7, 8),
        disposition="append",
        prefix_match_tokens=4,
    )
    third = _turn(
        "three",
        (1, 2, 3, 4, 5, 9),
        (10,),
        disposition="realign",
        prefix_match_tokens=5,
        realign_from_token=5,
        realigned_masked_tokens=1,
    )
    run = materialize_tito_trajectory(
        _result(
            (first, second, third),
            (_attempt("one"), _attempt("two"), _attempt("three")),
        ),
        reward=1.0,
        max_context_tokens=7,
    )

    assert [sample.tokens for sample in run.segments] == [
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5, 9, 10],
    ]
    assert run.segments[1].loss_mask == [0, 0, 0, 0, 0, 0, 1]
    assert run.metadata["tito_retention_dropped_turn_count"] == 1
    assert run.metadata["tito_retention_dropped_trainable_tokens"] == 3
    assert [
        item["tito_physical_split_index"]
        for item in run.metadata["tito_materialized_segments"]
    ] == [0, 1]


@pytest.mark.parametrize("limit", [0, -1])
def test_training_sequence_limit_must_be_positive(limit: int) -> None:
    turn = _turn("one", (1, 2), (3, 4))
    with pytest.raises(ValueError, match="max_context_tokens"):
        materialize_tito_trajectory(
            _result((turn,), (_attempt("one"),)),
            reward=1.0,
            max_context_tokens=limit,
        )


def test_sidecar_metrics_reach_common_step_reducer() -> None:
    turn = _turn("one", (1, 2), (3, 4))
    run = materialize_tito_trajectory(
        _result((turn,), (_attempt("one"),)),
        reward=1.0,
    )
    group = rollout_to_prompt_group(
        Rollout(runs=[run]),
        advantage_fn=lambda rewards: rewards,
    )
    assert group is not None

    metrics = compute_step_metrics(
        prompt_groups=[group],
        fwd_bwd_results=[],
        optim_result=SimpleNamespace(metrics={}),
        n_accum=1,
        timing_metrics={},
    )

    assert metrics["tito/trajectory/count"] == 1
    assert metrics["tito/turn/count"] == 1
    assert metrics["tito/turn/input_tokens_mean"] == 2
    assert metrics["tito/turn/output_tokens_mean"] == 2
    assert metrics["tito/turn/runtime_seconds_mean"] == pytest.approx(0.1)
    assert "tito/debug/calls/total" not in metrics
