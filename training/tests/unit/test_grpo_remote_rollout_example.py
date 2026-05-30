from __future__ import annotations

import pytest

pytest.importorskip("eval_protocol")

from eval_protocol.models import EvaluateResult, EvaluationRow, Message

from training.examples.rl.grpo_remote_rollout.convert import evaluation_row_to_rollout_sample
from training.examples.rl.grpo_remote_rollout.reward import grade_row, score_text


def test_score_text_extracts_boxed_answer():
    score, reason = score_text("We compute 40 + 2 = \\boxed{42}.", "42")

    assert score == 1.0
    assert "42" in reason


def test_grade_row_uses_final_assistant_message():
    row = EvaluationRow(
        messages=[
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="The answer is \\boxed{4}."),
        ]
    )

    graded = grade_row(row, {"answer": "4"})

    assert graded.evaluation_result is not None
    assert graded.evaluation_result.score == 1.0


def test_convert_token_turn_traces_to_rollout_sample():
    row = EvaluationRow(
        messages=[
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="4"),
            Message(role="user", content="Try again with a box."),
            Message(role="assistant", content="\\boxed{4}"),
        ]
    )
    row.evaluation_result = EvaluateResult(score=1.0, reason="correct")
    row.execution_metadata.extra = {
        "token_turn_traces": [
            {
                "prompt_ids": [1, 2],
                "completion_ids": [3, 4],
                "completion_logprobs": [-0.1, -0.2],
                "finish_reason": "stop",
            },
            {
                "prompt_ids": [1, 2, 3, 4, 5],
                "completion_ids": [6],
                "completion_logprobs": [-0.3],
                "finish_reason": "stop",
            },
        ]
    }

    sample = evaluation_row_to_rollout_sample(row, tokenizer_id="test-tokenizer")

    assert sample is not None
    assert sample.tokens == [1, 2, 3, 4, 5, 6]
    assert sample.logprobs == [0.0, 0.0, -0.1, -0.2, 0.0, -0.3]
    assert sample.loss_mask == [0, 0, 1, 1, 0, 1]
    assert sample.reward == 1.0


def test_convert_prompt_only_traces_with_payloads():
    row = EvaluationRow(
        messages=[
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="4"),
            Message(role="user", content="Try again with a box."),
            Message(role="assistant", content="\\boxed{4}"),
        ]
    )
    row.evaluation_result = EvaluateResult(score=1.0, reason="correct")
    row.execution_metadata.extra = {
        "token_turn_traces": [
            {"turn": 0, "prompt_ids": [1, 2], "finish_reason": "stop"},
            {"turn": 1, "prompt_ids": [1, 2, 3, 4, 5], "finish_reason": "stop"},
        ],
        "assistant_turn_payloads": [
            {
                "assistant_turn_index": 0,
                "completion_token_ids": [3, 4],
                "completion_logprobs": [-0.1, -0.2],
            },
            {
                "assistant_turn_index": 1,
                "completion_token_ids": [6],
                "completion_logprobs": [-0.3],
            },
        ],
    }

    sample = evaluation_row_to_rollout_sample(row, tokenizer_id="test-tokenizer")

    assert sample is not None
    assert sample.tokens == [1, 2, 3, 4, 5, 6]
    assert sample.logprobs == [0.0, 0.0, -0.1, -0.2, 0.0, -0.3]
    assert sample.loss_mask == [0, 0, 1, 1, 0, 1]
    assert sample.reward == 1.0
