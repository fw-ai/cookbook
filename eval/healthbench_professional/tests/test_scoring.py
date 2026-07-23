from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from healthbench_professional import judge, scoring


class FakeResponses:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=next(self.outputs))


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.responses = FakeResponses(outputs)


def test_grader_template_matches_openai_simple_evals_reference() -> None:
    assert hashlib.sha256(judge.GRADER_TEMPLATE.encode()).hexdigest() == (
        "2adffd51fd259554ebcd036ad1072d4aa2b7ce3aec2bbffe36271f911632ed3c"
    )


def test_signed_points_use_positive_points_as_denominator() -> None:
    rubrics = [
        {"criterion_text": "Includes the requested fact.", "points": 6},
        {"criterion_text": "Includes unsafe advice.", "points": -3},
    ]
    assert scoring.calculate_score(
        rubrics, [{"criteria_met": True}, {"criteria_met": False}]
    ) == pytest.approx(1.0)
    assert scoring.calculate_score(
        rubrics, [{"criteria_met": True}, {"criteria_met": True}]
    ) == pytest.approx(0.5)
    assert scoring.calculate_score(
        rubrics, [{"criteria_met": False}, {"criteria_met": True}]
    ) == pytest.approx(-0.5)


def test_score_requires_one_boolean_grade_per_rubric() -> None:
    rubric = [{"criterion_text": "Synthetic criterion.", "points": 1}]
    with pytest.raises(ValueError, match="one grading response"):
        scoring.calculate_score(rubric, [])
    with pytest.raises(ValueError, match="boolean criteria_met"):
        scoring.calculate_score(rubric, [{"criteria_met": 1}])


def test_length_adjustment_is_centered_and_not_sample_clipped() -> None:
    assert scoring.calculate_length_adjusted_score(1.0, "x" * 2000) == pytest.approx(
        1.0
    )
    assert scoring.calculate_length_adjusted_score(1.0, "x" * 2500) == pytest.approx(
        0.9853
    )
    # Canonical simple-evals behavior bonuses responses shorter than the center.
    assert scoring.calculate_length_adjusted_score(1.0, "x" * 1500) == pytest.approx(
        1.0147
    )
    assert scoring.calculate_length_adjusted_score(-0.5, "x" * 2500) == pytest.approx(
        -0.5147
    )


def test_score_response_emits_sparse_slice_metrics_without_clipping() -> None:
    metrics = scoring.score_response(
        [{"criterion_text": "Synthetic criterion.", "points": 1}],
        [{"criteria_met": True}],
        "short",
        slices={
            "use_case": "consult",
            "type": "good_faith",
            "difficulty": "typical",
            "specialty": "Synthetic / General",
        },
    )
    assert metrics["overall_score_length_adjusted"] > 1.0
    assert (
        metrics["specialty_synthetic_general"]
        == metrics["overall_score_length_adjusted"]
    )
    assert (
        metrics["source_good_faith_typical"] == metrics["overall_score_length_adjusted"]
    )


def test_judge_defaults_to_canonical_responses_settings_and_one_rubric_call() -> None:
    client = FakeClient(
        ['```json\n{"explanation":"synthetic","criteria_met":true}\n```']
    )
    grade = judge.judge_rubric_item(
        client,
        [{"role": "user", "content": "Synthetic prompt"}],
        "Synthetic answer",
        scoring.RubricItem("UNIQUE_SYNTHETIC_CRITERION", 2),
        config=judge.JudgeConfig(retry_delay_seconds=0),
    )

    assert grade == {"explanation": "synthetic", "criteria_met": True}
    assert len(client.responses.calls) == 1
    call = client.responses.calls[0]
    assert call["model"] == "gpt-5.4-2026-03-05"
    assert call["reasoning"] == {"effort": "low"}
    assert call["input"][0]["role"] == "user"
    assert "UNIQUE_SYNTHETIC_CRITERION" in call["input"][0]["content"]


def test_judge_retry_is_bounded() -> None:
    client = FakeClient(
        [
            "not json",
            '{"explanation":"still wrong","criteria_met":"yes"}',
            '{"explanation":"recovered","criteria_met":false}',
        ]
    )
    grade = judge.judge_rubric_item(
        client,
        [{"role": "user", "content": "Synthetic prompt"}],
        "Synthetic answer",
        scoring.RubricItem("Synthetic criterion", 1),
        config=judge.JudgeConfig(max_attempts=3, retry_delay_seconds=0),
    )
    assert grade["criteria_met"] is False
    assert len(client.responses.calls) == 3


def test_grade_response_makes_one_independent_call_per_rubric() -> None:
    client = FakeClient(
        [
            '{"explanation":"first","criteria_met":true}',
            '{"explanation":"second","criteria_met":true}',
        ]
    )
    source = {
        "messages": [{"role": "user", "content": "Synthetic prompt"}],
        "rubric_items": [
            {"criterion_text": "FIRST_UNIQUE_CRITERION", "points": 4},
            {"criterion_text": "SECOND_UNIQUE_CRITERION", "points": -2},
        ],
        "use_case": "consult",
        "type": "good_faith",
        "difficulty": "difficult",
        "specialty": "synthetic",
    }
    result = judge.grade_healthbench_response(
        client,
        source,
        "Synthetic answer",
        config=judge.JudgeConfig(retry_delay_seconds=0),
    )
    assert len(client.responses.calls) == 2
    assert "FIRST_UNIQUE_CRITERION" in client.responses.calls[0]["input"][0]["content"]
    assert "SECOND_UNIQUE_CRITERION" in client.responses.calls[1]["input"][0]["content"]
    assert result["metrics"]["overall_score"] == pytest.approx(0.5)
    assert len(result["rubric_grades"]) == 2
    assert result["judge"] == {
        "model": "gpt-5.4-2026-03-05",
        "reasoning_effort": "low",
        "max_attempts": 3,
        "max_workers": 8,
    }
    assert result["slices"]["specialty"] == "synthetic"


def test_judge_config_rejects_noncanonical_settings() -> None:
    with pytest.raises(ValueError, match="must use gpt-5.4"):
        judge.JudgeConfig(model="different-model")
    with pytest.raises(ValueError, match="low reasoning"):
        judge.JudgeConfig(reasoning_effort="high")
    with pytest.raises(ValueError, match="between 1 and 5"):
        judge.JudgeConfig(max_attempts=6)


def test_harbor_reward_contains_only_the_primary_adjusted_scalar() -> None:
    result = {
        "metrics": {
            "overall_score_length_adjusted": 0.625,
            "overall_score": 0.75,
            "specialty_synthetic": 0.625,
        }
    }
    assert judge.harbor_reward(result) == {"reward": 0.625}

    with pytest.raises(ValueError, match="overall_score_length_adjusted"):
        judge.harbor_reward({"metrics": {"overall_score": 0.75}})


def test_answer_must_match_validated_trajectory_before_judging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.json"
    answer_path = tmp_path / "answer.txt"
    trajectory_path = tmp_path / "trajectory.json"
    source_path.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "Synthetic prompt"}],
                "rubric_items": [
                    {"criterion_text": "Synthetic criterion", "points": 1}
                ],
                "use_case": "consult",
                "type": "good_faith",
                "difficulty": "typical",
                "specialty": "synthetic",
            }
        ),
        encoding="utf-8",
    )
    answer_path.write_text("answer A", encoding="utf-8")
    monkeypatch.setattr(
        judge,
        "load_atif_trajectory",
        lambda _: SimpleNamespace(visible_text="answer B"),
    )
    client = FakeClient([])
    with pytest.raises(ValueError, match="does not exactly match"):
        judge.grade_answer_files(
            source_path=source_path,
            answer_path=answer_path,
            trajectory_path=trajectory_path,
            client=client,
        )
    assert client.responses.calls == []
