from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

from healthbench_professional import adapter


def synthetic_record(index: int = 0) -> dict:
    return {
        "id": f"synthetic-{index}",
        "conversation": {
            "messages": [
                {"role": "system", "content": "Answer carefully."},
                {"role": "user", "content": f"Synthetic question {index}: café?"},
            ]
        },
        "rubric_items": [
            {"criterion_text": "States the synthetic fact.", "points": 6},
            {"criterion_text": "Includes unsafe advice.", "points": -3},
        ],
        "use_case": "consult",
        "type": "good_faith",
        "difficulty": "typical",
        "specialty": "synthetic medicine",
        "physician_response": "private synthetic reference",
        "canary_string": "synthetic canary must be removed",
    }


def test_canonical_dataset_identity_is_pinned() -> None:
    assert adapter.DATASET_ID == "openai/healthbench-professional"
    assert adapter.DATASET_REVISION == "349962fd46dd02343a0d8a606491baf59154ea1a"
    assert (
        adapter.DATASET_SHA256
        == "d44b08e6e952e04c945e2c406f02533d9e7a989a84e35820ee7efdff20c9e4e2"
    )
    assert adapter.EXPECTED_ROWS == 525


def test_generated_task_round_trips_messages_and_hides_reference_fields(
    tmp_path: Path,
) -> None:
    record = synthetic_record()
    task = adapter.HealthBenchTask.from_record(record)
    output = adapter.HealthBenchProfessionalAdapter(tmp_path).generate_task(task)

    instruction = json.loads((output / "instruction.md").read_text(encoding="utf-8"))
    source_text = (output / "tests/source.json").read_text(encoding="utf-8")
    source = json.loads(source_text)

    assert instruction == {
        "schema_version": 1,
        "messages": record["conversation"]["messages"],
    }
    assert source["messages"] == record["conversation"]["messages"]
    assert source["rubric_items"] == record["rubric_items"]
    assert source["dataset_revision"] == adapter.DATASET_REVISION
    assert "physician_response" not in source
    assert "canary_string" not in source
    assert record["physician_response"] not in source_text
    assert record["canary_string"] not in source_text


def test_parser_rejects_conversation_that_does_not_end_with_user() -> None:
    record = synthetic_record()
    record["conversation"]["messages"][-1]["role"] = "assistant"
    with pytest.raises(ValueError, match="end in a user message"):
        adapter.HealthBenchTask.from_record(record)


def test_parser_rejects_rows_without_positive_point_rubrics() -> None:
    record = synthetic_record()
    record["rubric_items"] = [
        {"criterion_text": "Synthetic undesirable behavior.", "points": -1}
    ]
    with pytest.raises(ValueError, match="positive-point rubric"):
        adapter.HealthBenchTask.from_record(record)


def test_verify_dataset_file_rejects_wrong_sha(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        adapter.verify_dataset_file(path)


def test_canonical_loader_requires_exactly_525_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        "".join(json.dumps(synthetic_record(index)) + "\n" for index in range(524)),
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter, "verify_dataset_file", lambda _: None)
    with pytest.raises(ValueError, match="Expected 525.*found 524"):
        adapter.load_canonical_records(path)


def test_canonical_loader_accepts_exactly_525_sanitized_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        "".join(json.dumps(synthetic_record(index)) + "\n" for index in range(525)),
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter, "verify_dataset_file", lambda _: None)
    tasks = adapter.load_canonical_records(path)
    assert len(tasks) == 525
    assert tasks[-1].dataset_id == "synthetic-524"


def test_generate_tasks_rejects_duplicate_ids(tmp_path: Path) -> None:
    task = adapter.HealthBenchTask.from_record(synthetic_record())
    with pytest.raises(ValueError, match="Duplicate HealthBench dataset id"):
        adapter.HealthBenchProfessionalAdapter(tmp_path).generate_tasks([task, task])


def test_template_sentinels_render_and_verifier_modules_are_packaged(
    tmp_path: Path,
) -> None:
    template_dir = tmp_path / "template"
    (template_dir / "tests").mkdir(parents=True)
    (template_dir / "task.toml").write_text(
        "\n".join(
            [
                'name = "__TASK_NAME__"',
                'id = "__DATASET_ID__"',
                'index = "__DATASET_INDEX__"',
                'question_sha = "__QUESTION_SHA256__"',
                'use_case = "__USE_CASE__"',
                'kind = "__INTERACTION_TYPE__"',
                'difficulty = "__DIFFICULTY__"',
                'specialty = "__SPECIALTY__"',
                'key = "${OPENAI_API_KEY}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (template_dir / "tests/test.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    task = adapter.HealthBenchTask.from_record(synthetic_record(), dataset_index=17)
    output = adapter.HealthBenchProfessionalAdapter(
        tmp_path / "generated", template_dir=template_dir
    ).generate_task(task)

    task_toml = (output / "task.toml").read_text(encoding="utf-8")
    assert re.search(r"__[A-Z0-9_]+__", task_toml) is None
    assert 'index = "17"' in task_toml
    assert 'specialty = "synthetic medicine"' in task_toml
    assert 'key = "${OPENAI_API_KEY}"' in task_toml
    for filename in ("scoring.py", "judge.py", "trajectory.py"):
        assert (output / "tests" / filename).is_file()


def test_limit_task_ids_and_overwrite_controls(tmp_path: Path) -> None:
    tasks = [
        adapter.HealthBenchTask.from_record(synthetic_record(index))
        for index in range(3)
    ]
    task_adapter = adapter.HealthBenchProfessionalAdapter(tmp_path)
    generated = task_adapter.generate_tasks(
        tasks,
        task_ids=["synthetic-0", "synthetic-2"],
        limit=1,
    )
    assert [path.name for path in generated] == ["healthbench_professional_synthetic-0"]

    marker = generated[0] / "stale.txt"
    marker.write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        task_adapter.generate_task(tasks[0])
    task_adapter.generate_task(tasks[0], overwrite=True)
    assert not marker.exists()

    task_adapter.generate_tasks(tasks, overwrite=True)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    task_adapter.generate_tasks(tasks, limit=1, overwrite=True)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "healthbench_professional_synthetic-0",
        "unrelated",
    ]

    with pytest.raises(ValueError, match="Unknown HealthBench task ids"):
        task_adapter.generate_tasks(tasks, task_ids=["not-present"])


def test_actual_harbor_template_renders_as_valid_toml(tmp_path: Path) -> None:
    task = adapter.HealthBenchTask.from_record(synthetic_record(), dataset_index=9)
    output = adapter.HealthBenchProfessionalAdapter(
        tmp_path, template_dir=adapter.DEFAULT_TEMPLATE_DIR
    ).generate_task(task)
    task_toml_text = (output / "task.toml").read_text(encoding="utf-8")
    task_toml = tomllib.loads(task_toml_text)
    assert re.search(r"__[A-Z0-9_]+__", task_toml_text) is None
    assert task_toml["task"]["name"] == "openai/healthbench-professional-synthetic-0"
    assert task_toml["metadata"]["dataset_id"] == "synthetic-0"
    assert task_toml["metadata"]["dataset_index"] == "9"
    assert (output / "tests/test.sh").is_file()
