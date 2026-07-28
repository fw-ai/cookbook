from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXAMPLE_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "rl" / "terminal_bench_glm5p2"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_terminal_bench_recipe_matches_proven_rllm_revision():
    assert _sha256(EXAMPLE_DIR / "train.py") == (
        "eea4d14565e6986ab59a37bc29e034c0cdd2103263772f656572a69cb131eba1"
    )
    assert _sha256(EXAMPLE_DIR / "train_fireworks_glm5p2.sh") == (
        "d13b98c9ed9c492d74320cbc2123ed2958a7e56b1b8bce5c43a831fec537ed6d"
    )


def test_terminal_bench_curriculum_is_complete_and_ordered():
    selection = json.loads((EXAMPLE_DIR / "data" / "selection.json").read_text())
    task_ids = [item["task_id"] for item in selection["selected"]]

    assert len(task_ids) == 48
    assert len(set(task_ids)) == 48
    assert task_ids[:3] == [
        "fhir-access-policy",
        "vinaigrette-signature-forge",
        "task-190047",
    ]
    assert task_ids[-1] == "spatial-network-audit"

    tasks_dir = EXAMPLE_DIR / "data" / "tasks"
    assert {path.name for path in tasks_dir.iterdir() if path.is_dir()} == set(task_ids)
    for task_id in task_ids:
        task_dir = tasks_dir / task_id
        for relative_path in (
            "task.toml",
            "instruction.md",
            "environment/Dockerfile",
            "tests/test.sh",
        ):
            assert (task_dir / relative_path).is_file()
