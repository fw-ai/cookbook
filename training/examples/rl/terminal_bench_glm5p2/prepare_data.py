"""Register the bundled 48-task curriculum and pinned Terminal-Bench 2.1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_TRAIN_DATASET = "tb-opencode-medium-48"
DEFAULT_EVAL_DATASET = "terminal-bench@2.1"
DEFAULT_EVAL_SOURCE = "terminal-bench/terminal-bench-2-1@6"
EXPECTED_TRAIN_TASKS = 48
EXPECTED_EVAL_TASKS = 89

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_TASKS_DIR = EXAMPLE_DIR / "data" / "tasks"
DEFAULT_SELECTION = EXAMPLE_DIR / "data" / "selection.json"
REQUIRED_TASK_FILES = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "tests/test.sh",
)


def load_task_ids(selection_path: Path, tasks_dir: Path) -> list[str]:
    """Read and validate the ordered curriculum selection."""
    selection = json.loads(selection_path.read_text())
    task_ids = [str(item["task_id"]) for item in selection["selected"]]

    if len(task_ids) != EXPECTED_TRAIN_TASKS:
        raise ValueError(
            f"{selection_path} selects {len(task_ids)} tasks; "
            f"expected {EXPECTED_TRAIN_TASKS}"
        )
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"{selection_path} contains duplicate task IDs")

    for task_id in task_ids:
        task_dir = tasks_dir / task_id
        missing = [
            relative_path
            for relative_path in REQUIRED_TASK_FILES
            if not (task_dir / relative_path).is_file()
        ]
        if missing:
            raise ValueError(
                f"Task {task_id!r} is incomplete under {tasks_dir}: "
                f"missing {', '.join(missing)}"
            )
    return task_ids


def register_train_dataset(
    *,
    dataset_name: str,
    split: str,
    selection_path: Path,
    tasks_dir: Path,
) -> list[dict]:
    """Register the selected Harbor tasks without changing their order."""
    from rllm.data import DatasetRegistry
    from rllm.integrations.harbor.dataset_loader import harbor_task_to_row

    task_ids = load_task_ids(selection_path, tasks_dir)
    rows = []
    for task_id in task_ids:
        row = harbor_task_to_row(tasks_dir / task_id)
        if row is None:
            raise ValueError(f"Harbor rejected bundled task {task_id!r}")
        if str(row.get("task_id")) != task_id:
            raise ValueError(
                f"Bundled task {task_id!r} registered as {row.get('task_id')!r}"
            )
        rows.append(row)

    DatasetRegistry.register_dataset(
        name=dataset_name,
        data=rows,
        split=split,
        source="cookbook:terminal_bench_glm5p2/data/tasks",
        description=(
            "Fixed 48-task OpenCode curriculum used by the GLM-5.2 "
            "Terminal-Bench synchronous GRPO experiment"
        ),
        category="agentic",
    )
    return rows


def register_eval_dataset(*, dataset_name: str, split: str, source: str) -> list[dict]:
    """Pull and register the immutable 89-task Terminal-Bench 2.1 suite."""
    from rllm.data import DatasetRegistry
    from rllm.integrations.harbor.dataset_loader import load_harbor_dataset

    rows = load_harbor_dataset(source)
    if len(rows) != EXPECTED_EVAL_TASKS:
        raise ValueError(
            f"{source} resolved to {len(rows)} tasks; expected {EXPECTED_EVAL_TASKS}"
        )
    task_ids = [str(row.get("task_id")) for row in rows]
    if any(not task_id for task_id in task_ids):
        raise ValueError(f"{source} contains an empty task_id")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"{source} contains duplicate task IDs")

    DatasetRegistry.register_dataset(
        name=dataset_name,
        data=rows,
        split=split,
        source=f"harbor:{source}",
        description="Pinned 89-task Terminal-Bench 2.1 evaluation suite",
        category="agentic",
    )
    return rows


def assert_disjoint(train_rows: list[dict], eval_rows: list[dict]) -> None:
    """Reject train/eval task leakage."""
    train_ids = {str(row["task_id"]) for row in train_rows}
    eval_ids = {str(row["task_id"]) for row in eval_rows}
    overlap = sorted(train_ids & eval_ids)
    if overlap:
        raise ValueError(
            f"Training and evaluation overlap on {len(overlap)} task IDs; "
            f"first overlap: {overlap[0]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dataset",
        default=os.environ.get("TB_CURRICULUM_DATASET", DEFAULT_TRAIN_DATASET),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--eval-dataset",
        default=os.environ.get("TB_VAL_DATASET", DEFAULT_EVAL_DATASET),
    )
    parser.add_argument("--eval-split", default="default")
    parser.add_argument(
        "--eval-source",
        default=os.environ.get("TB_BENCHMARK_SOURCE", DEFAULT_EVAL_SOURCE),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Register only the bundled train split (local validation only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = register_train_dataset(
        dataset_name=args.train_dataset,
        split=args.train_split,
        selection_path=args.selection,
        tasks_dir=args.tasks_dir,
    )
    print(
        f"Registered {args.train_dataset}/{args.train_split}: {len(train_rows)} tasks"
    )

    if args.skip_eval:
        return

    eval_rows = register_eval_dataset(
        dataset_name=args.eval_dataset,
        split=args.eval_split,
        source=args.eval_source,
    )
    assert_disjoint(train_rows, eval_rows)
    print(
        f"Registered {args.eval_dataset}/{args.eval_split}: "
        f"{len(eval_rows)} tasks; train/eval overlap: 0"
    )


if __name__ == "__main__":
    main()
