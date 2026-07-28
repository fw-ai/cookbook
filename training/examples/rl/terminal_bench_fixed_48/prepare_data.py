"""Register the bundled 48-task curriculum and pinned Terminal-Bench 2.1."""

from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
import tempfile
from pathlib import Path

DEFAULT_TRAIN_DATASET = "terminal-bench-fixed-48"
DEFAULT_EVAL_DATASET = "terminal-bench@2.1"
DEFAULT_EVAL_SOURCE = "terminal-bench/terminal-bench-2-1@6"
EXPECTED_TRAIN_TASKS = 48
EXPECTED_EVAL_TASKS = 89
TASK_ARCHIVE_SHA256 = "bad2aac7f21cb4a104419642c12ec38d83534a3fa3a5bbcbf794d91ea4665b35"

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_TASK_ARCHIVE = EXAMPLE_DIR / "data" / "tasks.tar.gz"
DEFAULT_CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
DEFAULT_TASKS_DIR = (
    DEFAULT_CACHE_ROOT
    / "rllm"
    / "terminal_bench_fixed_48"
    / TASK_ARCHIVE_SHA256[:12]
    / "tasks"
)
DEFAULT_TASK_IDS = EXAMPLE_DIR / "data" / "task_ids.txt"
REQUIRED_TASK_FILES = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "tests/test.sh",
)


def ensure_bundled_tasks(archive_path: Path, tasks_dir: Path) -> Path:
    """Verify and safely extract the immutable task archive once."""
    if tasks_dir.is_dir():
        return tasks_dir
    if tasks_dir.exists():
        raise ValueError(f"Task destination exists but is not a directory: {tasks_dir}")

    with archive_path.open("rb") as archive_file:
        digest = hashlib.file_digest(archive_file, "sha256").hexdigest()
    if digest != TASK_ARCHIVE_SHA256:
        raise ValueError(
            f"{archive_path} has SHA-256 {digest}; expected {TASK_ARCHIVE_SHA256}"
        )

    tasks_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".terminal-bench-fixed-48-", dir=tasks_dir.parent
    ) as temporary_dir:
        extraction_root = Path(temporary_dir)
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if (
                    not parts
                    or parts[0] != "tasks"
                    or Path(member.name).is_absolute()
                    or ".." in parts
                    or member.issym()
                    or member.islnk()
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(
                        f"Unsafe member in bundled task archive: {member.name!r}"
                    )
            # Keep the explicit checks above for clear errors, then apply
            # Python's restrictive data filter as a second safety boundary.
            archive.extractall(extraction_root, filter="data")

        extracted_tasks = extraction_root / "tasks"
        if not extracted_tasks.is_dir():
            raise ValueError(f"{archive_path} does not contain a tasks/ directory")
        try:
            extracted_tasks.rename(tasks_dir)
        except OSError:
            # Another process may have populated the same content-addressed cache.
            if not tasks_dir.is_dir():
                raise

    return tasks_dir


def load_task_ids(task_ids_path: Path, tasks_dir: Path) -> list[str]:
    """Read and validate the ordered curriculum manifest."""
    task_ids = [
        line.strip() for line in task_ids_path.read_text().splitlines() if line.strip()
    ]

    if len(task_ids) != EXPECTED_TRAIN_TASKS:
        raise ValueError(
            f"{task_ids_path} lists {len(task_ids)} tasks; "
            f"expected {EXPECTED_TRAIN_TASKS}"
        )
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"{task_ids_path} contains duplicate task IDs")

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
    task_ids_path: Path,
    tasks_dir: Path,
) -> list[dict]:
    """Register the fixed Harbor tasks without changing their order."""
    from rllm.data import DatasetRegistry
    from rllm.integrations.harbor.dataset_loader import harbor_task_to_row

    task_ids = load_task_ids(task_ids_path, tasks_dir)
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
        source="cookbook:terminal_bench_fixed_48/data/tasks.tar.gz",
        description="Fixed 48-task OpenCode Terminal-Bench curriculum",
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
        default=os.environ.get("TB_TRAIN_DATASET", DEFAULT_TRAIN_DATASET),
    )
    parser.add_argument(
        "--train-split", default=os.environ.get("TB_TRAIN_SPLIT", "train")
    )
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--task-archive", type=Path, default=DEFAULT_TASK_ARCHIVE)
    parser.add_argument("--task-ids", type=Path, default=DEFAULT_TASK_IDS)
    parser.add_argument(
        "--eval-dataset",
        default=os.environ.get("TB_VAL_DATASET", DEFAULT_EVAL_DATASET),
    )
    parser.add_argument(
        "--eval-split", default=os.environ.get("TB_VAL_SPLIT", "default")
    )
    parser.add_argument(
        "--eval-source",
        default=os.environ.get("TB_VAL_SOURCE", DEFAULT_EVAL_SOURCE),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Register only the bundled train split (local validation only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_bundled_tasks(args.task_archive, args.tasks_dir)
    train_rows = register_train_dataset(
        dataset_name=args.train_dataset,
        split=args.train_split,
        task_ids_path=args.task_ids,
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
