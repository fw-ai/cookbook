"""Generate immutable HealthBench Professional tasks in Harbor format."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sysconfig
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_ID = "openai/healthbench-professional"
DATASET_REVISION = "349962fd46dd02343a0d8a606491baf59154ea1a"
DATASET_FILENAME = "healthbench_professional_eval.jsonl"
DATASET_SHA256 = "d44b08e6e952e04c945e2c406f02533d9e7a989a84e35820ee7efdff20c9e4e2"
EXPECTED_ROWS = 525
INSTRUCTION_SCHEMA_VERSION = 1
SLICE_FIELDS = ("use_case", "type", "difficulty", "specialty")

_SOURCE_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"
_INSTALLED_TEMPLATE_DIR = (
    Path(sysconfig.get_path("data"))
    / "share"
    / "healthbench-professional"
    / "templates"
)
DEFAULT_TEMPLATE_DIR = (
    _SOURCE_TEMPLATE_DIR if _SOURCE_TEMPLATE_DIR.is_dir() else _INSTALLED_TEMPLATE_DIR
)
DEFAULT_OUTPUT_DIR = Path("datasets") / "healthbench-professional"


@dataclass(frozen=True)
class HealthBenchTask:
    """The only fields needed to prompt and grade one public benchmark row."""

    dataset_id: str
    messages: list[dict[str, Any]]
    rubric_items: list[dict[str, Any]]
    slices: dict[str, str]
    dataset_index: int = 0

    @classmethod
    def from_record(
        cls, record: Mapping[str, Any], *, dataset_index: int = 0
    ) -> HealthBenchTask:
        try:
            dataset_id = record["id"]
            conversation = record["conversation"]
            rubric_items = record["rubric_items"]
        except KeyError as exc:
            raise ValueError(
                f"HealthBench row is missing required field {exc.args[0]!r}"
            ) from exc

        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("HealthBench row id must be a non-empty string")
        if (
            isinstance(dataset_index, bool)
            or not isinstance(dataset_index, int)
            or dataset_index < 0
        ):
            raise ValueError("HealthBench dataset index must be a non-negative integer")
        if not isinstance(conversation, Mapping):
            raise ValueError(
                f"HealthBench row {dataset_id!r} conversation must be an object"
            )

        messages = conversation.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"HealthBench row {dataset_id!r} must have conversation messages"
            )
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ValueError(
                    f"HealthBench row {dataset_id!r} message {index} must be an object"
                )
            if not isinstance(message.get("role"), str) or not isinstance(
                message.get("content"), str
            ):
                raise ValueError(
                    f"HealthBench row {dataset_id!r} message {index} requires string role and content"
                )
        if messages[-1]["role"] != "user":
            raise ValueError(
                f"HealthBench row {dataset_id!r} conversation must end in a user message"
            )

        if not isinstance(rubric_items, list) or not rubric_items:
            raise ValueError(f"HealthBench row {dataset_id!r} must have rubric items")
        for index, rubric in enumerate(rubric_items):
            if not isinstance(rubric, Mapping):
                raise ValueError(
                    f"HealthBench row {dataset_id!r} rubric {index} must be an object"
                )
            if not isinstance(rubric.get("criterion_text"), str):
                raise ValueError(
                    f"HealthBench row {dataset_id!r} rubric {index} requires criterion_text"
                )
            points = rubric.get("points")
            if isinstance(points, bool) or not isinstance(points, (int, float)):
                raise ValueError(
                    f"HealthBench row {dataset_id!r} rubric {index} requires numeric points"
                )
        if (
            sum(rubric["points"] for rubric in rubric_items if rubric["points"] > 0)
            <= 0
        ):
            raise ValueError(
                f"HealthBench row {dataset_id!r} requires positive-point rubric items"
            )

        slices: dict[str, str] = {}
        for field in SLICE_FIELDS:
            value = record.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"HealthBench row {dataset_id!r} requires non-empty {field}"
                )
            slices[field] = value

        # Copy only the benchmark inputs and grading dependencies. In particular,
        # physician_response and canary_string never enter generated tasks.
        return cls(
            dataset_id=dataset_id,
            messages=copy.deepcopy(messages),
            rubric_items=copy.deepcopy(rubric_items),
            slices=slices,
            dataset_index=dataset_index,
        )

    def instruction_payload(self) -> dict[str, Any]:
        return {
            "schema_version": INSTRUCTION_SCHEMA_VERSION,
            "messages": copy.deepcopy(self.messages),
        }

    def source_payload(self) -> dict[str, Any]:
        return {
            "schema_version": INSTRUCTION_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "dataset_repo": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_index": self.dataset_index,
            "messages": copy.deepcopy(self.messages),
            "rubric_items": copy.deepcopy(self.rubric_items),
            **self.slices,
        }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset_file(path: Path, *, expected_sha256: str = DATASET_SHA256) -> None:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"HealthBench dataset SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def download_dataset(*, cache_dir: Path | None = None) -> Path:
    """Download the exact public dataset artifact from its immutable revision."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised by integration setup
        raise RuntimeError(
            "huggingface-hub is required to download HealthBench Professional"
        ) from exc

    path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=DATASET_FILENAME,
            revision=DATASET_REVISION,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    verify_dataset_file(path)
    return path


def parse_records(lines: Iterable[str]) -> list[HealthBenchTask]:
    """Parse JSONL rows into sanitized tasks without performing network I/O."""

    tasks: list[HealthBenchTask] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on HealthBench line {line_number}") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"HealthBench line {line_number} must contain an object")
        tasks.append(HealthBenchTask.from_record(record, dataset_index=len(tasks)))
    return tasks


def load_canonical_records(path: Path) -> list[HealthBenchTask]:
    """Load the pinned artifact, enforcing both byte identity and row count."""

    verify_dataset_file(path)
    with path.open(encoding="utf-8") as handle:
        tasks = parse_records(handle)
    if len(tasks) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} HealthBench rows, found {len(tasks)}"
        )
    return tasks


def _task_id_slug(dataset_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_id).strip("._-")
    if not slug:
        raise ValueError(
            f"Cannot derive a task directory from dataset id {dataset_id!r}"
        )
    return slug


def _task_directory_name(dataset_id: str) -> str:
    return f"healthbench_professional_{_task_id_slug(dataset_id)}"


class HealthBenchProfessionalAdapter:
    """Write sanitized HealthBench rows into Harbor task directories."""

    def __init__(self, task_dir: Path, *, template_dir: Path | None = None) -> None:
        self.task_dir = Path(task_dir)
        self.template_dir = Path(template_dir) if template_dir is not None else None

    @staticmethod
    def _template_replacements(task: HealthBenchTask) -> dict[str, str]:
        task_name = _task_id_slug(task.dataset_id)
        question_sha256 = hashlib.sha256(
            task.messages[-1]["content"].encode("utf-8")
        ).hexdigest()

        # Sentinels occur inside TOML basic strings. JSON string escaping (minus
        # the surrounding quotes) is compatible for the values used here.
        def escaped(value: object) -> str:
            return json.dumps(str(value), ensure_ascii=False)[1:-1]

        return {
            "__TASK_NAME__": escaped(task_name),
            "__DATASET_ID__": escaped(task.dataset_id),
            "__DATASET_INDEX__": escaped(task.dataset_index),
            "__QUESTION_SHA256__": escaped(question_sha256),
            "__USE_CASE__": escaped(task.slices["use_case"]),
            "__INTERACTION_TYPE__": escaped(task.slices["type"]),
            "__DIFFICULTY__": escaped(task.slices["difficulty"]),
            "__SPECIALTY__": escaped(task.slices["specialty"]),
        }

    @staticmethod
    def _render_template_files(
        output_dir: Path, replacements: Mapping[str, str]
    ) -> None:
        sentinel_pattern = re.compile(r"__[A-Z0-9_]+__")
        for path in output_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for sentinel, value in replacements.items():
                content = content.replace(sentinel, value)
            remaining = sorted(set(sentinel_pattern.findall(content)))
            if remaining:
                raise ValueError(
                    f"Unrendered template sentinels in {path}: {remaining}"
                )
            path.write_text(content, encoding="utf-8")

    @staticmethod
    def _copy_verifier_modules(tests_dir: Path) -> None:
        package_dir = Path(__file__).resolve().parent
        for filename in ("scoring.py", "judge.py", "trajectory.py"):
            source = package_dir / filename
            if not source.is_file():
                raise FileNotFoundError(
                    f"Required HealthBench verifier module not found: {source}"
                )
            shutil.copy2(source, tests_dir / filename)

    def generate_task(self, task: HealthBenchTask, *, overwrite: bool = False) -> Path:
        output_dir = self.task_dir / _task_directory_name(task.dataset_id)
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"HealthBench task already exists: {output_dir}; pass overwrite=True to replace it"
                )
            shutil.rmtree(output_dir)
        if self.template_dir is not None:
            if not self.template_dir.is_dir():
                raise FileNotFoundError(
                    f"HealthBench Harbor template not found: {self.template_dir}"
                )
            shutil.copytree(self.template_dir, output_dir, dirs_exist_ok=True)
            self._render_template_files(output_dir, self._template_replacements(task))
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        instruction = (
            json.dumps(task.instruction_payload(), ensure_ascii=False, indent=2) + "\n"
        )
        (output_dir / "instruction.md").write_text(instruction, encoding="utf-8")

        tests_dir = output_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        self._copy_verifier_modules(tests_dir)
        source = json.dumps(task.source_payload(), ensure_ascii=False, indent=2) + "\n"
        (tests_dir / "source.json").write_text(source, encoding="utf-8")
        return output_dir

    def _remove_generated_tasks(self) -> None:
        """Remove every task previously generated in this dataset directory."""
        if not self.task_dir.is_dir():
            return
        for path in self.task_dir.iterdir():
            if not path.name.startswith("healthbench_professional_"):
                continue
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    def generate_tasks(
        self,
        tasks: Sequence[HealthBenchTask],
        *,
        limit: int | None = None,
        task_ids: Sequence[str] | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("limit must be a positive integer")

        seen_ids: set[str] = set()
        for task in tasks:
            if task.dataset_id in seen_ids:
                raise ValueError(
                    f"Duplicate HealthBench dataset id {task.dataset_id!r}"
                )
            seen_ids.add(task.dataset_id)

        selected = list(tasks)
        if task_ids is not None:
            requested = list(dict.fromkeys(task_ids))
            missing = sorted(set(requested) - seen_ids)
            if missing:
                raise ValueError(f"Unknown HealthBench task ids: {missing}")
            requested_set = set(requested)
            selected = [task for task in selected if task.dataset_id in requested_set]
        if limit is not None:
            selected = selected[:limit]

        # ``prepare --overwrite`` replaces the prepared dataset as a whole. In
        # particular, a limited smoke preparation must not leave older task
        # directories behind for a later unbounded Harbor run to discover.
        if overwrite:
            self._remove_generated_tasks()

        generated: list[Path] = []
        for task in selected:
            generated.append(self.generate_task(task, overwrite=overwrite))
        return generated

    def generate_canonical_dataset(
        self,
        dataset_path: Path | None = None,
        *,
        limit: int | None = None,
        task_ids: Sequence[str] | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        path = dataset_path if dataset_path is not None else download_dataset()
        return self.generate_tasks(
            load_canonical_records(Path(path)),
            limit=limit,
            task_ids=task_ids,
            overwrite=overwrite,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    adapter = HealthBenchProfessionalAdapter(
        args.output_dir, template_dir=args.template_dir
    )
    generated = adapter.generate_canonical_dataset(
        args.dataset_file,
        limit=args.limit,
        task_ids=args.task_ids,
        overwrite=args.overwrite,
    )
    print(
        f"Generated {len(generated)} HealthBench Professional tasks in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
