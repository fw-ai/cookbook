"""Pinned DABstep membership and reference-style adaptive task selection."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.examples.rl.harbor.prepare_opencode_tasks import task_output_path

_UNSIGNED_NUMERIC_PATTERNS = (
    r"r'(\d*\.\d+|\d+\.?\d*)%?'",
    r'r"(\d*\.\d+|\d+\.?\d*)%?"',
)
_SIGNED_NUMERIC_PATTERNS = (
    r"r'([+-]?(?:\d*\.\d+|\d+\.?\d*))%?'",
    r'r"([+-]?(?:\d*\.\d+|\d+\.?\d*))%?"',
)


@dataclass(frozen=True)
class DABstepManifest:
    train_tasks: tuple[str, ...]
    holdout_tasks: tuple[str, ...]
    reference_step_tasks: tuple[str, ...]
    profile: dict[str, dict[str, Any]]
    content_sha256: dict[str, str]
    source: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> DABstepManifest:
        source_path = Path(path).expanduser()
        document = json.loads(source_path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1:
            raise ValueError("DABstep manifest schema_version must be 1")
        tasks = document.get("tasks") or {}
        manifest = cls(
            train_tasks=tuple(tasks.get("train") or ()),
            holdout_tasks=tuple(tasks.get("holdout") or ()),
            reference_step_tasks=tuple(tasks.get("reference_step0") or ()),
            profile=dict(document.get("profile") or {}),
            content_sha256=dict(document.get("content_sha256") or {}),
            source=dict(document.get("source") or {}),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if len(self.train_tasks) != 67:
            raise ValueError(
                f"DABstep manifest must contain 67 train tasks, got {len(self.train_tasks)}"
            )
        if len(self.holdout_tasks) != 8:
            raise ValueError(
                f"DABstep manifest must contain 8 holdout tasks, got {len(self.holdout_tasks)}"
            )
        if len(self.reference_step_tasks) != 6:
            raise ValueError(
                "DABstep manifest must contain the six reference step-0 tasks"
            )
        train = set(self.train_tasks)
        holdout = set(self.holdout_tasks)
        if len(train) != len(self.train_tasks) or len(holdout) != len(
            self.holdout_tasks
        ):
            raise ValueError("DABstep task membership contains duplicates")
        if train & holdout:
            raise ValueError("DABstep train and holdout membership overlap")
        if not set(self.reference_step_tasks) <= train:
            raise ValueError("DABstep reference step tasks must be training tasks")
        selected = train | holdout
        if not selected <= self.profile.keys():
            missing = sorted(selected - self.profile.keys())
            raise ValueError(f"DABstep profile is missing tasks: {missing}")
        if not selected <= self.content_sha256.keys():
            missing = sorted(selected - self.content_sha256.keys())
            raise ValueError(f"DABstep content hashes are missing tasks: {missing}")
        for name in selected:
            digest = self.content_sha256[name]
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(f"DABstep task {name!r} has an invalid SHA-256 digest")

    def verify_task_root(self, root: str | Path) -> None:
        task_root = Path(root).expanduser().resolve()
        for name in (*self.train_tasks, *self.holdout_tasks):
            task_path = task_root / name
            if not task_path.is_dir():
                raise FileNotFoundError(
                    f"DABstep task directory is missing: {task_path}"
                )
            actual = _directory_sha256(task_path)
            expected = self.content_sha256[name]
            if actual != expected:
                raise ValueError(
                    f"DABstep task content drift for {name}: "
                    f"expected {expected}, got {actual}"
                )
            _validate_numeric_scorer(task_path)


def _scorer_path(task: Path) -> Path:
    scorer = task_output_path(task, "tests", "scorer.py")
    if not scorer.is_file():
        raise ValueError(f"DABstep task {task.name!r} has no tests/scorer.py")
    return scorer


def _validate_numeric_scorer(task: Path) -> None:
    source = _scorer_path(task).read_text(encoding="utf-8")
    if any(pattern in source for pattern in _UNSIGNED_NUMERIC_PATTERNS):
        raise ValueError(
            f"DABstep task {task.name!r} has a sign-insensitive numeric scorer; "
            "prepare it with prepare_dabstep_tasks and refresh the manifest"
        )


def make_numeric_scorer_sign_sensitive(task: Path) -> None:
    """Patch the known upstream DABstep numeric regex."""
    scorer = _scorer_path(task)
    source = scorer.read_text(encoding="utf-8")
    unsigned_matches = sum(
        source.count(pattern) for pattern in _UNSIGNED_NUMERIC_PATTERNS
    )
    signed_matches = sum(source.count(pattern) for pattern in _SIGNED_NUMERIC_PATTERNS)
    if unsigned_matches == 0 and signed_matches == 1:
        return
    if unsigned_matches != 1 or signed_matches:
        raise ValueError(
            f"DABstep task {task.name!r} does not contain exactly one known "
            "unsigned numeric scorer pattern"
        )
    for unsigned, signed in zip(
        _UNSIGNED_NUMERIC_PATTERNS,
        _SIGNED_NUMERIC_PATTERNS,
        strict=True,
    ):
        if unsigned in source:
            source = source.replace(unsigned, signed, 1)
            break
    scorer.write_text(source, encoding="utf-8")


def _directory_sha256(path: Path) -> str:
    """Hash a task directory with the algorithm used to create the manifest."""
    try:
        from dirhash import dirhash
    except ImportError as exc:
        raise RuntimeError(
            "DABstep manifest verification requires the example-only "
            "'dirhash' dependency; install it alongside Harbor"
        ) from exc
    return str(dirhash(str(path), "sha256"))


@dataclass
class _TaskStats:
    attempts: int = 0
    solved: int = 0

    @property
    def pass_rate(self) -> float:
        return (self.solved + 1.0) / (self.attempts + 2.0)


def usable_group_probability(pass_rate: float, group_size: int) -> float:
    return 1.0 - pass_rate**group_size - (1.0 - pass_rate) ** group_size


class AdaptiveTaskSelector:
    """Select distinct tasks per global batch using the reference yield weight."""

    def __init__(
        self,
        *,
        task_rows: list[dict[str, Any]],
        profile: dict[str, dict[str, Any]],
        group_size: int,
        groups_per_batch: int,
        seed: int,
        solved_reward: float = 1.0,
    ) -> None:
        if group_size < 1 or groups_per_batch < 1:
            raise ValueError("group_size and groups_per_batch must be >= 1")
        by_name = {str(row.get("task_name")): row for row in task_rows}
        if len(by_name) != len(task_rows):
            raise ValueError(
                "adaptive selector task rows must have unique task_name values"
            )
        if groups_per_batch > len(by_name):
            raise ValueError("groups_per_batch cannot exceed the task pool")
        self._rows = [(name, by_name[name]) for name in sorted(by_name)]
        self._stats = {
            name: _TaskStats(
                attempts=int((profile.get(name) or {}).get("attempts") or 0),
                solved=int((profile.get(name) or {}).get("solved") or 0),
            )
            for name, _ in self._rows
        }
        self._group_size = group_size
        self._groups_per_batch = groups_per_batch
        self._solved_reward = solved_reward
        self._rng = random.Random(seed)
        self._selections: dict[int, list[tuple[str, dict[str, Any]]]] = {}
        self._observations: dict[int, dict[int, float | None]] = {}
        self._updated_groups: set[int] = set()
        self._lock = asyncio.Lock()

    async def row_for_group(self, group_index: int) -> dict[str, Any]:
        if group_index < 0:
            raise ValueError("group_index must be non-negative")
        batch_index, slot = divmod(group_index, self._groups_per_batch)
        async with self._lock:
            if batch_index not in self._selections:
                self._selections[batch_index] = self._sample_batch()
            return dict(self._selections[batch_index][slot][1])

    async def record(
        self,
        group_index: int,
        sample_index: int,
        reward: float | None,
    ) -> None:
        batch_index, slot = divmod(group_index, self._groups_per_batch)
        async with self._lock:
            selections = self._selections.get(batch_index)
            if selections is None:
                raise RuntimeError(f"DABstep batch {batch_index} was not selected")
            if not 0 <= sample_index < self._group_size:
                raise ValueError("sample_index is outside the DABstep group")
            values = self._observations.setdefault(group_index, {})
            values[sample_index] = None if reward is None else float(reward)
            if (
                len(values) == self._group_size
                and group_index not in self._updated_groups
            ):
                task_name = selections[slot][0]
                stats = self._stats[task_name]
                stats.attempts += len(values)
                stats.solved += sum(
                    value is not None and value >= self._solved_reward
                    for value in values.values()
                )
                self._updated_groups.add(group_index)

    def selected_names(self) -> dict[int, tuple[str, ...]]:
        return {
            batch: tuple(name for name, _ in selections)
            for batch, selections in self._selections.items()
        }

    def state_dict(self) -> dict[str, Any]:
        """Return the persistent selection state at a quiescent stage boundary."""

        return {
            "schema_version": 1,
            "task_names": [name for name, _ in self._rows],
            "group_size": self._group_size,
            "groups_per_batch": self._groups_per_batch,
            "solved_reward": self._solved_reward,
            "stats": {
                name: {"attempts": stats.attempts, "solved": stats.solved}
                for name, stats in self._stats.items()
            },
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore persistent state before any selection for a resumed stage."""

        if state.get("schema_version") != 1:
            raise ValueError("adaptive selector state schema_version must be 1")
        expected_names = [name for name, _ in self._rows]
        if state.get("task_names") != expected_names:
            raise ValueError("adaptive selector state task membership does not match")
        if state.get("group_size") != self._group_size:
            raise ValueError("adaptive selector state group_size does not match")
        if state.get("groups_per_batch") != self._groups_per_batch:
            raise ValueError("adaptive selector state groups_per_batch does not match")
        if state.get("solved_reward") != self._solved_reward:
            raise ValueError("adaptive selector state solved_reward does not match")

        raw_stats = state.get("stats")
        if not isinstance(raw_stats, dict) or set(raw_stats) != set(expected_names):
            raise ValueError(
                "adaptive selector state stats do not match task membership"
            )
        restored: dict[str, _TaskStats] = {}
        for name in expected_names:
            values = raw_stats[name]
            if not isinstance(values, dict):
                raise ValueError(f"adaptive selector stats for {name!r} are invalid")
            attempts = values.get("attempts")
            solved = values.get("solved")
            if (
                not isinstance(attempts, int)
                or isinstance(attempts, bool)
                or not isinstance(solved, int)
                or isinstance(solved, bool)
                or attempts < 0
                or solved < 0
                or solved > attempts
            ):
                raise ValueError(f"adaptive selector stats for {name!r} are invalid")
            restored[name] = _TaskStats(attempts=attempts, solved=solved)

        def as_tuple(value: Any) -> Any:
            if isinstance(value, list):
                return tuple(as_tuple(item) for item in value)
            return value

        try:
            self._rng.setstate(as_tuple(state["rng_state"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("adaptive selector RNG state is invalid") from error
        self._stats = restored
        self._selections.clear()
        self._observations.clear()
        self._updated_groups.clear()

    def _sample_batch(self) -> list[tuple[str, dict[str, Any]]]:
        remaining = [
            (
                name,
                row,
                usable_group_probability(self._stats[name].pass_rate, self._group_size),
            )
            for name, row in self._rows
        ]
        selected: list[tuple[str, dict[str, Any]]] = []
        for _ in range(self._groups_per_batch):
            total = sum(weight for _, _, weight in remaining)
            if total <= 0.0:
                choice = self._rng.randrange(len(remaining))
            else:
                threshold = self._rng.random() * total
                cumulative = 0.0
                choice = len(remaining) - 1
                for index, (_, _, weight) in enumerate(remaining):
                    cumulative += weight
                    if cumulative >= threshold:
                        choice = index
                        break
            name, row, _ = remaining.pop(choice)
            selected.append((name, row))
        return selected


def rows_for_tasks(
    rows: list[dict[str, Any]],
    task_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    by_name = {str(row.get("task_name")): row for row in rows}
    missing = [name for name in task_names if name not in by_name]
    if missing:
        raise ValueError(f"Harbor dataset is missing DABstep tasks: {missing}")
    return [dict(by_name[name]) for name in task_names]


__all__ = [
    "AdaptiveTaskSelector",
    "DABstepManifest",
    "make_numeric_scorer_sign_sensitive",
    "rows_for_tasks",
    "usable_group_probability",
]
