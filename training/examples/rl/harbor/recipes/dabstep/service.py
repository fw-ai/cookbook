"""First-class DABstep rows and progressive Harbor environment preparation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from training.examples.rl.harbor.pi.constants import (
    PI_HARBOR_IMPORT_PATH,
    PINNED_PI_VERSION,
)
from training.examples.rl.harbor.pi.prepare_tasks import prepare as prepare_pi_tasks
from training.examples.rl.harbor.recipes.dabstep.manifest import (
    make_numeric_scorer_sign_sensitive,
)
from training.examples.rl.harbor.tito.e2b_templates import prebuild_e2b_templates
from training.examples.rl.harbor.tito.trial import (
    DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
    load_harbor_rows,
)

EXPECTED_DEFAULT_SPLIT_TASKS = 450
DEFAULT_WAVE_SIZE = 64
_TASK_INDEX_FIELD = "_dabstep_task_index"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_sha256(path: Path) -> str:
    try:
        from dirhash import dirhash
    except ImportError as exc:
        raise RuntimeError(
            "DABstep service preparation requires the example-only 'dirhash' dependency"
        ) from exc
    return str(dirhash(str(path), "sha256"))


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class FrozenDABstepDataset:
    task_root: Path
    task_names: tuple[str, ...]
    evaluation_tasks: tuple[str, ...]
    content_sha256: dict[str, str]
    scorer_sha256: dict[str, str]
    manifest_path: Path
    manifest_sha256: str

    def rollout_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "id": f"dabstep-{index:04d}-{name}",
                "task_name": name,
                _TASK_INDEX_FIELD: index,
            }
            for index, name in enumerate(self.task_names)
        ]

    def evaluation_rows(self) -> list[dict[str, Any]]:
        index_by_name = {name: index for index, name in enumerate(self.task_names)}
        return [
            {
                "id": f"dabstep-eval-{name}",
                "task_name": name,
                _TASK_INDEX_FIELD: index_by_name[name],
            }
            for name in self.evaluation_tasks
        ]


def freeze_default_split(
    task_root: str | Path,
    *,
    manifest_path: str | Path,
    evaluation_tasks: tuple[str, ...] | None = None,
    hash_workers: int = 8,
) -> FrozenDABstepDataset:
    """Freeze and verify the complete materialized DABstep default split."""

    root = Path(task_root).expanduser().resolve()
    output = Path(manifest_path).expanduser().resolve()
    snapshot_path = root / ".fw-ai-package-snapshot.json"
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"DABstep package snapshot is missing: {snapshot_path}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    task_records = list(snapshot.get("tasks") or ())
    names = tuple(str(record.get("name")) for record in task_records)
    if len(names) != EXPECTED_DEFAULT_SPLIT_TASKS or len(names) != len(set(names)):
        raise ValueError(
            "DABstep default split must contain exactly "
            f"{EXPECTED_DEFAULT_SPLIT_TASKS} unique tasks"
        )
    dataset = snapshot.get("dataset") or {}
    selection = snapshot.get("selection") or {}
    if (
        int(dataset.get("source_task_count") or 0) != EXPECTED_DEFAULT_SPLIT_TASKS
        or int(selection.get("count") or 0) != EXPECTED_DEFAULT_SPLIT_TASKS
        or dataset.get("parent_status") != "match"
        or dataset.get("stored_ref") != dataset.get("recomputed_ref")
    ):
        raise ValueError("DABstep package snapshot does not certify the default split")
    actual_names = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "task.toml").is_file()
    }
    if actual_names != set(names):
        raise ValueError(
            "materialized DABstep task membership differs from its snapshot"
        )
    resolved_evaluation_tasks = (
        names[:4] if evaluation_tasks is None else evaluation_tasks
    )
    if (
        len(resolved_evaluation_tasks) != 4
        or len(set(resolved_evaluation_tasks)) != 4
        or not set(resolved_evaluation_tasks) <= set(names)
    ):
        raise ValueError("DABstep service evaluation requires four dataset tasks")
    if hash_workers < 1:
        raise ValueError("hash_workers must be positive")

    task_paths = [root / name for name in names]
    with ThreadPoolExecutor(max_workers=hash_workers) as executor:
        content_hashes = list(executor.map(_directory_sha256, task_paths))
    content_sha256 = dict(zip(names, content_hashes, strict=True))
    scorer_sha256 = {
        name: _sha256(root / name / "tests" / "scorer.py") for name in names
    }
    document = {
        "schema_version": 1,
        "dataset": "adyen/DABstep",
        "source_snapshot_sha256": _sha256(snapshot_path),
        "source_ref": dataset["stored_ref"],
        "task_count": len(names),
        "task_names": list(names),
        "task_order_sha256": hashlib.sha256(
            ("\n".join(names) + "\n").encode()
        ).hexdigest(),
        "evaluation_tasks": list(resolved_evaluation_tasks),
        "content_sha256": content_sha256,
        "scorer_sha256": scorer_sha256,
    }
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != document:
            raise ValueError(f"frozen DABstep manifest drifted: {output}")
    else:
        _atomic_json(output, document)
    return FrozenDABstepDataset(
        task_root=root,
        task_names=names,
        evaluation_tasks=resolved_evaluation_tasks,
        content_sha256=content_sha256,
        scorer_sha256=scorer_sha256,
        manifest_path=output,
        manifest_sha256=_sha256(output),
    )


def shuffle_dataset_for_run(
    dataset: FrozenDABstepDataset,
    *,
    seed: int,
    order_path: str | Path,
) -> FrozenDABstepDataset:
    """Persist one seeded run order and return the dataset in that order."""

    names = list(dataset.task_names)
    random.Random(seed).shuffle(names)
    ordered_names = tuple(names)
    document = {
        "schema_version": 1,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "shuffle_seed": seed,
        "task_count": len(ordered_names),
        "task_names": list(ordered_names),
        "task_order_sha256": hashlib.sha256(
            ("\n".join(ordered_names) + "\n").encode()
        ).hexdigest(),
    }
    output = Path(order_path).expanduser().resolve()
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != document:
            raise ValueError(f"DABstep run order drifted: {output}")
    else:
        _atomic_json(output, document)
    return replace(dataset, task_names=ordered_names)


class ProgressiveDABstepTasks:
    """Prepare and qualify one ordered task wave before its first rollout."""

    def __init__(
        self,
        dataset: FrozenDABstepDataset,
        *,
        run_root: str | Path,
        start_task_index: int = 0,
        wave_size: int = DEFAULT_WAVE_SIZE,
        template_concurrency: int = 8,
        template_timeout_seconds: float = 1_800.0,
        context_limit: int,
        output_limit: int,
        trial_config: Any | None = None,
        tool_timeout_seconds: int = DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
    ) -> None:
        if wave_size < 1:
            raise ValueError("wave_size must be positive")
        if not 0 <= start_task_index <= len(dataset.task_names):
            raise ValueError("start_task_index is outside the frozen dataset")
        self.dataset = dataset
        self.run_root = Path(run_root).expanduser().resolve()
        self.wave_size = wave_size
        self.template_concurrency = template_concurrency
        self.template_timeout_seconds = template_timeout_seconds
        self.context_limit = context_limit
        self.output_limit = output_limit
        self.trial_config = trial_config
        self.tool_timeout_seconds = tool_timeout_seconds
        self._rows: dict[int, dict[str, Any]] = {}
        self._ready_waves: set[int] = set()
        self._train_wave = start_task_index // wave_size - 1
        self._lock: asyncio.Lock | None = None

    async def row_for(
        self,
        placeholder: dict[str, Any],
        *,
        evaluation: bool,
    ) -> dict[str, Any]:
        index = placeholder.get(_TASK_INDEX_FIELD)
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("DABstep rollout row has no integer task index")
        if not 0 <= index < len(self.dataset.task_names):
            raise ValueError("DABstep rollout task index is outside the manifest")
        wave = index // self.wave_size
        lock = self._lock
        if lock is None:
            lock = self._lock = asyncio.Lock()
        async with lock:
            if not evaluation:
                if wave > self._train_wave + 1:
                    raise RuntimeError(
                        "DABstep producer crossed a progressive-build wave before "
                        "the prior wave was exhausted"
                    )
                self._train_wave = max(self._train_wave, wave)
            if wave not in self._ready_waves:
                await self._prepare_wave(wave)
        return dict(self._rows[index])

    async def _prepare_wave(self, wave: int) -> None:
        start = wave * self.wave_size
        stop = min(start + self.wave_size, len(self.dataset.task_names))
        names = self.dataset.task_names[start:stop]
        if not names:
            raise ValueError(f"DABstep wave {wave} is empty")
        wave_root = self.run_root / "prepared" / f"wave-{wave:03d}"
        wave_state = self.run_root / "waves" / f"wave-{wave:03d}.json"
        if not wave_root.exists():
            prepared = await asyncio.to_thread(
                prepare_pi_tasks,
                self.dataset.task_root,
                wave_root,
                PINNED_PI_VERSION,
                task_names=names,
            )
            await asyncio.to_thread(
                lambda: [make_numeric_scorer_sign_sensitive(task) for task in prepared]
            )
        rows = await asyncio.to_thread(
            load_harbor_rows,
            wave_root,
            task_names=list(names),
            n_tasks=None,
        )
        if [str(row["task_name"]) for row in rows] != list(names):
            raise RuntimeError(f"Harbor changed DABstep wave {wave} order")
        records = await prebuild_e2b_templates(
            rows,
            trials_dir=self.run_root / "template-trials" / f"wave-{wave:03d}",
            agent_import_path=PI_HARBOR_IMPORT_PATH,
            agent_version=PINNED_PI_VERSION,
            agent_provider="fireworks-tito",
            context_limit=self.context_limit,
            output_limit=self.output_limit,
            trial_config=self.trial_config,
            max_concurrency=self.template_concurrency,
            timeout_seconds=self.template_timeout_seconds,
            tool_timeout_seconds=self.tool_timeout_seconds,
        )
        prepared_hashes = await asyncio.gather(
            *(asyncio.to_thread(_directory_sha256, wave_root / name) for name in names)
        )
        scorer_hashes = {
            name: _sha256(wave_root / name / "tests" / "scorer.py") for name in names
        }
        document = {
            "schema_version": 1,
            "wave": wave,
            "start_task_index": start,
            "stop_task_index": stop,
            "task_names": list(names),
            "source_content_sha256": {
                name: self.dataset.content_sha256[name] for name in names
            },
            "prepared_content_sha256": dict(zip(names, prepared_hashes, strict=True)),
            "prepared_scorer_sha256": scorer_hashes,
            "templates": [
                {
                    "task_name": record.task_name,
                    "template_name": record.template_name,
                    "existed": record.existed,
                }
                for record in records
            ],
            "status": "ready",
        }
        if wave_state.exists():
            prior = json.loads(wave_state.read_text(encoding="utf-8"))
            comparable = dict(document)
            comparable["templates"] = [
                {**record, "existed": prior_record.get("existed", record["existed"])}
                for record, prior_record in zip(
                    document["templates"], prior.get("templates") or (), strict=True
                )
            ]
            if prior != comparable:
                raise ValueError(f"DABstep prepared wave drifted: {wave_state}")
        else:
            _atomic_json(wave_state, document)
        self._rows.update(
            {start + offset: dict(row) for offset, row in enumerate(rows)}
        )
        self._ready_waves.add(wave)


def make_progressive_rollout_factory(task_waves: ProgressiveDABstepTasks, base_factory):
    """Adapt placeholder DABstep rows without coupling the async-RL loop."""

    def factory(setup):
        base_rollout = base_factory(setup)

        async def rollout(sample_prompt, *, evaluation: bool = False, **context):
            row = await task_waves.row_for(sample_prompt, evaluation=evaluation)
            return await base_rollout(row, evaluation=evaluation, **context)

        rollout.aclose = base_rollout.aclose
        return rollout

    return factory


__all__ = [
    "DEFAULT_WAVE_SIZE",
    "EXPECTED_DEFAULT_SPLIT_TASKS",
    "FrozenDABstepDataset",
    "ProgressiveDABstepTasks",
    "freeze_default_split",
    "make_progressive_rollout_factory",
    "shuffle_dataset_for_run",
]
