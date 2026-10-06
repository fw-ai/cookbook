"""Durable spill of accepted PromptGroups for crash-safe async RL resume.

When ``ASYNC_RL_TRAIN_SPILL_DIR`` is set, each accepted group is pickled under
``{dir}/batch-{id}/group-{sequence:06d}.pkl`` and indexed by ``manifest.json``.
On restart the producer reloads a partial/full batch so rollouts are not redone.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable

from training.utils.rl.losses import PromptGroup

logger = logging.getLogger(__name__)

SPILL_ENV = "ASYNC_RL_TRAIN_SPILL_DIR"
_MANIFEST = "manifest.json"


@dataclass(frozen=True, slots=True)
class SpilledGroup:
    sequence: int
    source_token: Hashable
    submit_version: int
    group: PromptGroup


@dataclass(frozen=True, slots=True)
class SpillBatch:
    batch_id: int
    target_groups: int
    chunk_targets: tuple[int, ...]
    groups: tuple[SpilledGroup, ...]

    @property
    def realized_groups(self) -> int:
        return len(self.groups)

    @property
    def complete(self) -> bool:
        return self.realized_groups >= self.target_groups


def spill_root_from_env() -> Path | None:
    raw = (os.environ.get(SPILL_ENV) or "").strip()
    if not raw:
        return None
    return Path(raw)


def batch_dir(root: Path, batch_id: int) -> Path:
    return root / f"batch-{batch_id}"


def write_group(
    root: Path,
    *,
    batch_id: int,
    sequence: int,
    group: PromptGroup,
    source_token: Hashable,
    submit_version: int,
    target_groups: int,
    chunk_targets: tuple[int, ...],
) -> Path:
    """Atomically persist one accepted group and refresh the batch manifest."""

    bdir = batch_dir(root, batch_id)
    bdir.mkdir(parents=True, exist_ok=True)
    fname = f"group-{sequence:06d}.pkl"
    path = bdir / fname
    payload = {
        "sequence": sequence,
        "source_token": source_token,
        "submit_version": submit_version,
        "group": group,
    }
    _atomic_pickle(path, payload)

    manifest = _read_manifest(bdir) or {
        "batch_id": batch_id,
        "target_groups": target_groups,
        "chunk_targets": list(chunk_targets),
        "groups": [],
    }
    if int(manifest.get("batch_id", batch_id)) != batch_id:
        raise RuntimeError(
            f"spill manifest batch_id mismatch: {manifest.get('batch_id')} != {batch_id}"
        )
    manifest["target_groups"] = int(target_groups)
    manifest["chunk_targets"] = list(chunk_targets)
    groups = [
        g for g in manifest.get("groups", []) if int(g["sequence"]) != sequence
    ]
    groups.append({"sequence": sequence, "file": fname})
    groups.sort(key=lambda g: int(g["sequence"]))
    manifest["groups"] = groups
    _atomic_json(bdir / _MANIFEST, manifest)
    return path


def load_batch(root: Path, batch_id: int) -> SpillBatch | None:
    bdir = batch_dir(root, batch_id)
    manifest = _read_manifest(bdir)
    if manifest is None:
        return None
    loaded: list[SpilledGroup] = []
    for entry in manifest.get("groups", []):
        sequence = int(entry["sequence"])
        path = bdir / entry["file"]
        if not path.is_file():
            logger.warning("spill missing group file %s; skipping", path)
            continue
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        loaded.append(
            SpilledGroup(
                sequence=int(payload["sequence"]),
                source_token=payload["source_token"],
                submit_version=int(payload.get("submit_version", 0)),
                group=payload["group"],
            )
        )
    if not loaded:
        return None
    loaded.sort(key=lambda g: g.sequence)
    return SpillBatch(
        batch_id=int(manifest["batch_id"]),
        target_groups=int(manifest["target_groups"]),
        chunk_targets=tuple(int(x) for x in manifest["chunk_targets"]),
        groups=tuple(loaded),
    )


def clear_batch(root: Path, batch_id: int) -> None:
    bdir = batch_dir(root, batch_id)
    if bdir.is_dir():
        shutil.rmtree(bdir)
        logger.info("cleared train spill batch-%d", batch_id)


def _read_manifest(bdir: Path) -> dict[str, Any] | None:
    path = bdir / _MANIFEST
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


__all__ = [
    "SPILL_ENV",
    "SpillBatch",
    "SpilledGroup",
    "batch_dir",
    "clear_batch",
    "load_batch",
    "spill_root_from_env",
    "write_group",
]
