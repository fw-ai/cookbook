"""Checkpoint, resume, and sampler weight-sync plumbing for cookbook loops.

Remote checkpoint rows are authoritative. The cookbook only persists the one
piece of client state the trainer cannot know: a mapping from trainer job and
checkpoint step to the next raw dataset row::

    {"trainer-job-id": {"10": 80, "20": 160}}

Recipes never choose LoRA/full or base/delta sampler formats. ``sync_weights``
delegates that choice to the SDK, while final promotable saves force a complete
export behind this boundary.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Protocol

import training.utils.fileio as fileio

DATALOADER_CURSOR_BASE_NAME = "dataloader.json"
# Compatibility for callers that imported the old constant.
DATALOADER_BASE_NAME = DATALOADER_CURSOR_BASE_NAME
DATALOADER_HISTORY_KEEP = 20

_RESUMABLE_TYPE_SUFFIXES = ("TRAINING", "TRAINING_LORA")
_SESSION_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")
_RUN_SCOPED_NAME_RE = re.compile(r"^run-[0-9a-f]{32}[:-](.+)$")
_CHECKPOINT_RESOURCE_RE = re.compile(
    r"(?:^|/)(?:rlorTrainerJobs|trainingSessions)/([^/]+)/checkpoints/([^/]+)$"
)
_STEP_NAME_RE = re.compile(r"^step-(\d+)$")
_PROMOTABLE_FRESHNESS_TOLERANCE_S = 10.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResumeInfo:
    """The resolved trainer step and raw-row cursor for a run."""

    step: int = 0
    row_cursor: int = 0
    source_job_id: str | None = None

    @property
    def data_consumed(self) -> int:
        """Backward-compatible name for :attr:`row_cursor`."""
        return self.row_cursor


class _CheckpointControlPlane(Protocol):
    def list_checkpoints(self, job_id: str, *, page_size: int = 200) -> list[dict]: ...

    def promote_checkpoint(
        self,
        *,
        name: str,
        output_model_id: str,
        base_model: str,
        hot_load_deployment_id: str | None = None,
    ) -> dict: ...


def validate_warm_start_config(
    *,
    warm_start_from_adapter: str | None,
    init_from_checkpoint: str | Mapping[str, Any] | None,
    lora_rank: int,
) -> None:
    """Validate cross-field constraints on weights-only warm starts."""
    if warm_start_from_adapter and init_from_checkpoint is not None:
        raise ValueError(
            "warm_start_from_adapter and init_from_checkpoint are mutually exclusive"
        )
    if warm_start_from_adapter and lora_rank == 0:
        raise ValueError(
            "warm_start_from_adapter requires lora_rank > 0. "
            "For full-param warm-start, set cfg.base_model to the promoted model "
            "resource name; the training session initializes from it directly."
        )


def _short_name(resource_name: str) -> str:
    return resource_name.rstrip("/").rsplit("/", 1)[-1]


def _logical_name(short: str) -> str:
    """Strip the trainer's sampler-session suffix from a checkpoint id."""
    return _SESSION_SUFFIX_RE.sub("", short)


def _public_logical_name(short: str) -> str:
    """Strip a public serverless run prefix and sampler-session suffix."""
    match = _RUN_SCOPED_NAME_RE.match(short)
    if match:
        short = match.group(1)
    return _logical_name(short)


def _logical_name_for_run(short: str, current_run_id: str | None = None) -> str:
    if current_run_id:
        for separator in ("-", ":"):
            prefix = f"{current_run_id}{separator}"
            if short.startswith(prefix):
                return _logical_name(short[len(prefix) :])
    return _public_logical_name(short)


def _is_run_scoped_name(short: str) -> bool:
    return _RUN_SCOPED_NAME_RE.match(short) is not None


def _belongs_to_run(short: str, current_run_id: str | None = None) -> bool:
    if current_run_id is None:
        return not _is_run_scoped_name(short)
    return short.startswith(f"{current_run_id}-") or short.startswith(
        f"{current_run_id}:"
    )


def _is_resumable_row(row: Mapping[str, Any]) -> bool:
    checkpoint_type = row.get("checkpointType", "") or ""
    return any(checkpoint_type.endswith(suffix) for suffix in _RESUMABLE_TYPE_SUFFIXES)


def _parse_iso_time(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _newest_first(rows: list[dict]) -> list[dict]:
    """Sort CP rows by parsed ``createTime`` rather than RFC3339 text."""
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(
        rows,
        key=lambda row: _parse_iso_time(row.get("createTime")) or epoch,
        reverse=True,
    )


def _checkpoint_step(name: str) -> int:
    match = _STEP_NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Checkpoint {name!r} does not use the cookbook step-N naming contract"
        )
    return int(match.group(1))


def _checkpoint_ref(
    spec: str | Mapping[str, Any],
    *,
    default_job_id: str,
) -> tuple[str, str]:
    """Resolve a list-checkpoints row, resource name, ``job:name``, or name."""
    if isinstance(spec, Mapping):
        if not _is_resumable_row(spec):
            raise ValueError("init_from_checkpoint must reference a resumable checkpoint")
        value = spec.get("name")
        if not isinstance(value, str) or not value:
            raise ValueError("Checkpoint row is missing its resource name")
        spec = value

    resource_match = _CHECKPOINT_RESOURCE_RE.search(spec)
    if resource_match:
        return resource_match.group(1), resource_match.group(2)

    if ":" in spec and not spec.startswith(("gs://", "tinker://", "/")):
        job_id, name = spec.split(":", 1)
        if job_id and name:
            return job_id, name
    return default_job_id, _short_name(spec)


class TrainingCheckpoints:
    """One cookbook boundary for checkpoint RPCs and sampler weight sync."""

    def __init__(
        self,
        client: Any,
        fw_client: _CheckpointControlPlane,
        *,
        trainer_id: str,
        log_path: str,
        lora_rank: int = 0,
        serverless: bool = False,
        save_appear_timeout_s: float = 90.0,
        save_poll_s: float = 3.0,
        current_run_id: str | None = None,
        **legacy_options: Any,
    ) -> None:
        # ``save_stabilize_s`` used to control a CP poll after DCP saves. DCP
        # results now give us the stored checkpoint name directly.
        legacy_options.pop("save_stabilize_s", None)
        if legacy_options:
            unexpected = ", ".join(sorted(legacy_options))
            raise TypeError(f"Unexpected checkpoint options: {unexpected}")
        self._client = client
        self._fw_client = fw_client
        self._trainer_id = trainer_id
        self._log_path = log_path
        self._lora_rank = lora_rank
        self._serverless = serverless
        self._current_run_id = current_run_id if serverless else None
        self._save_appear_timeout_s = save_appear_timeout_s
        self._save_poll_s = save_poll_s

    def list(self) -> list[dict]:
        """Return the control plane's authoritative checkpoint rows."""
        return self._fw_client.list_checkpoints(self._trainer_id)

    def save(
        self,
        step: int,
        *,
        resumable: bool,
        promotable: bool,
        row_cursor: int | None = None,
    ) -> None:
        """Save trainer state and/or a complete promotable export.

        When ``row_cursor`` is supplied for a resumable save, it is recorded
        under the step returned by the trainer. Omitting it performs no local
        cursor I/O, which is the sync/async RL contract.
        """
        if step < 0:
            raise ValueError("step must be >= 0")
        if not (resumable or promotable):
            raise ValueError("save() requires at least one of resumable/promotable")
        if row_cursor is not None and row_cursor < 0:
            raise ValueError("row_cursor must be >= 0")

        name = f"step-{step}"
        if resumable:
            started = time.time()
            logger.info("Saving resumable checkpoint %r...", name)
            result = self._client.save_state(name)
            actual_name = self._saved_checkpoint_name(result, fallback=name)
            if row_cursor is None:
                logger.info(
                    "Resumable checkpoint %r saved (%.1fs)",
                    actual_name,
                    time.time() - started,
                )
            else:
                actual_step = _checkpoint_step(actual_name)
                self._write_row_cursor(self._trainer_id, actual_step, row_cursor)
                logger.info(
                    "Resumable checkpoint %r saved (%.1fs, row cursor %d)",
                    actual_name,
                    time.time() - started,
                    row_cursor,
                )

        if promotable:
            self._save_promotable(name)

    def sync_weights(
        self,
        step: int,
        hotload: Callable[[str], Any],
    ) -> str:
        """Save and hot-load the current sampler weights.

        No checkpoint format is supplied: the SDK selects a full LoRA adapter
        for LoRA runs and manages the base/delta chain for full-parameter runs.
        """
        if step < 0:
            raise ValueError("step must be >= 0")
        name = f"step-{step}"
        started = time.time()
        saved = self._client.save_weights_for_sampler(name)
        path = getattr(saved, "path", None)
        if not isinstance(path, str) or not path:
            raise RuntimeError(f"Weight sync {name!r} returned no snapshot path")
        hotload(path)
        logger.info("Weights synced at step %d (%.1fs)", step, time.time() - started)
        return path

    def resume(
        self,
        *,
        init_from_checkpoint: str | Mapping[str, Any] | None = None,
        warm_start_from_adapter: str | None = None,
        dataloader_cursor: int | None = None,
        auto_latest: bool = True,
    ) -> ResumeInfo:
        """Resolve and load a checkpoint, returning step and raw-row cursor.

        ``init_from_checkpoint`` accepts a row returned by :meth:`list`, its
        full resource name, ``job_id:step-N``, or a bare ``step-N``. When
        ``dataloader_cursor`` is supplied, local cursor state is not read.
        Set ``auto_latest=False`` when the caller owns checkpoint selection.
        """
        validate_warm_start_config(
            warm_start_from_adapter=warm_start_from_adapter,
            init_from_checkpoint=init_from_checkpoint,
            lora_rank=self._lora_rank,
        )
        if dataloader_cursor is not None and dataloader_cursor < 0:
            raise ValueError("dataloader_cursor must be >= 0")

        if init_from_checkpoint is not None:
            source_job_id, stored_name = _checkpoint_ref(
                init_from_checkpoint,
                default_job_id=self._trainer_id,
            )
            if self._serverless and source_job_id != self._trainer_id:
                raise ValueError(
                    "serverless init_from_checkpoint cannot reference another "
                    f"session ({source_job_id!r}); checkpoints on the shared pool "
                    "are isolated"
                )
            logical_name = self._trainer_logical_name(stored_name)
            step = _checkpoint_step(logical_name)
            load_source = None if source_job_id == self._trainer_id else source_job_id
            self._load_checkpoint(logical_name, source_job_id=load_source)
            return ResumeInfo(
                step=step,
                row_cursor=self._resolve_row_cursor(
                    source_job_id,
                    step,
                    override=dataloader_cursor,
                ),
                source_job_id=source_job_id,
            )

        latest = self._latest_resumable() if auto_latest else None
        if latest is not None:
            logical_name = self._trainer_logical_name(_short_name(latest["name"]))
            step = _checkpoint_step(logical_name)
            self._load_checkpoint(logical_name, source_job_id=None)
            return ResumeInfo(
                step=step,
                row_cursor=self._resolve_row_cursor(
                    self._trainer_id,
                    step,
                    override=dataloader_cursor,
                ),
                source_job_id=self._trainer_id,
            )

        if warm_start_from_adapter:
            logger.info("Starting with an explicit HF adapter")
            started = time.time()
            self._client.load_adapter(warm_start_from_adapter)
            logger.info("Adapter loaded (%.1fs)", time.time() - started)

        return ResumeInfo(row_cursor=dataloader_cursor or 0)

    def promote_latest(
        self,
        output_model_id: str,
        base_model: str,
        *,
        hot_load_deployment_id: str | None = None,
    ) -> dict:
        """Promote the newest promotable row returned by the control plane."""
        rows = _newest_first(
            [
                row
                for row in self.list()
                if row.get("promotable") and self._row_matches_current_run(row)
            ]
        )
        if not rows:
            raise RuntimeError(
                f"No promotable checkpoints found for trainer job {self._trainer_id!r}. "
                "Save a promotable checkpoint first."
            )
        full_name = rows[0]["name"]
        logger.info(
            "Promoting checkpoint %s -> %s",
            _short_name(full_name),
            output_model_id,
        )
        return self._fw_client.promote_checkpoint(
            name=full_name,
            output_model_id=output_model_id,
            base_model=base_model,
            hot_load_deployment_id=hot_load_deployment_id,
        )

    def _load_checkpoint(self, name: str, *, source_job_id: str | None) -> None:
        path = self._client.resolve_checkpoint_path(
            name,
            source_job_id=source_job_id,
        )
        logger.info("Resuming from checkpoint %s", path)
        started = time.time()
        self._client.load_state_with_optimizer(path)
        logger.info("Checkpoint loaded (%.1fs)", time.time() - started)

    def _latest_resumable(self) -> dict | None:
        rows = [
            row
            for row in self.list()
            if _is_resumable_row(row) and self._row_matches_current_run(row)
        ]
        rows = _newest_first(rows)
        return rows[0] if rows else None

    def _save_promotable(self, name: str) -> None:
        if self._promotable_row(name) is not None:
            logger.info("Promotable checkpoint %r already exists", name)
            return
        started_at = datetime.now(timezone.utc)
        started = time.time()
        # A complete export is required for promotion. This is intentionally
        # hidden here; ordinary weight sync delegates base/delta selection.
        self._client.save_weights_for_sampler(name, checkpoint_type="base")
        logger.info("Promotable checkpoint %r saved (%.1fs)", name, time.time() - started)
        self._wait_for_promotable(name=name, save_started=started_at)

    def _wait_for_promotable(self, *, name: str, save_started: datetime) -> None:
        deadline = time.time() + self._save_appear_timeout_s
        while time.time() < deadline:
            try:
                if self._promotable_row(name, min_create_time=save_started) is not None:
                    return
            except Exception as error:
                logger.debug("Checkpoint list failed while waiting for %r: %s", name, error)
            time.sleep(self._save_poll_s)
        logger.warning(
            "Timed out after %.0fs waiting for promotable checkpoint %r to surface",
            self._save_appear_timeout_s,
            name,
        )

    def _promotable_row(
        self,
        name: str,
        *,
        min_create_time: datetime | None = None,
    ) -> dict | None:
        rows = _newest_first([row for row in self.list() if row.get("promotable")])
        freshness_floor = None
        if min_create_time is not None:
            freshness_floor = min_create_time - timedelta(
                seconds=_PROMOTABLE_FRESHNESS_TOLERANCE_S
            )
        for row in rows:
            if not self._row_matches_current_run(row):
                continue
            if self._trainer_logical_name(_short_name(row.get("name", ""))) != name:
                continue
            create_time = _parse_iso_time(row.get("createTime"))
            if (
                freshness_floor is not None
                and create_time is not None
                and create_time < freshness_floor
            ):
                continue
            return row
        return None

    def _row_matches_current_run(self, row: Mapping[str, Any]) -> bool:
        if not self._serverless:
            return True
        return _belongs_to_run(_short_name(row.get("name", "")), self._current_run_id)

    def _trainer_logical_name(self, short: str) -> str:
        if self._serverless:
            return _logical_name_for_run(short, self._current_run_id)
        return _logical_name(short)

    @staticmethod
    def _saved_checkpoint_name(result: Any, *, fallback: str) -> str:
        path = getattr(result, "path", None)
        if not isinstance(path, str) or not path:
            return fallback
        short = _short_name(path)
        return short if _STEP_NAME_RE.fullmatch(short) else fallback

    def _resolve_row_cursor(
        self,
        job_id: str,
        step: int,
        *,
        override: int | None,
    ) -> int:
        if override is not None:
            return override
        return self._read_row_cursor(job_id, step)

    def _dataloader_path(self) -> str:
        return fileio.join(self._log_path, DATALOADER_CURSOR_BASE_NAME)

    def _read_cursor_store(self) -> dict[str, dict[str, int]]:
        path = self._dataloader_path()
        raw = fileio.read_text(path)
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            logger.warning("Corrupt %s (%s); treating as empty", path, error)
            return {}
        if not isinstance(data, dict):
            logger.warning("Invalid cursor mapping in %s; treating as empty", path)
            return {}

        # Migrate the former {"step-N": cursor} shape into the only local
        # state shape supported now: {job_id: {step: cursor}}.
        if all(not isinstance(value, dict) for value in data.values()):
            migrated: dict[str, int] = {}
            for name, cursor in data.items():
                try:
                    migrated[str(_checkpoint_step(str(name)))] = int(cursor)
                except (TypeError, ValueError):
                    continue
            return {self._trainer_id: migrated} if migrated else {}

        normalized: dict[str, dict[str, int]] = {}
        try:
            for job_id, steps in data.items():
                if not isinstance(steps, dict):
                    raise TypeError
                normalized[str(job_id)] = {
                    str(int(step)): int(cursor) for step, cursor in steps.items()
                }
        except (TypeError, ValueError):
            logger.warning("Invalid cursor mapping in %s; treating as empty", path)
            return {}
        return normalized

    def _write_row_cursor(self, job_id: str, step: int, row_cursor: int) -> None:
        data = self._read_cursor_store()
        job_steps = data.setdefault(job_id, {})
        job_steps[str(step)] = row_cursor
        ordered = sorted(job_steps.items(), key=lambda item: int(item[0]))
        data[job_id] = dict(ordered[-DATALOADER_HISTORY_KEEP:])
        fileio.makedirs(self._log_path)
        fileio.write_json(self._dataloader_path(), data)

    def _read_row_cursor(self, job_id: str, step: int) -> int:
        return self._read_cursor_store().get(job_id, {}).get(str(step), 0)
