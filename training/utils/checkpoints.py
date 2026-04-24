"""Unified checkpoint API for cookbook training loops.

Collapses the old three-way split (DCP / sampler-full / sampler-LoRA) behind
two user-facing axes: ``resumable`` and ``promotable``. The control plane
(``FireworksClient.list_checkpoints(job_id)``) is the source of truth for
what checkpoints exist, their type, and promotability. The only
locally-persisted file is ``dataloader.json``, which maps checkpoint name
to the cookbook's ``data_consumed`` counter (no server-side
representation).

See issue fw-ai/fireworks#23495 for the full design.

Usage::

    ckpt = TrainingCheckpoints(client, rlor_mgr,
                               trainer_id=job_id, log_path=cfg.log_path,
                               lora_rank=cfg.lora_rank)

    resume_info = ckpt.resume(
        init_from_checkpoint=cfg.init_from_checkpoint,
        warm_start_from_adapter=cfg.warm_start_from_adapter,
    )

    # periodic (RL / SFT)
    ckpt.save(f"step-{step}", resumable=True, promotable=False,
              data_consumed=data_consumed)

    # final
    ckpt.save(f"step-{step}", resumable=True, promotable=True,
              data_consumed=data_consumed)
    if cfg.output_model_id:
        ckpt.promote_latest(cfg.output_model_id, cfg.base_model)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import training.utils.fileio as fileio

DATALOADER_BASE_NAME = "dataloader.json"
DATALOADER_HISTORY_KEEP = 20

_RESUMABLE_TYPE_SUFFIXES = ("TRAINING", "TRAINING_LORA")

logger = logging.getLogger(__name__)


# -- Public types --------------------------------------------------------------


@dataclass
class ResumeInfo:
    """Resolved resume state returned by :meth:`TrainingCheckpoints.resume`."""

    step: int = 0
    data_consumed: int = 0
    source_job_id: str | None = None


class _CheckpointLister(Protocol):
    def list_checkpoints(self, job_id: str, *, page_size: int = 200) -> list[dict]: ...

    def promote_checkpoint(
        self,
        job_id: str,
        checkpoint_id: str,
        output_model_id: str,
        base_model: str,
        *,
        hot_load_deployment_id: str | None = None,
    ) -> dict: ...


# -- Helpers -------------------------------------------------------------------


def validate_warm_start_config(
    *,
    warm_start_from_adapter: str | None,
    init_from_checkpoint: str | None,
    lora_rank: int,
) -> None:
    """Validate cross-field constraints on warm-start inputs.

    Full-param warm-start is handled via ``cfg.base_model`` at session init,
    not via this API — this helper only checks the LoRA adapter path.
    """
    if warm_start_from_adapter and init_from_checkpoint:
        raise ValueError(
            "warm_start_from_adapter and init_from_checkpoint are mutually exclusive"
        )
    if warm_start_from_adapter and lora_rank == 0:
        raise ValueError(
            "warm_start_from_adapter requires lora_rank > 0. "
            "For full-param warm-start, set cfg.base_model to the promoted model "
            "resource name — the training session will initialize from it directly."
        )


def _short_name(resource_name: str) -> str:
    """Extract the trailing checkpoint id from a full resource name."""
    return resource_name.rstrip("/").rsplit("/", 1)[-1]


def _resolved_save_name(save_result: Any, *, fallback: str) -> str:
    """Extract the server-authoritative short checkpoint name from save_state's
    response. Tinker's ``SaveWeightsResponse`` returns a ``path`` URI
    (e.g. ``tinker://.../step-0``); its basename is what the control plane's
    ``list_checkpoints`` will surface. Falls back to the caller name if the
    response shape is unexpected (older clients, tests with naive mocks).
    """
    if save_result is None:
        return fallback
    path = getattr(save_result, "path", None)
    if path:
        return _short_name(str(path))
    snapshot_name = getattr(save_result, "snapshot_name", None)
    if snapshot_name:
        return _short_name(str(snapshot_name))
    return fallback


_SESSION_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")


def _logical_name(short: str) -> str:
    """Strip the server-appended session suffix from a sampler checkpoint id.

    Sampler writes (``INFERENCE_*`` rows) get an 8-hex-char session id
    appended by the trainer (e.g. ``"step-5"`` -> ``"step-5-45dda197"``).
    DCP (``TRAINING*``) rows keep the original caller-supplied name. For
    skip-if-exists the caller's logical name is what should match.
    """
    return _SESSION_SUFFIX_RE.sub("", short)


def _is_resumable_row(row: dict) -> bool:
    ctype = row.get("checkpointType", "") or ""
    return any(ctype.endswith(suffix) for suffix in _RESUMABLE_TYPE_SUFFIXES)


def _newest_first(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: r.get("createTime", ""), reverse=True)


def _parse_cross_job(spec: str) -> tuple[str | None, str]:
    """Parse ``"job_id:checkpoint_name"`` or a plain path/name."""
    if ":" in spec and not spec.startswith(("gs://", "/")):
        job_id, name = spec.split(":", 1)
        return job_id, name
    return None, spec


# -- Main class ----------------------------------------------------------------


class TrainingCheckpoints:
    """Single cookbook-side checkpoint manager.

    Holds a reference to the live training client (for save / load RPCs on
    the trainer pod) and a control-plane client (for the authoritative list
    of checkpoints and for ``:promote``).
    """

    def __init__(
        self,
        client: Any,
        fw_client: _CheckpointLister,
        *,
        trainer_id: str,
        log_path: str,
        lora_rank: int = 0,
    ) -> None:
        self._client = client
        self._fw_client = fw_client
        self._trainer_id = trainer_id
        self._log_path = log_path
        self._lora_rank = lora_rank

    # -- Save --------------------------------------------------------------

    def save(
        self,
        name: str,
        *,
        resumable: bool,
        promotable: bool,
        data_consumed: int | None = None,
    ) -> None:
        """Save a checkpoint with the requested capabilities.

        ``resumable=True`` writes a DCP checkpoint (weights + optimizer).
        ``promotable=True`` writes a sampler checkpoint. The sampler write is
        skipped if a row with the same name already exists on the control
        plane with ``promotable=True`` (e.g. produced earlier by
        ``WeightSyncer.save_and_hotload`` in an RL loop).

        ``data_consumed`` is persisted to ``dataloader.json`` keyed on
        ``name`` so the corresponding resume call can recover the cookbook's
        rollouts-consumed counter. Ignored when ``resumable=False``.
        """
        if not (resumable or promotable):
            raise ValueError("save() requires at least one of resumable/promotable")

        t0 = time.time()
        if resumable:
            logger.info("Saving DCP checkpoint '%s'...", name)
            result = self._client.save_state(name)
            logger.info("DCP checkpoint '%s' saved (%.1fs)", name, time.time() - t0)
            if data_consumed is not None:
                # Key dataloader.json by what the server actually wrote. The
                # trainer may rename to its internal step counter (e.g. the
                # caller passes "step-42" but the service stores as "step-0").
                # Resume reads the row name from the control plane, so the
                # stored key must match that — not the caller's logical name.
                actual_name = _resolved_save_name(result, fallback=name)
                self._write_dataloader(actual_name, data_consumed)
                if actual_name != name:
                    logger.info(
                        "DCP server-returned name %r differs from caller name %r; "
                        "dataloader.json keyed on server name for resume alignment.",
                        actual_name, name,
                    )

        if promotable:
            if self._promotable_exists(name):
                logger.info(
                    "Sampler checkpoint '%s' already promotable on control plane — "
                    "skipping redundant save.",
                    name,
                )
            else:
                t1 = time.time()
                logger.info("Saving sampler checkpoint '%s'...", name)
                self._client.save_weights_for_sampler_ext(name, checkpoint_type="base")
                logger.info(
                    "Sampler checkpoint '%s' saved (%.1fs)", name, time.time() - t1
                )

    # -- Resume ------------------------------------------------------------

    def resume(
        self,
        *,
        init_from_checkpoint: str | None = None,
        warm_start_from_adapter: str | None = None,
    ) -> ResumeInfo | None:
        """Determine resume state and load weights into the live client.

        Priority:

        1. ``init_from_checkpoint`` — explicit cross-job DCP load (weights
           and optimizer). Step counter resets to 0.
        2. Newest resumable row on the control plane — auto-resume.
        3. ``warm_start_from_adapter`` — HF PEFT adapter (weights only).
        4. Fresh start (returns ``None``).
        """
        validate_warm_start_config(
            warm_start_from_adapter=warm_start_from_adapter,
            init_from_checkpoint=init_from_checkpoint,
            lora_rank=self._lora_rank,
        )

        if init_from_checkpoint:
            source_job_id, dcp_name = _parse_cross_job(init_from_checkpoint)
            path = self._client.resolve_checkpoint_path(
                dcp_name, source_job_id=source_job_id
            )
            logger.info(
                "Starting at step 0 with weights loaded from %s "
                "(no resume — step counter resets)",
                path,
            )
            t0 = time.time()
            self._client.load_state_with_optimizer(path)
            logger.info("Checkpoint loaded (%.1fs)", time.time() - t0)
            return ResumeInfo(step=0, data_consumed=0, source_job_id=source_job_id)

        latest = self._latest_resumable()
        if latest:
            short = _short_name(latest["name"])
            path = self._client.resolve_checkpoint_path(
                short, source_job_id=self._trainer_id
            )
            logger.info("Resuming from control-plane row: %s", short)
            t0 = time.time()
            self._client.load_state_with_optimizer(path)
            logger.info("Checkpoint loaded: %s (%.1fs)", path, time.time() - t0)
            return ResumeInfo(
                step=_step_from_name(short),
                data_consumed=self._read_dataloader(short),
                source_job_id=self._trainer_id,
            )

        if warm_start_from_adapter:
            logger.info("Fresh start with HF adapter: %s", warm_start_from_adapter)
            t0 = time.time()
            self._client.load_adapter(warm_start_from_adapter)
            logger.info("Adapter loaded (%.1fs)", time.time() - t0)
            return ResumeInfo(step=0, data_consumed=0, source_job_id=None)

        logger.info("Starting at step 0 from base model (no checkpoint)")
        return None

    # -- Promote -----------------------------------------------------------

    def promote_latest(
        self,
        output_model_id: str,
        base_model: str,
        *,
        hot_load_deployment_id: str | None = None,
    ) -> dict:
        """Promote the newest promotable row on the control plane.

        No local lookup. Works identically for full and LoRA runs; in LoRA
        runs this transparently picks up the most recent
        ``save_and_hotload`` row without requiring an explicit final
        sampler save.
        """
        rows = _newest_first(
            [r for r in self._list_checkpoints() if r.get("promotable")]
        )
        if not rows:
            raise RuntimeError(
                f"No promotable checkpoints found for trainer job '{self._trainer_id}'. "
                "Call save(promotable=True) or weight_syncer.save_and_hotload() first."
            )
        name = _short_name(rows[0]["name"])
        logger.info("Promoting newest promotable checkpoint: %s -> %s", name, output_model_id)
        return self._fw_client.promote_checkpoint(
            self._trainer_id,
            name,
            output_model_id,
            base_model,
            hot_load_deployment_id=hot_load_deployment_id,
        )

    # -- Internal ----------------------------------------------------------

    def _list_checkpoints(self) -> list[dict]:
        return self._fw_client.list_checkpoints(self._trainer_id)

    def _latest_resumable(self) -> dict | None:
        try:
            rows = [r for r in self._list_checkpoints() if _is_resumable_row(r)]
        except Exception as e:
            logger.warning(
                "Control-plane list_checkpoints failed (%s); treating as fresh start. "
                "Override with init_from_checkpoint if this is wrong.",
                e,
            )
            return None
        rows = _newest_first(rows)
        return rows[0] if rows else None

    def _promotable_exists(self, name: str) -> bool:
        """Check if ``name`` is already on the control plane as promotable.

        Failures are non-fatal: on error we proceed with the save (GCS
        overwrite is safe), trading dedup for forward progress.
        """
        try:
            rows = self._list_checkpoints()
        except Exception as e:
            logger.warning(
                "Control-plane list_checkpoints failed during skip-check (%s); "
                "proceeding with sampler write.",
                e,
            )
            return False
        return any(
            r.get("promotable") and _logical_name(_short_name(r.get("name", ""))) == name
            for r in rows
        )

    def _dataloader_path(self) -> str:
        return fileio.join(self._log_path, DATALOADER_BASE_NAME)

    def _read_all_dataloader(self) -> dict[str, int]:
        path = self._dataloader_path()
        raw = fileio.read_text(path)
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Corrupt %s (%s); treating as empty.", path, e)
            return {}
        return {k: int(v) for k, v in data.items()}

    def _write_dataloader(self, name: str, data_consumed: int) -> None:
        data = self._read_all_dataloader()
        data[name] = data_consumed
        if len(data) > DATALOADER_HISTORY_KEEP:
            ordered = sorted(data.items(), key=lambda kv: _step_from_name(kv[0]))
            data = dict(ordered[-DATALOADER_HISTORY_KEEP:])
        fileio.makedirs(self._log_path)
        fileio.write_json(self._dataloader_path(), data)

    def _read_dataloader(self, name: str) -> int:
        return self._read_all_dataloader().get(name, 0)


def _step_from_name(name: str) -> int:
    """Parse an integer step from a ``"step-N"`` checkpoint name."""
    if name.startswith("step-"):
        try:
            return int(name.removeprefix("step-"))
        except ValueError:
            pass
    return 0
