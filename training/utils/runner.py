"""Runner contract for cookbook orchestration.

Defines the file-based contract the orchestration layer consumes.
All file paths are optional -- when unset, the corresponding output is skipped.

Optional inputs (via ``RunnerConfig`` fields or environment variables):

* ``status_file``  / ``COOKBOOK_STATUS_FILE``  -- training status + progress
* ``metadata_file`` / ``COOKBOOK_METADATA_FILE`` -- accumulated tokens + accelerator-seconds
* ``metrics_file``  / ``COOKBOOK_METRICS_FILE``  -- append-only JSONL per-step metrics
* ``output_model_path`` / ``COOKBOOK_OUTPUT_MODEL_PATH`` -- where to write final model info

File formats:

``status_file`` (protojson-compatible ``google.rpc.Status``, overwritten each update)::

    {"code": 0, "message": "training",
     "details": [{"@type": "type.googleapis.com/gateway.JobProgress", "percent": 5}]}

``metadata_file`` (JSON, overwritten each update)::

    {"metadata": {"tokens": 120000, "accelerator_seconds": {"NVIDIA_H100_80GB": 3600}}}

``metrics_file`` (JSONL, appended each step)::

    {"step": 1, "train/ce_loss": 2.3, "train/ppl": 10.0, ...}

``output_model_path`` (JSON, written once at completion)::

    {"model_id": "accounts/.../models/my-model", "checkpoint": "step-100",
     "job_id": "job-abc"}
"""

from __future__ import annotations

import errno
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from training.utils import fileio

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


_GRPC_OK = 0
_GRPC_CANCELLED = 1
_GRPC_INVALID_ARGUMENT = 3
_GRPC_FAILED_PRECONDITION = 9
_GRPC_INTERNAL = 13
_GRPC_UNAVAILABLE = 14
_JOB_PROGRESS_TYPE_URL = "type.googleapis.com/gateway.JobProgress"
_ERROR_INFO_TYPE_URL = "type.googleapis.com/google.rpc.ErrorInfo"
_ERROR_INFO_DOMAIN = "training.fireworks.ai"
_INTERNAL_ERROR_MESSAGE = "Internal error"
_CANCELLED_ERROR_MESSAGE = "Training run cancelled"
_SIGNAL_CANCELLATION_EXIT_CODES = frozenset({130, 143})
_SIGNAL_CANCELLATION_MESSAGES = frozenset({"Terminated by SIGINT", "Terminated by SIGTERM"})

_STATUS_TO_GRPC_CODE: dict[RunStatus, int] = {
    RunStatus.PENDING: _GRPC_OK,
    RunStatus.RUNNING: _GRPC_OK,
    RunStatus.COMPLETED: _GRPC_OK,
    RunStatus.FAILED: _GRPC_FAILED_PRECONDITION,
}


class UserConfigError(Exception):
    """Base class for user-actionable configuration errors raised inside a recipe.

    When a recipe raises one of these inside a ``RunnerIO`` context, the runner
    records the failure as ``INVALID_ARGUMENT`` (user-fixable) instead of the
    generic ``FAILED_PRECONDITION``, so the control plane preserves the
    actionable message and category (FIR2-1774).
    """


class DatasetError(UserConfigError):
    """Raised when a dataset is malformed or yields no trainable examples.

    Typical causes: invalid JSONL/preference-row shape, every row filtered by
    sequence length, or no assistant tokens selected by ``train_on_what`` /
    per-message ``weight``. The control plane surfaces this as a user-fixable
    dataset error instead of sanitizing it to Internal error.
    """


# Keep aligned with firetitan managed dataset validation so customers see one message.
NO_VALID_TRAINING_EXAMPLES_MESSAGE = (
    "No valid training examples remained after tokenization. Verify that the "
    "dataset contains trainable assistant messages for the selected "
    "train_on_what setting and that examples fit the configured maximum "
    "sequence length, then retry."
)
NO_VALID_PREFERENCE_PAIRS_MESSAGE = (
    "No valid preference pairs remained after tokenization. Verify that the "
    "dataset contains valid chosen and rejected examples that fit the "
    "configured maximum sequence length, then retry."
)


class WandbConfigError(UserConfigError):
    """Raised when Weights & Biases auth/config is invalid (bad key, entity, or project)."""


def _is_managed_api_network_unavailable(error: BaseException) -> bool:
    """Return whether a managed API connection failed with ENETUNREACH."""
    current: BaseException | None = error
    seen: set[int] = set()
    has_managed_api_error = False
    has_network_unreachable = False
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        error_type = type(current)
        module_root = error_type.__module__.partition(".")[0]
        has_managed_api_error = (
            has_managed_api_error
            or error_type.__name__ == "APIConnectionError"
            and module_root in ("fireworks", "tinker")
        )
        if isinstance(current, OSError) and current.errno == errno.ENETUNREACH:
            has_network_unreachable = True
        current = current.__cause__ or current.__context__
    return has_managed_api_error and has_network_unreachable


def _is_recipe_cancellation(error: BaseException) -> bool:
    """Return whether a recipe surfaced an explicit user/platform cancellation.

    Cookbook signal handlers raise ``SystemExit`` so ``RunnerIO`` can flush
    status/metadata during cleanup. Treat only those conventional signal exits
    as cancellation; other ``SystemExit`` values may be user-input failures and
    must keep the safe unknown-exception path.
    """

    if isinstance(error, KeyboardInterrupt):
        return True
    if not isinstance(error, SystemExit):
        return False
    code = error.code
    if isinstance(code, int) and not isinstance(code, bool):
        return code in _SIGNAL_CANCELLATION_EXIT_CODES
    if isinstance(code, str):
        return code.strip() in _SIGNAL_CANCELLATION_MESSAGES
    return False


@dataclass
class RunnerConfig:
    """Optional orchestration outputs.

    Paths can be set directly or read from environment variables.
    Direct values take precedence over env vars.
    """

    status_file: str | None = None
    metadata_file: str | None = None
    metrics_file: str | None = None
    output_model_path: str | None = None

    def resolve(self) -> RunnerConfig:
        """Return a copy with env-var fallbacks applied."""
        return RunnerConfig(
            status_file=self.status_file or os.environ.get("COOKBOOK_STATUS_FILE"),
            metadata_file=self.metadata_file
            or os.environ.get("COOKBOOK_METADATA_FILE"),
            metrics_file=self.metrics_file or os.environ.get("COOKBOOK_METRICS_FILE"),
            output_model_path=self.output_model_path
            or os.environ.get("COOKBOOK_OUTPUT_MODEL_PATH"),
        )

    @property
    def enabled(self) -> bool:
        """True if any output path is configured."""
        return any(
            [
                self.status_file,
                self.metadata_file,
                self.metrics_file,
                self.output_model_path,
            ]
        )


class RunnerIO:
    """Writes orchestration contract files.

    Constructed once per recipe run.  Call methods at the appropriate
    points in the training loop to keep the orchestration layer informed.

    All write operations are best-effort -- failures are logged but never
    raise so the training loop is not interrupted.
    """

    def __init__(self, config: RunnerConfig | None = None):
        cfg = (config or RunnerConfig()).resolve()
        self._status_file = cfg.status_file
        self._metadata_file = cfg.metadata_file
        self._metrics_file = cfg.metrics_file
        self._output_model_path = cfg.output_model_path

        self._tokens_processed: int = 0
        self._training_start: float | None = None
        self._accelerator_type: str | None = None
        self._accelerator_count: int | None = None
        self._last_step: int = 0
        self._last_total_steps: int = 0
        self._serverless: bool = False

    # -- context manager -------------------------------------------------------

    def __enter__(self) -> "RunnerIO":
        return self

    def __exit__(self, exc_type: object, exc_val: object, tb: object) -> bool:
        if exc_type is not None:
            error_code = None
            error_message = str(exc_val)
            error_info: dict[str, Any] | None = None
            if isinstance(exc_val, BaseException) and _is_recipe_cancellation(exc_val):
                # SIGINT/SIGTERM exits and KeyboardInterrupt are cancellations.
                # FireTitan only catches Exception, so these BaseException exits
                # must be classified here or they surface as generic failures.
                error_code = _GRPC_CANCELLED
                error_message = _CANCELLED_ERROR_MESSAGE
                error_info = {
                    "@type": _ERROR_INFO_TYPE_URL,
                    "reason": "CANCELLED",
                    "domain": _ERROR_INFO_DOMAIN,
                    "metadata": {"version": "1", "source": "managed", "category": "signal"},
                }
            elif isinstance(exc_val, SystemExit):
                # Non-signal SystemExit values may carry user/config text; keep
                # the safe internal status instead of leaking it to the file.
                error_code = _GRPC_INTERNAL
                error_message = _INTERNAL_ERROR_MESSAGE
            elif isinstance(exc_val, UserConfigError):
                # User-config errors (bad W&B credentials, etc.) are user-actionable;
                # surface them as INVALID_ARGUMENT instead of the generic
                # FAILED_PRECONDITION so the control plane preserves the actionable
                # message and category (FIR2-1774).
                error_code = _GRPC_INVALID_ARGUMENT
            elif isinstance(exc_val, BaseException) and _is_managed_api_network_unavailable(exc_val):
                error_code = _GRPC_UNAVAILABLE
            self.write_status(
                RunStatus.FAILED,
                step=self._last_step,
                total_steps=self._last_total_steps,
                error=error_message,
                error_code=error_code,
                error_info=error_info,
            )
            self.write_metadata()
        return False  # never suppress the exception

    # -- status ----------------------------------------------------------------

    def write_status(
        self,
        status: RunStatus,
        *,
        step: int = 0,
        total_steps: int = 0,
        message: str = "",
        error: str | None = None,
        error_code: int | None = None,
        error_info: dict[str, Any] | None = None,
    ) -> None:
        self._last_step = step
        self._last_total_steps = total_steps
        if not self._status_file:
            return
        grpc_code = (
            error_code
            if error_code is not None
            else _STATUS_TO_GRPC_CODE.get(status, _GRPC_OK)
        )
        status_message = error or message or status.value
        payload: dict[str, Any] = {
            "code": grpc_code,
            "message": status_message,
        }
        details: list[dict[str, Any]] = []
        if total_steps > 0:
            percent = int(step / total_steps * 100)
            details.append({"@type": _JOB_PROGRESS_TYPE_URL, "percent": percent})
        if error_info is not None:
            details.append(error_info)
        if details:
            payload["details"] = details
        self._write_json(self._status_file, payload)

    def report_rendering_progress(
        self, current: int, total: int, *, label: str = "rendering data"
    ) -> None:
        """Log and write status for data rendering progress."""
        pct = int(100.0 * current / total) if total else 0
        logger.info("%s: %d/%d (%d%%)", label, current, total, pct)
        self.write_status(RunStatus.PENDING, message=f"{label} ({pct}%)")

    # -- metadata --------------------------------------------------------------

    def set_accelerator_info(
        self,
        accelerator_type: str | None = None,
        accelerator_count: int | None = None,
        *,
        profile: Any | None = None,
    ) -> None:
        if profile is not None:
            if accelerator_type is None:
                accelerator_type = getattr(profile, "accelerator_type", None)
            if accelerator_count is None:
                accelerator_count = getattr(profile, "accelerator_count", None)
        self._accelerator_type = accelerator_type
        self._accelerator_count = accelerator_count

    def mark_serverless(self) -> None:
        """Record that this run trained on the shared serverless pool.

        Written into metadata.json so the control plane skips its own per-token
        billing leg: the pooled trainer already bills these tokens per-token, and
        billing them again on the CP side would double-charge the customer. Call
        this only on the serverless execution path (after the session attaches to
        the pool), never from the config request alone -- a run that falls back to
        a dedicated trainer must still be billed by the control plane.
        """
        self._serverless = True

    def start_training(self) -> None:
        """Mark training start for accelerator-seconds calculation."""
        self._training_start = time.monotonic()

    def write_metadata(self) -> None:
        if not self._metadata_file:
            return
        accel_seconds: dict[str, int] = {}
        if self._training_start is not None:
            wall_seconds = time.monotonic() - self._training_start
            n_devices = self._accelerator_count or 1
            total = round(wall_seconds * n_devices)
            accel_type = self._accelerator_type or "UNKNOWN"
            accel_seconds[accel_type] = total
        payload: dict[str, Any] = {
            "metadata": {
                "tokens": self._tokens_processed,
                "accelerator_seconds": accel_seconds,
                # When true, the serverless pool trainer already billed these
                # tokens per-token; the control plane skips its token-billing
                # leg to avoid double-charging.
                "serverless": self._serverless,
            }
        }
        self._write_json(self._metadata_file, payload)

    def set_tokens_processed(self, tokens: int) -> None:
        """Set the canonical processed-token total used by metadata output."""
        self._tokens_processed = max(0, int(tokens))

    # -- metrics ---------------------------------------------------------------

    def append_metrics(
        self, step: int, metrics: dict[str, Any], *, tokens: int = 0
    ) -> None:
        if tokens:
            self._tokens_processed += tokens
        if not self._metrics_file:
            return
        record = {"step": step}
        for k, v in metrics.items():
            if isinstance(v, float) and (v != v):  # NaN guard
                v = None
            record[k] = v
        self._append_jsonl(self._metrics_file, record)

    # -- output model ----------------------------------------------------------

    def write_output_model(
        self,
        *,
        model_id: str | None = None,
        checkpoint: str | None = None,
        job_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self._output_model_path:
            return
        payload: dict[str, Any] = {}
        if model_id:
            payload["model_id"] = model_id
        if checkpoint:
            payload["checkpoint"] = checkpoint
        if job_id:
            payload["job_id"] = job_id
        if extra:
            payload.update(extra)
        self._write_json(self._output_model_path, payload)

    # -- helpers ---------------------------------------------------------------

    def _write_json(self, path: str, data: dict[str, Any]) -> None:
        try:
            fileio.write_json(path, data)
        except Exception:
            logger.warning("Failed to write %s", path, exc_info=True)

    def _append_jsonl(self, path: str, record: dict[str, Any]) -> None:
        try:
            fileio.append_jsonl(path, record)
        except Exception:
            logger.warning("Failed to append to %s", path, exc_info=True)
