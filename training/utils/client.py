"""ReconnectableClient -- wraps FiretitanTrainingClient with dispatch + wait.

Provides a clean API surface over the tinker training client.  Each method
dispatches one request and blocks until the result is ready, with a
configurable timeout to prevent indefinite hangs.

Tinker's internal polling already retries transient HTTP errors (408, 5xx).
However, 404 ("Trainer job not found or not running") can occur transiently
when the gateway routing hasn't stabilized for a freshly-RUNNING trainer.
This module adds retry logic for NotFoundError (404) around the dispatch +
wait cycle so the orchestrator doesn't crash on these transient routing
hiccups.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, TypeVar

import tinker
from fireworks.training.sdk.client import (
    FiretitanServiceClient,
    FiretitanTrainingClient,
    GradAccNormalization,
)
from fireworks.training.sdk.trainer import TrainerJobManager, TrainerServiceEndpoint
import tinker.lib.api_future_impl as tinker_api_future_impl
from tinker.types.future_retrieve_request import FutureRetrieveRequest as _FutureRetrieveRequest

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

DEFAULT_TIMEOUT_S: int = 3600
"""Default timeout for forward / forward_backward / optim_step (60 min)."""

DCP_TIMEOUT_S: int = 2700
"""Default timeout for save_state / load_state_with_optimizer (45 min)."""

_NOT_FOUND_MAX_RETRIES: int = 6
"""Max number of retries when a NotFoundError (404) is raised during result
retrieval.  Covers transient Tinker API gateway routing gaps that appear
right after a trainer transitions to JOB_STATE_RUNNING."""

_NOT_FOUND_BASE_DELAY_S: float = 2.0
"""Base delay (seconds) for exponential back-off on NotFoundError retries."""

_NOT_FOUND_MAX_DELAY_S: float = 30.0
"""Cap on the back-off delay between NotFoundError retries."""


def _retry_on_not_found(fn: Callable[[], _T], *, timeout: int) -> _T:
    """Execute *fn* and retry on ``tinker.NotFoundError`` with back-off.

    The 404 "Trainer job not found or not running" error can surface
    transiently when the Tinker gateway hasn't fully registered a
    freshly-RUNNING trainer.  Rather than crashing immediately (which
    kills the orchestrator pod and wastes an expensive training step
    already in flight), this helper retries the full dispatch-then-wait
    cycle up to ``_NOT_FOUND_MAX_RETRIES`` times with capped exponential
    back-off.

    Non-404 errors propagate immediately.
    """
    last_exc: tinker.NotFoundError | None = None
    for attempt in range(_NOT_FOUND_MAX_RETRIES + 1):
        try:
            return fn()
        except tinker.NotFoundError as exc:
            last_exc = exc
            if attempt >= _NOT_FOUND_MAX_RETRIES:
                break
            delay = min(
                _NOT_FOUND_BASE_DELAY_S * (2 ** attempt),
                _NOT_FOUND_MAX_DELAY_S,
            )
            logger.warning(
                "Transient 404 on result retrieval (attempt %d/%d), "
                "retrying in %.1fs: %s",
                attempt + 1,
                _NOT_FOUND_MAX_RETRIES,
                delay,
                exc,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _install_tinker_future_retrieve_compat() -> None:
    current = getattr(tinker_api_future_impl, "FutureRetrieveRequest", None)
    if current is None or getattr(current, "_fw_cookbook_compat", False):
        return

    def _compat_future_retrieve_request(*args, **kwargs):
        kwargs.pop("allow_metadata_only", None)
        return _FutureRetrieveRequest(*args, **kwargs)

    _compat_future_retrieve_request._fw_cookbook_compat = True
    tinker_api_future_impl.FutureRetrieveRequest = _compat_future_retrieve_request


_install_tinker_future_retrieve_compat()


class ReconnectableClient:
    """Training client wrapper: dispatch + wait with timeout.

    Each API call dispatches a single request to the trainer and blocks
    until the result is ready (or the timeout expires).  Transient 404
    errors from the Tinker gateway are retried with exponential back-off;
    all other failures propagate to the caller.

    For LoRA GRPO with KL regularisation, a single LoRA trainer can serve
    both policy and reference logprobs.  Use :meth:`create_base_reference`
    to obtain a second ``ReconnectableClient`` that shares the same trainer
    job but creates a ``base-<hex>`` model handle (adapters disabled).
    This avoids provisioning a second trainer purely for reference forward
    passes.
    """

    def __init__(
        self,
        rlor_mgr: TrainerJobManager,
        job_id: str,
        base_model: str,
        lora_rank: int = 0,
        fw_api_key: str | None = None,
        default_timeout: int = DEFAULT_TIMEOUT_S,
        endpoint: TrainerServiceEndpoint | None = None,
        *,
        base_only: bool = False,
    ):
        self._rlor_mgr = rlor_mgr
        self._job_id = job_id
        self._base_model = base_model
        self._lora_rank = lora_rank
        self._base_only = base_only
        self._fw_api_key = fw_api_key or os.environ.get("FIREWORKS_API_KEY")
        self._default_timeout = default_timeout
        self._endpoint: TrainerServiceEndpoint | None = None
        self._client: FiretitanTrainingClient | None = None
        self._closed = False
        if endpoint:
            self._use_endpoint(endpoint)
        else:
            self._connect()

    def create_base_reference(self) -> ReconnectableClient:
        """Create a reference client sharing this trainer's job.

        Returns a new ``ReconnectableClient`` backed by a ``base-<hex>``
        model handle on the same trainer.  Forward passes through the
        returned client run with all LoRA adapters disabled, giving
        base-model logprobs without a second GPU allocation.
        """
        assert self._endpoint is not None, "policy client must be connected first"
        return ReconnectableClient(
            rlor_mgr=self._rlor_mgr,
            job_id=self._job_id,
            base_model=self._base_model,
            lora_rank=0,
            fw_api_key=self._fw_api_key,
            default_timeout=self._default_timeout,
            endpoint=self._endpoint,
            base_only=True,
        )

    @property
    def inner(self) -> FiretitanTrainingClient:
        assert self._client is not None
        return self._client

    @property
    def endpoint(self) -> TrainerServiceEndpoint:
        assert self._endpoint is not None
        return self._endpoint

    @property
    def job_id(self) -> str:
        return self._job_id

    def forward(self, data, loss_fn):
        return _retry_on_not_found(
            lambda: self._client.forward(data, loss_fn).result(
                timeout=self._default_timeout,
            ),
            timeout=self._default_timeout,
        )

    def forward_backward(self, data, loss_fn: str = "cross_entropy", loss_fn_config=None):
        return _retry_on_not_found(
            lambda: self._client.forward_backward(
                data, loss_fn, loss_fn_config=loss_fn_config,
            ).result(timeout=self._default_timeout),
            timeout=self._default_timeout,
        )

    def forward_backward_custom(self, data, loss_fn):
        return _retry_on_not_found(
            lambda: self._client.forward_backward_custom(data, loss_fn).result(
                timeout=self._default_timeout,
            ),
            timeout=self._default_timeout,
        )

    def optim_step(
        self,
        params,
        grad_accumulation_normalization: str | GradAccNormalization | None = None,
    ):
        kwargs: dict = {}
        if grad_accumulation_normalization is not None:
            kwargs["grad_accumulation_normalization"] = _normalize_grad_accumulation_normalization(
                grad_accumulation_normalization
            )
        return _retry_on_not_found(
            lambda: self._client.optim_step(params, **kwargs).result(
                timeout=self._default_timeout,
            ),
            timeout=self._default_timeout,
        )

    def save_state(self, name: str, timeout: int = DCP_TIMEOUT_S):
        return self._client.save_state(name).result(timeout=timeout)

    def load_state(self, path: str, timeout: int = DCP_TIMEOUT_S):
        """Load model weights only (optimizer state is reset to zero).

        Use this when resuming from a checkpoint but starting with a fresh
        optimizer — e.g. after a large learning rate change.  The server
        loads the full DCP checkpoint then clears all Adam momentum (m)
        and variance (v) buffers so the next ``optim_step`` starts fresh.
        """
        return self._client.load_state(path).result(timeout=timeout)

    def load_state_with_optimizer(self, path: str, timeout: int = DCP_TIMEOUT_S):
        return self._client.load_state_with_optimizer(path).result(timeout=timeout)

    def save_weights_for_sampler_ext(
        self, name: str, checkpoint_type: str | None = None, timeout: int = DCP_TIMEOUT_S
    ):
        return self.inner.save_weights_for_sampler_ext(name, checkpoint_type=checkpoint_type)

    def resolve_checkpoint_path(self, name: str, source_job_id: str | None = None) -> str:
        return self.inner.resolve_checkpoint_path(name, source_job_id=source_job_id)

    def list_checkpoints(self) -> list[str]:
        return self.inner.list_checkpoints()

    def close(self, timeout: float = 5.0) -> None:
        """Stop local Tinker background tasks for this trainer client.

        The underlying Tinker holder owns background heartbeat / telemetry tasks.
        Best-effort flush queued telemetry, then stop those tasks before remote
        trainer cleanup so local tasks do not continue talking to a trainer
        that has already been deleted.
        """
        if self._closed:
            return
        self._closed = True

        client = self._client
        self._client = None
        self._endpoint = None
        if client is None:
            return

        holder = client.holder
        if holder is None:
            return

        telemetry = holder.get_telemetry()
        if telemetry is not None:
            try:
                # trigger_flush = getattr(telemetry, "_trigger_flush", None)
                telemetry._trigger_flush()
                telemetry._wait_until_drained_sync()
            except Exception as e:
                logger.warning(
                    "ReconnectableClient.close: failed to drain telemetry for job %s: %s",
                    self._job_id,
                    e,
                )
            finally:
                try:
                    telemetry.stop()
                except Exception as e:
                    logger.warning(
                        "ReconnectableClient.close: failed to stop telemetry for job %s: %s",
                        self._job_id,
                        e,
                    )

        try:
            cleanup_future = holder.run_coroutine_threadsafe(holder._async_cleanup())
            cleanup_future.result(timeout=timeout)
        except Exception as e:
            logger.warning(
                "ReconnectableClient.close: failed to stop holder for job %s: %s",
                self._job_id,
                e,
            )

    def __enter__(self) -> ReconnectableClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- Internal --------------------------------------------------------------

    def _use_endpoint(self, ep: TrainerServiceEndpoint) -> None:
        svc = FiretitanServiceClient(
            base_url=ep.base_url,
            api_key=self._fw_api_key,
        )
        if self._base_only:
            self._client = svc.create_base_training_client(
                base_model=self._base_model,
            )
        else:
            self._client = svc.create_training_client(
                base_model=self._base_model,
                lora_rank=self._lora_rank,
            )
        self._endpoint = ep

    def _connect(self) -> None:
        ep = self._rlor_mgr.wait_for_existing(self._job_id)
        self._use_endpoint(ep)


def _normalize_grad_accumulation_normalization(
    value: str | GradAccNormalization,
) -> GradAccNormalization:
    """Accept legacy string values and convert them to the SDK enum."""
    if isinstance(value, GradAccNormalization):
        return value
    try:
        return GradAccNormalization(str(value).lower())
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in GradAccNormalization)
        raise ValueError(
            f"Unknown grad_accumulation_normalization '{value}'. Expected one of: {valid}"
        ) from exc
