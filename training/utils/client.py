"""ReconnectableClient -- wraps FiretitanTrainingClient with dispatch + wait.

Provides a clean API surface over the tinker training client.  Each method
dispatches one request and blocks until the result is ready, with a
configurable timeout to prevent indefinite hangs.

Tinker's internal polling already retries transient HTTP errors (408, 5xx).
404 ("Trainer job not found or not running") is *not* retried by tinker
because in pure tinker semantics a 404 is permanent.  The Fireworks API
gateway, however, transiently returns 404 while its DynamoDB-backed route
table is catching up to a freshly-RUNNING trainer — the trainer pod itself
is healthy and the request never reaches it.

Two-part fix lives in this module:

1. **Data-plane warmup** at connect time (``_wait_for_data_plane_ready``):
   after the SDK's ``wait_for_existing`` returns (which only verifies
   control-plane state + a single ``/healthz`` probe), we issue real
   request-path calls (``get_info``) until we see ``N`` consecutive
   successes.  This proves the gateway route is globally visible before
   the orchestrator starts a training step.

2. **Verified retry on 404** at request time (``_retry_on_transient_not_found``):
   if a 404 still surfaces during training, we re-query the control plane.
   - Job state is still ``JOB_STATE_RUNNING`` → routing race, retry with
     bounded backoff.
   - Anything else (deleted, failed, paused) → re-raise immediately so we
     fail-fast on a true not-found instead of burning ~90s of backoff.
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

# -- Data-plane warmup --------------------------------------------------------

_WARMUP_TIMEOUT_S: float = 120.0
"""Max seconds to spend warming up the gateway route after connect."""

_WARMUP_REQUIRED_SUCCESSES: int = 3
"""Consecutive successful request-path probes required before declaring ready."""

_WARMUP_PROBE_INTERVAL_S: float = 1.0
"""Sleep between successful probes (forces fresh connection-pool entries)."""

_WARMUP_BACKOFF_MAX_S: float = 8.0
"""Cap on the backoff sleep after a failed warmup probe."""

# -- Verified retry on transient 404 ------------------------------------------

_NOT_FOUND_MAX_RETRIES: int = 4
"""Max bounded retries on a 404 once we've confirmed (via control plane)
that the job is still RUNNING.  Lower than a naive client-side retry budget
because the warmup at connect time should have already eliminated the
freshly-RUNNING window."""

_NOT_FOUND_BASE_DELAY_S: float = 2.0
"""Base delay for exponential back-off on confirmed-transient 404 retries."""

_NOT_FOUND_MAX_DELAY_S: float = 15.0
"""Cap on the back-off delay between confirmed-transient 404 retries."""


def _retry_on_transient_not_found(
    fn: Callable[[], _T],
    *,
    is_running: Callable[[], bool],
    job_id: str,
) -> _T:
    """Execute *fn* and retry only on a *confirmed-transient* ``NotFoundError``.

    On a 404 we call ``is_running()`` (a control-plane probe).  If the job
    is still in ``JOB_STATE_RUNNING`` the 404 is attributable to a stale
    gateway route and we retry with bounded exponential back-off.  If
    ``is_running()`` returns False (job deleted / failed / paused) or the
    control-plane probe itself raises, we surface the original 404 immediately
    so the orchestrator can fail-fast and resume from the last DCP checkpoint
    instead of waiting out an open-ended retry budget.
    """
    last_exc: tinker.NotFoundError | None = None
    for attempt in range(_NOT_FOUND_MAX_RETRIES + 1):
        try:
            return fn()
        except tinker.NotFoundError as exc:
            last_exc = exc
            if attempt >= _NOT_FOUND_MAX_RETRIES:
                logger.error(
                    "Job %s: 404 retries exhausted (%d attempts), giving up: %s",
                    job_id, _NOT_FOUND_MAX_RETRIES, exc,
                )
                break
            try:
                still_running = is_running()
            except Exception as probe_exc:
                logger.error(
                    "Job %s: 404 received but control-plane probe failed (%s); "
                    "treating 404 as terminal to avoid masking a real error.",
                    job_id, probe_exc,
                )
                raise exc
            if not still_running:
                logger.error(
                    "Job %s: 404 received and control plane reports job is no "
                    "longer RUNNING; treating as terminal.",
                    job_id,
                )
                raise exc
            delay = min(
                _NOT_FOUND_BASE_DELAY_S * (2 ** attempt),
                _NOT_FOUND_MAX_DELAY_S,
            )
            logger.warning(
                "Job %s: confirmed-transient 404 (attempt %d/%d, control plane "
                "still reports RUNNING) — retrying in %.1fs: %s",
                job_id, attempt + 1, _NOT_FOUND_MAX_RETRIES, delay, exc,
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
    until the result is ready (or the timeout expires).  Two narrowly-scoped
    safeguards address a Fireworks-gateway routing race:

    * On connect, ``_wait_for_data_plane_ready`` issues real request-path
      probes until the gateway → trainer route is stable.  This eliminates
      the "freshly-RUNNING window" where ``healthz`` succeeds but
      ``forward_backward`` 404s.
    * If a 404 still surfaces during training, it is retried only after
      the control plane confirms the job is still ``JOB_STATE_RUNNING``.
      Real not-founds (deleted / failed / paused) propagate immediately.

    All other failures propagate to the caller.

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
        return _retry_on_transient_not_found(
            lambda: self._client.forward(data, loss_fn).result(
                timeout=self._default_timeout,
            ),
            is_running=self._is_job_running,
            job_id=self._job_id,
        )

    def forward_backward(self, data, loss_fn: str = "cross_entropy", loss_fn_config=None):
        return _retry_on_transient_not_found(
            lambda: self._client.forward_backward(
                data, loss_fn, loss_fn_config=loss_fn_config,
            ).result(timeout=self._default_timeout),
            is_running=self._is_job_running,
            job_id=self._job_id,
        )

    def forward_backward_custom(self, data, loss_fn):
        return _retry_on_transient_not_found(
            lambda: self._client.forward_backward_custom(data, loss_fn).result(
                timeout=self._default_timeout,
            ),
            is_running=self._is_job_running,
            job_id=self._job_id,
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
        return _retry_on_transient_not_found(
            lambda: self._client.optim_step(params, **kwargs).result(
                timeout=self._default_timeout,
            ),
            is_running=self._is_job_running,
            job_id=self._job_id,
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
        self._wait_for_data_plane_ready()

    def _wait_for_data_plane_ready(
        self,
        *,
        timeout_s: float = _WARMUP_TIMEOUT_S,
        required_successes: int = _WARMUP_REQUIRED_SUCCESSES,
    ) -> None:
        """Probe the actual request path until the gateway route is stable.

        ``TrainerJobManager.wait_for_existing`` only verifies that the
        control plane reports ``JOB_STATE_RUNNING`` and that a single
        ``/api/v1/healthz`` call through the gateway succeeds.  In practice,
        the gateway's DynamoDB-backed route lookup for the *generic* request
        path can lag the healthz path and across replicas / connection-pool
        entries, producing transient 404s on the first real call.

        We send ``required_successes`` consecutive ``get_info`` requests
        (which exercise the same gateway → trainer routing as ``forward`` /
        ``forward_backward``).  Only after that do we hand the client to
        the orchestrator.

        404s during warmup are treated as expected — we wait and retry.
        Non-404 errors propagate (the trainer is genuinely broken) so we
        don't silently mask real failures.
        """
        if self._client is None:
            return

        deadline = time.monotonic() + timeout_s
        attempt = 0
        consecutive_successes = 0
        last_404: tinker.NotFoundError | None = None

        while time.monotonic() < deadline:
            try:
                self._client.get_info()
            except tinker.NotFoundError as exc:
                last_404 = exc
                consecutive_successes = 0
                attempt += 1
                delay = min(
                    _NOT_FOUND_BASE_DELAY_S * (2 ** (attempt - 1)),
                    _WARMUP_BACKOFF_MAX_S,
                )
                logger.info(
                    "Job %s: warmup probe %d returned 404 (gateway route not "
                    "yet stable), retrying in %.1fs",
                    self._job_id, attempt, delay,
                )
                time.sleep(delay)
                continue
            except Exception as exc:
                logger.warning(
                    "Job %s: warmup probe failed with non-404 (%s); skipping "
                    "warmup and letting the next real call surface the error.",
                    self._job_id, exc,
                )
                return

            consecutive_successes += 1
            if consecutive_successes >= required_successes:
                logger.info(
                    "Job %s: data plane ready after %d successful warmup probes",
                    self._job_id, consecutive_successes,
                )
                return
            time.sleep(_WARMUP_PROBE_INTERVAL_S)

        if last_404 is not None:
            raise TimeoutError(
                f"Job {self._job_id}: gateway route did not stabilise within "
                f"{timeout_s:.0f}s (last error: {last_404}). The trainer pod "
                f"may be running but the API gateway cannot route to it."
            ) from last_404
        logger.warning(
            "Job %s: warmup loop exited without %d consecutive successes; "
            "proceeding anyway.",
            self._job_id, required_successes,
        )

    def _is_job_running(self) -> bool:
        """Control-plane probe used by the verified-404-retry path.

        Returns True iff the control plane reports ``JOB_STATE_RUNNING``.
        Any exception propagates so the caller can decide whether to treat
        the original 404 as terminal.
        """
        job = self._rlor_mgr.get(self._job_id)
        return job.get("state", "") == "JOB_STATE_RUNNING"

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
