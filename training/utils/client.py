"""ReconnectableClient -- wraps FiretitanTrainingClient with dispatch + wait.

Provides a clean API surface over the tinker training client.  Each method
dispatches one request and blocks until the result is ready, with a
configurable timeout to prevent indefinite hangs.

No explicit retry or reconnect logic -- tinker's internal polling already
retries transient HTTP errors (408, 5xx).  If the call fails (410, timeout,
connection error), the exception propagates so the training loop can crash
cleanly and resume from the last DCP checkpoint.
"""

from __future__ import annotations

import logging
import os

from fireworks.training.sdk.client import (
    FiretitanServiceClient,
    FiretitanTrainingClient,
    GradAccNormalization,
)
from fireworks.training.sdk.trainer import TrainerJobManager, TrainerServiceEndpoint
import tinker.lib.api_future_impl as tinker_api_future_impl
from tinker.types.future_retrieve_request import FutureRetrieveRequest as _FutureRetrieveRequest

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S: int = 3600
"""Default timeout for forward / forward_backward / optim_step (60 min)."""

DCP_TIMEOUT_S: int = 2700
"""Default timeout for save_state / load_state_with_optimizer (45 min)."""


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
    until the result is ready (or the timeout expires).  No retry, no
    reconnect -- failures propagate to the caller.

    For LoRA GRPO with KL regularisation, a single LoRA trainer can serve
    both policy and reference logprobs.  Use :meth:`create_base_reference`
    to obtain a second ``ReconnectableClient`` that shares the same trainer
    job *and the same session* but operates on a ``base-<hex>`` model handle
    (no LoRA adapter).  Sharing the service client is essential — creating
    a second :class:`FiretitanServiceClient` would create a second session
    and reset the trainer, unloading the policy LoRA.

    The two clients must not dispatch concurrent requests; the trainer
    serialises ops per session.  The cookbook RL loop already alternates
    policy and reference forward passes synchronously.
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
        service: FiretitanServiceClient | None = None,
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
        self._service: FiretitanServiceClient | None = service
        self._client: FiretitanTrainingClient | None = None
        self._owns_service = service is None
        self._closed = False
        if endpoint:
            self._use_endpoint(endpoint)
        else:
            self._connect()

    def create_base_reference(self) -> ReconnectableClient:
        """Create a base-only reference client sharing this trainer's session.

        Returns a new :class:`ReconnectableClient` that reuses the same
        :class:`FiretitanServiceClient` (and therefore the same session) as
        this client, but issues forward passes against a ``base-<hex>``
        model handle (LoRA adapter disabled).

        The returned client must NOT be used concurrently with the policy
        client; calls are serialised through the shared session.  Do not
        call ``forward_backward`` or ``optim_step`` on it.
        """
        assert self._endpoint is not None and self._service is not None, (
            "policy client must be connected before creating a base reference"
        )
        return ReconnectableClient(
            rlor_mgr=self._rlor_mgr,
            job_id=self._job_id,
            base_model=self._base_model,
            lora_rank=0,
            fw_api_key=self._fw_api_key,
            default_timeout=self._default_timeout,
            endpoint=self._endpoint,
            service=self._service,
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
        return self._client.forward(data, loss_fn).result(
            timeout=self._default_timeout,
        )

    def forward_backward(self, data, loss_fn: str = "cross_entropy", loss_fn_config=None):
        return self._client.forward_backward(data, loss_fn, loss_fn_config=loss_fn_config).result(
            timeout=self._default_timeout,
        )

    def forward_backward_custom(self, data, loss_fn):
        return self._client.forward_backward_custom(data, loss_fn).result(
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
        return self._client.optim_step(params, **kwargs).result(
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

    def load_adapter(self, adapter_path: str, timeout: int = DCP_TIMEOUT_S):
        """Load HF PEFT adapter weights into the current LoRA session.

        Weights-only load (no optimizer, no LR schedule, no data cursor).
        Intended for cross-job warm-start from a promoted Model resource
        or an uploaded HF adapter.

        Args:
            adapter_path: ``gs://`` URI or absolute local path to an HF
                PEFT adapter directory (must contain
                ``adapter_model.safetensors``).

        Raises:
            AttributeError: underlying SDK does not expose ``load_adapter``
                (requires fireworks-ai-python with the load_adapter client
                method; see PR #122 in stainless-sdks/fireworks-ai-python).
        """
        return self._client.load_adapter(adapter_path).result(timeout=timeout)

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

        When this client shares its :class:`FiretitanServiceClient` with
        another :class:`ReconnectableClient` (e.g. created via
        :meth:`create_base_reference`), this client does not own the holder
        and skips the holder/telemetry teardown — the owning client will
        clean it up.
        """
        if self._closed:
            return
        self._closed = True

        client = self._client
        self._client = None
        self._endpoint = None
        if client is None or not self._owns_service:
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
        if self._service is None:
            self._service = FiretitanServiceClient(
                base_url=ep.base_url,
                api_key=self._fw_api_key,
            )
        if self._base_only:
            self._client = self._service.create_base_training_client(
                base_model=self._base_model,
            )
        else:
            self._client = self._service.create_training_client(
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
