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
import warnings
from enum import Enum
from types import SimpleNamespace
from typing import Any

try:
    from fireworks.training.sdk.client import (
        FiretitanServiceClient,
        FiretitanTrainingClient,
        GradAccNormalization,
    )
except ImportError:
    from fireworks.training.sdk.client import (
        FiretitanServiceClient,
        FiretitanTrainingClient,
    )

    class GradAccNormalization(str, Enum):
        NUM_SEQUENCES = "num_sequences"
        NUM_LOSS_TOKENS = "num_loss_tokens"
from fireworks.training.sdk.trainer import TrainerJobManager, TrainerServiceEndpoint
import tinker.lib.api_future_impl as tinker_api_future_impl
from tinker.types.future_retrieve_request import FutureRetrieveRequest as _FutureRetrieveRequest

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S: int = 3600
"""Default timeout for forward / forward_backward / optim_step (60 min)."""

DCP_TIMEOUT_S: int = 2700
"""Default timeout for save_state / load_state_with_optimizer (45 min)."""

_DEFAULT_FBC_OUTPUT = "logprobs"
_DEFAULT_FBC_POOLING = "mean"
"""Legacy SFT/ORPO/DPO recipes use these defaults and must not pass embedding kwargs."""

_MIN_SDK_FOR_CONTRASTIVE = "1.2.0a78"
"""First published ``fireworks-ai[training]`` release with ``forward_backward_contrastive``
in an importable cookbook (a77 has the method but lacks other symbols utils imports)."""


def _install_tinker_future_retrieve_compat() -> None:
    """Keep metadata-only disabled until all trainer versions support it."""
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

    The KL/DPO reference is provisioned by the SDK
    (``service.create_reference_client``) and wrapped inline by each recipe
    via :meth:`from_training_client` (``base_only=True``). The SDK decides
    whether the reference reuses the policy session (LoRA) or runs on a
    separate frozen reference trainer (full-parameter / explicit reference
    shape), and owns that trainer's lifecycle — the cookbook never manages a
    second service.
    """

    def __init__(
        self,
        rlor_mgr: TrainerJobManager | None,
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

    @classmethod
    def from_training_client(
        cls,
        client: FiretitanTrainingClient,
        *,
        base_model: str,
        lora_rank: int = 0,
        job_id: str,
        default_timeout: int = DEFAULT_TIMEOUT_S,
        service: FiretitanServiceClient | None = None,
        base_only: bool = False,
    ) -> "ReconnectableClient":
        self = cls.__new__(cls)
        self._rlor_mgr = None
        self._job_id = job_id
        self._base_model = base_model
        self._lora_rank = lora_rank
        self._base_only = base_only
        self._fw_api_key = None
        self._default_timeout = default_timeout
        self._endpoint = SimpleNamespace(job_id=job_id, base_url="")
        self._service = service
        self._client = client
        self._owns_service = False
        self._closed = False
        return self

    def _require_client(self) -> FiretitanTrainingClient:
        if self._client is None:
            raise RuntimeError("ReconnectableClient is closed or not connected")
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

    def forward_backward_custom(
        self,
        data,
        loss_fn,
        *,
        output: str = "logprobs",
        pooling: str = "mean",
    ):
        """Client-side custom loss over a single forward.

        ``output`` selects what the trainer returns to the client loss_fn:

          - ``"logprobs"`` (default, used by SFT/ORPO/DPO).
          - ``"embedding"``: pooled per-datum vectors.
          - ``"cos_similarity_matrix"``: rows of the in-batch ``[N, N]``
            cosine-similarity matrix.

        ``pooling`` only applies to the embedding outputs ("mean" or "last").
        """
        if output == _DEFAULT_FBC_OUTPUT and pooling == _DEFAULT_FBC_POOLING:
            # Preserve the pre-embedding call shape so older pinned SDKs and
            # every existing SFT/ORPO/DPO recipe keep working.
            fb = self._client.forward_backward_custom(data, loss_fn)
        else:
            fb = self._client.forward_backward_custom(
                data, loss_fn, output=output, pooling=pooling,
            )
        return fb.result(timeout=self._default_timeout)

    def forward_backward_contrastive(
        self,
        data,
        *,
        num_queries: int,
        temperature: float,
        pooling: str = "last",
        num_extra_negatives: int = 0,
    ):
        """Server-side bidirectional InfoNCE in one round trip.

        ``data`` is laid out as ``[Q_0..Q_{B-1}, D_0..D_{B-1}]`` (plus an
        optional ``num_extra_negatives`` tail of unpaired hard negatives). The
        trainer pools, L2-normalizes, builds the similarity matrix, and runs
        cross-entropy + backward itself, returning a metrics dict with ``loss``.
        """
        fb_contrastive = getattr(self._client, "forward_backward_contrastive", None)
        if fb_contrastive is None:
            raise RuntimeError(
                f"forward_backward_contrastive requires fireworks-ai[training]>="
                f"{_MIN_SDK_FOR_CONTRASTIVE}. Upgrade the Training SDK or use "
                "output_mode='embedding' or 'cos_similarity_matrix'."
            )
        return fb_contrastive(
            data,
            num_queries=num_queries,
            temperature=temperature,
            pooling=pooling,
            num_extra_negatives=num_extra_negatives,
        ).result(timeout=self._default_timeout)

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

    # -- Non-blocking submit API (for pipelining) ------------------------------
    #
    # The methods above block on the result. These return the raw future so a
    # caller can overlap fwd/bwd + optim across steps (the SFT pipeline). Pair
    # with ``future.result(timeout=...)``. They exist so callers use named
    # methods instead of reaching through the wrapper.

    def submit_forward_backward(self, data, loss_fn: str = "cross_entropy", loss_fn_config=None):
        return self._require_client().forward_backward(data, loss_fn, loss_fn_config=loss_fn_config)

    def submit_optim_step(
        self,
        params,
        grad_accumulation_normalization: str | GradAccNormalization | None = None,
    ):
        kwargs: dict = {}
        if grad_accumulation_normalization is not None:
            kwargs["grad_accumulation_normalization"] = _normalize_grad_accumulation_normalization(
                grad_accumulation_normalization
            )
        return self._require_client().optim_step(params, **kwargs)

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

        Weights-only (no optimizer, no LR schedule, no data cursor).
        """
        return self._client.load_adapter(adapter_path).result(timeout=timeout)

    def save_weights_for_sampler_ext(
        self, name: str, checkpoint_type: str | None = None, timeout: int = DCP_TIMEOUT_S
    ) -> Any:
        """Deprecated compatibility shim.

        New cookbook code should call :meth:`save_weights_for_sampler`, which
        returns the SDK's standard ``path`` field. This wrapper exists only for
        old callers that still read ``snapshot_name``.
        """
        warnings.warn(
            "ReconnectableClient.save_weights_for_sampler_ext() is deprecated; "
            "use save_weights_for_sampler() and read the returned .path instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        saved = self.save_weights_for_sampler(
            name,
            checkpoint_type=checkpoint_type,
            timeout=timeout,
        )
        return SimpleNamespace(path=saved.path, snapshot_name=saved.path)

    def save_weights_for_sampler(
        self,
        name: str,
        ttl_seconds: int | None = None,
        *,
        checkpoint_type: str | None = None,
        timeout: int = DCP_TIMEOUT_S,
    ) -> Any:
        return self._require_client().save_weights_for_sampler(
            name,
            ttl_seconds=ttl_seconds,
            checkpoint_type=checkpoint_type,
        ).result(timeout=timeout)

    def save_weights_and_get_sampler(
        self,
        name: str,
        *,
        checkpoint_type: str | None = None,
        tokenizer: object | None = None,
        concurrency_controller: object | None = None,
        timeout: int = DCP_TIMEOUT_S,
    ) -> Any:
        """Save sampler weights, hot-load them, and return the refreshed sampler.

        FireTitan's one-call equivalent of tinker's
        ``save_weights_and_get_sampling_client``: it saves a sampler snapshot on
        the trainer and hot-loads it into the colocated inference deployment,
        then returns the (client-side-tokenizing) ``DeploymentSampler``.
        Behavior is equivalent to saving a sampler snapshot and then creating a
        deployment sampler from the returned snapshot identity. FireTitan can't
        return an in-service sampler the way tinker does because sampling runs
        on a separate deployment. Blocks until the hot-load is ready. Requires
        the SDK-managed service.
        """
        if self._service is None:
            raise RuntimeError("save_weights_and_get_sampler requires the managed service")
        saved = self.save_weights_for_sampler(
            name,
            checkpoint_type=checkpoint_type,
            timeout=timeout,
        )
        return self._service.create_deployment_sampler(
            model_path=saved.path,
            tokenizer=tokenizer,
            concurrency_controller=concurrency_controller,
        )

    def resolve_checkpoint_path(self, name: str, source_job_id: str | None = None) -> str:
        return self._require_client().resolve_checkpoint_path(name, source_job_id=source_job_id)

    def list_checkpoints(self) -> list[str]:
        return self._require_client().list_checkpoints()

    def unload_model(self, timeout: float = 30.0) -> None:
        """POST ``/api/v1/unload_model`` to drop this client's LoRA session."""
        if self._closed or self._client is None or not self._owns_service:
            return
        if self._lora_rank == 0:
            # Full-param: no LoRA session to remove, and the op's api-side
            # hook clears request-sequencing state that cross-job reads need.
            return
        try:
            from tinker.types.unload_model_request import UnloadModelRequest
            from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
        except ImportError:
            logger.debug("unload_model: tinker types unavailable — skipping")
            return

        client = self._client
        holder = client.holder
        if holder is None:
            return

        try:
            model_id = client._guaranteed_model_id()
        except Exception:
            return

        async def _do_unload():
            with holder.aclient(ClientConnectionPoolType.TRAIN) as async_client:
                return await async_client.models.unload(
                    request=UnloadModelRequest(model_id=model_id),
                )

        try:
            fut = holder.run_coroutine_threadsafe(_do_unload())
            fut.result(timeout=timeout)
            logger.info(
                "unload_model: dropped server session %s on job %s",
                model_id, self._job_id,
            )
        except Exception as e:
            logger.warning(
                "unload_model: failed for job %s model %s: %s",
                self._job_id, model_id, e,
            )

    def close(self, timeout: float = 5.0) -> None:
        """Drop the server-side session, then stop local Tinker background tasks.

        Shared-service clients (e.g. base-reference) skip both steps — the
        owning client will clean up.
        """
        if self._closed:
            return

        # Unload first while the holder is still alive to dispatch the POST.
        self.unload_model(timeout=timeout)

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
        if self._rlor_mgr is None:
            raise RuntimeError("ReconnectableClient cannot connect without a TrainerJobManager")
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
