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

from fireworks.training.sdk.client import FiretitanServiceClient, FiretitanTrainingClient
from fireworks.training.sdk.trainer import TrainerJobManager, TrainerServiceEndpoint
import tinker.lib.api_future_impl as tinker_api_future_impl
from tinker.types.future_retrieve_request import FutureRetrieveRequest as _FutureRetrieveRequest

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S: int = 600
"""Default timeout for forward / forward_backward / optim_step (10 min)."""

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
    """

    def __init__(
        self,
        rlor_mgr: TrainerJobManager,
        job_id: str,
        base_model: str,
        lora_rank: int = 0,
        api_key: str = "tml-local",
        fw_api_key: str | None = None,
        default_timeout: int = DEFAULT_TIMEOUT_S,
        endpoint: TrainerServiceEndpoint | None = None,
    ):
        self._rlor_mgr = rlor_mgr
        self._job_id = job_id
        self._base_model = base_model
        self._lora_rank = lora_rank
        self._api_key = api_key
        self._fw_api_key = fw_api_key or os.environ.get("FIREWORKS_API_KEY")
        self._default_timeout = default_timeout
        self._endpoint: TrainerServiceEndpoint | None = None
        self._client: FiretitanTrainingClient | None = None
        if endpoint:
            self._use_endpoint(endpoint)
        else:
            self._connect()

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

    def forward_backward_custom(self, data, loss_fn):
        return self._client.forward_backward_custom(data, loss_fn).result(
            timeout=self._default_timeout,
        )

    def optim_step(self, params):
        return self._client.optim_step(params).result(
            timeout=self._default_timeout,
        )

    def save_state(self, name: str, timeout: int = DCP_TIMEOUT_S):
        return self._client.save_state(name).result(timeout=timeout)

    def load_state_with_optimizer(self, path: str, timeout: int = DCP_TIMEOUT_S):
        return self._client.load_state_with_optimizer(path).result(timeout=timeout)

    def save_weights_for_sampler_ext(self, name: str, checkpoint_type: str | None = None, timeout: int = DCP_TIMEOUT_S):
        return self.inner.save_weights_for_sampler_ext(name, checkpoint_type=checkpoint_type)

    def resolve_checkpoint_path(self, name: str, source_job_id: str | None = None) -> str:
        return self.inner.resolve_checkpoint_path(name, source_job_id=source_job_id)

    def list_checkpoints(self) -> list[str]:
        return self.inner.list_checkpoints()

    # -- Internal --------------------------------------------------------------

    def _use_endpoint(self, ep: TrainerServiceEndpoint) -> None:
        kwargs: dict = {}
        if self._fw_api_key:
            kwargs["default_headers"] = {
                "X-API-Key": self._fw_api_key,
                "Authorization": f"Bearer {self._fw_api_key}",
            }
        svc = FiretitanServiceClient(
            base_url=ep.base_url, api_key=self._api_key, **kwargs,
        )
        self._client = svc.create_training_client(
            base_model=self._base_model,
            lora_rank=self._lora_rank,
        )
        self._endpoint = ep

    def _connect(self) -> None:
        ep = self._rlor_mgr.wait_for_existing(self._job_id)
        self._use_endpoint(ep)
