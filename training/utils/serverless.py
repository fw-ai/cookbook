"""Serverless SFT setup helpers.

The serverless counterpart to ``build_service_client`` in
``training/utils/service.py``: connects to a shared, already-running pooled
trainer through the gateway serverless surface
(``{FIREWORKS_BASE_URL}/training/v1/serverless``) instead of provisioning a
dedicated trainer, and routes checkpoint list/promote through the session-scoped
endpoints. Returns the same handles ``sft_loop.main`` consumes for the dedicated
path so the training loop is identical from there on.
"""

from __future__ import annotations

from typing import Any, NoReturn

from fireworks.training.sdk import FireworksClient, FiretitanServiceClient

from training.utils.account import (
    FireworksAccountProvenanceError,
    assert_expected_fireworks_account,
)
from training.utils.checkpoints import TrainingCheckpoints
from training.utils.client import DEFAULT_TIMEOUT_S, ReconnectableClient


_DEFAULT_LORA_ALPHA = object()


def _close_after_account_guard_failure(
    service: FiretitanServiceClient,
    error: Exception,
) -> NoReturn:
    """Close a possibly connected service, then raise the provenance error."""

    try:
        service.close()
    except Exception as close_error:
        error.add_note(f"serverless service cleanup also failed: {close_error}")
        raise error from close_error
    raise error


def assert_serverless_session_account(
    service: FiretitanServiceClient,
    expected_account_id: str,
) -> str:
    """Assert that a created serverless session belongs to the gated account."""

    session_name = getattr(service, "training_session_name", None)
    parts = session_name.split("/") if isinstance(session_name, str) else []
    if (
        len(parts) != 4
        or parts[0] != "accounts"
        or parts[1] != expected_account_id
        or parts[2] != "trainingSessions"
        or not parts[3]
    ):
        _close_after_account_guard_failure(
            service,
            FireworksAccountProvenanceError(
                "serverless training session account differs from the "
                "pre-creation account gate: "
                f"session_name={session_name!r}, "
                f"expected_account={expected_account_id!r}"
            ),
        )
    return session_name


def resolve_serverless_account_before_session(
    *,
    api_key: str,
    base_url: str,
    additional_headers: dict[str, str] | None,
    expected_account_id: str | None,
) -> str:
    """Resolve account identity before constructing a session-owning service."""

    return assert_expected_fireworks_account(
        api_key=api_key,
        base_url=base_url,
        additional_headers=additional_headers,
        expected_account_id=expected_account_id,
    )


def create_lora_training_client_for_account(
    service: FiretitanServiceClient,
    *,
    expected_account_id: str,
    base_model: str,
    rank: int,
    alpha: int | None | object = _DEFAULT_LORA_ALPHA,
) -> Any:
    """Create one LoRA client and assert its session account.

    Callers must obtain ``expected_account_id`` through
    :func:`resolve_serverless_account_before_session` before constructing
    ``service``. The resource-name assertion is defense in depth against an SDK
    or routing discrepancy during session/model creation.
    """

    create_kwargs: dict[str, Any] = {
        "base_model": base_model,
        "rank": rank,
    }
    if alpha is not _DEFAULT_LORA_ALPHA:
        create_kwargs["alpha"] = alpha
    try:
        training_client = service.create_lora_training_client(**create_kwargs)
        assert_serverless_session_account(service, expected_account_id)
    except FireworksAccountProvenanceError:
        # The post-create guard already closes the service.
        raise
    except Exception as error:
        try:
            service.close()
        except Exception as close_error:
            error.add_note(f"serverless service cleanup also failed: {close_error}")
        raise
    return training_client


class ServerlessCheckpointClient:
    """Adapts the SDK's session-scoped checkpoint endpoints to the control-plane
    client protocol that ``TrainingCheckpoints`` expects (``_CheckpointLister``).

    In serverless mode checkpoint list/promote target the owning
    ``TrainingSession`` (``accounts/{a}/trainingSessions/{s}/checkpoints``)
    rather than the ``rlorTrainerJobs`` path. Save/load still go through the
    live training client to the pooled trainer; only list + promote diverge.
    """

    def __init__(self, fw_client: FireworksClient, account_id: str) -> None:
        self._fw = fw_client
        self._account_id = account_id

    def _session_name(self, session_id: str) -> str:
        return f"accounts/{self._account_id}/trainingSessions/{session_id}"

    def list_checkpoints(self, job_id: str, *, page_size: int = 200) -> list[dict]:
        # ``job_id`` is the TrainingCheckpoints trainer_id, which is the
        # owning TrainingSession id in serverless mode.
        return self._fw.list_training_session_checkpoints(
            self._session_name(job_id), page_size=page_size
        )

    def promote_checkpoint(
        self,
        job_id: str | None = None,
        checkpoint_id: str | None = None,
        output_model_id: str | None = None,
        base_model: str | None = None,
        *,
        name: str | None = None,
        hot_load_deployment_id: str | None = None,
    ) -> dict:
        # TrainingCheckpoints.promote_latest passes the full session checkpoint
        # resource name via ``name=``; hot_load_deployment_id is not used here.
        if name is None:
            raise ValueError(
                "serverless promotion requires the full session checkpoint "
                "resource name (name=accounts/.../trainingSessions/.../checkpoints/...)"
            )
        return self._fw.promote_session_checkpoint(
            name=name, output_model_id=output_model_id, base_model=base_model
        )


def setup_serverless_training(
    cfg,
    *,
    api_key,
    base_url,
    additional_headers,
    stack,
    expected_account_id: str | None = None,
):
    """Build the training + checkpoint handles for a serverless SFT run.

    Returns ``(service, client, ckpt, session_id, max_seq_len)``. The caller
    registers ``service.close`` for teardown; the internal control-plane client
    used for checkpoint list/promote is registered on the provided ``stack``
    (an ``ExitStack``) here, so it is closed on teardown too. Requires
    ``cfg.lora_rank > 0`` and a concrete ``cfg.max_seq_len`` (there is no training
    shape to resolve sequence length from on this path).
    """
    if cfg.lora_rank <= 0:
        raise ValueError(
            "serverless mode requires lora_rank > 0 (the pool is LoRA-only)."
        )
    if not cfg.max_seq_len:
        raise ValueError(
            "serverless mode requires Config.max_seq_len to be set "
            "(there is no training shape to resolve it from)."
        )
    guarded_account_id = resolve_serverless_account_before_session(
        api_key=api_key,
        base_url=base_url,
        additional_headers=additional_headers,
        expected_account_id=(
            expected_account_id
            if expected_account_id is not None
            else getattr(cfg, "expected_account_id", None)
        ),
    )
    service = FiretitanServiceClient(
        base_url=f"{base_url}/training/v1/serverless",
        api_key=api_key,
        default_headers=additional_headers or None,
    )
    training_client = create_lora_training_client_for_account(
        service,
        expected_account_id=guarded_account_id,
        base_model=cfg.base_model,
        rank=cfg.lora_rank,
        alpha=getattr(cfg, "lora_alpha", 32),
    )
    # The API gateway now returns run-scoped model ids
    # ("{run_id}:train:{seq}"). The owning CP TrainingSession remains on the
    # service holder and is the resource used for checkpoint list/promote.
    session_id = getattr(service, "training_session_id", None)
    if not session_id:
        raise RuntimeError(
            "serverless service did not expose a training_session_id; "
            "cannot resolve the training session id for checkpoint promotion."
        )
    client = ReconnectableClient.from_training_client(
        training_client,
        base_model=cfg.base_model,
        lora_rank=cfg.lora_rank,
        job_id=session_id,
        default_timeout=cfg.step_timeout or DEFAULT_TIMEOUT_S,
        service=service,
    )
    # Checkpoint list/promote use the session-scoped endpoints on the regular
    # gateway (not the serverless surface base_url). cp_client holds a persistent
    # sync httpx client, so register its close on teardown like the dedicated
    # path closes its service client.
    cp_client = FireworksClient(
        api_key=api_key, base_url=base_url, additional_headers=additional_headers
    )
    stack.callback(cp_client.close)
    ckpt = TrainingCheckpoints(
        client,
        ServerlessCheckpointClient(cp_client, guarded_account_id),
        trainer_id=session_id,
        log_path=cfg.log_path,
        lora_rank=cfg.lora_rank,
        serverless=True,
        current_run_id=getattr(training_client, "run_id", None),
    )
    return service, client, ckpt, session_id, cfg.max_seq_len
