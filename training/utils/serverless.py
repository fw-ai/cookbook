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

from fireworks.training.sdk import FireworksClient, FiretitanServiceClient

from training.utils.checkpoints import TrainingCheckpoints
from training.utils.client import DEFAULT_TIMEOUT_S, ReconnectableClient


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


def setup_serverless_training(cfg, *, api_key, base_url, additional_headers, stack):
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
    service = FiretitanServiceClient(
        base_url=f"{base_url}/training/v1/serverless",
        api_key=api_key,
        default_headers=additional_headers or None,
    )
    training_client = service.create_lora_training_client(
        cfg.base_model, rank=cfg.lora_rank
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
        ServerlessCheckpointClient(cp_client, cp_client.account_id),
        trainer_id=session_id,
        log_path=cfg.log_path,
        lora_rank=cfg.lora_rank,
        serverless=True,
        current_run_id=getattr(training_client, "run_id", None),
    )
    return service, client, ckpt, session_id, cfg.max_seq_len
