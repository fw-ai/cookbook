"""Standalone Fireworks infrastructure provisioning for training recipes.

Run from this directory with its own uv environment, for example:

    uv run python provision.py --recipe rl

All recipes are defined in the sibling ``fireworks.yaml`` file.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType, UnionType
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints

import hydra
import yaml
from fireworks.training.sdk import FiretitanServiceClient
from fireworks.training.sdk.client import FiretitanTrainingClient
from omegaconf import DictConfig, OmegaConf

from training.utils import (
    AdaptiveConcurrencyController,
    ConcurrencyConfig,
    DeployConfig,
    ReconnectableClient,
    TrainerConfig,
    build_service_client,
    load_deployment_tokenizer,
    read_api_extra_headers_env,
)
from training.utils.rl.grpo import validate_grpo_config

ProvisionMode = Literal["sft", "rl", "distillation", "dpo"]
PROVISION_MODES: tuple[ProvisionMode, ...] = ("sft", "rl", "distillation", "dpo")
# User-facing mode aliases that canonicalize to a real mode. RFT and RL are the
# same training mode, so "rft" is accepted everywhere and resolves to "rl".
MODE_ALIASES: dict[str, ProvisionMode] = {"rft": "rl"}
FIREWORKS_YAML = Path(__file__).with_name("fireworks.yaml")

# Top-level YAML keys. Recipe entries live under the explicit `recipe` block.
RESERVED_SECTIONS: frozenset[str] = frozenset({"common", "trainers", "deployments", "recipe"})

logger = logging.getLogger(__name__)

TRAINER_HEALTHY_STATES = {
    "JOB_STATE_RUNNING",
    "JOB_STATE_READY",
    "JOB_STATE_CREATING",
    "JOB_STATE_PENDING",
}
DEPLOYMENT_HEALTHY_STATES = {"READY"}
# Terminal trainer states that mean the job is fully torn down. A deleted RLOR
# trainer is surfaced on the public API as JOB_STATE_ARCHIVED (a retention
# tombstone: no resources/billing), so it counts as cleaned up.
_TRAINER_CLEANED_STATES = {
    "JOB_STATE_DELETED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_ARCHIVED",
}
# Trainer display names must be shorter than 64 characters (SDK validation).
_DISPLAY_NAME_MAX_LENGTH = 63


@dataclass
class ResourceStatus:
    kind: str
    resource_id: str
    state: str
    healthy: bool
    detail: str = ""


@dataclass
class FireworksProvisionInfra:
    """Resources created for a provision run."""

    mode: ProvisionMode
    service: FiretitanServiceClient
    training_client: FiretitanTrainingClient
    policy: ReconnectableClient
    policy_job_id: str
    deployment_id: str | None
    max_seq_len: int | None
    accelerator_type: str | None
    accelerator_count: int | None
    training_profile: Any
    reference: ReconnectableClient | None = None
    reference_job_id: str | None = None
    sampler: Any | None = None
    concurrency_controller: Any | None = None
    teacher_models: list[str] = field(default_factory=list)
    teacher_job_ids: list[str] = field(default_factory=list)
    teacher_base_urls: list[str] = field(default_factory=list)
    cleanup_callbacks: list[Callable[[], None]] = field(default_factory=list)
    trainer_manager: Any | None = None
    deployment_manager: Any | None = None
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for cleanup in reversed(self.cleanup_callbacks):
            cleanup()
        self.service.close()

    def check_status(self) -> list[ResourceStatus]:
        statuses = [_check_trainer_status(self.trainer_manager, self.policy_job_id)]
        if self.reference_job_id:
            statuses.append(_check_trainer_status(self.trainer_manager, self.reference_job_id))
        for teacher_job_id in self.teacher_job_ids:
            statuses.append(_check_trainer_status(self.trainer_manager, teacher_job_id))
        if self.deployment_id:
            statuses.append(_check_deployment_status(self.deployment_manager, self.deployment_id))
        return statuses

    def unhealthy_statuses(self) -> list[ResourceStatus]:
        return [status for status in self.check_status() if not status.healthy]


def init_fireworks_infra(
    mode: ProvisionMode,
    cfg: Any,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    additional_headers: dict[str, str] | None = None,
    cleanup_on_close: bool = False,
    cleanup_existing: bool = False,
    cleanup_deployment_on_close: str | None = None,
) -> FireworksProvisionInfra:
    """Create Fireworks training infrastructure for one recipe mode."""
    if mode not in PROVISION_MODES:
        raise ValueError(f"mode must be one of {PROVISION_MODES}, got {mode!r}")
    resolved_api_key = api_key or os.environ["FIREWORKS_API_KEY"]
    resolved_base_url = base_url or os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    resolved_headers = additional_headers
    if resolved_headers is None:
        resolved_headers = read_api_extra_headers_env()

    if mode == "sft":
        return _init_sft_infra(
            cfg,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            additional_headers=resolved_headers,
            cleanup_on_close=cleanup_on_close,
            cleanup_existing=cleanup_existing,
            cleanup_deployment_on_close=cleanup_deployment_on_close,
        )
    if mode == "rl":
        return _init_rl_infra(
            cfg,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            additional_headers=resolved_headers,
            cleanup_on_close=cleanup_on_close,
            cleanup_existing=cleanup_existing,
            cleanup_deployment_on_close=cleanup_deployment_on_close,
        )
    if mode == "dpo":
        return _init_dpo_infra(
            cfg,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            additional_headers=resolved_headers,
            cleanup_on_close=cleanup_on_close,
            cleanup_existing=cleanup_existing,
            cleanup_deployment_on_close=cleanup_deployment_on_close,
        )
    elif mode == "distillation":
        return _init_distillation_infra(
            cfg,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            additional_headers=resolved_headers,
            cleanup_on_close=cleanup_on_close,
            cleanup_existing=cleanup_existing,
            cleanup_deployment_on_close=cleanup_deployment_on_close,
        )
    else:
        raise ValueError(f"mode must be one of {PROVISION_MODES}, got {mode!r}")


def run_until_interrupted(
    mode: ProvisionMode,
    cfg: Any,
    *,
    progress_interval_s: float = 15.0,
    health_check_interval_s: float = 60.0,
    cleanup_existing: bool = False,
) -> FireworksProvisionInfra:
    """Start infrastructure, print heartbeat progress, and clean up on exit."""
    begin_cleanup, restore_signals = _install_cleanup_signal_handlers()
    infra: FireworksProvisionInfra | None = None
    started_at = time.monotonic()
    try:
        infra = init_fireworks_infra(
            mode,
            cfg,
            cleanup_on_close=True,
            cleanup_existing=cleanup_existing,
            cleanup_deployment_on_close="delete",
        )
        next_health_check_at = started_at
        logger.info("Fireworks %s infra is running. Press Ctrl+C to clean up.", mode)
        while True:
            print(_format_progress(infra, started_at), flush=True)
            now = time.monotonic()
            if now >= next_health_check_at:
                unhealthy = infra.unhealthy_statuses()
                if unhealthy:
                    print(_format_unhealthy_report(unhealthy), flush=True)
                next_health_check_at = now + health_check_interval_s
            time.sleep(min(progress_interval_s, max(0.1, next_health_check_at - time.monotonic())))
    except KeyboardInterrupt:
        if infra is None:
            # Interrupted mid-provision. The SDK already tore down the partially
            # created trainer/deployment while unwinding (and the signal shield
            # kept further Ctrl+C from aborting that teardown).
            print(
                "\nInterrupted during provisioning; the partially created trainer/deployment "
                "have been torn down.",
                flush=True,
            )
        else:
            print("\nInterrupted; cleaning up Fireworks infra...", flush=True)
    finally:
        try:
            if infra is not None:
                begin_cleanup()
                _cleanup_until_complete(infra, started_at=started_at, progress_interval_s=progress_interval_s)
        finally:
            restore_signals()
    if infra is None:
        raise SystemExit(130)
    return infra


def _init_sft_infra(
    cfg: Any,
    *,
    api_key: str,
    base_url: str,
    additional_headers: dict[str, str] | None,
    cleanup_on_close: bool,
    cleanup_existing: bool,
    cleanup_deployment_on_close: str | None,
) -> FireworksProvisionInfra:
    if getattr(cfg, "serverless", False):
        raise ValueError("provision.py does not support SFT serverless mode; use the SFT recipe directly.")
    if not cfg.tokenizer_model:
        raise ValueError("Config.tokenizer_model is required for SFT provisioning.")
    service = _build_managed_service(
        cfg,
        api_key=api_key,
        base_url=base_url,
        additional_headers=additional_headers,
        tokenizer_model=cfg.tokenizer_model,
        deployment=None,
        cleanup_on_close=cleanup_on_close,
        cleanup_existing=cleanup_existing,
        cleanup_deployment_on_close=cleanup_deployment_on_close,
        reference_required=False,
    )
    try:
        training_client, policy = _make_policy(service, cfg)
        return _infra_from_service(
            mode="sft",
            service=service,
            training_client=training_client,
            policy=policy,
            sampler=None,
        )
    except BaseException:
        service.close()
        raise


def _init_rl_infra(
    cfg: Any,
    *,
    api_key: str,
    base_url: str,
    additional_headers: dict[str, str] | None,
    cleanup_on_close: bool,
    cleanup_existing: bool,
    cleanup_deployment_on_close: str | None,
) -> FireworksProvisionInfra:
    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required for RL provisioning.")
    validate_grpo_config(
        kl_beta=cfg.kl_beta,
        eps_clip=getattr(cfg, "eps_clip", 0.2),
        eps_clip_high=getattr(cfg, "eps_clip_high", None),
        reference_training_shape_id=getattr(
            cfg.trainer, "reference_training_shape_id", None
        ),
        reference_job_id=getattr(cfg.trainer, "reference_job_id", None),
        ppo_n_minibatches=getattr(cfg, "ppo_n_minibatches", 1),
    )
    service = _build_managed_service(
        cfg,
        api_key=api_key,
        base_url=base_url,
        additional_headers=additional_headers,
        tokenizer_model=cfg.deployment.tokenizer_model,
        deployment=cfg.deployment,
        cleanup_on_close=cleanup_on_close,
        cleanup_existing=cleanup_existing,
        cleanup_deployment_on_close=cleanup_deployment_on_close,
        reference_required=cfg.kl_beta > 0,
    )
    try:
        training_client, policy = _make_policy(service, cfg)
        reference = None
        reference_job_id = None
        if cfg.kl_beta > 0:
            reference = ReconnectableClient.from_training_client(
                service.create_reference_client(
                    cfg.base_model,
                    lora_rank=cfg.lora_rank,
                ),
                base_model=cfg.base_model,
                lora_rank=0,
                job_id=service.reference_client_job_id,
                default_timeout=cfg.step_timeout or 3600,
                service=service,
                base_only=True,
            )
            reference_job_id = service.reference_trainer_job_id
        sampler, concurrency_controller = _create_sampler(service, cfg)
        return _infra_from_service(
            mode="rl",
            service=service,
            training_client=training_client,
            policy=policy,
            reference=reference,
            reference_job_id=reference_job_id,
            sampler=sampler,
            concurrency_controller=concurrency_controller,
        )
    except BaseException:
        service.close()
        raise


def _init_dpo_infra(
    cfg: Any,
    *,
    api_key: str,
    base_url: str,
    additional_headers: dict[str, str] | None,
    cleanup_on_close: bool,
    cleanup_existing: bool,
    cleanup_deployment_on_close: str | None,
) -> FireworksProvisionInfra:
    if not cfg.tokenizer_model:
        raise ValueError("Config.tokenizer_model is required for DPO provisioning.")
    # DPO is offline preference training: a policy trainer plus a frozen
    # reference (always required), and no rollout deployment or sampler.
    service = _build_managed_service(
        cfg,
        api_key=api_key,
        base_url=base_url,
        additional_headers=additional_headers,
        tokenizer_model=cfg.tokenizer_model,
        deployment=None,
        cleanup_on_close=cleanup_on_close,
        cleanup_existing=cleanup_existing,
        cleanup_deployment_on_close=cleanup_deployment_on_close,
        reference_required=True,
    )
    try:
        training_client, policy = _make_policy(service, cfg)
        reference = ReconnectableClient.from_training_client(
            service.create_reference_client(cfg.base_model, lora_rank=cfg.lora_rank),
            base_model=cfg.base_model,
            lora_rank=0,
            job_id=service.reference_client_job_id,
            default_timeout=cfg.step_timeout or 3600,
            service=service,
            base_only=True,
        )
        return _infra_from_service(
            mode="dpo",
            service=service,
            training_client=training_client,
            policy=policy,
            reference=reference,
            reference_job_id=service.reference_trainer_job_id,
            sampler=None,
        )
    except BaseException:
        service.close()
        raise


def _init_distillation_infra(
    cfg: Any,
    *,
    api_key: str,
    base_url: str,
    additional_headers: dict[str, str] | None,
    cleanup_on_close: bool,
    cleanup_existing: bool,
    cleanup_deployment_on_close: str | None,
) -> FireworksProvisionInfra:
    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required for distillation provisioning.")
    service = _build_managed_service(
        cfg,
        api_key=api_key,
        base_url=base_url,
        additional_headers=additional_headers,
        tokenizer_model=cfg.deployment.tokenizer_model,
        deployment=cfg.deployment,
        cleanup_on_close=cleanup_on_close,
        cleanup_existing=cleanup_existing,
        cleanup_deployment_on_close=cleanup_deployment_on_close,
        reference_required=False,
    )
    cleanup_callbacks: list[Callable[[], None]] = []
    try:
        training_client, policy = _make_policy(service, cfg)
        sampler, concurrency_controller = _create_sampler(service, cfg)
        return _infra_from_service(
            mode="distillation",
            service=service,
            training_client=training_client,
            policy=policy,
            sampler=sampler,
            concurrency_controller=concurrency_controller,
            teacher_models=[cfg.teacher_model] if getattr(cfg, "teacher_model", "") else [],
            teacher_job_ids=[],
            teacher_base_urls=[],
            cleanup_callbacks=cleanup_callbacks,
        )
    except BaseException:
        for cleanup in reversed(cleanup_callbacks):
            cleanup()
        service.close()
        raise


def _build_managed_service(
    cfg: Any,
    *,
    api_key: str,
    base_url: str,
    additional_headers: dict[str, str] | None,
    tokenizer_model: str,
    deployment: Any | None,
    cleanup_on_close: bool,
    cleanup_existing: bool,
    cleanup_deployment_on_close: str | None,
    reference_required: bool,
) -> FiretitanServiceClient:
    should_cleanup_trainer = cleanup_on_close and (cleanup_existing or cfg.trainer.job_id is None)
    should_cleanup_deployment = cleanup_on_close and deployment is not None and (
        cleanup_existing or deployment.deployment_id is None
    )
    resolved_deployment_cleanup = None
    if should_cleanup_deployment:
        resolved_deployment_cleanup = cleanup_deployment_on_close or "scale_to_zero"
    return build_service_client(
        api_key=api_key,
        base_url=base_url,
        additional_headers=additional_headers,
        base_model=cfg.base_model,
        tokenizer_model=tokenizer_model,
        lora_rank=cfg.lora_rank,
        max_context_length=cfg.max_seq_len,
        learning_rate=cfg.learning_rate,
        trainer=cfg.trainer,
        deployment=deployment,
        hotload_timeout_s=getattr(cfg, "weight_sync_timeout", None),
        cleanup_trainer_on_close=should_cleanup_trainer,
        cleanup_deployment_on_close=resolved_deployment_cleanup,
        reference_required=reference_required,
    )


def _make_policy(
    service: FiretitanServiceClient,
    cfg: Any,
) -> tuple[FiretitanTrainingClient, ReconnectableClient]:
    training_client = service.create_training_client(cfg.base_model, lora_rank=cfg.lora_rank)
    policy = ReconnectableClient.from_training_client(
        training_client,
        base_model=cfg.base_model,
        lora_rank=cfg.lora_rank,
        job_id=service.trainer_job_id,
        default_timeout=cfg.step_timeout or 3600,
        service=service,
    )
    return training_client, policy


def _create_sampler(service: FiretitanServiceClient, cfg: Any) -> tuple[Any, Any]:
    tokenizer = load_deployment_tokenizer(cfg.deployment)
    initial_window = cfg.concurrency.initial_window or (8 * (cfg.deployment.replica_count or 1))
    concurrency_controller = AdaptiveConcurrencyController(
        initial_window=initial_window,
        min_window=cfg.concurrency.min_window,
        max_window=cfg.concurrency.max_window,
        prefill_queue_target=cfg.concurrency.prefill_queue_target,
        adjustment_interval=cfg.concurrency.rollout_adjustment_interval,
    )
    sampler = service.create_deployment_sampler(
        tokenizer=tokenizer,
        concurrency_controller=concurrency_controller,
    )
    return sampler, concurrency_controller


def _check_trainer_status(manager: Any | None, job_id: str) -> ResourceStatus:
    if manager is None:
        return ResourceStatus("trainer", job_id, "UNKNOWN", False, "trainer manager is unavailable")
    try:
        payload = manager.get(job_id)
    except Exception as exc:  # noqa: BLE001 - status reporting should include API failures
        return ResourceStatus("trainer", job_id, "ERROR", False, str(exc))
    state = str(payload.get("state") or "UNKNOWN")
    return ResourceStatus(
        kind="trainer",
        resource_id=job_id,
        state=state,
        healthy=state in TRAINER_HEALTHY_STATES,
        detail=str(payload.get("error") or payload.get("errorMessage") or payload.get("statusMessage") or ""),
    )


def _check_deployment_status(manager: Any | None, deployment_id: str) -> ResourceStatus:
    if manager is None:
        return ResourceStatus("deployment", deployment_id, "UNKNOWN", False, "deployment manager is unavailable")
    try:
        deployment = manager.get(deployment_id)
    except Exception as exc:  # noqa: BLE001 - status reporting should include API failures
        return ResourceStatus("deployment", deployment_id, "ERROR", False, str(exc))
    if deployment is None:
        return ResourceStatus("deployment", deployment_id, "NOT_FOUND", False)
    state = str(getattr(deployment, "state", "UNKNOWN") or "UNKNOWN")
    return ResourceStatus(
        kind="deployment",
        resource_id=deployment_id,
        state=state,
        healthy=state in DEPLOYMENT_HEALTHY_STATES,
    )


def _infra_from_service(
    *,
    mode: ProvisionMode,
    service: FiretitanServiceClient,
    training_client: FiretitanTrainingClient,
    policy: ReconnectableClient,
    reference: ReconnectableClient | None = None,
    reference_job_id: str | None = None,
    sampler: Any | None = None,
    concurrency_controller: Any | None = None,
    teacher_models: list[str] | None = None,
    teacher_job_ids: list[str] | None = None,
    teacher_base_urls: list[str] | None = None,
    cleanup_callbacks: list[Callable[[], None]] | None = None,
) -> FireworksProvisionInfra:
    deployment_id = service.managed_deployment_id
    handle = getattr(service, "_managed_handle", None)
    trainer_manager = getattr(handle, "trainer_manager", None)
    deployment_manager = getattr(handle, "deployment_manager", None)
    logger.info(
        "Fireworks %s infra ready (trainer=%s, deployment=%s)",
        mode,
        service.trainer_job_id,
        deployment_id,
    )
    return FireworksProvisionInfra(
        mode=mode,
        service=service,
        training_client=training_client,
        policy=policy,
        reference=reference,
        sampler=sampler,
        concurrency_controller=concurrency_controller,
        policy_job_id=service.trainer_job_id,
        reference_job_id=reference_job_id,
        deployment_id=deployment_id,
        max_seq_len=service.max_context_length,
        accelerator_type=service.accelerator_type,
        accelerator_count=service.accelerator_count,
        training_profile=service.training_profile,
        teacher_models=teacher_models or [],
        teacher_job_ids=teacher_job_ids or [],
        teacher_base_urls=teacher_base_urls or [],
        cleanup_callbacks=cleanup_callbacks or [],
        trainer_manager=trainer_manager,
        deployment_manager=deployment_manager,
    )


def _format_progress(infra: FireworksProvisionInfra, started_at: float) -> str:
    elapsed_s = int(time.monotonic() - started_at)
    parts = [
        f"[{elapsed_s:>5}s] Fireworks {infra.mode} infra alive",
        f"trainer={infra.policy_job_id}",
    ]
    if infra.deployment_id:
        parts.append(f"deployment={infra.deployment_id}")
    if infra.reference_job_id:
        parts.append(f"reference={infra.reference_job_id}")
    if infra.teacher_job_ids:
        parts.append(f"teachers={','.join(infra.teacher_job_ids)}")
    if infra.accelerator_type:
        parts.append(f"accelerator={infra.accelerator_type}x{infra.accelerator_count or '?'}")
    if infra.max_seq_len:
        parts.append(f"max_seq_len={infra.max_seq_len}")
    return " | ".join(parts)


def _format_unhealthy_report(statuses: list[ResourceStatus]) -> str:
    lines = ["One or more Fireworks resources are no longer running:"]
    for status in statuses:
        line = f"- {status.kind} {status.resource_id}: state={status.state}"
        if status.detail:
            line = f"{line} ({status.detail})"
        lines.append(line)
    lines.append("Continuing to monitor remaining resources.")
    return "\n".join(lines)


def _cleanup_until_complete(
    infra: FireworksProvisionInfra,
    *,
    started_at: float,
    progress_interval_s: float,
) -> None:
    try:
        infra.close()
    except Exception as exc:  # noqa: BLE001 - keep the cleanup monitor alive
        print(f"Cleanup request raised an error; continuing to monitor resources: {exc}", flush=True)
    while True:
        statuses = _cleanup_statuses(infra)
        print(_format_cleanup_progress(infra, statuses, started_at), flush=True)
        if all(status.healthy for status in statuses):
            return
        time.sleep(progress_interval_s)


def _cleanup_statuses(infra: FireworksProvisionInfra) -> list[ResourceStatus]:
    statuses = [_check_trainer_cleanup_status(infra.trainer_manager, infra.policy_job_id)]
    if infra.reference_job_id:
        statuses.append(_check_trainer_cleanup_status(infra.trainer_manager, infra.reference_job_id))
    for teacher_job_id in infra.teacher_job_ids:
        statuses.append(_check_trainer_cleanup_status(infra.trainer_manager, teacher_job_id))
    if infra.deployment_id:
        statuses.append(_check_deployment_cleanup_status(infra.deployment_manager, infra.deployment_id))
    return statuses


def _check_trainer_cleanup_status(manager: Any | None, job_id: str) -> ResourceStatus:
    if manager is None:
        return ResourceStatus("trainer", job_id, "UNKNOWN", True, "trainer manager is unavailable")
    try_get = getattr(manager, "try_get", None)
    try:
        payload = try_get(job_id) if try_get is not None else manager.get(job_id)
    except Exception as exc:  # noqa: BLE001 - cleanup polling should keep reporting status
        return ResourceStatus("trainer", job_id, "ERROR", False, str(exc))
    if payload is None:
        return ResourceStatus("trainer", job_id, "NOT_FOUND", True)
    state = str(payload.get("state") or "UNKNOWN")
    return ResourceStatus(
        kind="trainer",
        resource_id=job_id,
        state=state,
        healthy=state in _TRAINER_CLEANED_STATES,
        detail=str(payload.get("error") or payload.get("errorMessage") or payload.get("statusMessage") or ""),
    )


def _check_deployment_cleanup_status(manager: Any | None, deployment_id: str) -> ResourceStatus:
    if manager is None:
        return ResourceStatus("deployment", deployment_id, "UNKNOWN", True, "deployment manager is unavailable")
    try:
        deployment = manager.get(deployment_id)
    except Exception as exc:  # noqa: BLE001 - cleanup polling should keep reporting status
        return ResourceStatus("deployment", deployment_id, "ERROR", False, str(exc))
    if deployment is None:
        return ResourceStatus("deployment", deployment_id, "NOT_FOUND", True)
    state = str(getattr(deployment, "state", "UNKNOWN") or "UNKNOWN")
    return ResourceStatus("deployment", deployment_id, state, state in {"DELETED"})


def _format_cleanup_progress(
    infra: FireworksProvisionInfra,
    statuses: list[ResourceStatus],
    started_at: float,
) -> str:
    elapsed_s = int(time.monotonic() - started_at)
    lines = [f"[{elapsed_s:>5}s] Cleaning up Fireworks {infra.mode} infra"]
    for status in statuses:
        marker = "done" if status.healthy else "waiting"
        detail = f" ({status.detail})" if status.detail else ""
        lines.append(f"- {status.kind} {status.resource_id}: {marker}, state={status.state}{detail}")
    return "\n".join(lines)


def _install_cleanup_signal_handlers() -> tuple[Callable[[], None], Callable[[], None]]:
    previous_handlers: dict[int, Any] = {}
    cleanup_started = False

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        nonlocal cleanup_started
        if cleanup_started:
            # Cleanup is already underway (SDK teardown or the cleanup monitor).
            # Swallow every further Ctrl+C/SIGTERM so teardown runs to completion
            # no matter how many times the user presses Ctrl+C.
            print(
                f"\nReceived {signal.Signals(signum).name}; cleanup is already in progress and "
                "will not be interrupted. Please wait until resources are erased.",
                flush=True,
            )
            return
        # First signal: enter shielded cleanup mode immediately, then raise once
        # so the in-flight provision/monitor unwinds into the cleanup path. From
        # here on, repeated signals are swallowed by the branch above.
        cleanup_started = True
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _handle_signal)
    except ValueError:
        return lambda: None, lambda: None

    def _begin_cleanup() -> None:
        nonlocal cleanup_started
        cleanup_started = True

    def _restore() -> None:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    return _begin_cleanup, _restore


def _load_yaml_provision(
    *,
    mode: str | None,
    recipe: str | None,
    path: Path = FIREWORKS_YAML,
    overrides: list[str] | None = None,
) -> tuple[ProvisionMode, Any]:
    doc = _load_config_doc(path, overrides=overrides)
    return _load_provision_doc(doc, mode=mode, recipe=recipe)


def _load_provision_doc(
    doc: dict[str, Any],
    *,
    mode: str | None,
    recipe: str | None,
) -> tuple[ProvisionMode, Any]:
    if not isinstance(doc, dict):
        raise ValueError("fireworks provision config must be a YAML mapping")

    options = _pop_hydra_runtime_options(doc)
    recipe = recipe if recipe is not None else options.get("recipe")
    recipe_name, recipe_cfg = _select_recipe(doc, mode=mode, recipe=recipe)
    resolved_mode = _canonical_mode(recipe_cfg.get("mode") or mode or _mode_from_recipe_name(recipe_name))
    if resolved_mode not in PROVISION_MODES:
        raise ValueError(f"recipe {recipe_name!r} mode must be one of {PROVISION_MODES}, got {resolved_mode!r}")
    cfg = _build_recipe_config(doc, recipe_name=recipe_name, recipe_cfg=recipe_cfg, mode=resolved_mode)
    return resolved_mode, cfg


def load_yaml_provision(
    *,
    mode: str | None,
    recipe: str | None,
    path: Path = FIREWORKS_YAML,
    overrides: list[str] | None = None,
) -> tuple[ProvisionMode, Any]:
    """Load a Fireworks provisioning YAML and return its resolved mode/config."""
    return _load_yaml_provision(mode=mode, recipe=recipe, path=path, overrides=overrides)


def _load_config_doc(path: Path, *, overrides: list[str] | None) -> dict[str, Any]:
    if overrides:
        return _load_hydra_config_doc(path, overrides=overrides)
    with path.open(encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    if not isinstance(doc, dict):
        raise ValueError("fireworks provision config must be a YAML mapping")
    return doc


def _load_hydra_config_doc(path: Path, *, overrides: list[str]) -> dict[str, Any]:
    config_path = path.resolve()
    config_name = config_path.stem if config_path.suffix in {".yaml", ".yml"} else config_path.name
    with hydra.initialize_config_dir(config_dir=str(config_path.parent), version_base="1.3"):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    doc = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(doc, dict):
        raise ValueError("fireworks provision config must be a YAML mapping")
    return doc


def _select_recipe(
    doc: dict[str, Any],
    *,
    mode: str | None,
    recipe: str | None,
) -> tuple[str, dict[str, Any]]:
    recipe_entries = doc.get("recipe")
    if recipe_entries is None:
        raise ValueError("no recipe defined; add a 'recipe:' mapping to fireworks.yaml")
    if not isinstance(recipe_entries, dict):
        raise ValueError("recipe must be a mapping of recipe name to config")
    if not recipe_entries:
        raise ValueError("no recipe defined; add an entry under 'recipe:' (e.g. 'rl:')")
    if recipe is not None:
        if recipe not in recipe_entries:
            available = ", ".join(sorted(recipe_entries))
            raise ValueError(f"recipe {recipe!r} not found; available recipe entries: {available}")
        return recipe, _as_recipe_mapping(recipe, recipe_entries[recipe], mode=mode)
    if mode is not None:
        matches = [
            (name, cfg)
            for name, cfg in recipe_entries.items()
            if isinstance(cfg, dict) and (cfg.get("mode") or _mode_from_recipe_name(name)) == mode
        ]
        if len(matches) == 1:
            name, cfg = matches[0]
            return name, _as_recipe_mapping(name, cfg, mode=None)
        if not matches:
            raise ValueError(f"no recipe with mode {mode!r}; pass --recipe")
        raise ValueError(f"multiple recipe entries use mode {mode!r}; pass --recipe")
    if len(recipe_entries) != 1:
        available = ", ".join(sorted(recipe_entries))
        raise ValueError(f"multiple recipe entries are defined; pass --recipe (available: {available})")
    name, cfg = next(iter(recipe_entries.items()))
    return name, _as_recipe_mapping(name, cfg, mode=None)


def _as_recipe_mapping(name: str, cfg: Any, *, mode: str | None) -> dict[str, Any]:
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"recipe {name!r} must be a mapping, got {type(cfg).__name__}")
    recipe_cfg = copy.deepcopy(cfg)
    if mode is not None:
        recipe_cfg["mode"] = mode
    return recipe_cfg


def _canonical_mode(value: str | None) -> str:
    """Resolve mode aliases (e.g. ``rft`` -> ``rl``) to a canonical mode."""
    if not value:
        return ""
    return MODE_ALIASES.get(value, value)


def _mode_from_recipe_name(name: str) -> ProvisionMode | None:
    for alias, target in MODE_ALIASES.items():
        if name == alias or name.startswith(f"{alias}_"):
            return target
    for provision_mode in PROVISION_MODES:
        if name == provision_mode or name.startswith(f"{provision_mode}_"):
            return provision_mode
    return None


def _build_recipe_config(
    doc: dict[str, Any],
    *,
    recipe_name: str,
    recipe_cfg: dict[str, Any],
    mode: ProvisionMode,
) -> Any:
    common_cfg = copy.deepcopy(doc.get("common") or {})
    merged = _deep_merge(common_cfg, copy.deepcopy(recipe_cfg))
    if mode == "rl":
        removed_loss_options = sorted(
            {
                "policy_loss",
                "loss_path",
                "dapo",
                "dro",
                "gspo",
                "cispo",
                "ratio_log_cap",
                "separate_tis",
                "use_rollout_logprobs",
            }.intersection(merged)
        )
        if removed_loss_options:
            raise ValueError(
                "The generic RL recipe uses direct client-side GRPO and does not accept "
                f"alternate loss settings: {', '.join(removed_loss_options)}. "
                "Fork its documented loss call for another algorithm."
            )
    base_model = _required(merged, "base_model")
    tokenizer_model = merged.get("tokenizer_model")
    trainer, trainer_deployment_ref, trainer_base_model = _resolve_trainer_config(
        doc,
        merged.get("trainer"),
        merged.get("reference_trainer"),
        default_base_model=base_model,
    )
    base_model = trainer_base_model
    deployment_ref = merged.get("deployment") or trainer_deployment_ref
    deployment = _resolve_deploy_config(doc, deployment_ref, tokenizer_model=tokenizer_model)
    values = {
        **merged,
        "log_path": str(merged.get("log_path") or f"./provision_logs/{recipe_name}"),
        "base_model": base_model,
        "learning_rate": merged.get("learning_rate", 1e-5),
        "max_seq_len": merged.get("max_seq_len"),
        "lora_rank": merged.get("lora_rank", 0),
        "step_timeout": merged.get("step_timeout", 0),
        "trainer": trainer,
    }
    if mode == "sft":
        from training.recipes import sft_loop

        return _make_dataclass_config(
            sft_loop.Config,
            {
                **values,
                "tokenizer_model": _required_value(tokenizer_model, "tokenizer_model"),
                "renderer_name": merged.get("renderer_name", ""),
                "serverless": bool(merged.get("serverless", False)),
            },
        )
    elif mode == "rl":
        from training.recipes import rl_loop

        cfg = _make_dataclass_config(
            rl_loop.Config,
            {
                **values,
                "kl_beta": merged.get("kl_beta", 0.001),
                "eps_clip": merged.get("eps_clip", 0.2),
                "eps_clip_high": merged.get("eps_clip_high"),
                "ppo_n_minibatches": merged.get("ppo_n_minibatches", 1),
                "deployment": deployment,
                "concurrency": _resolve_concurrency_config(merged.get("concurrency")),
                "weight_sync_timeout": merged.get("weight_sync_timeout", 600),
            },
        )
        validate_grpo_config(
            kl_beta=cfg.kl_beta,
            eps_clip=cfg.eps_clip,
            eps_clip_high=cfg.eps_clip_high,
            reference_training_shape_id=cfg.trainer.reference_training_shape_id,
            reference_job_id=cfg.trainer.reference_job_id,
            reference_configured=merged.get("reference_trainer") is not None,
            ppo_n_minibatches=cfg.ppo_n_minibatches,
        )
        return cfg
    elif mode == "dpo":
        from training.recipes import dpo_loop

        return _make_dataclass_config(
            dpo_loop.Config,
            {
                **values,
                "tokenizer_model": _required_value(tokenizer_model, "tokenizer_model"),
                "renderer_name": merged.get("renderer_name", ""),
                "beta": merged.get("beta", 0.1),
            },
        )
    elif mode == "distillation":
        from training.recipes import distillation_loop

        teacher_model = _required_value(merged.get("teacher_model"), "teacher_model")
        return _make_dataclass_config(
            distillation_loop.Config,
            {
                **values,
                "teacher_model": teacher_model,
                "deployment": deployment,
                "concurrency": _resolve_concurrency_config(merged.get("concurrency")),
                "weight_sync_timeout": merged.get("weight_sync_timeout", 600),
            },
        )
    else:
        raise ValueError(f"mode must be one of {PROVISION_MODES}, got {mode!r}")


def _resolve_trainer_config(
    doc: dict[str, Any],
    trainer_ref: Any,
    reference_ref: Any = None,
    *,
    default_base_model: str,
) -> tuple[TrainerConfig, Any, str]:
    trainer_data = _resolve_named_section(doc, "trainers", trainer_ref)
    base_model = trainer_data.pop("base_model", None) or default_base_model
    deployment_ref = trainer_data.pop("weight_sync_deployment", None)
    if "deployment" in trainer_data:
        raise ValueError(
            "trainer field 'deployment' is ambiguous; use "
            "'weight_sync_deployment' for the deployment that receives hot-loaded weights"
        )
    if reference_ref is not None:
        reference_data = _resolve_named_section(doc, "trainers", reference_ref)
        trainer_data.setdefault("reference_training_shape_id", reference_data.get("training_shape_id"))
        trainer_data.setdefault("reference_job_id", reference_data.get("job_id") or reference_data.get("training_job_id"))
    if "training_job_id" in trainer_data and "job_id" not in trainer_data:
        trainer_data["job_id"] = trainer_data.pop("training_job_id")
    return _make_dataclass_config(TrainerConfig, trainer_data), deployment_ref, base_model


def _resolve_deploy_config(
    doc: dict[str, Any],
    deployment_ref: Any,
    *,
    tokenizer_model: str | None,
) -> DeployConfig:
    deployment_data = _resolve_named_section(doc, "deployments", deployment_ref)
    if tokenizer_model is not None:
        deployment_data.setdefault("tokenizer_model", tokenizer_model)
    return _make_dataclass_config(DeployConfig, deployment_data)


def _resolve_concurrency_config(value: Any) -> ConcurrencyConfig:
    if value is None:
        return ConcurrencyConfig()
    if not isinstance(value, dict):
        raise ValueError("concurrency must be a mapping")
    return _make_dataclass_config(ConcurrencyConfig, value)


def _resolve_named_section(doc: dict[str, Any], section: str, ref: Any) -> dict[str, Any]:
    if ref is None:
        return {}
    if isinstance(ref, str):
        values = doc.get(section) or {}
        if ref not in values:
            raise ValueError(f"{section[:-1]} {ref!r} not found in {section}")
        return copy.deepcopy(values[ref] or {})
    if isinstance(ref, dict):
        return copy.deepcopy(ref)
    raise ValueError(f"{section[:-1]} reference must be a string or mapping")


def _make_dataclass_config(cls: type, values: dict[str, Any]) -> Any:
    allowed = {field.name for field in dataclasses.fields(cls)}
    type_hints = get_type_hints(cls)
    return cls(
        **{
            key: _coerce_config_value(type_hints.get(key), value)
            for key, value in values.items()
            if key in allowed
        }
    )


def _coerce_config_value(target_type: Any, value: Any) -> Any:
    if target_type is None:
        return value
    if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
        return _make_dataclass_config(target_type, value)
    origin = get_origin(target_type)
    if origin in (list, tuple) and isinstance(value, list):
        args = get_args(target_type)
        if not args:
            return value
        item_type = args[0]
        return [_coerce_config_value(item_type, item) for item in value]
    if origin in (UnionType, Union):
        return _coerce_union_config_value(target_type, value)
    return value


def _coerce_union_config_value(target_type: Any, value: Any) -> Any:
    for option in get_args(target_type):
        if option is type(None):
            continue
        if dataclasses.is_dataclass(option) and isinstance(value, dict):
            return _make_dataclass_config(option, value)
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _required(values: dict[str, Any], key: str) -> Any:
    return _required_value(values.get(key), key)


def _required_value(value: Any, name: str) -> Any:
    if value in (None, ""):
        raise ValueError(f"{name} is required")
    return value


def _resolve_config_path(config: str | None, config_name: str | None) -> Path:
    """Resolve which YAML config file to load.

    ``--config`` wins when given. Otherwise ``--config-name NAME`` maps to
    ``NAME.yaml`` next to this script. With neither, default to fireworks.yaml.
    """
    if config is not None and config_name is not None:
        raise ValueError("pass either --config or --config-name, not both")
    if config is not None:
        return Path(config)
    if config_name is not None:
        return FIREWORKS_YAML.with_name(f"{config_name}.yaml")
    return FIREWORKS_YAML


def resolve_config_path(config: str | None, config_name: str | None) -> Path:
    """Public wrapper around CLI config path resolution."""
    return _resolve_config_path(config, config_name)


def _pop_hydra_runtime_options(doc: dict[str, Any]) -> dict[str, Any]:
    options = doc.pop("provision_cli", {}) or {}
    if not isinstance(options, dict):
        raise ValueError("provision_cli must be a mapping when provided")
    return options


@hydra.main(config_path=".", config_name="fireworks", version_base="1.3")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    doc = OmegaConf.to_container(config, resolve=True)
    if not isinstance(doc, dict):
        raise ValueError("fireworks provision config must be a YAML mapping")
    mode, cfg = _load_provision_doc(mode=None, recipe=None, doc=doc)
    run_until_interrupted(
        mode,
        cfg,
    )


if __name__ == "__main__":
    main()
