#!/usr/bin/env python3
"""Distillation training loop with sampled-token OPD supervision.

The student samples responses from its own hot-loaded deployment.  A teacher
deployment then scores those exact token sequences.  Training uses Tinker's
server-side ``importance_sampling`` loss with per-token advantages equal to
``teacher_logprob - sampling_logprob`` on response tokens.

This is intentionally not a reward-model RL loop: there is no outcome reward
and no reference trainer.  The dense teacher/student log-ratio is the OPD
signal.

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.distillation_loop
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import signal
import time as _time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

import tinker

from fireworks.training.sdk import DeploymentConfig, DeploymentManager
from fireworks.training.sdk.client import GradAccNormalization
from fireworks.training.sdk.deployment import AdaptiveConcurrencyController, DeploymentSampler
from training.utils import (
    CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO,
    DEFAULT_ADAM,
    ConcurrencyConfig,
    DeployConfig,
    RLPromptDataset,
    ReconnectableClient,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    TrainerConfig,
    WandBConfig,
    build_service_client,
    load_jsonl_dataset,
    load_deployment_tokenizer,
    log_metrics_json,
    prepare_sampling_messages,
    read_api_extra_headers_env,
    setup_wandb,
    validate_config,
    wandb_finish,
    wandb_log,
)
from training.utils.checkpoints import TrainingCheckpoints, validate_warm_start_config
from training.utils.distillation import (
    MultiTeacherConfig,
    OPDPromptGroup,
    TeacherConfig,
    build_opd_server_datums,
    combine_opd_prompt_groups,
)

# Re-export distillation eval helpers from the recipe module for existing examples.
from training.utils.distillation.eval import (  # noqa: F401
    evaluate_teacher_trace_logprob_gap,
    extract_final_answer,
    expected_final_answer,
    make_teacher_trace_logprob_gap_eval,
    normalize_final_answer,
    validate_opd_trace_result,
    validate_privileged_opd_dataset,
)
from training.utils.distillation.sampling import (
    _align_completion_logprobs,
    _align_response_logprobs,
    _build_teacher_scoring_tokens,
    _score_with_teacher,
    _teacher_messages_for_row,
    _tokenize_teacher_prompt,
)
from training.utils.rl.train import TrainStepFns, run_rl_loop
from training.utils.timer import flush_timing, timer

logger = logging.getLogger(__name__)

DEPLOYMENT_ID_MAX_LENGTH = 63
TEACHER_ID_HASH_CHARS = 10


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TeacherTrainerConfig:
    """A forward-only teacher trainer used as a distillation supervision source.

    ``base_model`` is the teacher model; ``trainer`` carries its launch settings
    (training shape, region, reattach job id). Provisioned as a separate trainer
    job from the student and scored via its endpoint.
    """

    base_model: str = ""
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    """Student model ID. Matches the default used by ``rl_loop.py``."""

    rollout_base_model: str | None = None
    dataset: str = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"

    teacher_model: str = ""
    """Teacher base model or deployment ID on the same tokenizer as ``base_model``."""

    teacher_trainers: list[TeacherTrainerConfig] = field(default_factory=list)
    """Forward-only teacher trainers. Each entry is provisioned as its own trainer
    job (its ``base_model`` is the teacher) alongside the student; the loop scores
    sampled tokens against these trainers' endpoints rather than a teacher inference
    deployment. One entry is single-teacher; multiple entries are multi-teacher."""

    teacher_deployment_id: str | None = None
    """Deployment ID to create/reuse when ``teacher_model`` is a base model."""

    teacher_deployment_shape: str | None = None
    """Optional teacher deployment shape. Defaults to the resolved student deployment shape."""

    teacher_replica_count: int = 1
    """Replica count for an auto-created frozen teacher deployment."""

    teacher_deployment_timeout_s: float = 5400
    """Timeout for an auto-created frozen teacher deployment to become ready."""

    teacher_inference_url: str | None = None
    """Optional inference base URL for the teacher. Defaults to FIREWORKS_BASE_URL."""

    teacher_top_logprobs: int = 0
    """Optional top-logprob count to request while scoring. The loss needs only sampled-token logprobs."""

    multi_teacher: MultiTeacherConfig | None = None
    """Optional multi-TARGET routing config. When set (non-empty ``teachers``) it
    overrides the single ``teacher_model``: each prompt is scored by exactly ONE
    teacher, chosen by ``row[multi_teacher.route_key]`` (value == a teacher's
    ``route_value`` when set, else its ``model``). One student, N frozen
    teacher deployments; the async sampling window interleaves scoring across
    them. Not a per-prompt mixture. Leave ``None`` for the single-teacher
    default."""

    learning_rate: float = 1e-5
    opd_loss_scale: float = 1.0
    """Scalar multiplier on ``teacher_logprob - sampling_logprob``."""

    ratio_log_cap: float = 20.0
    """Numerical clamp for the server-side importance ratio log."""

    completions_per_prompt: int = 1
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    max_rows: int = 100
    max_seq_len: int | None = None
    """Max sequence length for sampling and training."""

    lora_rank: int = 0

    prompt_groups_per_step: int = 1
    """Number of prompt groups per optimizer step."""

    grad_accumulation_normalization: GradAccNormalization | str | None = GradAccNormalization.NUM_LOSS_TOKENS
    """Normalization mode for accumulated gradients at optim_step."""

    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    """Concurrency control for inference sampling."""

    init_from_checkpoint: str | None = None
    """Load pretrained DCP weights on a fresh dataset."""

    warm_start_from_adapter: str | None = None
    """GCS URI of an HF PEFT adapter directory for LoRA warm start."""

    output_model_id: str | None = None
    save_final_checkpoint: bool = True

    step_timeout: int = 0
    """Timeout in seconds for forward_backward / optim_step calls."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync_interval: int = 1
    weight_sync_before_training: bool = False
    weight_sync_timeout: int = 600
    dcp_save_interval: int = 0
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="distillation-tinker"))
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    """Optional orchestration outputs written during training."""

    post_training_eval: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    """Optional validation callback run after training/weight-sync and before cleanup."""

    step_eval: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    """Optional validation callback run during training after synced weights are serving."""

    step_eval_interval: int = 0
    """Run ``step_eval`` every N synced optimizer steps. Disabled when <= 0."""

    eval_before_training: bool = False
    """Run ``step_eval`` once at step 0 before sampling starts."""


# ---------------------------------------------------------------------------
# Teacher deployment helpers
# ---------------------------------------------------------------------------


def _is_base_model_resource(model: str) -> bool:
    return "/models/" in model and "/deployments/" not in model and "/deployedModels/" not in model


def _stable_model_hash(model: str) -> str:
    return hashlib.sha256(model.encode("utf-8")).hexdigest()[:TEACHER_ID_HASH_CHARS]


def _default_teacher_deployment_id(base_model: str, *, include_hash: bool = False) -> str:
    slug = base_model.rstrip("/").split("/")[-1]
    safe_slug = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")
    base_id = f"distillation-teacher-{safe_slug}"
    if not include_hash:
        return base_id[:DEPLOYMENT_ID_MAX_LENGTH]

    suffix = _stable_model_hash(base_model)
    prefix_len = DEPLOYMENT_ID_MAX_LENGTH - len(suffix) - 1
    return f"{base_id[:prefix_len]}-{suffix}"


def _teacher_metric_slug(model: str) -> str:
    slug = model.rstrip("/").split("/")[-1]
    safe_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", slug).strip("-")
    return f"{safe_slug}-{_stable_model_hash(model)}"


def _teacher_deployment_id_for_spec(
    spec: TeacherConfig,
    *,
    single_teacher: bool,
    single_teacher_deployment_id: str | None,
) -> str:
    if single_teacher and single_teacher_deployment_id:
        return single_teacher_deployment_id
    if spec.deployment_id:
        return spec.deployment_id
    return _default_teacher_deployment_id(spec.model, include_hash=not single_teacher)


@dataclass(frozen=True)
class _TeacherDeploymentRequest:
    deployment_id: str
    deploy_cfg: DeploymentConfig
    info: Any


@dataclass(frozen=True)
class _TeacherRoutingEntry:
    spec: TeacherConfig
    resolved_model: str
    sampler: DeploymentSampler
    top_logprobs: int
    metric_slug: str


def _request_frozen_teacher_deployment(
    deploy_mgr: DeploymentManager,
    *,
    base_model: str,
    deployment_id: str,
    deployment_shape: str | None,
    replica_count: int,
    cleanup: Callable[[str], None] | None = None,
) -> _TeacherDeploymentRequest:
    """Request a frozen teacher deployment for scoring privileged prompts."""
    existing = deploy_mgr.get(deployment_id)
    if existing is not None:
        state = getattr(existing, "state", None)
        if state in {"FAILED", "DELETED", "DELETING"}:
            raise RuntimeError(
                f"Teacher deployment {deployment_id!r} is in terminal state {state!r}. "
                "Use a different teacher_deployment_id or restore/delete the old resource."
            )
        logger.info("Re-using frozen teacher deployment: %s", deployment_id)
        deploy_cfg = DeploymentConfig(
            deployment_id=deployment_id,
            base_model=base_model,
            deployment_shape=deployment_shape,
            min_replica_count=replica_count,
            max_replica_count=replica_count,
            hot_load_bucket_type=None,
            enable_hot_load=False,
        )
        return _TeacherDeploymentRequest(
            deployment_id=deployment_id,
            deploy_cfg=deploy_cfg,
            info=existing,
        )

    deploy_cfg = DeploymentConfig(
        deployment_id=deployment_id,
        base_model=base_model,
        deployment_shape=deployment_shape,
        enable_hot_load=False,
        hot_load_bucket_type=None,
        min_replica_count=replica_count,
        max_replica_count=replica_count,
    )
    info = deploy_mgr.create_or_get(deploy_cfg)
    if cleanup is not None:
        cleanup(deployment_id)
    logger.info("Requested frozen teacher deployment: %s", deployment_id)
    return _TeacherDeploymentRequest(
        deployment_id=deployment_id,
        deploy_cfg=deploy_cfg,
        info=info,
    )


def _wait_frozen_teacher_deployment(
    deploy_mgr: DeploymentManager,
    request: _TeacherDeploymentRequest,
    *,
    timeout_s: float,
) -> str:
    ready = deploy_mgr.wait_for_ready(request.deployment_id, timeout_s=timeout_s)
    return ready.inference_model or f"accounts/{deploy_mgr.account_id}/deployments/{request.deployment_id}"


def _wait_frozen_teacher_deployments(
    deploy_mgr: DeploymentManager,
    requests_by_model: dict[str, _TeacherDeploymentRequest],
    *,
    timeout_s: float,
) -> dict[str, str]:
    """Wait for frozen teacher deployments concurrently.

    Thread-safety: callers complete all deployment create/reuse requests before
    entering this helper. Worker threads only perform independent readiness
    polling via ``DeploymentManager.wait_for_ready`` and do not mutate shared
    deployment request state.
    """
    if not requests_by_model:
        return {}

    max_workers = len(requests_by_model)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_by_model = {
            model: executor.submit(
                _wait_frozen_teacher_deployment,
                deploy_mgr,
                request,
                timeout_s=timeout_s,
            )
            for model, request in requests_by_model.items()
        }
        return {
            model: future.result()
            for model, future in futures_by_model.items()
        }


def _resolve_teacher_specs(cfg: Config) -> list[TeacherConfig]:
    """Normalize single- and multi-teacher config into a list of teacher specs.

    Single ``teacher_model`` (no ``multi_teacher``) is the backward-compatible
    default and yields one spec. With ``multi_teacher`` set, its ``teachers``
    list is used as-is.
    """
    if cfg.multi_teacher is not None:
        if not cfg.multi_teacher.teachers:
            raise ValueError("Config.multi_teacher requires at least one teacher.")
        return list(cfg.multi_teacher.teachers)
    if not cfg.teacher_model:
        raise ValueError("Config.teacher_model is required unless Config.multi_teacher is set.")
    return [TeacherConfig(model=cfg.teacher_model)]


def _validate_teacher_tokenizers(
    teacher_specs: list[TeacherConfig],
    *,
    student_tokenizer_model: str,
) -> None:
    """Validate optional teacher tokenizer declarations against the student."""
    for spec in teacher_specs:
        if spec.tokenizer_model is None:
            continue
        if spec.tokenizer_model != student_tokenizer_model:
            raise ValueError(
                "TeacherConfig.tokenizer_model must match deployment.tokenizer_model "
                f"for sampled-token OPD: teacher {spec.model!r} uses "
                f"{spec.tokenizer_model!r}, student uses {student_tokenizer_model!r}."
            )


def _teacher_top_logprobs(spec: TeacherConfig, default_top_logprobs: int) -> int:
    """Return the per-teacher top-logprobs setting for scoring requests."""
    if default_top_logprobs < 0:
        raise ValueError("Config.teacher_top_logprobs must be non-negative.")
    if spec.top_logprobs is None:
        return default_top_logprobs
    return spec.top_logprobs


def _teacher_deployment_shape_for_spec(
    spec: TeacherConfig,
    cfg: Config,
    *,
    student_deployment_shape: str,
) -> str | None:
    if spec.deployment_shape is not None:
        return spec.deployment_shape
    if cfg.teacher_deployment_shape is not None:
        return cfg.teacher_deployment_shape
    if spec.model == cfg.base_model:
        return student_deployment_shape
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: Any | None = None,
    deploy_mgr: DeploymentManager | None = None,
    cancel_on_exit: bool = False,
    cleanup_on_exit: bool | None = None,
):
    if cleanup_on_exit is not None:
        warnings.warn(
            "distillation_loop.main(cleanup_on_exit=...) is deprecated; use cancel_on_exit=...",
            DeprecationWarning,
            stacklevel=2,
        )
        cancel_on_exit = cleanup_on_exit

    cfg = config
    runner = RunnerIO(cfg.runner)

    # Convert SIGTERM/SIGINT into exceptions so the finally block runs cleanup.
    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s; raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(
        cfg.base_model,
        cfg.dataset,
        deploy=cfg.deployment,
        output_model_id=cfg.output_model_id,
    )
    validate_warm_start_config(
        warm_start_from_adapter=cfg.warm_start_from_adapter,
        init_from_checkpoint=cfg.init_from_checkpoint,
        lora_rank=cfg.lora_rank,
    )
    if not cfg.deployment.tokenizer_model:
        raise ValueError(
            "deployment.tokenizer_model is required for client-side tokenization. "
            "Use the HuggingFace tokenizer that matches both student and teacher."
        )
    if cfg.teacher_replica_count <= 0:
        raise ValueError("Config.teacher_replica_count must be positive.")
    if cfg.weight_sync_interval not in (0, 1):
        raise ValueError(
            "Distillation supports weight_sync_interval 0 or 1. Use 1 for strict "
            "TML-style on-policy distillation, or 0 to disable sampler sync."
        )

    teacher_specs = _resolve_teacher_specs(cfg)
    _validate_teacher_tokenizers(
        teacher_specs,
        student_tokenizer_model=cfg.deployment.tokenizer_model,
    )
    for teacher_spec in teacher_specs:
        _teacher_top_logprobs(teacher_spec, cfg.teacher_top_logprobs)

    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step

    setup_wandb(
        cfg.wandb,
        {
            "teacher_model": cfg.teacher_model or teacher_specs[0].model,
            "teacher_models": [spec.model for spec in teacher_specs],
            "teacher_routes": [spec.routing_value for spec in teacher_specs],
            "teacher_route_key": cfg.multi_teacher.route_key if cfg.multi_teacher else None,
            "opd_loss_scale": cfg.opd_loss_scale,
            "ratio_log_cap": cfg.ratio_log_cap,
            "completions_per_prompt": completions_per_prompt,
            "prompt_groups_per_step": prompt_groups_per_step,
            "lr": cfg.learning_rate,
        },
    )

    # -- Setup infrastructure -----------------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
        )

    runner.write_status(RunStatus.PENDING, message="provisioning")

    with runner, ExitStack() as stack:
        service = build_service_client(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
            base_model=cfg.base_model,
            tokenizer_model=cfg.deployment.tokenizer_model,
            lora_rank=cfg.lora_rank,
            max_context_length=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            trainer=cfg.trainer,
            deployment=cfg.deployment,
            hotload_timeout_s=cfg.weight_sync_timeout,
            cleanup_trainer_on_close=cancel_on_exit,
            cleanup_deployment_on_close=(
                CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO if cancel_on_exit else None
            ),
            reference_required=False,
        )
        stack.callback(service.close)
        training_client = service.create_training_client(cfg.base_model, lora_rank=cfg.lora_rank)

        runner.set_accelerator_info(
            service.accelerator_type,
            service.accelerator_count,
            profile=service.training_profile,
        )

        policy_job_id = service.trainer_job_id
        policy = ReconnectableClient.from_training_client(
            training_client,
            base_model=cfg.base_model,
            lora_rank=cfg.lora_rank,
            job_id=policy_job_id,
            default_timeout=cfg.step_timeout or 3600,
            service=service,
        )
        tokenizer = load_deployment_tokenizer(cfg.deployment)
        max_seq_len = service.max_context_length
        deployment_id = service.deployment_id
        student_model = f"accounts/{deploy_mgr.account_id}/deployments/{deployment_id}"

        # Provision one frozen deployment per distinct base-model teacher; keyed
        # by teacher ``model`` so duplicate models reuse one deployment. Non-base
        # teachers resolve to themselves.
        single_teacher = len(teacher_specs) == 1
        teacher_model_out: dict[str, str] = {}
        teacher_requests: dict[str, _TeacherDeploymentRequest] = {}
        deployment_id_to_teacher_model: dict[str, str] = {}
        _provisioned: set[str] = set()
        for spec in teacher_specs:
            if not _is_base_model_resource(spec.model):
                teacher_model_out[spec.model] = spec.model
                continue
            if spec.model in _provisioned:
                continue
            _provisioned.add(spec.model)
            teacher_deployment_shape = _teacher_deployment_shape_for_spec(
                spec,
                cfg,
                student_deployment_shape=service.deployment_shape,
            )
            teacher_deployment_id = _teacher_deployment_id_for_spec(
                spec,
                single_teacher=single_teacher,
                single_teacher_deployment_id=cfg.teacher_deployment_id,
            )
            existing_teacher_model = deployment_id_to_teacher_model.get(teacher_deployment_id)
            if existing_teacher_model is not None and existing_teacher_model != spec.model:
                raise ValueError(
                    "Teacher deployment ids must be unique per base model: "
                    f"{teacher_deployment_id!r} is used by both "
                    f"{existing_teacher_model!r} and {spec.model!r}."
                )
            deployment_id_to_teacher_model[teacher_deployment_id] = spec.model
            teacher_request = _request_frozen_teacher_deployment(
                deploy_mgr,
                base_model=spec.model,
                deployment_id=teacher_deployment_id,
                deployment_shape=teacher_deployment_shape,
                replica_count=cfg.teacher_replica_count,
                cleanup=(
                    lambda deployment_id: stack.callback(deploy_mgr.scale_to_zero, deployment_id)
                    if cancel_on_exit
                    else None
                ),
            )
            teacher_requests[spec.model] = teacher_request
        teacher_model_out.update(
            _wait_frozen_teacher_deployments(
                deploy_mgr,
                teacher_requests,
                timeout_s=cfg.teacher_deployment_timeout_s,
            )
        )

        initial_window = cfg.concurrency.initial_window or (8 * (cfg.deployment.replica_count or 1))
        concurrency_controller = AdaptiveConcurrencyController(
            initial_window=initial_window,
            min_window=cfg.concurrency.min_window,
            max_window=cfg.concurrency.max_window,
            prefill_queue_target=cfg.concurrency.prefill_queue_target,
        )
        logger.info(
            "Concurrency: adaptive (initial=%d, range=%d-%d, target_pq=%.2fs)",
            initial_window,
            cfg.concurrency.min_window,
            cfg.concurrency.max_window,
            cfg.concurrency.prefill_queue_target,
        )
        student_sampler = service.create_deployment_sampler(
            tokenizer=tokenizer,
            concurrency_controller=concurrency_controller,
        )
        # Build one scoring sampler per teacher spec. Multi-target routing: each
        # prompt is scored by exactly one teacher chosen by ``row[route_key]``.
        # ``route_to_entry`` maps the configured routing value to its
        # (resolved_model, sampler). The primary (first) teacher backs the
        # single-teacher fields used by eval contexts.
        teacher_entries: list[_TeacherRoutingEntry] = []
        route_to_entry: dict[str, _TeacherRoutingEntry] = {}
        for spec in teacher_specs:
            resolved = teacher_model_out.get(spec.model, spec.model)
            sampler = DeploymentSampler(
                inference_url=cfg.teacher_inference_url or base_url,
                model=resolved,
                api_key=api_key,
                tokenizer=tokenizer,
            )
            routing_entry = _TeacherRoutingEntry(
                spec=spec,
                resolved_model=resolved,
                sampler=sampler,
                top_logprobs=_teacher_top_logprobs(spec, cfg.teacher_top_logprobs),
                metric_slug=_teacher_metric_slug(spec.routing_value),
            )
            teacher_entries.append(routing_entry)
            route_to_entry[spec.routing_value] = routing_entry
        primary_teacher = teacher_entries[0]
        teacher_model = primary_teacher.resolved_model
        teacher_sampler = primary_teacher.sampler
        is_multi_teacher = len(teacher_entries) > 1
        teacher_route_key = cfg.multi_teacher.route_key if cfg.multi_teacher is not None else "teacher"

        # Per-teacher visibility so skewed routing / idle teachers are observable.
        # ``scored`` is cumulative attempts (skew); ``inflight`` is the live gauge
        # (saturation). Keyed by configured route value so route labels stay stable.
        teacher_scored: dict[str, int] = {entry.spec.routing_value: 0 for entry in teacher_entries}
        teacher_inflight: dict[str, int] = {entry.spec.routing_value: 0 for entry in teacher_entries}

        async def _score_routed(
            teacher_entry: _TeacherRoutingEntry,
            scoring_tokens: list[int],
            *,
            prompt_len: int,
            response_len: int,
        ) -> list[float] | None:
            teacher_key = teacher_entry.spec.routing_value
            teacher_inflight[teacher_key] += 1
            try:
                return await _score_with_teacher(
                    teacher_entry.sampler,
                    scoring_tokens,
                    prompt_len=prompt_len,
                    response_len=response_len,
                    top_logprobs=teacher_entry.top_logprobs,
                    http_timeout=cfg.deployment.sample_timeout,
                )
            finally:
                teacher_inflight[teacher_key] -= 1
                teacher_scored[teacher_key] += 1

        ckpt = TrainingCheckpoints(
            policy,
            service,
            trainer_id=policy_job_id,
            log_path=cfg.log_path,
            lora_rank=cfg.lora_rank,
        )

        if is_multi_teacher:
            logger.info(
                "Distillation training: %d teachers routed by row[%r] (%s) | completions_per_prompt=%d | groups_per_step=%d",
                len(teacher_entries),
                teacher_route_key,
                ", ".join(f"{entry.spec.routing_value}->{entry.resolved_model}" for entry in teacher_entries),
                completions_per_prompt,
                prompt_groups_per_step,
            )
        logger.info(
            "Distillation training: teacher=%s | completions_per_prompt=%d | groups_per_step=%d",
            teacher_model,
            completions_per_prompt,
            prompt_groups_per_step,
        )

        # -- Resume ---------------------------------------------------------------

        resume_info = ckpt.resume(
            init_from_checkpoint=cfg.init_from_checkpoint,
            warm_start_from_adapter=cfg.warm_start_from_adapter,
        )
        step_offset = resume_info.step if resume_info else 0
        wandb_log({"train/step": step_offset}, step_offset)

        if cfg.weight_sync_before_training:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            saved = policy.save_weights_for_sampler_ext(name, checkpoint_type="base")
            service.hotload_sampler_snapshot(saved.snapshot_name)

        # -- Prepare sampling and training --------------------------------------

        raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        all_rows = raw_dataset * cfg.epochs
        rl_dataset = RLPromptDataset(all_rows, prompts_per_step=prompt_groups_per_step)
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)
        server_loss_config = {"ratio_log_cap": cfg.ratio_log_cap}

        sample_kwargs: dict[str, Any] = {
            "max_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "max_seq_len": max_seq_len,
            "http_timeout": cfg.deployment.sample_timeout,
            "logprobs": True,
        }

        # -- Sample one prompt (VISIBLE -- customise this) ----------------------

        async def sample_one_prompt(row: dict) -> OPDPromptGroup | None:
            """Sample student completions and score them with the teacher prompt."""
            messages = row.get("messages", [])
            input_messages = prepare_sampling_messages(messages)
            if not input_messages:
                return None

            # Route this prompt to exactly one teacher. Single-teacher -> primary.
            if is_multi_teacher:
                route_value = row.get(teacher_route_key)
                scoring_entry = route_to_entry.get(route_value) if isinstance(route_value, str) else None
                if scoring_entry is None:
                    logger.warning(
                        "Skipping prompt: route key %r=%r not in configured teachers %s",
                        teacher_route_key,
                        route_value,
                        sorted(route_to_entry),
                    )
                    return None
            else:
                scoring_entry = primary_teacher

            teacher_messages = _teacher_messages_for_row(
                row,
                input_messages,
                teacher_messages_key=scoring_entry.spec.teacher_messages_key,
            )

            try:
                teacher_prompt_tokens = _tokenize_teacher_prompt(tokenizer, teacher_messages)
            except Exception as exc:
                logger.warning("Teacher prompt tokenization failed: %s", exc)
                return None

            try:
                sampled = await student_sampler.sample_with_tokens(
                    messages=input_messages,
                    n=completions_per_prompt,
                    **sample_kwargs,
                )
            except Exception as exc:
                logger.warning("Student sampling failed: %s", exc)
                return None

            sampled = [s for s in sampled if len(s.full_tokens) >= 2]
            if not sampled:
                return None

            teacher_scores: list[list[float] | None] = [None] * len(sampled)
            teacher_task_indices: list[int] = []
            teacher_tasks = []
            for idx, sample in enumerate(sampled):
                teacher_scoring_tokens = _build_teacher_scoring_tokens(
                    teacher_prompt_tokens,
                    sample.full_tokens,
                    student_prompt_len=sample.prompt_len,
                )
                if teacher_scoring_tokens is None:
                    continue
                scoring_tokens, response_len = teacher_scoring_tokens
                teacher_task_indices.append(idx)
                teacher_tasks.append(
                    _score_routed(
                        scoring_entry,
                        scoring_tokens,
                        prompt_len=len(teacher_prompt_tokens),
                        response_len=response_len,
                    )
                )
            if teacher_tasks:
                teacher_results = await asyncio.gather(*teacher_tasks)
                for idx, teacher_lp in zip(teacher_task_indices, teacher_results, strict=True):
                    teacher_scores[idx] = teacher_lp

            policy_data: list[tinker.Datum] = []
            teacher_logprobs: list[list[float]] = []
            sampling_logprobs: list[list[float]] = []
            rewards: list[float] = []
            completion_lens: list[int] = []
            truncated: list[bool] = []
            prompt_len = sampled[0].prompt_len

            for sample, teacher_lp in zip(sampled, teacher_scores, strict=True):
                if teacher_lp is None:
                    continue
                if not sample.inference_logprobs:
                    logger.warning("Skipping OPD sample without student inference logprobs")
                    continue

                tokens = sample.full_tokens
                target_tokens = tokens[1:]
                target_len = len(target_tokens)
                aligned_sampling_lp = _align_completion_logprobs(
                    list(sample.inference_logprobs),
                    prompt_len=sample.prompt_len,
                    target_len=target_len,
                    echoed=getattr(sample, "logprobs_echoed", False),
                )
                if aligned_sampling_lp is None:
                    logger.warning("Skipping OPD sample with incomplete student logprobs")
                    continue
                aligned_teacher_lp = _align_response_logprobs(
                    teacher_lp,
                    prompt_len=sample.prompt_len,
                    target_len=target_len,
                )
                if aligned_teacher_lp is None:
                    logger.warning("Skipping OPD sample with incomplete teacher logprobs")
                    continue

                policy_data.append(
                    tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData(
                                data=target_tokens,
                                dtype="int64",
                                shape=[target_len],
                            ),
                        },
                    )
                )
                teacher_logprobs.append(aligned_teacher_lp)
                sampling_logprobs.append(aligned_sampling_lp)
                rewards.append(0.0)
                completion_lens.append(sample.completion_len)
                truncated.append(sample.finish_reason == "length")

            if not policy_data:
                return None

            return OPDPromptGroup(
                data=policy_data,
                teacher_logprobs=teacher_logprobs,
                sampling_logprobs=sampling_logprobs,
                prompt_len=prompt_len,
                rewards=rewards,
                completion_lens=completion_lens,
                truncated=truncated,
            )

        # -- Eval callbacks --------------------------------------------------------

        latest_eval_metrics: dict[str, Any] = {}

        def _eval_context(step: int) -> dict[str, Any]:
            return {
                "config": cfg,
                "dataset": raw_dataset,
                "student_sampler": student_sampler,
                "teacher_sampler": teacher_sampler,
                "student_model": student_model,
                "teacher_model": teacher_model,
                "teacher_samplers": {
                    entry.spec.routing_value: entry.sampler for entry in teacher_entries
                },
                "teacher_models": {
                    entry.spec.routing_value: entry.resolved_model for entry in teacher_entries
                },
                "teacher_messages_keys": {
                    entry.spec.routing_value: entry.spec.teacher_messages_key for entry in teacher_entries
                },
                "teacher_route_key": teacher_route_key,
                "is_multi_teacher": is_multi_teacher,
                "tokenizer": tokenizer,
                "global_step": step,
                "max_seq_len": max_seq_len,
            }

        def _run_eval_callback(
            callback: Callable[[dict[str, Any]], dict[str, Any]],
            *,
            step: int,
            phase: str,
        ) -> dict[str, Any]:
            logger.info("[step %d] %s eval: starting...", step, phase)
            t0 = _time.time()
            metrics = callback(_eval_context(step)) or {}
            logger.info("[step %d] %s eval: done (%.1fs)", step, phase, _time.time() - t0)
            if metrics:
                runner.append_metrics(step, metrics)
                wandb_log(metrics, step)
                fixed_gap = metrics.get("eval/opd_fixed_target_student_minus_teacher_nll")
                abs_gap = metrics.get("eval/opd_fixed_target_abs_logprob_gap")
                if fixed_gap is not None or abs_gap is not None:
                    logger.info(
                        "[step %d] eval logprob gap: student-teacher NLL=%s | abs=%s",
                        step,
                        f"{float(fixed_gap):.4f}" if fixed_gap is not None else "n/a",
                        f"{float(abs_gap):.4f}" if abs_gap is not None else "n/a",
                    )
                trace_gap = metrics.get("eval/opd_trace_student_minus_teacher_nll")
                trace_final_gap = metrics.get("eval/opd_trace_final_student_minus_teacher_nll")
                student_gen_acc = metrics.get("eval/opd_trace_student_generation_accuracy")
                if trace_gap is not None or trace_final_gap is not None or student_gen_acc is not None:
                    logger.info(
                        "[step %d] eval teacher-trace gap: full=%s | final=%s | student_gen_acc=%s",
                        step,
                        f"{float(trace_gap):.4f}" if trace_gap is not None else "n/a",
                        f"{float(trace_final_gap):.4f}" if trace_final_gap is not None else "n/a",
                        f"{float(student_gen_acc):.3f}" if student_gen_acc is not None else "n/a",
                    )
            return metrics

        def _run_step_eval(step: int, *, phase: str) -> None:
            nonlocal latest_eval_metrics
            if cfg.step_eval is None:
                return
            latest_eval_metrics = _run_eval_callback(cfg.step_eval, step=step, phase=phase)

        if cfg.eval_before_training and cfg.step_eval is not None:
            _run_step_eval(step_offset, phase="pre-training")

        # -- Training callbacks ----------------------------------------------------

        def train_step(
            step: int,
            prompt_groups: list[OPDPromptGroup],
            loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            if not prompt_groups:
                raise ValueError("train_step requires at least one prompt group")

            data, teacher_lp, prompt_lens, sampling_lp = combine_opd_prompt_groups(prompt_groups)
            opd_datums, opd_input_metrics = build_opd_server_datums(
                data,
                teacher_lp,
                sampling_lp,
                prompt_lens,
                loss_scale=cfg.opd_loss_scale,
            )

            t0 = _time.time()
            with timer("fwd_bwd"):
                fwd_bwd_result = policy.forward_backward(
                    opd_datums,
                    "importance_sampling",
                    loss_fn_config=server_loss_config,
                )
            logger.info("[step %d] fwd_bwd: done (%.1fs)", step + 1, _time.time() - t0)

            t0 = _time.time()
            with timer("optim_step"):
                optim_result = policy.optim_step(
                    adam_params,
                    grad_accumulation_normalization=cfg.grad_accumulation_normalization,
                )
            step += 1
            logger.info("[step %d] optim_step: done (%.1fs)", step, _time.time() - t0)

            rollouts_completed = step - step_offset
            dcp_interval = cfg.dcp_save_interval
            if dcp_interval > 0 and rollouts_completed > 0 and rollouts_completed % dcp_interval == 0:
                data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    rollouts_completed * prompt_groups_per_step
                )
                ckpt.save(
                    f"step-{step}",
                    resumable=True,
                    promotable=False,
                    data_consumed=data_consumed,
                )

            metrics: dict[str, Any] = dict(flush_timing())
            metrics["train/step"] = step
            for key, value in opd_input_metrics.items():
                metrics[f"train/{key}"] = value

            if fwd_bwd_result and hasattr(fwd_bwd_result, "metrics"):
                for key, value in fwd_bwd_result.metrics.items():
                    metrics[f"train/{key}"] = value
            if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
                for key, value in optim_result.metrics.items():
                    metrics[f"train/{key}"] = value

            sampled_kl = metrics.get("train/opd_sampled_reverse_kl", 0.0)
            active_tokens = int(metrics.get("train/opd_active_tokens", 0.0))
            logger.info(
                "Step %d | sampled reverse KL: %.4f | OPD advantage: %.4f | tokens=%d",
                step,
                sampled_kl,
                metrics.get("train/opd_advantage", 0.0),
                active_tokens,
            )
            log_metrics_json(
                step,
                opd_sampled_reverse_kl=sampled_kl,
                opd_advantage=metrics.get("train/opd_advantage", 0.0),
                active_tokens=active_tokens,
            )
            wandb_log(metrics, step)

            total_steps = len(rl_dataset) - step_offset
            runner.append_metrics(step, metrics, tokens=active_tokens)
            runner.write_status(
                RunStatus.RUNNING,
                step=step,
                total_steps=total_steps,
                message="training",
            )
            runner.write_metadata()

            if cfg.weight_sync_interval == 1:
                logger.info("[step %d] weight_sync: saving + loading...", step)
                t0 = _time.time()
                with timer("weight_sync"):
                    saved = policy.save_weights_for_sampler_ext(f"step-{step}")
                    service.hotload_sampler_snapshot(saved.snapshot_name)
                logger.info("[step %d] weight_sync: done (%.1fs)", step, _time.time() - t0)
                if (
                    cfg.step_eval is not None
                    and cfg.step_eval_interval > 0
                    and step % cfg.step_eval_interval == 0
                ):
                    _run_step_eval(step, phase="post-sync")

            return step, metrics

        def _loop_metrics_callback(loop_metrics: dict) -> None:
            cc_summary = concurrency_controller.step_completed()
            for key, value in cc_summary.items():
                loop_metrics[f"concurrency/{key}"] = value
            if is_multi_teacher:
                for entry in teacher_entries:
                    teacher_key = entry.spec.routing_value
                    loop_metrics[f"teacher_route/{entry.metric_slug}/scored"] = teacher_scored[
                        teacher_key
                    ]
                    loop_metrics[f"teacher_route/{entry.metric_slug}/inflight"] = teacher_inflight[
                        teacher_key
                    ]
            wandb_log(loop_metrics, step=loop_metrics.get("train/step", 0))

        # -- Run loop -------------------------------------------------------------

        remaining_rows = []
        for i_step in range(step_offset, len(rl_dataset)):
            remaining_rows.extend(rl_dataset.get_batch(i_step))

        total_steps = len(rl_dataset) - step_offset
        runner.start_training()
        runner.write_status(RunStatus.RUNNING, total_steps=total_steps, message="training")

        global_step = asyncio.run(
            run_rl_loop(
                sample_fns=(sample_one_prompt(row) for row in remaining_rows),
                train_fns=TrainStepFns(train_step=train_step),
                prompt_groups_per_step=prompt_groups_per_step,
                global_step=step_offset,
                metrics_callback=_loop_metrics_callback,
                weight_sync_interval=cfg.weight_sync_interval,
            )
        )

        # -- Final checkpoint --------------------------------------------------

        if cfg.save_final_checkpoint and global_step > step_offset:
            try:
                checkpoint_name = f"step-{global_step}"
                data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    (global_step - step_offset) * prompt_groups_per_step
                )
                ckpt.save(
                    checkpoint_name,
                    resumable=True,
                    promotable=True,
                    data_consumed=data_consumed,
                )
                if cfg.output_model_id:
                    ckpt.promote_latest(cfg.output_model_id, cfg.base_model)
                    runner.write_output_model(
                        model_id=cfg.output_model_id,
                        checkpoint=checkpoint_name,
                        job_id=policy_job_id,
                    )
            except Exception as exc:
                logger.warning("Failed to save final checkpoint: %s", exc)

        eval_metrics: dict[str, Any] = dict(latest_eval_metrics)
        if cfg.post_training_eval is not None:
            eval_metrics = _run_eval_callback(
                cfg.post_training_eval,
                step=global_step,
                phase="post-training",
            )

        runner.write_status(
            RunStatus.COMPLETED,
            step=global_step,
            total_steps=total_steps,
            message="done",
        )
        runner.write_metadata()
        logger.info("Distillation training complete: %d steps", global_step)
        wandb_finish()
        return {
            "steps": global_step,
            "policy_job_id": policy_job_id,
            "deployment_id": deployment_id,
            "eval": eval_metrics,
            "max_seq_len": max_seq_len,
            "training_shape_id": cfg.trainer.training_shape_id,
            "deployment_shape": service.deployment_shape,
        }


def _env_with_legacy(primary_name: str, legacy_name: str, default: str = "") -> str:
    return os.environ.get(primary_name, os.environ.get(legacy_name, default))


def _multi_teacher_from_env() -> MultiTeacherConfig | None:
    """Parse ``DISTILLATION_TEACHERS`` for routed distillation.

    ``DISTILLATION_TEACHER_ROUTE_KEY`` names the dataset row key whose value
    selects the teacher model for each prompt (default ``teacher``). Legacy
    ``OPD_TEACHERS`` and ``OPD_TEACHER_ROUTE_KEY`` are accepted as fallbacks.
    Returns ``None`` when no routed-teacher env is set.
    """
    raw = _env_with_legacy("DISTILLATION_TEACHERS", "OPD_TEACHERS").strip()
    if not raw:
        return None
    teachers = [TeacherConfig(model=m.strip()) for m in raw.split(",") if m.strip()]
    route_key = _env_with_legacy(
        "DISTILLATION_TEACHER_ROUTE_KEY",
        "OPD_TEACHER_ROUTE_KEY",
        "teacher",
    )
    return MultiTeacherConfig(teachers=teachers, route_key=route_key)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        log_path="./distillation_logs",
        base_model="accounts/fireworks/models/qwen3-8b",
        teacher_model=_env_with_legacy("DISTILLATION_TEACHER_MODEL", "OPD_TEACHER_MODEL"),
        multi_teacher=_multi_teacher_from_env(),
        trainer=TrainerConfig(
            training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200",
        ),
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
    )
    main(cfg)
