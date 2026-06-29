#!/usr/bin/env python3
"""Distillation training loop with sampled reverse-KL and top-K SDFT modes.

The student samples responses from its own hot-loaded deployment.  A teacher
deployment then scores those exact token sequences.  Sampled reverse KL uses
per-token dense rewards equal to ``teacher_logprob - sampling_logprob`` on
response tokens.  The recipe sends those rewards through Tinker's server-side
``importance_sampling`` loss to account for rollout/current-policy drift.

This is intentionally not a reward-model RL loop: there is no outcome reward
and no reference trainer.  The teacher/student log-ratio is the distillation
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
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

import tinker

from fireworks.training.sdk.client import GradAccNormalization
from fireworks.training.sdk.deployment import AdaptiveConcurrencyController, DeploymentConfig
from training.utils import (
    CLEANUP_DEPLOYMENT_ON_CLOSE_DELETE,
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
    DistillMode,
    MultiTeacherConfig,
    OPDPromptGroup,
    TeacherConfig,
)
from training.utils.distillation.objectives import (
    DistillationObjectiveSettings,
    TeacherScoringFns,
    TeacherSourceContext,
    create_distillation_objective,
    validate_distillation_objective_settings,
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
    _score_teacher_topk,
    _score_with_teacher,
    _teacher_messages_for_row,
    _tokenize_teacher_prompt,
)
from training.train_loop import TrainStepFns, run_batched_training_loop
from training.utils.timer import flush_timing, timer

logger = logging.getLogger(__name__)

DEPLOYMENT_ID_MAX_LENGTH = 63
TEACHER_ID_HASH_CHARS = 10
MAX_SDFT_TOP_LOGPROBS = 5


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


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

    teacher_deployment_id: str | None = None
    """Deployment ID to create/reuse when ``teacher_model`` is a base model."""

    teacher_deployment_shape: str | None = None
    """Optional teacher deployment shape. Defaults to the resolved student deployment shape."""

    teacher_replica_count: int = 1
    """Replica count for an auto-created teacher inference deployment."""

    teacher_deployment_timeout_s: float = 5400
    """Timeout for an auto-created teacher inference deployment to become ready."""

    teacher_inference_url: str | None = None
    """Optional inference base URL for the teacher. Defaults to FIREWORKS_BASE_URL."""

    distill_mode: DistillMode | str = DistillMode.SAMPLED_REVERSE_KL
    """Distillation objective. ``TOPK_FORWARD_KL`` uses inference-source top-K SDFT."""

    sdft_top_k: int = 5
    """Teacher top-K size for ``TOPK_FORWARD_KL`` SDFT."""

    multi_teacher: MultiTeacherConfig | None = None
    """Optional multi-teacher config.

    ``SAMPLED_REVERSE_KL`` routes each prompt to one teacher. Multi-teacher SDFT scores
    each rollout with all teachers and blends sparse top-K distributions.
    """

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

    grad_accumulation_normalization: GradAccNormalization | str | None = None
    """Optional server-side normalization for accumulated gradients.
    ``None`` leaves accumulated gradients unchanged."""

    grad_clip_norm: float = 0.0
    """Max gradient norm for clipping. 0 disables clipping."""

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
class _TeacherRoutingEntry:
    spec: TeacherConfig
    resolved_model: str
    sampler: Any
    metric_slug: str


@dataclass(frozen=True)
class _TeacherRuntime:
    entries: list[_TeacherRoutingEntry]
    route_to_entry: dict[str, _TeacherRoutingEntry]
    primary: _TeacherRoutingEntry
    route_key: str

    @property
    def is_multi_teacher(self) -> bool:
        return len(self.entries) > 1


def _resolve_teacher_specs(cfg: Config) -> list[TeacherConfig]:
    """Normalize single- and multi-teacher config into a list of teacher specs.

    Single ``teacher_model`` (no ``multi_teacher``) yields one spec. With
    ``multi_teacher`` set, its ``teachers`` list is used as-is.
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


def _teacher_deployment_shape_for_spec(
    spec: TeacherConfig,
    cfg: Config,
) -> str | None:
    if spec.deployment_shape is not None:
        return spec.deployment_shape
    if cfg.teacher_deployment_shape is not None:
        return cfg.teacher_deployment_shape
    return None


def _resolve_teacher_runtime(
    *,
    cfg: Config,
    teacher_specs: list[TeacherConfig],
    service: Any,
    tokenizer: Any,
    base_url: str,
    cancel_on_exit: bool,
) -> _TeacherRuntime:
    """Resolve teacher model specs to inference samplers used for scoring."""
    single_teacher = len(teacher_specs) == 1
    resolved_models: dict[str, str] = {}
    samplers: dict[str, Any] = {}
    deployment_id_to_teacher_model: dict[str, str] = {}

    for spec in teacher_specs:
        if spec.model in samplers:
            continue

        if not _is_base_model_resource(spec.model):
            resolved_models[spec.model] = spec.model
            samplers[spec.model] = service.create_deployment_sampler_for_model(
                spec.model,
                tokenizer=tokenizer,
                inference_url=cfg.teacher_inference_url or base_url,
            )
            continue

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

        sampler = service.create_inference_deployment_sampler(
            DeploymentConfig(
                deployment_id=teacher_deployment_id,
                base_model=spec.model,
                deployment_shape=_teacher_deployment_shape_for_spec(spec, cfg),
                min_replica_count=cfg.teacher_replica_count,
                max_replica_count=cfg.teacher_replica_count,
                hot_load_bucket_type=None,
                enable_hot_load=False,
                for_training=True,
            ),
            timeout_s=cfg.teacher_deployment_timeout_s,
            cleanup_on_close=CLEANUP_DEPLOYMENT_ON_CLOSE_DELETE if cancel_on_exit else None,
            tokenizer=tokenizer,
        )
        resolved_models[spec.model] = sampler.model
        samplers[spec.model] = sampler

    entries: list[_TeacherRoutingEntry] = []
    route_to_entry: dict[str, _TeacherRoutingEntry] = {}
    for spec in teacher_specs:
        entry = _TeacherRoutingEntry(
            spec=spec,
            resolved_model=resolved_models[spec.model],
            sampler=samplers[spec.model],
            metric_slug=_teacher_metric_slug(spec.routing_value),
        )
        entries.append(entry)
        route_to_entry[spec.routing_value] = entry

    return _TeacherRuntime(
        entries=entries,
        route_to_entry=route_to_entry,
        primary=entries[0],
        route_key=cfg.multi_teacher.route_key if cfg.multi_teacher is not None else "teacher",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: Any | None = None,
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
    distill_mode = DistillMode(cfg.distill_mode)
    objective_settings = DistillationObjectiveSettings(
        mode=distill_mode,
        top_k=cfg.sdft_top_k,
        has_multi_teacher=cfg.multi_teacher is not None,
        max_top_logprobs=MAX_SDFT_TOP_LOGPROBS,
    )
    validate_distillation_objective_settings(objective_settings)

    teacher_specs = _resolve_teacher_specs(cfg)
    _validate_teacher_tokenizers(
        teacher_specs,
        student_tokenizer_model=cfg.deployment.tokenizer_model,
    )
    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step

    setup_wandb(
        cfg.wandb,
        {
            "teacher_model": cfg.teacher_model or teacher_specs[0].model,
            "teacher_models": [spec.model for spec in teacher_specs],
            "teacher_routes": [spec.routing_value for spec in teacher_specs],
            "teacher_route_key": cfg.multi_teacher.route_key if cfg.multi_teacher else None,
            "teacher_blend_mode": (
                "probability_union"
                if cfg.multi_teacher and distill_mode == DistillMode.TOPK_FORWARD_KL
                else None
            ),
            "teacher_blend_weights": (
                {spec.routing_value: spec.blend_weight for spec in teacher_specs}
                if cfg.multi_teacher and distill_mode == DistillMode.TOPK_FORWARD_KL
                else None
            ),
            "distill_mode": distill_mode.value,
            "sdft_top_k": cfg.sdft_top_k,
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
                CLEANUP_DEPLOYMENT_ON_CLOSE_DELETE if cancel_on_exit else None
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
        student_model = student_sampler.model

        teacher_runtime = _resolve_teacher_runtime(
            cfg=cfg,
            teacher_specs=teacher_specs,
            service=service,
            tokenizer=tokenizer,
            base_url=base_url,
            cancel_on_exit=cancel_on_exit,
        )
        teacher_entries = teacher_runtime.entries
        route_to_entry = teacher_runtime.route_to_entry
        primary_teacher = teacher_runtime.primary
        teacher_model = primary_teacher.resolved_model
        teacher_sampler = primary_teacher.sampler
        is_multi_teacher = teacher_runtime.is_multi_teacher
        teacher_route_key = teacher_runtime.route_key

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
                    top_logprobs=0,
                    http_timeout=cfg.deployment.sample_timeout,
                )
            finally:
                teacher_inflight[teacher_key] -= 1
                teacher_scored[teacher_key] += 1

        async def _score_topk_routed(
            teacher_entry: _TeacherRoutingEntry,
            scoring_tokens: list[int],
            *,
            prompt_len: int,
            response_len: int,
            top_k: int,
        ):
            if top_k > MAX_SDFT_TOP_LOGPROBS:
                raise ValueError(
                    "Config.sdft_top_k exceeds the inference top_logprobs limit "
                    f"({MAX_SDFT_TOP_LOGPROBS})."
                )
            teacher_key = teacher_entry.spec.routing_value
            teacher_inflight[teacher_key] += 1
            try:
                return await _score_teacher_topk(
                    teacher_entry.sampler,
                    scoring_tokens,
                    prompt_len=prompt_len,
                    response_len=response_len,
                    top_logprobs=top_k,
                    http_timeout=cfg.deployment.sample_timeout,
                    tokenizer=tokenizer,
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
            if distill_mode == DistillMode.TOPK_FORWARD_KL:
                logger.info(
                    "Multi-teacher SDFT blend: mode=%s | weights=%s",
                    "probability_union",
                    ", ".join(
                        f"{entry.spec.routing_value}:{entry.spec.blend_weight:g}"
                        for entry in teacher_entries
                    ),
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
        adam_kwargs = dict(DEFAULT_ADAM)
        adam_kwargs["grad_clip_norm"] = cfg.grad_clip_norm
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **adam_kwargs)
        server_loss_config = {"ratio_log_cap": cfg.ratio_log_cap}

        objective = create_distillation_objective(
            objective_settings,
            loss_scale=cfg.opd_loss_scale,
            server_loss_config=server_loss_config,
        )

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

            # Sampled reverse KL routes each prompt to one teacher. Top-K
            # forward KL uses all configured teachers and blends their sparse
            # distributions.
            if distill_mode == DistillMode.TOPK_FORWARD_KL and is_multi_teacher:
                scoring_entries = list(teacher_entries)
            elif is_multi_teacher:
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
                scoring_entries = [scoring_entry]
            else:
                scoring_entries = [primary_teacher]

            teacher_sources: list[TeacherSourceContext] = []
            for scoring_entry in scoring_entries:
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
                teacher_sources.append(
                    TeacherSourceContext(
                        prompt_tokens=teacher_prompt_tokens,
                        weight=scoring_entry.spec.blend_weight,
                    )
                )

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

            async def _score_teacher_logprobs(
                source_idx: int,
                scoring_tokens: list[int],
                prompt_len: int,
                response_len: int,
            ) -> list[float] | None:
                return await _score_routed(
                    scoring_entries[source_idx],
                    scoring_tokens,
                    prompt_len=prompt_len,
                    response_len=response_len,
                )

            async def _score_teacher_topk(
                source_idx: int,
                scoring_tokens: list[int],
                prompt_len: int,
                response_len: int,
                top_k: int,
            ):
                return await _score_topk_routed(
                    scoring_entries[source_idx],
                    scoring_tokens,
                    prompt_len=prompt_len,
                    response_len=response_len,
                    top_k=top_k,
                )

            teacher_scores = await objective.collect_teacher_scores(
                sampled,
                teacher_sources,
                TeacherScoringFns(
                    logprobs=_score_teacher_logprobs,
                    topk=_score_teacher_topk,
                ),
            )
            return objective.build_prompt_group(
                sampled,
                teacher_scores,
                teacher_sources,
                warning=logger.warning,
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

        sdft_shapes_logged = False

        def train_step(
            step: int,
            prompt_groups: list[OPDPromptGroup],
            loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            nonlocal sdft_shapes_logged
            if not prompt_groups:
                raise ValueError("train_step requires at least one prompt group")

            train_batch = objective.build_train_batch(
                prompt_groups,
                step=step,
                include_shape_record=not sdft_shapes_logged,
            )
            if train_batch.shape_record is not None:
                logger.info(
                    "SDFT shapes: top_k_logprobs shape=%s | "
                    "top_k_indices shape=%s | target_tokens shape=%s | weights shape=%s",
                    train_batch.shape_record["top_k_logprobs_shape"],
                    train_batch.shape_record["top_k_indices_shape"],
                    train_batch.shape_record["target_tokens_shape"],
                    train_batch.shape_record["weights_shape"],
                )
                sdft_shapes_logged = True

            t0 = _time.time()
            with timer("fwd_bwd"):
                fwd_bwd_result = policy.forward_backward(
                    train_batch.datums,
                    train_batch.loss_name,
                    loss_fn_config=train_batch.loss_fn_config,
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
            for key, value in train_batch.input_metrics.items():
                metrics[f"train/{key}"] = value

            if fwd_bwd_result and hasattr(fwd_bwd_result, "metrics"):
                for key, value in fwd_bwd_result.metrics.items():
                    metrics[f"train/{key}"] = value
            if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
                for key, value in optim_result.metrics.items():
                    metrics[f"train/{key}"] = value

            step_summary = objective.summarize_step(metrics, step=step)
            active_tokens = step_summary.active_tokens
            logger.info(step_summary.log_message, *step_summary.log_args)
            log_metrics_json(step, **step_summary.json_metrics)
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
            run_batched_training_loop(
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


def _multi_teacher_from_env() -> MultiTeacherConfig | None:
    """Parse ``DISTILLATION_TEACHERS`` for routed distillation.

    ``DISTILLATION_TEACHER_ROUTE_KEY`` names the dataset row key whose value
    selects the teacher model for each prompt (default ``teacher``).
    Returns ``None`` when no routed-teacher env is set.
    """
    raw = os.environ.get("DISTILLATION_TEACHERS", "").strip()
    if not raw:
        return None
    teachers = [TeacherConfig(model=m.strip()) for m in raw.split(",") if m.strip()]
    route_key = os.environ.get("DISTILLATION_TEACHER_ROUTE_KEY", "teacher")
    return MultiTeacherConfig(teachers=teachers, route_key=route_key)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        log_path="./distillation_logs",
        base_model="accounts/fireworks/models/qwen3-8b",
        teacher_model=os.environ.get("DISTILLATION_TEACHER_MODEL", ""),
        multi_teacher=_multi_teacher_from_env(),
        trainer=TrainerConfig(
            training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200",
        ),
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
    )
    main(cfg)
