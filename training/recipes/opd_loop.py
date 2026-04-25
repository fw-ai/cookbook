#!/usr/bin/env python3
"""Sampled-token on-policy distillation (OPD) training loop.

The student samples responses from its own hot-loaded deployment.  A teacher
deployment then scores those exact token sequences.  Training uses Tinker's
server-side ``importance_sampling`` loss with per-token advantages equal to
``teacher_logprob - sampling_logprob`` on response tokens.

This is intentionally not a reward-model RL loop: there is no outcome reward
and no reference trainer.  The dense teacher/student log-ratio is the OPD
signal.

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.opd_loop
"""

from __future__ import annotations

import os
import signal
import asyncio
import logging
import warnings
import re
import time as _time
from contextlib import ExitStack
from typing import Any, Callable
from dataclasses import field, dataclass

import tinker
import transformers

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.client import GradAccNormalization
from fireworks.training.sdk.deployment import AdaptiveConcurrencyController, DeploymentSampler
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils import (
    DEFAULT_ADAM,
    ConcurrencyConfig,
    DeployConfig,
    InfraConfig,
    RLPromptDataset,
    ResourceCleanup,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    WeightSyncConfig,
    load_jsonl_dataset,
    log_metrics_json,
    prepare_sampling_messages,
    read_api_extra_headers_env,
    setup_wandb,
    validate_config,
    wandb_finish,
    wandb_log,
)
from training.utils.checkpoint_utils import (
    CheckpointKind,
    resolve_resume,
    save_checkpoint,
    validate_warm_start_config,
)
from training.utils.opd import (
    OPDPromptGroup,
    build_opd_server_datums,
    combine_opd_prompt_groups,
)

# Re-export the OPD eval helpers from the recipe module for existing examples.
from training.utils.opd_eval import (  # noqa: F401
    evaluate_teacher_trace_logprob_gap,
    extract_final_answer,
    expected_final_answer,
    make_teacher_trace_logprob_gap_eval,
    normalize_final_answer,
    validate_opd_trace_result,
    validate_privileged_opd_dataset,
)
from training.utils.opd_sampling import (
    _align_completion_logprobs,
    _align_response_logprobs,
    _build_teacher_scoring_tokens,
    _score_with_teacher,
    _teacher_messages_for_row,
    _tokenize_teacher_prompt,
)
from training.utils.infra import request_deployment, wait_deployment
from training.utils.rl import setup_infra
from training.utils.rl.train import TrainStepFns, run_rl_loop
from training.utils.timer import flush_timing, timer

logger = logging.getLogger(__name__)


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
    """Replica count for an auto-created frozen teacher deployment."""

    teacher_deployment_timeout_s: float = 5400
    """Timeout for an auto-created frozen teacher deployment to become ready."""

    teacher_inference_url: str | None = None
    """Optional inference base URL for the teacher. Defaults to FIREWORKS_BASE_URL."""

    teacher_top_logprobs: int = 0
    """Optional top-logprob count to request while scoring. The loss needs only sampled-token logprobs."""

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

    policy_job_id: str | None = None
    """Pre-created RLOR policy trainer job ID (skip creation if set)."""

    init_from_checkpoint: str | None = None
    """Load pretrained DCP weights on a fresh dataset."""

    warm_start_from_adapter: str | None = None
    """GCS URI of an HF PEFT adapter directory for LoRA warm start."""

    output_model_id: str | None = None
    save_final_checkpoint: bool = True

    step_timeout: int = 0
    """Timeout in seconds for forward_backward / optim_step calls."""

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="opd-tinker"))
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


def _default_teacher_deployment_id(base_model: str) -> str:
    slug = base_model.rstrip("/").split("/")[-1]
    safe_slug = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")
    return f"opd-teacher-{safe_slug}"[:63]


@dataclass(frozen=True)
class _TeacherDeploymentRequest:
    deployment_id: str
    deploy_cfg: DeployConfig
    info: Any


def _request_frozen_teacher_deployment(
    deploy_mgr: DeploymentManager,
    *,
    infra_cfg: InfraConfig,
    base_model: str,
    deployment_id: str,
    deployment_shape: str | None,
    replica_count: int,
    cleanup: ResourceCleanup | None,
) -> _TeacherDeploymentRequest:
    """Request a frozen teacher deployment with the same helper path as RL rollouts."""
    existing = deploy_mgr.get(deployment_id)
    if existing is not None:
        state = getattr(existing, "state", None)
        if state in {"FAILED", "DELETED", "DELETING"}:
            raise RuntimeError(
                f"Teacher deployment {deployment_id!r} is in terminal state {state!r}. "
                "Use a different teacher_deployment_id or restore/delete the old resource."
            )
        logger.info("Re-using frozen teacher deployment: %s", deployment_id)
        deploy_cfg = DeployConfig(deployment_id=deployment_id)
        return _TeacherDeploymentRequest(
            deployment_id=deployment_id,
            deploy_cfg=deploy_cfg,
            info=existing,
        )

    deploy_cfg = DeployConfig(
        deployment_id=deployment_id,
        deployment_shape=deployment_shape,
        enable_hot_load=False,
        hot_load_bucket_type=None,
        replica_count=replica_count,
    )
    info = request_deployment(deploy_mgr, deploy_cfg, base_model, infra_cfg)
    if cleanup is not None:
        cleanup.deployment(deployment_id, action="scale_to_zero")
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
    request.deploy_cfg.deployment_timeout_s = timeout_s
    ready = wait_deployment(deploy_mgr, request.info, request.deploy_cfg)
    return ready.inference_model or f"accounts/{deploy_mgr.account_id}/deployments/{request.deployment_id}"


def _make_teacher_deployment_provisioner(
    cfg: Config,
    deploy_mgr: DeploymentManager,
    *,
    cleanup: ResourceCleanup | None,
    model_out: dict[str, str],
) -> Callable[[str | None], tuple[str, Callable[[], None]] | None]:
    """Build a setup hook for an auto-created frozen teacher deployment."""

    def _provision(resolved_deployment_shape: str | None) -> tuple[str, Callable[[], None]] | None:
        if not _is_base_model_resource(cfg.teacher_model):
            return None

        deployment_id = cfg.teacher_deployment_id or _default_teacher_deployment_id(cfg.teacher_model)
        deployment_shape = cfg.teacher_deployment_shape or resolved_deployment_shape
        request = _request_frozen_teacher_deployment(
            deploy_mgr,
            infra_cfg=cfg.infra,
            base_model=cfg.teacher_model,
            deployment_id=deployment_id,
            deployment_shape=deployment_shape,
            replica_count=cfg.teacher_replica_count,
            cleanup=cleanup,
        )

        def _wait_and_capture() -> None:
            model_out["teacher_model"] = _wait_frozen_teacher_deployment(
                deploy_mgr,
                request,
                timeout_s=cfg.teacher_deployment_timeout_s,
            )

        return "teacher_deployment", _wait_and_capture

    return _provision


def _resolve_teacher_model_for_scoring(
    cfg: Config,
    infra_teacher_model: str | None,
) -> str:
    """Resolve ``cfg.teacher_model`` to an inference target for frozen scoring."""
    if not _is_base_model_resource(cfg.teacher_model):
        return cfg.teacher_model
    if not infra_teacher_model:
        raise RuntimeError("OPD teacher deployment was not provisioned")
    return infra_teacher_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
    deploy_mgr: DeploymentManager | None = None,
    cancel_on_exit: bool = False,
    cleanup_on_exit: bool | None = None,
):
    if cleanup_on_exit is not None:
        warnings.warn(
            "opd_loop.main(cleanup_on_exit=...) is deprecated; use cancel_on_exit=...",
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

    if not cfg.teacher_model:
        raise ValueError("Config.teacher_model is required for OPD training.")

    validate_config(
        cfg.base_model,
        cfg.dataset,
        cfg.weight_sync,
        cfg.deployment,
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

    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step

    setup_wandb(
        cfg.wandb,
        {
            "teacher_model": cfg.teacher_model,
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

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
        )
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
        )

    runner.write_status(RunStatus.PENDING, message="provisioning")

    def _on_trainer_status(msg: str) -> None:
        runner.write_status(RunStatus.PENDING, message=msg)

    with runner, ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup, ExitStack() as stack:
        teacher_provisioners = []
        teacher_model_out: dict[str, str] = {}
        if _is_base_model_resource(cfg.teacher_model):
            teacher_provisioners.append(
                _make_teacher_deployment_provisioner(
                    cfg,
                    deploy_mgr,
                    cleanup=cleanup if cancel_on_exit else None,
                    model_out=teacher_model_out,
                )
            )

        # Shapes + trainer + deployment + trainer clients. OPD does not need a
        # reference trainer because teacher scores come from inference.
        infra = setup_infra(
            rlor_mgr=rlor_mgr,
            deploy_mgr=deploy_mgr,
            base_model=cfg.base_model,
            infra_cfg=cfg.infra,
            deploy_cfg=cfg.deployment,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            step_timeout=cfg.step_timeout,
            policy_job_id=cfg.policy_job_id,
            needs_reference=False,
            needs_inference=True,
            role_prefix="opd",
            api_key=api_key,
            cleanup=cleanup if cancel_on_exit else None,
            on_status=_on_trainer_status,
            post_shape_provisioners=teacher_provisioners,
        )
        for closeable in infra.closeables:
            stack.callback(closeable.close)

        runner.set_accelerator_info(profile=infra.policy_profile)
        wandb_log(infra.boot_metrics, step=0)

        policy = infra.policy
        policy_job_id = infra.policy_job_id
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.deployment.tokenizer_model,
            trust_remote_code=True,
        )

        initial_window = cfg.concurrency.initial_window or (8 * infra.deployment_gpu_count)
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
        student_sampler = DeploymentSampler(
            inference_url=deploy_mgr.inference_url,
            model=infra.inference_model,
            api_key=api_key,
            tokenizer=tokenizer,
            concurrency_controller=concurrency_controller,
        )
        teacher_model = _resolve_teacher_model_for_scoring(
            cfg,
            teacher_model_out.get("teacher_model"),
        )
        teacher_sampler = DeploymentSampler(
            inference_url=cfg.teacher_inference_url or base_url,
            model=teacher_model,
            api_key=api_key,
            tokenizer=tokenizer,
        )
        weight_syncer = WeightSyncer(
            policy_client=policy.inner,
            deploy_mgr=deploy_mgr,
            deployment_id=infra.deployment_id,
            base_model=cfg.rollout_base_model or cfg.base_model,
            hotload_timeout=cfg.weight_sync.weight_sync_timeout,
            first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
            lora_rank=cfg.lora_rank,
        )

        logger.info(
            "OPD training: teacher=%s | completions_per_prompt=%d | groups_per_step=%d",
            teacher_model,
            completions_per_prompt,
            prompt_groups_per_step,
        )

        # -- Resume ---------------------------------------------------------------

        resume_info = resolve_resume(
            policy,
            cfg.log_path,
            cfg.init_from_checkpoint,
            cfg.warm_start_from_adapter,
        )
        step_offset = resume_info.step if resume_info else 0
        wandb_log({"train/step": step_offset}, step_offset)

        if cfg.weight_sync.weight_sync_before_training and infra.deployment_id:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            weight_syncer.save_and_hotload(name, checkpoint_type="base")

        # -- Prepare sampling and training --------------------------------------

        raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        all_rows = raw_dataset * cfg.epochs
        rl_dataset = RLPromptDataset(all_rows, prompts_per_step=prompt_groups_per_step)
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)
        server_loss_config = {"ratio_log_cap": cfg.ratio_log_cap}

        sample_kwargs: dict[str, Any] = {
            "max_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "max_seq_len": infra.max_seq_len,
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
            teacher_messages = _teacher_messages_for_row(row, input_messages)

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
                    _score_with_teacher(
                        teacher_sampler,
                        scoring_tokens,
                        prompt_len=len(teacher_prompt_tokens),
                        response_len=response_len,
                        top_logprobs=cfg.teacher_top_logprobs,
                        http_timeout=cfg.deployment.sample_timeout,
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
                "student_model": infra.inference_model,
                "teacher_model": teacher_model,
                "tokenizer": tokenizer,
                "global_step": step,
                "max_seq_len": infra.max_seq_len,
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
            dcp_interval = cfg.weight_sync.dcp_save_interval
            if dcp_interval > 0 and rollouts_completed > 0 and rollouts_completed % dcp_interval == 0:
                data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    rollouts_completed * prompt_groups_per_step
                )
                save_checkpoint(
                    policy,
                    f"step-{step}",
                    cfg.log_path,
                    {
                        "step": step,
                        "data_consumed": data_consumed,
                        "source_job_id": policy_job_id,
                    },
                    kind=CheckpointKind.STATE,
                    base_model=cfg.base_model,
                    training_shape=infra.training_shape_id,
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
            return step, metrics

        def _weight_sync(step: int) -> None:
            logger.info("[step %d] weight_sync: saving + loading...", step)
            t0 = _time.time()
            with timer("weight_sync"):
                weight_syncer.save_and_hotload(f"step-{step}")
            logger.info("[step %d] weight_sync: done (%.1fs)", step, _time.time() - t0)
            if (
                cfg.step_eval is not None
                and cfg.step_eval_interval > 0
                and step % cfg.step_eval_interval == 0
            ):
                _run_step_eval(step, phase="post-sync")

        def _loop_metrics_callback(loop_metrics: dict) -> None:
            cc_summary = concurrency_controller.step_completed()
            for key, value in cc_summary.items():
                loop_metrics[f"concurrency/{key}"] = value
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
                weight_sync_fn=_weight_sync if cfg.weight_sync.weight_sync_interval > 0 else None,
                weight_sync_interval=cfg.weight_sync.weight_sync_interval,
            )
        )

        # -- Final checkpoint --------------------------------------------------

        if cfg.save_final_checkpoint and global_step > step_offset:
            try:
                checkpoint_name = f"step-{global_step}"
                data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    (global_step - step_offset) * prompt_groups_per_step
                )
                paths = save_checkpoint(
                    policy,
                    checkpoint_name,
                    cfg.log_path,
                    {
                        "step": global_step,
                        "data_consumed": data_consumed,
                        "source_job_id": policy_job_id,
                    },
                    kind=CheckpointKind.BOTH,
                    base_model=cfg.base_model,
                    training_shape=infra.training_shape_id,
                )
                if cfg.output_model_id:
                    rlor_mgr.promote_checkpoint(
                        policy_job_id,
                        paths["sampler_path"],
                        cfg.output_model_id,
                        cfg.base_model,
                    )
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
        logger.info("OPD training complete: %d steps", global_step)
        wandb_finish()
        return {
            "steps": global_step,
            "policy_job_id": policy_job_id,
            "eval": eval_metrics,
            "max_seq_len": infra.max_seq_len,
            "training_shape_id": infra.training_shape_id,
            "deployment_shape": infra.deployment_shape,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        log_path="./opd_logs",
        base_model="accounts/fireworks/models/qwen3-8b",
        teacher_model=os.environ.get("OPD_TEACHER_MODEL", ""),
        infra=InfraConfig(
            training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200",
        ),
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
    )
    main(cfg)
