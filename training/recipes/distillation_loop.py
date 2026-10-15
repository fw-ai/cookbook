#!/usr/bin/env python3
"""On-policy distillation: the student learns from a frozen teacher on its own rollouts.

Each step has three visible phases:

1. the student samples completions for a batch of prompts and the teacher
   scores those exact tokens,
2. one server-side ``importance_sampling`` forward/backward whose per-token
   advantage is ``teacher_logprob - sampling_logprob`` (sampled reverse KL),
   then one optimizer step,
3. hotload the new student weights before sampling the next batch.

There is no outcome reward and no reference model: the teacher/student
log-ratio is the whole training signal. A row's ``teacher_messages`` give the
teacher privileged context the student never sees. Fork this file to route
prompts to several teachers or change the objective; for top-K SDFT on
serverless see the external Fireworks Tinker-Cookbook fork linked in
``training/docs/sdft.md``.

Usage:
    export FIREWORKS_API_KEY=...
    export DISTILLATION_TEACHER_MODEL=accounts/fireworks/models/qwen3-32b
    python -m recipes.distillation_loop
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import signal
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

from fireworks.training import sdk
from fireworks.training.sdk.deployment import AdaptiveConcurrencyController
from training.utils import (
    CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO,
    DEFAULT_ADAM,
    ConcurrencyConfig,
    DeployConfig,
    ReconnectableClient,
    TrainerConfig,
    WandBConfig,
    build_service_client,
    load_deployment_tokenizer,
    load_jsonl_dataset,
    log_metrics,
    log_metrics_json,
    make_weight_sync,
    prepare_sampling_messages,
    read_api_extra_headers_env,
    setup_wandb,
    validate_config,
    wandb_finish,
)
from training.utils.checkpoints import TrainingCheckpoints
from training.utils.client import DEFAULT_TIMEOUT_S
from training.utils.dataloader import CursorDataLoader
from training.utils.distillation import (
    OPDPromptGroup,
    build_opd_datums,
    build_opd_prompt_group,
)
from training.utils.distillation.eval import EvalContext, EvalFn
from training.utils.distillation.sampling import (
    score_teacher_logprobs,
    teacher_messages_for_row,
    tokenize_teacher_prompt,
)
from training.utils.distillation.teacher import (
    DEFAULT_TEACHER_DEPLOYMENT_TIMEOUT_S,
    create_teacher_sampler,
)
from training.utils.logging import log_run_resources
from training.utils.metrics import (
    forward_backward_metrics,
    optimizer_metrics,
    sampler_metrics,
    weight_sync_metrics,
)
from training.utils.termination import TerminatedBySignal
from training.utils.timer import elapsed_timer, flush_timing

logger = logging.getLogger(__name__)


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    """Student model ID."""
    dataset: str = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"

    teacher_model: str = ""
    """Teacher on the student's tokenizer: a base model (deployed for scoring)
    or an existing deployment / deployed model (used directly)."""
    teacher_deployment_id: str | None = None
    """Deployment ID to create or reuse when ``teacher_model`` is a base model."""
    teacher_deployment_shape: str | None = None
    """Teacher deployment shape; ``None`` lets the deployment API choose."""
    teacher_replica_count: int = 1
    teacher_deployment_timeout_s: float = DEFAULT_TEACHER_DEPLOYMENT_TIMEOUT_S

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
    lora_alpha: int | None = 32
    """LoRA alpha scaling factor. Ignored when ``lora_rank == 0``.

    Defaults to ``32`` to match Tinker and the Training API SDK client
    ``DEFAULT_LORA_ALPHA``. Override when you need a different scaling factor."""

    prompt_groups_per_step: int = 1
    """Prompts sampled and trained on per optimizer step."""
    grad_clip_norm: float = 0.0
    """Max gradient norm for clipping. 0 disables clipping."""
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    """Adaptive request concurrency for student sampling."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    step_timeout: int = 0
    """Timeout in seconds for forward_backward / optim_step calls. 0 = default."""
    weight_sync_timeout: int = 600
    dcp_save_interval: int = 0
    eval_every: int = 0
    """Run ``main(eval_fn=...)`` before training and every N steps. 0 disables."""
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="distillation-tinker"))

    init_from_checkpoint: str | None = None
    """Load pretrained DCP weights on a fresh dataset."""
    save_final_checkpoint: bool = True
    output_model_id: str | None = None


def main(
    config: Config,
    *,
    eval_fn: EvalFn | None = None,
    cancel_on_exit: bool = False,
) -> dict[str, Any]:
    """Run on-policy distillation.

    ``eval_fn`` is an optional async callback that receives an
    :class:`~training.utils.distillation.eval.EvalContext` and returns
    ``eval/*`` metrics. ``cancel_on_exit`` deletes the trainer and scales the
    deployments this run created to zero on exit.
    """
    cfg = config
    if not cfg.teacher_model:
        raise ValueError("Config.teacher_model is required.")
    if not cfg.deployment.tokenizer_model:
        raise ValueError(
            "deployment.tokenizer_model is required for client-side tokenization. "
            "Use the HuggingFace tokenizer shared by the student and teacher."
        )
    if cfg.teacher_replica_count < 1:
        raise ValueError("teacher_replica_count must be >= 1.")
    if cfg.prompt_groups_per_step < 1:
        raise ValueError("prompt_groups_per_step must be >= 1.")

    def _signal_handler(signum, _):
        name = signal.Signals(signum).name
        raise TerminatedBySignal(name)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(
        cfg.base_model,
        cfg.dataset,
        deploy=cfg.deployment,
        output_model_id=cfg.output_model_id,
    )
    setup_wandb(
        cfg.wandb,
        {
            "teacher_model": cfg.teacher_model,
            "opd_loss_scale": cfg.opd_loss_scale,
            "ratio_log_cap": cfg.ratio_log_cap,
            "completions_per_prompt": cfg.completions_per_prompt,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "max_completion_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "lr": cfg.learning_rate,
        },
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()
    with ExitStack() as stack:
        tokenizer = load_deployment_tokenizer(cfg.deployment)
        service = build_service_client(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
            base_model=cfg.base_model,
            tokenizer_model=cfg.deployment.tokenizer_model,
            max_lora_rank=cfg.lora_rank,
            max_context_length=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            trainer=cfg.trainer,
            deployment=cfg.deployment,
            hotload_timeout_s=cfg.weight_sync_timeout,
            cleanup_trainer_on_close=cancel_on_exit,
            cleanup_deployment_on_close=(CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO if cancel_on_exit else None),
            reference_required=False,
        )
        stack.callback(service.close)

        training_client = service.create_training_client(
            cfg.base_model,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
        )
        policy = ReconnectableClient.from_training_client(
            training_client,
            base_model=cfg.base_model,
            lora_rank=cfg.lora_rank,
            job_id=service.trainer_job_id,
            default_timeout=cfg.step_timeout or DEFAULT_TIMEOUT_S,
            service=service,
        )
        publish_weights = make_weight_sync(policy, service, cfg.deployment)

        concurrency = AdaptiveConcurrencyController(
            initial_window=(cfg.concurrency.initial_window or 8 * (cfg.deployment.replica_count or 1)),
            min_window=cfg.concurrency.min_window,
            max_window=cfg.concurrency.max_window,
            prefill_queue_target=cfg.concurrency.prefill_queue_target,
            adjustment_interval=cfg.concurrency.rollout_adjustment_interval,
        )
        student_sampler = service.create_deployment_sampler(
            tokenizer=tokenizer,
            concurrency_controller=concurrency,
        )
        stack.callback(student_sampler.close)
        teacher_sampler = create_teacher_sampler(
            service,
            cfg.teacher_model,
            tokenizer=tokenizer,
            inference_url=base_url,
            deployment_id=cfg.teacher_deployment_id,
            deployment_shape=cfg.teacher_deployment_shape,
            replica_count=cfg.teacher_replica_count,
            timeout_s=cfg.teacher_deployment_timeout_s,
            cleanup_on_close=cancel_on_exit,
        )
        stack.callback(teacher_sampler.close)

        checkpoint = TrainingCheckpoints(
            policy,
            service,
            trainer_id=service.trainer_job_id,
            log_path=cfg.log_path,
            lora_rank=cfg.lora_rank,
        )
        resume_info = checkpoint.resume(init_from_checkpoint=cfg.init_from_checkpoint)
        step_offset = resume_info.step if resume_info else 0
        prior_rows_consumed = resume_info.data_consumed if resume_info else 0
        log_metrics({"train/step": step_offset}, step=step_offset)

        # Strict on-policy: the student deployment serves the trainer's
        # weights before the first rollout and after every optimizer step.
        with elapsed_timer("weight_sync") as span:
            publish_weights(f"step-{step_offset}", checkpoint_type="base")
        logger.info("[step %d] initial weight sync (%.1fs)", step_offset, span.elapsed)
        flush_timing()
        log_run_resources(service)

        rows = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        row_loader = CursorDataLoader(rows, start_cursor=prior_rows_consumed, epochs=cfg.epochs)
        row_iterator = iter(row_loader)

        sample_kwargs: dict[str, Any] = {
            "max_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "max_seq_len": service.max_context_length,
            "http_timeout": cfg.deployment.sample_timeout,
            "logprobs": True,
        }

        # -- Rollout function (VISIBLE -- customize this) ---------------------

        async def sample_one_prompt(row: dict) -> OPDPromptGroup | None:
            """Sample the student, then score its tokens under the teacher's prompt."""
            messages = prepare_sampling_messages(row.get("messages", []))
            if not messages:
                return None
            try:
                teacher_prompt = tokenize_teacher_prompt(tokenizer, teacher_messages_for_row(row, messages))
                sampled = await student_sampler.sample_with_tokens(
                    messages=messages,
                    n=cfg.completions_per_prompt,
                    **sample_kwargs,
                )
            except Exception as error:
                logger.warning("Skipping row: %s", error)
                return None
            teacher_logprobs = await score_teacher_logprobs(
                teacher_sampler,
                teacher_prompt,
                sampled,
                http_timeout=cfg.deployment.sample_timeout,
            )
            return build_opd_prompt_group(sampled, teacher_logprobs)

        eval_metrics: dict[str, Any] = {}

        async def run_eval(step: int) -> None:
            nonlocal eval_metrics
            started = time.monotonic()
            eval_metrics = await eval_fn(
                EvalContext(
                    config=cfg,
                    rows=rows,
                    student_sampler=student_sampler,
                    teacher_sampler=teacher_sampler,
                    tokenizer=tokenizer,
                    step=step,
                    max_seq_len=service.max_context_length,
                )
            )
            logger.info("[step %d] eval done (%.1fs)", step, time.monotonic() - started)
            log_metrics(eval_metrics, step=step)

        evaluate = eval_fn is not None and cfg.eval_every > 0
        logger.info(
            "Distillation: teacher %s | %d prompts x %d completions per step",
            teacher_sampler.model,
            cfg.prompt_groups_per_step,
            cfg.completions_per_prompt,
        )

        # -- Training loop (VISIBLE algorithm phases) -------------------------

        adam_params = sdk.AdamParams(
            learning_rate=cfg.learning_rate,
            **{**DEFAULT_ADAM, "grad_clip_norm": cfg.grad_clip_norm},
        )

        async def run_training() -> int:
            step = step_offset
            if evaluate:
                await run_eval(step)
            while batch := list(itertools.islice(row_iterator, cfg.prompt_groups_per_step)):
                # 1. Student rollouts, scored token by token by the teacher.
                with elapsed_timer("rollout"):
                    results = await asyncio.gather(*(sample_one_prompt(item.value) for item in batch))
                prompt_groups = [group for group in results if group is not None]
                for item in batch:
                    row_loader.mark_resolved(item.index)
                if not prompt_groups:
                    continue

                # 2. One server-side importance-sampling step on the
                #    teacher-minus-student advantages.
                datums, opd_metrics = build_opd_datums(prompt_groups, loss_scale=cfg.opd_loss_scale)
                with elapsed_timer("fwd_bwd"):
                    fwd_bwd_result = policy.forward_backward(
                        datums,
                        "importance_sampling",
                        loss_fn_config={"ratio_log_cap": cfg.ratio_log_cap},
                    )
                with elapsed_timer("optim_step"):
                    optim_result = policy.optim_step(adam_params)
                step += 1

                # 3. Publish the new student before the next rollout batch.
                with elapsed_timer("weight_sync"):
                    sync_result = publish_weights(f"step-{step}")

                metrics = flush_timing()
                metrics.update(forward_backward_metrics([fwd_bwd_result]))
                metrics.update(optimizer_metrics(optim_result))
                metrics.update({f"train/{k}": v for k, v in opd_metrics.items()})
                metrics.update(sampler_metrics(student_sampler.drain_metrics()))
                metrics.update(weight_sync_metrics(sync_result))
                metrics.update({f"concurrency/{k}": v for k, v in concurrency.step_completed().items()})
                metrics["train/step"] = step
                logger.info(
                    "Step %d | sampled reverse KL: %.4f | OPD advantage: %.4f | tokens=%d",
                    step,
                    opd_metrics["opd_sampled_reverse_kl"],
                    opd_metrics["opd_advantage"],
                    opd_metrics["opd_active_tokens"],
                )
                log_metrics_json(
                    step,
                    opd_sampled_reverse_kl=opd_metrics["opd_sampled_reverse_kl"],
                    opd_advantage=opd_metrics["opd_advantage"],
                    active_tokens=opd_metrics["opd_active_tokens"],
                )
                log_metrics(metrics, step=step)

                if evaluate and step % cfg.eval_every == 0:
                    await run_eval(step)
                if cfg.dcp_save_interval > 0 and (step - step_offset) % cfg.dcp_save_interval == 0:
                    checkpoint.save(
                        f"step-{step}",
                        resumable=True,
                        promotable=False,
                        data_consumed=row_loader.data_consumed,
                    )
            return step

        global_step = asyncio.run(run_training())

        has_trained_steps = global_step > step_offset
        has_advanced_dataset = row_loader.data_consumed > prior_rows_consumed
        if cfg.save_final_checkpoint and (has_trained_steps or has_advanced_dataset):
            checkpoint.save(
                f"step-{global_step}",
                resumable=True,
                promotable=has_trained_steps,
                data_consumed=row_loader.data_consumed,
            )
            if cfg.output_model_id and has_trained_steps:
                checkpoint.promote_latest(cfg.output_model_id, cfg.base_model)

        logger.info(
            "Distillation complete: %d steps (%d new)",
            global_step,
            global_step - step_offset,
        )
        wandb_finish(metrics_file=os.environ.get("COOKBOOK_METRICS_FILE"))
        return {
            "steps": global_step,
            "policy_job_id": service.trainer_job_id,
            "deployment_id": service.deployment_id,
            "eval": eval_metrics,
            "max_seq_len": service.max_context_length,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(
        Config(
            log_path="./distillation_logs",
            teacher_model=os.environ.get("DISTILLATION_TEACHER_MODEL", ""),
            trainer=TrainerConfig(
                training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200",
            ),
            deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
        )
    )
