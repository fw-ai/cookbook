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
from contextlib import ExitStack
from typing import Any
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
from training.utils.rl import setup_infra
from training.utils.rl.train import TrainStepFns, run_rl_loop
from training.utils.timer import flush_timing, timer

import time as _time

logger = logging.getLogger(__name__)


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    rollout_base_model: str | None = None
    dataset: str = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"

    teacher_model: str = ""
    """Teacher model/deployment ID on the same tokenizer as ``base_model``."""

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
    lora_rank: int = 0
    prompt_groups_per_step: int = 1

    grad_accumulation_normalization: GradAccNormalization | str | None = GradAccNormalization.NUM_LOSS_TOKENS

    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    policy_job_id: str | None = None
    init_from_checkpoint: str | None = None
    warm_start_from_adapter: str | None = None
    output_model_id: str | None = None
    save_final_checkpoint: bool = True
    step_timeout: int = 0

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="opd-tinker"))
    runner: RunnerConfig = field(default_factory=RunnerConfig)


def _align_completion_logprobs(
    completion_logprobs: list[float],
    *,
    prompt_len: int,
    target_len: int,
    echoed: bool,
) -> list[float]:
    """Align API completion logprobs to ``target_tokens`` length."""
    if echoed:
        aligned = list(completion_logprobs)
    else:
        response_start = max(0, prompt_len - 1)
        aligned = [0.0] * response_start + list(completion_logprobs)

    if len(aligned) < target_len:
        aligned.extend([0.0] * (target_len - len(aligned)))
    return aligned[:target_len]


def _extract_scored_token_logprobs(
    response: dict[str, Any],
    *,
    target_len: int,
) -> list[float] | None:
    """Extract echo logprobs for ``tokens[1:]`` from a completions response."""
    choices = response.get("choices", [])
    if not choices:
        return None

    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict):
        return None

    content = logprobs.get("content")
    if not isinstance(content, list) or len(content) < 2:
        return None

    # Echo responses include an unconditional first-token logprob, then one
    # logprob per next-token target.  A generated extra token may follow; trim it.
    target_content = content[1 : 1 + target_len]
    if len(target_content) < target_len:
        return None
    return [float(item.get("logprob", 0.0)) for item in target_content]


async def _score_with_teacher(
    sampler: DeploymentSampler,
    token_ids: list[int],
    *,
    top_logprobs: int,
    http_timeout: int,
) -> list[float] | None:
    """Score a full student sequence with the teacher deployment."""
    target_len = max(0, len(token_ids) - 1)
    if target_len == 0:
        return None

    request_kwargs: dict[str, Any] = {
        "logprobs": True,
        "echo": True,
        "raw_output": True,
        "http_timeout": http_timeout,
    }
    if top_logprobs > 0:
        request_kwargs["top_logprobs"] = top_logprobs

    try:
        response, _metrics = await sampler.async_completions_stream(
            prompt=token_ids,
            max_tokens=1,
            temperature=0.0,
            **request_kwargs,
        )
    except Exception as exc:
        logger.warning("Teacher scoring failed: %s", exc)
        return None

    return _extract_scored_token_logprobs(response, target_len=target_len)


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
    deploy_mgr: DeploymentManager | None = None,
    cancel_on_exit: bool = False,
):
    cfg = config
    runner = RunnerIO(cfg.runner)

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

    setup_wandb(
        cfg.wandb,
        {
            "teacher_model": cfg.teacher_model,
            "opd_loss_scale": cfg.opd_loss_scale,
            "ratio_log_cap": cfg.ratio_log_cap,
            "completions_per_prompt": cfg.completions_per_prompt,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "lr": cfg.learning_rate,
        },
    )

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
        student_sampler = DeploymentSampler(
            inference_url=deploy_mgr.inference_url,
            model=infra.inference_model,
            api_key=api_key,
            tokenizer=tokenizer,
            concurrency_controller=concurrency_controller,
        )
        teacher_sampler = DeploymentSampler(
            inference_url=cfg.teacher_inference_url or base_url,
            model=cfg.teacher_model,
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
            cfg.teacher_model,
            cfg.completions_per_prompt,
            cfg.prompt_groups_per_step,
        )

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

        raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        all_rows = raw_dataset * cfg.epochs
        rl_dataset = RLPromptDataset(all_rows, prompts_per_step=cfg.prompt_groups_per_step)
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)
        server_loss_config = {"ratio_log_cap": cfg.ratio_log_cap}

        sample_kwargs: dict[str, Any] = {
            "max_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "max_seq_len": infra.max_seq_len,
            "http_timeout": cfg.deployment.sample_timeout,
            "logprobs": True,
        }

        async def sample_one_prompt(row: dict) -> OPDPromptGroup | None:
            messages = row.get("messages", [])
            input_messages = prepare_sampling_messages(messages)
            if not input_messages:
                return None

            try:
                sampled = await student_sampler.sample_with_tokens(
                    messages=input_messages,
                    n=cfg.completions_per_prompt,
                    **sample_kwargs,
                )
            except Exception as exc:
                logger.warning("Student sampling failed: %s", exc)
                return None

            sampled = [s for s in sampled if len(s.full_tokens) >= 2]
            if not sampled:
                return None

            teacher_scores = await asyncio.gather(
                *[
                    _score_with_teacher(
                        teacher_sampler,
                        s.full_tokens,
                        top_logprobs=cfg.teacher_top_logprobs,
                        http_timeout=cfg.deployment.sample_timeout,
                    )
                    for s in sampled
                ]
            )

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
                teacher_logprobs.append(teacher_lp[:target_len])
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
                    rollouts_completed * cfg.prompt_groups_per_step
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

        def _loop_metrics_callback(loop_metrics: dict) -> None:
            cc_summary = concurrency_controller.step_completed()
            for key, value in cc_summary.items():
                loop_metrics[f"concurrency/{key}"] = value
            wandb_log(loop_metrics, step=loop_metrics.get("train/step", 0))

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
                prompt_groups_per_step=cfg.prompt_groups_per_step,
                global_step=step_offset,
                metrics_callback=_loop_metrics_callback,
                weight_sync_fn=_weight_sync if cfg.weight_sync.weight_sync_interval > 0 else None,
                weight_sync_interval=cfg.weight_sync.weight_sync_interval,
            )
        )

        if cfg.save_final_checkpoint and global_step > step_offset:
            try:
                checkpoint_name = f"step-{global_step}"
                data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    (global_step - step_offset) * cfg.prompt_groups_per_step
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

        runner.write_status(
            RunStatus.COMPLETED,
            step=global_step,
            total_steps=total_steps,
            message="done",
        )
        runner.write_metadata()
        logger.info("OPD training complete: %d steps", global_step)
        wandb_finish()
        return {"steps": global_step, "policy_job_id": policy_job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        log_path="./opd_logs",
        base_model="accounts/fireworks/models/qwen3-8b",
        teacher_model="accounts/fireworks/models/qwen3-235b-a22b",
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
    )
    main(cfg)
