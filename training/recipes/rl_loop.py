#!/usr/bin/env python3
"""A small, synchronous, strictly on-policy GRPO recipe.

Each iteration has three visible phases:

1. collect one rollout batch: ``completions_per_prompt`` calls of
   ``rollout_fn(row)`` per row, grouped per row and filtered,
2. train the complete batch with one GRPO forward/backward and one optimizer step,
3. hotload the new policy before collecting the next batch.

Rollouts plug in through the same ``rollout_fn_factory(setup)`` seam as
``recipes.async_rl_loop``, so one task's code drives either loop. The default,
:func:`single_turn_rollout_fn`, samples one completion of the row's
``messages`` and grades it with :func:`reward_fn`. ``prepare_batch_fn(policy,
batch)`` prepares the complete batch before training (IGPO sets group-local
token advantages there). Fork this file to change the objective; the
modules in ``training.utils.rl.algorithm`` share one interface.

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.rl_loop
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from fireworks.training import sdk
from fireworks.training.sdk.training_spec import (
    LRSchedulerSpec,
    compute_lr,
    default_constant_schedule,
    normalize_lr_scheduler_spec,
)
from training.renderer import get_text_content
from training.utils import (
    CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO,
    DEFAULT_ADAM,
    DeployConfig,
    ReconnectableClient,
    TrainerConfig,
    WandBConfig,
    build_renderer,
    build_service_client,
    load_deployment_tokenizer,
    load_jsonl_dataset,
    log_metrics,
    log_metrics_json,
    make_weight_sync,
    phase_span,
    prepare_sampling_messages,
    read_api_extra_headers_env,
    resolve_router_replay_enabled,
    setup_wandb,
    validate_config,
    wandb_finish,
)
from training.utils.checkpoints import TrainingCheckpoints, validate_warm_start_config
from training.utils.data import compute_advantages
from training.utils.dataloader import CursorDataLoader
from training.utils.logging import log_run_resources
from training.utils.metrics import sampler_metrics, weight_sync_metrics
from training.utils.rl.algorithm import grpo as objective
from training.utils.rl.common import inference_drift_metrics
from training.utils.rl.heldout import RolloutEvaluationFn, run_evaluation
from training.utils.rl.losses import (
    PolicyBatch,
    PromptGroup,
    combine_prompt_groups,
    has_reward_variance,
)
from training.utils.rl.metrics import compute_step_metrics, sync_step_perf_metrics
from training.utils.rl.rollout import (
    Rollout,
    rollout_to_prompt_group,
    single_turn_renderer_rollout,
)
from training.utils.rl.rollout.lifecycle import close_rollout_fn
from training.utils.rl.rollout.setup import (
    DynamicFilterFn,
    RolloutFn,
    RolloutFnFactory,
    RolloutSetup,
    call_rollout_fn,
    default_sample_kwargs,
    rollout_context_names,
)
from training.utils.rl.sync_batch import collect_prompt_groups
from training.utils.termination import close_in_thread, run_in_thread, with_signals
from training.utils.timer import elapsed_timer, flush_timing

logger = logging.getLogger(__name__)

PrepareBatchFn = Callable[[Any, PolicyBatch], tuple[PolicyBatch, dict[str, float]]]


@dataclass
class Config:
    log_path: str
    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str | None = None
    """JSONL path/URL; optional when passing ``rows=`` to ``main()``."""

    learning_rate: float = 1e-5
    lr_scheduler: LRSchedulerSpec = field(default_factory=default_constant_schedule)

    completions_per_prompt: int = 4
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    shuffle: bool = True
    seed: int = 0
    max_rows: int = 100
    max_seq_len: int | None = None
    lora_rank: int = 0
    lora_alpha: int | None = 32
    """LoRA alpha scaling factor (Tinker and SDK default 32); ignored when ``lora_rank == 0``."""
    renderer_name: str = ""
    """Renderer for the default single-turn rollout; empty infers it from the tokenizer."""

    prompt_groups_per_step: int = 1
    """Valid prompt groups collected for each optimizer step."""

    grad_clip_norm: float = 0.0
    eps_clip: float = 0.2
    """GRPO lower clip epsilon; also the upper one unless ``eps_clip_high`` is set."""
    eps_clip_high: float | None = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    dcp_save_interval: int = 0
    weight_sync_timeout: int = 600
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="grpo-tinker"))
    cleanup_on_exit: bool = True

    init_from_checkpoint: str | None = None
    warm_start_from_adapter: str | None = None
    """Initialize LoRA weights from a PEFT adapter; exclusive with ``init_from_checkpoint``."""
    save_final_checkpoint: bool = True
    output_model_id: str | None = None


# ---------------------------------------------------------------------------
# Default task -- replace with your own rollout_fn_factory
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    digits = re.search(r"(-?\d+)", match.group(1))
    return digits.group(1) if digits else None


def reward_fn(completion: str, row: dict) -> float:
    """Return 1.0 if the model's numeric answer matches the ground truth."""
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


def single_turn_rollout_fn(renderer_name: str = "") -> RolloutFnFactory:
    """One completion of ``row["messages"]`` per call, graded by :func:`reward_fn`."""

    def factory(setup: RolloutSetup) -> RolloutFn:
        renderer = build_renderer(setup.tokenizer, setup.tokenizer_id, renderer_name)

        async def rollout_fn(row: dict):
            return await single_turn_renderer_rollout(
                row,
                renderer=renderer,
                sample_with_prompt_tokens=setup.sampler.sample_with_prompt_tokens,
                message_builder=lambda r: prepare_sampling_messages(r.get("messages", [])),
                reward_fn=lambda r, message, _parsed: reward_fn(get_text_content(message), r),
                sample_kwargs=setup.sample_kwargs,
                tokenizer=setup.tokenizer,
            )

        return rollout_fn

    return factory


@with_signals
async def main(
    config: Config,
    *,
    rollout_fn_factory: RolloutFnFactory | None = None,
    rows: list[dict] | None = None,
    dynamic_filter_fn: DynamicFilterFn | None = has_reward_variance,
    advantage_fn: Callable[[list[float]], list[float]] = compute_advantages,
    prepare_batch_fn: PrepareBatchFn | None = None,
    evaluation_fn: RolloutEvaluationFn | None = None,
    evaluation_interval: int = 1,
) -> dict[str, Any]:
    """Collect, optionally prepare, train and publish one complete on-policy batch."""
    cfg = config
    policy_loss = objective.GRPOConfig(eps_clip=cfg.eps_clip, eps_clip_high=cfg.eps_clip_high)
    if rollout_fn_factory is None:
        rollout_fn_factory = single_turn_rollout_fn(cfg.renderer_name)
    if cfg.trainer.reference_training_shape_id or cfg.trainer.reference_job_id:
        raise ValueError(
            "rl_loop trains without a reference model; unset "
            "trainer.reference_training_shape_id and trainer.reference_job_id."
        )
    if cfg.completions_per_prompt < 2:
        raise ValueError("completions_per_prompt must be >= 2 for GRPO.")
    if cfg.prompt_groups_per_step < 1:
        raise ValueError("prompt_groups_per_step must be >= 1.")
    if evaluation_interval < 1:
        raise ValueError("evaluation_interval must be >= 1.")
    if rows is None and not cfg.dataset:
        raise ValueError("Provide either cfg.dataset or rows= to main().")
    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required.")

    validate_config(
        cfg.base_model,
        cfg.dataset,
        deploy=cfg.deployment,
        output_model_id=cfg.output_model_id,
        require_dataset=(rows is None),
    )
    validate_warm_start_config(
        warm_start_from_adapter=cfg.warm_start_from_adapter,
        init_from_checkpoint=cfg.init_from_checkpoint,
        lora_rank=cfg.lora_rank,
    )
    lr_scheduler = normalize_lr_scheduler_spec(cfg.lr_scheduler)
    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": cfg.completions_per_prompt,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "max_completion_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "tokenizer_id": cfg.deployment.tokenizer_model,
            "shuffle": cfg.shuffle,
            "seed": cfg.seed,
            "algorithm": "grpo",
            "trainer_loss": f"server_{objective.LOSS_FN}",
            "eps_clip": policy_loss.eps_clip,
            "eps_clip_high": policy_loss.upper_eps,
            "lr": cfg.learning_rate,
            "lr_schedule": lr_scheduler.type,
        },
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()
    async with AsyncExitStack() as stack:
        tokenizer = await run_in_thread(load_deployment_tokenizer, cfg.deployment)
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
            cleanup_trainer_on_close=cfg.cleanup_on_exit,
            cleanup_deployment_on_close=(CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO if cfg.cleanup_on_exit else None),
        )
        stack.push_async_callback(close_in_thread, service.close)

        training_client = await run_in_thread(
            service.create_training_client, cfg.base_model, lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha
        )
        # Decide R3 after the trainer is bound: its create_model response is
        # the authoritative MoE signal, so no GET model is needed (private
        # early-access base models reject that with 403).
        router_replay_enabled = await run_in_thread(
            resolve_router_replay_enabled,
            requested=True,
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
            base_model=cfg.base_model,
            training_client=training_client,
        )
        policy = ReconnectableClient.from_training_client(
            training_client,
            base_model=cfg.base_model,
            lora_rank=cfg.lora_rank,
            job_id=service.trainer_job_id,
            service=service,
        )
        publish_weights = make_weight_sync(policy, service, cfg.deployment)

        checkpoint = TrainingCheckpoints(
            policy,
            service,
            trainer_id=service.trainer_job_id,
            log_path=cfg.log_path,
            lora_rank=cfg.lora_rank,
        )
        resume_info = await run_in_thread(
            checkpoint.resume,
            init_from_checkpoint=cfg.init_from_checkpoint,
            warm_start_from_adapter=cfg.warm_start_from_adapter,
        )
        step_offset = resume_info.step if resume_info else 0
        prior_rows_consumed = resume_info.data_consumed if resume_info else 0
        log_metrics({"train/step": step_offset}, step=step_offset)

        # Strict on-policy: initialize the sampler from the trainer, then repeat
        # this sync after every update.
        with elapsed_timer("weight_sync") as span:
            await run_in_thread(publish_weights, f"step-{step_offset}", checkpoint_type="base")
        logger.info("[step %d] initial weight sync (%.1fs)", step_offset, span.elapsed)
        flush_timing()
        log_run_resources(service)

        sample_kwargs = default_sample_kwargs(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=service.max_context_length,
            http_timeout=cfg.deployment.sample_timeout,
            router_replay=router_replay_enabled,
        )

        sampler = await run_in_thread(service.create_deployment_sampler, tokenizer=tokenizer)
        stack.push_async_callback(close_in_thread, sampler.close)
        # One rollout object serves training and evaluation, so stateful
        # rollouts (sandbox pools, judges) keep a single concurrency budget.
        rollout_fn = rollout_fn_factory(
            RolloutSetup(
                sampler=sampler,
                tokenizer=tokenizer,
                tokenizer_id=cfg.deployment.tokenizer_model,
                sample_kwargs=sample_kwargs,
                completions_per_prompt=cfg.completions_per_prompt,
                inference_base_url=sampler.base_url,
                api_key=api_key,
                model=sampler.model,
            )
        )
        stack.push_async_callback(close_rollout_fn, rollout_fn)
        context_names = rollout_context_names(rollout_fn)

        if rows is None:
            rows = await run_in_thread(load_jsonl_dataset, cfg.dataset, cfg.max_rows)
        else:
            rows = list(rows)
        row_loader = CursorDataLoader(
            rows,
            start_cursor=prior_rows_consumed,
            epochs=cfg.epochs,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
        )
        row_iterator = iter(row_loader)
        remaining_rows = max(0, row_loader.total_items - prior_rows_consumed)
        total_steps_estimate = step_offset + math.ceil(remaining_rows / cfg.prompt_groups_per_step)

        async def sample_group(row: dict, *, cursor_index: int) -> PromptGroup | None:
            runs = await asyncio.gather(
                *(
                    call_rollout_fn(
                        rollout_fn,
                        row,
                        {"cursor_index": cursor_index, "rollout_idx": i, "sample_index": i},
                        context_names,
                    )
                    for i in range(cfg.completions_per_prompt)
                )
            )
            # Failed trajectories are dropped; group advantages need two survivors.
            runs = [run for run in runs if run is not None]
            if len(runs) < 2:
                return None
            return rollout_to_prompt_group(Rollout(runs=runs), advantage_fn=advantage_fn)

        async def evaluate(step: int) -> None:
            if evaluation_fn is None:
                return
            # Synchronous: training requests were drained with the last step, so
            # everything drained here is evaluation traffic.
            metrics = await run_evaluation(step, evaluation_fn, rollout_fn)
            metrics.update(sampler_metrics(sampler.drain_metrics(), prefix="eval/sampler"))
            log_metrics(metrics, step=step)

        logger.info(
            "Synchronous GRPO: %d prompt groups x %d completions per step",
            cfg.prompt_groups_per_step,
            cfg.completions_per_prompt,
        )
        adam_kwargs = dict(DEFAULT_ADAM)
        adam_kwargs["grad_clip_norm"] = cfg.grad_clip_norm

        async def run_training() -> int:
            step = step_offset
            await evaluate(step)
            while True:
                step_started = time.monotonic()
                with phase_span("rollout_batch", category="rollout", attributes={"next_step": step + 1}):
                    prompt_groups, row_indices, loop_stats = await collect_prompt_groups(
                        row_iterator,
                        target_size=cfg.prompt_groups_per_step,
                        sample_prompt=sample_group,
                        should_accept=dynamic_filter_fn,
                    )
                if not row_indices:
                    break
                if not prompt_groups:
                    for index in row_indices:
                        row_loader.mark_resolved(index)
                    continue

                train_started = time.monotonic()

                # 1. One server-side GRPO forward/backward, clipped against the
                #    rollout logprobs (batch.anchor_logprobs).
                batch = combine_prompt_groups(prompt_groups)
                batch_metrics: dict[str, float] = {}
                if prepare_batch_fn is not None:
                    batch, batch_metrics = await run_in_thread(prepare_batch_fn, policy, batch)
                # Fork recipes (utils in training.utils.rl): recompute anchors before
                # build_datums; apply tis.apply_tis(datums, batch, cap=2.0) afterwards.
                # Client-side loss: policy.forward_backward_custom(
                #     batch.data, objective.make_loss_fn(batch, policy_loss)).
                datums = await run_in_thread(objective.build_datums, batch, policy_loss)
                with elapsed_timer("fwd_bwd"):
                    fwd_bwd_result = await run_in_thread(
                        policy.forward_backward,
                        datums,
                        objective.LOSS_FN,
                        loss_fn_config=objective.loss_fn_config(policy_loss),
                    )
                fwd_bwd_result.metrics.update(objective.metrics(batch, fwd_bwd_result, policy_loss))
                fwd_bwd_result.metrics.update(inference_drift_metrics(batch, fwd_bwd_result))

                # 2. Exactly one optimizer mutation.
                next_step = step + 1
                step_lr = compute_lr(
                    lr_scheduler,
                    step=next_step,
                    base_lr=cfg.learning_rate,
                    total_steps=total_steps_estimate,
                )
                with elapsed_timer("optim_step"):
                    optim_result = await run_in_thread(
                        policy.optim_step,
                        sdk.AdamParams(learning_rate=step_lr, **adam_kwargs),
                        grad_accumulation_normalization=objective.NORMALIZATION,
                        emit_grad_norm_metrics=True,
                    )
                step = next_step

                # 3. Publish this policy before the next rollout batch.
                with elapsed_timer("weight_sync"):
                    sync_result = await run_in_thread(publish_weights, f"step-{step}")

                for index in row_indices:
                    row_loader.mark_resolved(index)

                loop_stats["train_wall_time"] = time.monotonic() - train_started
                loop_stats["step_wall_time"] = time.monotonic() - step_started
                metrics = compute_step_metrics(
                    prompt_groups=prompt_groups,
                    fwd_bwd_results=[fwd_bwd_result],
                    optim_result=optim_result,
                    timing_metrics=flush_timing(),
                    raw_rewards=loop_stats["all_raw_rewards"],
                )
                metrics.update(sync_step_perf_metrics(loop_stats, prompt_groups))
                metrics.update(batch_metrics)
                metrics.update(
                    {
                        "train/forward_backward_requests": 1,
                        "train/groups_per_request_mean": len(prompt_groups),
                        "train/groups_per_request_max": len(prompt_groups),
                    }
                )
                metrics.update(sampler_metrics(sampler.drain_metrics()))
                metrics.update(weight_sync_metrics(sync_result))
                metrics["train/step"] = step
                metrics["train/lr"] = step_lr
                reward = metrics.get("rollout/filtered_reward", 0.0)
                logger.info(
                    "Step %d | reward %.3f | clip_frac %.3f | inference_k3 %.4f",
                    step,
                    reward,
                    metrics.get("train/ppo_clip_frac", 0.0),
                    metrics.get("train/inference_k3", 0.0),
                )
                log_metrics_json(step, reward=reward)
                log_metrics(metrics, step=step)

                if cfg.dcp_save_interval > 0 and (step - step_offset) % cfg.dcp_save_interval == 0:
                    with elapsed_timer("dcp_save") as span:
                        await run_in_thread(
                            checkpoint.save,
                            f"step-{step}",
                            resumable=True,
                            promotable=False,
                            data_consumed=row_loader.data_consumed,
                        )
                    logger.info("[step %d] checkpoint saved (%.1fs)", step, span.elapsed)
                if (step - step_offset) % evaluation_interval == 0:
                    await evaluate(step)
            if (step - step_offset) % evaluation_interval:
                await evaluate(step)
            return step

        global_step = await run_training()

        has_trained_steps = global_step > step_offset
        has_advanced_dataset = row_loader.data_consumed > prior_rows_consumed
        if cfg.save_final_checkpoint and (has_trained_steps or has_advanced_dataset):
            await run_in_thread(
                checkpoint.save,
                f"step-{global_step}",
                resumable=True,
                promotable=has_trained_steps,
                data_consumed=row_loader.data_consumed,
            )
            if cfg.output_model_id and has_trained_steps:
                await run_in_thread(checkpoint.promote_latest, cfg.output_model_id, cfg.base_model)

        logger.info(
            "Synchronous RL training complete: %d steps (%d new)",
            global_step,
            global_step - step_offset,
        )
        wandb_finish(metrics_file=os.environ.get("COOKBOOK_METRICS_FILE"))
        return {
            "steps": global_step,
            "policy_job_id": service.trainer_job_id,
            "deployment_id": service.deployment_id,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(
        Config(
            log_path="./rl_logs",
            dataset=("https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"),
            deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
        )
    )
