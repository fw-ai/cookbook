#!/usr/bin/env python3
"""Synchronous SAO: token-level actor-critic with a projection-head critic.

Experimental: projection-head SAO is not validated for every model, so this
recipe lives with the other experimental loops.

SAO (Single-rollout Asynchronous Optimization, arXiv 2607.07508) trains a
LoRA actor and an independently trained LoRA critic of the same base on two
Fireworks trainers. The critic trains only MLP adapters plus a dedicated
projection head that returns one value per token; it never uses the
language-model head. For every rollout batch the recipe performs:

1. ``critic_updates`` clipped value updates on lambda returns, every update
   re-reading the critic while anchors and targets stay at the first read;
2. a fresh read of the updated critic and token-level GAE with SAO's
   per-response ``lambda = 1 - 1 / (gae_alpha * T)``;
3. one decoupled-importance-sampling (DIS) actor update whose ratio is
   against the sampler's recorded logprobs, then one actor optimizer step;
4. a hotload of the new actor before the next rollout batch.

DIS has no trainer built-in and the critic trains a projection head, so both
losses run client side (``training.utils.rl.algorithm.sao`` and
``training.utils.rl.critic``). Fork this file to change the objective.

Dataset format and reward customization match ``recipes.rl_loop``. The
default example expects rows with ``messages`` and ``ground_truth``.

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.experiment.ppo_value_head_loop
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import signal
import time
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
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
    setup_wandb,
    validate_config,
    wandb_finish,
)
from training.utils.checkpoints import TrainingCheckpoints
from training.utils.dataloader import CursorDataLoader
from training.utils.metrics import weight_sync_metrics
from training.utils.rl.algorithm.sao import (
    adaptive_gae_lambdas,
    behavior_logprob_rows,
    make_dis_loss_fn,
    validate_dis_bounds,
)
from training.utils.rl.critic import (
    CRITIC_PROJECTION_HEAD_DIM,
    advantage_metrics,
    critic_batch,
    predict_values,
    token_gae,
    train_critic,
    value_statistics,
)
from training.utils.rl.losses import PromptGroup, combine_prompt_groups
from training.utils.rl.metrics import compute_step_metrics, sync_step_perf_metrics
from training.utils.rl.rollout import (
    Rollout,
    model_input_to_token_ids,
    rollout_to_prompt_group,
    sampled_completion_to_rollout_run,
)
from training.utils.rl.sync_batch import collect_prompt_groups
from training.utils.termination import TerminatedBySignal
from training.utils.timer import elapsed_timer, flush_timing

logger = logging.getLogger(__name__)

QWEN3_4B = "accounts/fireworks/models/qwen3-4b"
QWEN3_4B_TOKENIZER = "Qwen/Qwen3-4B"
QWEN3_4B_LORA_SHAPE = "accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora"

ADAM = {"beta1": 0.9, "beta2": 0.98, "eps": 1e-8, "weight_decay": 0.01}
"""Adam settings of SAO's math-reasoning setup, shared by actor and critic."""


@dataclass
class Config:
    log_path: str
    actor_base_model: str = QWEN3_4B
    critic_base_model: str = QWEN3_4B
    dataset: str | None = None

    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-6
    lr_scheduler: LRSchedulerSpec = field(default_factory=default_constant_schedule)
    """Actor learning-rate schedule; the critic learning rate is constant."""
    grad_clip_norm: float = 1.0

    gamma: float = 1.0
    critic_lambda: float = 1.0
    """GAE lambda of the critic's return targets."""
    gae_alpha: float = 1.5
    """Actor GAE uses ``lambda = 1 - 1 / (gae_alpha * T)`` per response of T tokens."""
    value_clip: float = 0.2
    critic_updates: int = 2
    """Critic optimizer steps per rollout batch."""
    dis_low: float = 0.3
    dis_high: float = 5.0
    """DIS keeps tokens with ``1 - dis_low < ratio < 1 + dis_high``."""

    completions_per_prompt: int = 1
    prompt_groups_per_batch: int = 8
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    shuffle: bool = True
    seed: int = 0
    max_rows: int = 100
    max_seq_len: int | None = None
    lora_rank: int = 32
    lora_alpha: int | None = 32
    critic_lora_rank: int = 32
    """The critic trains MLP LoRA adapters, so this must be positive."""
    critic_lora_alpha: int | None = 32
    renderer_name: str = ""

    actor_trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(
            training_shape_id=QWEN3_4B_LORA_SHAPE,
        )
    )
    # Any compatible validated shape can provision the critic; the trainer's
    # projection_head_dim setting creates its projection head. A reattached
    # trainer must already have a one-dimensional head.
    critic_trainer: TrainerConfig = field(default_factory=TrainerConfig)
    deployment: DeployConfig = field(
        default_factory=lambda: DeployConfig(
            tokenizer_model=QWEN3_4B_TOKENIZER,
        )
    )
    dcp_save_interval: int = 0
    weight_sync_timeout: int = 600
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="ppo-value-head-tinker"))
    cleanup_on_exit: bool = True

    actor_init_from_checkpoint: str | None = None
    critic_init_from_checkpoint: str | None = None
    save_final_checkpoint: bool = True
    output_model_id: str | None = None


def extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    digits = re.search(r"(-?\d+)", match.group(1))
    return digits.group(1) if digits else None


def reward_fn(completion: str, row: dict) -> float:
    """Replace this function with the terminal reward for your task."""
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


def _response_text_for_grading(renderer, sampled) -> str:
    message, _termination = renderer.parse_response(sampled.full_tokens[sampled.prompt_len :])
    return get_text_content(message)


def should_accept(prompt_group: PromptGroup) -> bool:
    """An actor-critic update does not need within-prompt reward variance."""
    return bool(prompt_group.data)


def validate_settings(cfg: Config) -> None:
    """Static checks before any resource is created."""
    for name in ("gamma", "critic_lambda"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1].")
    for name in (
        "actor_learning_rate",
        "critic_learning_rate",
        "grad_clip_norm",
        "gae_alpha",
        "value_clip",
    ):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    validate_dis_bounds(cfg.dis_low, cfg.dis_high)
    for name in ("critic_updates", "critic_lora_rank"):
        value = getattr(cfg, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if cfg.completions_per_prompt < 1 or cfg.prompt_groups_per_batch < 1:
        raise ValueError("Rollout batch sizes must be positive.")
    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required.")
    if not (cfg.critic_trainer.training_shape_id or cfg.critic_trainer.job_id):
        raise ValueError(
            "critic_trainer.training_shape_id must identify a compatible validated "
            "shape, or set critic_trainer.job_id to reattach a trainer created with "
            "a one-dimensional projection head."
        )


def resume_position(actor_resume: Any, critic_resume: Any) -> tuple[int, int]:
    """Shared ``(step, rows consumed)`` of the resumed actor and critic."""
    actor = (actor_resume.step, actor_resume.data_consumed) if actor_resume else (0, 0)
    critic = (critic_resume.step, critic_resume.data_consumed) if critic_resume else (0, 0)
    if actor != critic:
        raise ValueError(
            "Actor and critic checkpoints are at different (step, rows) positions: "
            f"actor={actor}, critic={critic}. Resume an aligned pair."
        )
    return actor


def main(
    config: Config,
    *,
    sample_prompt_fn: Callable[..., Awaitable[PromptGroup | None]] | None = None,
    rows: list[dict] | None = None,
    reward: Callable[[str, dict], float] | None = None,
) -> dict[str, Any]:
    """Run SAO with separate actor and critic trainers.

    ``reward`` scores a sampled response against its source row; the default
    expects an integer inside ``<answer>...</answer>``. ``sample_prompt_fn(row,
    *, cursor_index)`` replaces the built-in sampler; injected rollouts provision
    no deployment and skip weight sync, and must carry rollout logprobs.
    """
    cfg = config
    grade_response = reward if reward is not None else reward_fn
    validate_settings(cfg)
    if rows is None and not cfg.dataset:
        raise ValueError("Provide either cfg.dataset or rows= to main().")

    def _signal_handler(signum, _):
        name = signal.Signals(signum).name
        raise TerminatedBySignal(name)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(
        cfg.actor_base_model,
        cfg.dataset,
        deploy=cfg.deployment,
        output_model_id=cfg.output_model_id,
        require_dataset=(rows is None),
    )
    lr_scheduler = normalize_lr_scheduler_spec(cfg.lr_scheduler)
    setup_wandb(
        cfg.wandb,
        {
            "algorithm": "sao_ppo_projection_value_head",
            "actor_base_model": cfg.actor_base_model,
            "critic_base_model": cfg.critic_base_model,
            "actor_training_shape": cfg.actor_trainer.training_shape_id,
            "critic_training_shape": cfg.critic_trainer.training_shape_id,
            "actor_lr": cfg.actor_learning_rate,
            "critic_lr": cfg.critic_learning_rate,
            "gamma": cfg.gamma,
            "critic_lambda": cfg.critic_lambda,
            "gae_alpha": cfg.gae_alpha,
            "critic_updates": cfg.critic_updates,
            "dis_low": cfg.dis_low,
            "dis_high": cfg.dis_high,
            "critic_lora_rank": cfg.critic_lora_rank,
        },
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    with ExitStack() as stack:
        uses_managed_sampler = sample_prompt_fn is None
        actor_service = build_service_client(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
            base_model=cfg.actor_base_model,
            tokenizer_model=cfg.deployment.tokenizer_model,
            max_lora_rank=cfg.lora_rank,
            max_context_length=cfg.max_seq_len,
            learning_rate=cfg.actor_learning_rate,
            trainer=cfg.actor_trainer,
            deployment=cfg.deployment if uses_managed_sampler else None,
            hotload_timeout_s=cfg.weight_sync_timeout,
            cleanup_trainer_on_close=cfg.cleanup_on_exit,
            cleanup_deployment_on_close=(CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO if cfg.cleanup_on_exit else None),
            reference_required=False,
        )
        stack.callback(actor_service.close)
        critic_service = build_service_client(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
            base_model=cfg.critic_base_model,
            tokenizer_model=cfg.deployment.tokenizer_model,
            max_lora_rank=cfg.critic_lora_rank,
            projection_head_dim=CRITIC_PROJECTION_HEAD_DIM,
            max_context_length=cfg.max_seq_len,
            learning_rate=cfg.critic_learning_rate,
            trainer=cfg.critic_trainer,
            deployment=None,
            cleanup_trainer_on_close=cfg.cleanup_on_exit,
            reference_required=False,
        )
        stack.callback(critic_service.close)

        actor = ReconnectableClient.from_training_client(
            actor_service.create_training_client(
                cfg.actor_base_model,
                lora_rank=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
            ),
            base_model=cfg.actor_base_model,
            lora_rank=cfg.lora_rank,
            job_id=actor_service.trainer_job_id,
            service=actor_service,
        )
        critic = ReconnectableClient.from_training_client(
            critic_service.create_training_client(
                cfg.critic_base_model,
                lora_rank=cfg.critic_lora_rank,
                lora_alpha=cfg.critic_lora_alpha,
                # MLP adapters plus the projection head; never the LM head.
                train_attn=False,
                train_mlp=True,
                train_unembed=False,
            ),
            base_model=cfg.critic_base_model,
            lora_rank=cfg.critic_lora_rank,
            job_id=critic_service.trainer_job_id,
            service=critic_service,
        )

        sampler = renderer = publish_weights = None
        if uses_managed_sampler:
            tokenizer = load_deployment_tokenizer(cfg.deployment)
            sampler = actor_service.create_deployment_sampler(tokenizer=tokenizer)
            stack.callback(sampler.close)
            renderer = build_renderer(
                tokenizer,
                cfg.deployment.tokenizer_model,
                cfg.renderer_name,
            )
            publish_weights = make_weight_sync(actor, actor_service, cfg.deployment)

        actor_checkpoint = TrainingCheckpoints(
            actor,
            actor_service,
            trainer_id=actor_service.trainer_job_id,
            log_path=str(Path(cfg.log_path) / "actor"),
            lora_rank=cfg.lora_rank,
        )
        critic_checkpoint = TrainingCheckpoints(
            critic,
            critic_service,
            trainer_id=critic_service.trainer_job_id,
            log_path=str(Path(cfg.log_path) / "critic"),
            lora_rank=cfg.critic_lora_rank,
        )
        actor_resume = actor_checkpoint.resume(init_from_checkpoint=cfg.actor_init_from_checkpoint)
        critic_resume = critic_checkpoint.resume(init_from_checkpoint=cfg.critic_init_from_checkpoint)
        step_offset, prior_rows_consumed = resume_position(actor_resume, critic_resume)

        if publish_weights is not None:
            with elapsed_timer("weight_sync"):
                publish_weights(f"step-{step_offset}", checkpoint_type="base")
            flush_timing()

        if rows is None:
            rows = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
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
        total_steps = step_offset + math.ceil(remaining_rows / cfg.prompt_groups_per_batch)

        sample_kwargs: dict[str, Any] = {
            "max_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "top_p": 1.0,
            "top_k": 0,
            "max_seq_len": actor_service.max_context_length,
            "http_timeout": cfg.deployment.sample_timeout,
            "logprobs": True,
        }

        async def sample_one_prompt(
            row: dict,
            *,
            cursor_index: int,
        ) -> PromptGroup | None:
            if sample_prompt_fn is not None:
                return await sample_prompt_fn(row, cursor_index=cursor_index)

            messages = prepare_sampling_messages(row.get("messages", []))
            if not messages:
                return None
            prompt_token_ids = model_input_to_token_ids(renderer.build_generation_prompt(messages))
            try:
                sampled = await sampler.sample_with_prompt_tokens(
                    prompt_token_ids,
                    n=cfg.completions_per_prompt,
                    stop=renderer.get_stop_sequences(),
                    **sample_kwargs,
                )
            except Exception as error:
                logger.warning("Sampling row %d failed: %s", cursor_index, error)
                return None
            if not sampled or len(sampled) != cfg.completions_per_prompt:
                return None

            runs = []
            for sample in sampled:
                terminal_reward = grade_response(
                    _response_text_for_grading(renderer, sample),
                    row,
                )
                run = sampled_completion_to_rollout_run(sample, reward=terminal_reward)
                if run is None:
                    return None
                runs.append(run)
            # The critic supplies the advantages; keep the raw terminal rewards.
            return rollout_to_prompt_group(
                Rollout(runs=runs),
                advantage_fn=lambda rewards: list(rewards),
            )

        def adam_params(learning_rate: float) -> sdk.AdamParams:
            return sdk.AdamParams(
                learning_rate=learning_rate,
                grad_clip_norm=cfg.grad_clip_norm,
                **ADAM,
            )

        critic_adam = adam_params(cfg.critic_learning_rate)

        async def run_training() -> int:
            step = step_offset
            while True:
                with phase_span(
                    "rollout_batch",
                    category="rollout",
                    attributes={"next_step": step + 1},
                ):
                    (
                        prompt_groups,
                        row_indices,
                        loop_stats,
                    ) = await collect_prompt_groups(
                        row_iterator,
                        target_size=cfg.prompt_groups_per_batch,
                        sample_prompt=sample_one_prompt,
                        should_accept=should_accept,
                    )
                if not row_indices:
                    break
                if not prompt_groups:
                    for index in row_indices:
                        row_loader.mark_resolved(index)
                    continue

                train_started = time.monotonic()
                batch = combine_prompt_groups(prompt_groups)
                critic_inputs = critic_batch(batch)
                masks = critic_inputs.action_masks

                # 1. Critic: clipped value updates on targets frozen at the first read.
                critic_update = train_critic(
                    critic,
                    critic_inputs,
                    updates=cfg.critic_updates,
                    gamma=cfg.gamma,
                    critic_lambda=cfg.critic_lambda,
                    value_clip=cfg.value_clip,
                    adam_params=critic_adam,
                )

                # 2. Actor advantages from a fresh read of the updated critic.
                with elapsed_timer("critic_value_refresh"):
                    values = predict_values(critic, critic_inputs)
                lambdas = adaptive_gae_lambdas(masks, alpha=cfg.gae_alpha)
                advantages, _ = token_gae(
                    critic_inputs.terminal_rewards,
                    values,
                    masks,
                    gamma=cfg.gamma,
                    lam=lambdas,
                )

                # 3. Actor: one DIS update against the sampler's logprobs.
                loss_fn = make_dis_loss_fn(
                    behavior_logprobs=behavior_logprob_rows(batch.rollout_logprobs, masks),
                    advantages=advantages,
                    action_masks=masks,
                    dis_low=cfg.dis_low,
                    dis_high=cfg.dis_high,
                )
                with elapsed_timer("actor_fwd_bwd"):
                    actor_result = actor.forward_backward_custom(batch.data, loss_fn)
                step += 1
                actor_lr = compute_lr(
                    lr_scheduler,
                    step=step,
                    base_lr=cfg.actor_learning_rate,
                    total_steps=max(1, total_steps),
                )
                with elapsed_timer("actor_optim_step"):
                    actor_optim = actor.optim_step(adam_params(actor_lr), emit_grad_norm_metrics=True)

                # 4. Publish the actor before the next rollout batch.
                sync_metrics: dict[str, Any] = {}
                if publish_weights is not None:
                    with elapsed_timer("weight_sync"):
                        sync_metrics = weight_sync_metrics(publish_weights(f"step-{step}"))

                for index in row_indices:
                    row_loader.mark_resolved(index)

                loop_stats["train_wall_time"] = time.monotonic() - train_started
                metrics = compute_step_metrics(
                    prompt_groups=prompt_groups,
                    fwd_bwd_results=[*critic_update.fwd_bwd_results, actor_result],
                    optim_result=actor_optim,
                    timing_metrics=flush_timing(),
                    raw_rewards=loop_stats["all_raw_rewards"],
                )
                metrics.update(sync_step_perf_metrics(loop_stats, prompt_groups))
                metrics.update(sync_metrics)
                metrics.update(advantage_metrics(advantages, values, masks))
                for key, value in value_statistics(
                    values, critic_inputs.terminal_rewards, masks, gamma=cfg.gamma
                ).items():
                    metrics[f"train/critic_post/{key}"] = value
                for update, loss in enumerate(critic_update.value_losses, start=1):
                    metrics[f"train/critic-value_loss/epoch_{update}"] = loss
                metrics.update(
                    {
                        "train/step": step,
                        "train/actor_updates": step,
                        "train/critic_updates": step * cfg.critic_updates,
                        "train/actor_learning_rate": actor_lr,
                        "train/critic_learning_rate": cfg.critic_learning_rate,
                        "train/gae_lambda_mean": sum(lambdas) / len(lambdas),
                        "train/gae_lambda_min": min(lambdas),
                        "train/critic-value_loss": critic_update.value_losses[0],
                        "train/critic-sequences": len(critic_inputs.data),
                        "train/critic-token-positions": sum(m.numel() for m in masks),
                        "train/critic-active-tokens": sum(int(m.sum()) for m in masks),
                    }
                )
                reward = metrics.get("rollout/filtered_reward", 0.0)
                value_mean = metrics["train/old_value_mean"]
                logger.info(
                    "SAO step %d | reward %.3f | value %.3f",
                    step,
                    reward,
                    value_mean,
                )
                log_metrics_json(step, reward=reward, value=value_mean)
                log_metrics(metrics, step=step)

                if cfg.dcp_save_interval > 0 and (step - step_offset) % cfg.dcp_save_interval == 0:
                    for checkpoint in (actor_checkpoint, critic_checkpoint):
                        checkpoint.save(
                            f"step-{step}",
                            resumable=True,
                            promotable=False,
                            data_consumed=row_loader.data_consumed,
                        )

            return step

        step = asyncio.run(run_training())

        has_trained_steps = step > step_offset
        has_advanced_dataset = row_loader.data_consumed > prior_rows_consumed
        if cfg.save_final_checkpoint and (has_trained_steps or has_advanced_dataset):
            actor_checkpoint.save(
                f"step-{step}",
                resumable=True,
                promotable=has_trained_steps,
                data_consumed=row_loader.data_consumed,
            )
            critic_checkpoint.save(
                f"step-{step}",
                resumable=True,
                promotable=False,
                data_consumed=row_loader.data_consumed,
            )
            if cfg.output_model_id and has_trained_steps:
                actor_checkpoint.promote_latest(cfg.output_model_id, cfg.actor_base_model)

        wandb_finish(metrics_file=os.environ.get("COOKBOOK_METRICS_FILE"))
        return {
            "actor_steps": step,
            "critic_steps": step,
            "critic_updates": (step - step_offset) * cfg.critic_updates,
            "actor_job_id": actor_service.trainer_job_id,
            "critic_job_id": critic_service.trainer_job_id,
            "deployment_id": actor_service.deployment_id if uses_managed_sampler else None,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main(
        Config(
            log_path="./ppo_logs",
            dataset=(
                "https://raw.githubusercontent.com/eval-protocol/python-sdk/" "main/development/gsm8k_sample.jsonl"
            ),
        )
    )
