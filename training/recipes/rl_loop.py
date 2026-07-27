#!/usr/bin/env python3
"""A small, synchronous GRPO training recipe.

Each iteration has four visible phases:

1. collect one rollout batch,
2. compute reference and old-policy logprobs,
3. run one GRPO forward/backward and one optimizer step,
4. hotload the new policy before collecting the next batch.

Fork this file to customize the reward, rollout, or loss. For independent
rollout and training workers with bounded off-policy sampling, use
``recipes.async_rl_loop`` instead.

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
import signal
import time
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Literal

import tinker
from fireworks.training.sdk.training_spec import (
    LRSchedulerSpec,
    compute_lr,
    default_constant_schedule,
    normalize_lr_scheduler_spec,
)
from tinker_cookbook.renderers import get_text_content

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
    prepare_sampling_messages,
    read_api_extra_headers_env,
    resolve_router_replay_enabled,
    setup_wandb,
    validate_config,
    wandb_finish,
)
from training.utils.checkpoints import TrainingCheckpoints
from training.utils.client import GradAccNormalization
from training.utils.dataloader import CursorDataLoader
from training.utils.rl import PromptGroup
from training.utils.rl.grpo import make_grpo_loss_fn, validate_grpo_config
from training.utils.rl.losses import combine_prompt_groups
from training.utils.rl.metrics import compute_step_metrics
from training.utils.rl.router_replay import warn_if_full_sequence_router_replay
from training.utils.rl.rollout import (
    Rollout,
    model_input_to_token_ids,
    rollout_to_prompt_group,
    sampled_completion_to_rollout_run,
)
from training.utils.rl.sync_batch import collect_prompt_groups
from training.utils.rl.tis import TISConfig
from training.utils.timer import elapsed_timer, flush_timing

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration shared with the synchronous parts of ``async_rl_loop``."""

    log_path: str
    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str | None = None
    """JSONL path/URL; optional when passing ``rows=`` to ``main()``."""

    learning_rate: float = 1e-5
    lr_scheduler: LRSchedulerSpec = field(default_factory=default_constant_schedule)
    kl_beta: float = 0.001
    """Reference-KL coefficient. Set to ``0`` to skip reference provisioning."""

    completions_per_prompt: int = 4
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    shuffle: bool = True
    seed: int = 0
    max_rows: int = 100
    max_seq_len: int | None = None
    lora_rank: int = 0
    renderer_name: str = ""
    """Cookbook renderer used to build rollout prompts and grade responses.

    Empty = infer from ``deployment.tokenizer_model`` (see
    :func:`training.utils.supervised.resolve_renderer_name`). Set this when the
    inferred default is not the format you want to roll out in --
    ``async_rl_loop`` has no equivalent because there the ``rollout_fn`` owns
    renderer construction, while this recipe owns the rollout itself."""

    prompt_groups_per_step: int = 1
    """Valid prompt groups collected for each optimizer step."""

    router_replay: bool = True
    router_replay_completion_only: bool = True
    """Replay serving expert routes for MoE alignment.

    Completion-only replay avoids the serving cost of ``echo=True`` while
    aligning the generated tokens used by the policy loss and KLD metrics.
    """

    grad_accumulation_normalization: GradAccNormalization | str | None = None
    grad_clip_norm: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    tis: TISConfig = field(default_factory=TISConfig)
    anchor_logp: Literal["old_policy", "rollout"] = "old_policy"
    """PPO anchor source; matches ``async_rl_loop.Config.anchor_logp``."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    dcp_save_interval: int = 0
    weight_sync_timeout: int = 600
    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="grpo-tinker")
    )
    cleanup_on_exit: bool = True

    init_from_checkpoint: str | None = None
    save_final_checkpoint: bool = True
    output_model_id: str | None = None


# ---------------------------------------------------------------------------
# Reward and filter -- customize these for your task
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


def _response_text_for_grading(renderer, sampled) -> str:
    """Parse the generated tokens and return the assistant response text."""
    message, _termination = renderer.parse_response(
        sampled.full_tokens[sampled.prompt_len :]
    )
    return get_text_content(message)


def should_accept(prompt_group: PromptGroup) -> bool:
    """Keep groups with non-zero reward variance."""
    return len(set(prompt_group.rewards)) > 1


def main(
    config: Config,
    *,
    sample_prompt_fn: Callable[..., Awaitable[PromptGroup | None]] | None = None,
    rows: list[dict] | None = None,
) -> dict[str, Any]:
    """Run strict on-policy GRPO.

    ``sample_prompt_fn(row, *, cursor_index)`` is the optional rollout
    customization boundary. It returns one trainer-ready ``PromptGroup`` or
    ``None`` for a recoverable row-level drop.
    """
    cfg = config
    validate_grpo_config(
        kl_beta=cfg.kl_beta,
        eps_clip=cfg.eps_clip,
        eps_clip_high=cfg.eps_clip_high,
        reference_training_shape_id=cfg.trainer.reference_training_shape_id,
        reference_job_id=cfg.trainer.reference_job_id,
        anchor_logp=cfg.anchor_logp,
    )
    if cfg.completions_per_prompt < 2:
        raise ValueError("completions_per_prompt must be >= 2 for GRPO.")
    if cfg.prompt_groups_per_step < 1:
        raise ValueError("prompt_groups_per_step must be >= 1.")
    if rows is None and not cfg.dataset:
        raise ValueError("Provide either cfg.dataset or rows= to main().")
    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required.")

    def _signal_handler(signum, _):
        name = signal.Signals(signum).name
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(
        cfg.base_model,
        cfg.dataset,
        deploy=cfg.deployment,
        output_model_id=cfg.output_model_id,
        require_dataset=(rows is None),
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
            "renderer_name": cfg.renderer_name,
            "shuffle": cfg.shuffle,
            "seed": cfg.seed,
            "algorithm": "grpo",
            "trainer_loss": "client",
            "kl_beta": cfg.kl_beta,
            "anchor_logp": cfg.anchor_logp,
            "lr": cfg.learning_rate,
            "lr_schedule": lr_scheduler.type,
        },
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()
    router_replay_enabled = sample_prompt_fn is None and resolve_router_replay_enabled(
        requested=cfg.router_replay,
        api_key=api_key,
        base_url=base_url,
        additional_headers=additional_headers,
        base_model=cfg.base_model,
    )
    if cfg.router_replay and sample_prompt_fn is None and not router_replay_enabled:
        logger.info("Router Replay skipped for dense model %s", cfg.base_model)
    if router_replay_enabled:
        warn_if_full_sequence_router_replay(cfg.router_replay_completion_only)

    with ExitStack() as stack:
        tokenizer = load_deployment_tokenizer(cfg.deployment)
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
            cleanup_trainer_on_close=cfg.cleanup_on_exit,
            cleanup_deployment_on_close=(
                CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO
                if cfg.cleanup_on_exit
                else None
            ),
            reference_required=cfg.kl_beta > 0,
        )
        stack.callback(service.close)

        policy = ReconnectableClient.from_training_client(
            service.create_training_client(
                cfg.base_model,
                lora_rank=cfg.lora_rank,
            ),
            base_model=cfg.base_model,
            lora_rank=cfg.lora_rank,
            job_id=service.trainer_job_id,
            service=service,
        )
        reference = None
        if cfg.kl_beta > 0:
            reference = ReconnectableClient.from_training_client(
                service.create_reference_client(
                    cfg.base_model,
                    lora_rank=cfg.lora_rank,
                ),
                base_model=cfg.base_model,
                lora_rank=0,
                job_id=service.reference_client_job_id,
                service=service,
                base_only=True,
            )

        sampler = None
        response_renderer = None
        if sample_prompt_fn is None:
            sampler = service.create_deployment_sampler(tokenizer=tokenizer)
            response_renderer = build_renderer(
                tokenizer,
                cfg.deployment.tokenizer_model,
                cfg.renderer_name,
            )

        checkpoint = TrainingCheckpoints(
            policy,
            service,
            trainer_id=service.trainer_job_id,
            log_path=cfg.log_path,
            lora_rank=cfg.lora_rank,
        )
        resume_info = checkpoint.resume(
            init_from_checkpoint=cfg.init_from_checkpoint,
        )
        step_offset = resume_info.step if resume_info else 0
        prior_rows_consumed = resume_info.data_consumed if resume_info else 0
        log_metrics({"train/step": step_offset}, step=step_offset)

        # The synchronous recipe is always strict on-policy: initialize the
        # sampler from the trainer, then repeat this sync after every update.
        with elapsed_timer("weight_sync") as span:
            saved = policy.save_weights_for_sampler(
                f"step-{step_offset}",
                checkpoint_type="base",
            )
            service.hotload_sampler_snapshot(saved.path)
        logger.info("[step %d] initial weight sync (%.1fs)", step_offset, span.elapsed)
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
        total_steps_estimate = step_offset + math.ceil(
            remaining_rows / cfg.prompt_groups_per_step
        )

        sample_kwargs: dict[str, Any] = {
            "max_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "top_p": 1.0,
            "top_k": 0,
            "max_seq_len": service.max_context_length,
            "http_timeout": cfg.deployment.sample_timeout,
            "logprobs": True,
        }
        if router_replay_enabled:
            sample_kwargs.update(
                include_routing_matrix=True,
                echo=not cfg.router_replay_completion_only,
            )

        # -- Rollout function (VISIBLE -- customize this) ---------------------

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
            model_input = response_renderer.build_generation_prompt(messages)
            prompt_token_ids = model_input_to_token_ids(model_input)
            try:
                sampled = await sampler.sample_with_prompt_tokens(
                    prompt_token_ids,
                    n=cfg.completions_per_prompt,
                    stop=response_renderer.get_stop_sequences(),
                    **sample_kwargs,
                )
            except Exception as error:
                logger.warning("Sampling row %d failed: %s", cursor_index, error)
                return None
            if not sampled or len(sampled) != cfg.completions_per_prompt:
                return None

            rewards = [
                reward_fn(
                    _response_text_for_grading(response_renderer, sample),
                    row,
                )
                for sample in sampled
            ]
            runs = []
            for sample, reward in zip(sampled, rewards, strict=True):
                run = sampled_completion_to_rollout_run(sample, reward=reward)
                if run is None:
                    return None
                runs.append(run)
            return rollout_to_prompt_group(
                Rollout(runs=runs),
                with_reference=(reference is not None),
                router_replay_completion_only=cfg.router_replay_completion_only,
            )

        logger.info(
            "Synchronous GRPO: %d prompt groups x %d completions per step",
            cfg.prompt_groups_per_step,
            cfg.completions_per_prompt,
        )

        # -- Synchronous training loop (VISIBLE algorithm phases) -------------

        adam_kwargs = dict(DEFAULT_ADAM)
        adam_kwargs["grad_clip_norm"] = cfg.grad_clip_norm

        async def run_training() -> int:
            step = step_offset
            while True:
                prompt_groups, row_indices, loop_stats = await collect_prompt_groups(
                    row_iterator,
                    target_size=cfg.prompt_groups_per_step,
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

                # 1. Reference and old-policy logprobs.
                if reference is not None:
                    with elapsed_timer("ref_forward"):
                        reference_data = [
                            datum
                            for group in prompt_groups
                            for datum in group.ref_data
                        ]
                        reference_result = reference.forward(
                            reference_data,
                            "cross_entropy",
                        )
                        offset = 0
                        for group in prompt_groups:
                            group_size = len(group.ref_data)
                            group.ref_logprobs = [
                                reference_result.loss_fn_outputs[offset + i][
                                    "logprobs"
                                ].data
                                for i in range(group_size)
                            ]
                            offset += group_size

                (
                    data,
                    advantages,
                    reference_logprobs,
                    prompt_lengths,
                    rollout_logprobs,
                    raw_inference_logprobs,
                ) = combine_prompt_groups(prompt_groups, include_raw=True)

                if cfg.anchor_logp == "old_policy":
                    with elapsed_timer("old_policy_forward"):
                        old_policy_result = policy.forward(data, "cross_entropy")
                        old_policy_logprobs = [
                            output["logprobs"].data
                            for output in old_policy_result.loss_fn_outputs
                        ]
                else:
                    if len(rollout_logprobs) != len(data) or any(
                        not row for row in rollout_logprobs
                    ):
                        raise ValueError(
                            "anchor_logp='rollout' requires one non-empty "
                            "rollout_logprobs row per training datum."
                        )
                    old_policy_logprobs = rollout_logprobs

                # 2. One GRPO forward/backward.
                # To switch to built-in PPO or another loss, replace this
                # direct call. See
                # skills/fireworks-training/references/rl-custom-loss.md.
                with elapsed_timer("fwd_bwd"):
                    fwd_bwd_result = policy.forward_backward_custom(
                        data,
                        make_grpo_loss_fn(
                            advantages=advantages,
                            ref_logprobs=reference_logprobs,
                            prompt_len=prompt_lengths,
                            inf_logprobs=rollout_logprobs,
                            old_policy_logprobs=old_policy_logprobs,
                            kl_beta=cfg.kl_beta,
                            eps_clip=cfg.eps_clip,
                            eps_clip_high=cfg.eps_clip_high,
                            tis_config=cfg.tis,
                            raw_inf_logprobs=raw_inference_logprobs,
                        ),
                    )

                # 3. Exactly one optimizer mutation.
                next_step = step + 1
                step_lr = compute_lr(
                    lr_scheduler,
                    step=next_step,
                    base_lr=cfg.learning_rate,
                    total_steps=total_steps_estimate,
                )
                with elapsed_timer("optim_step"):
                    optim_result = policy.optim_step(
                        tinker.AdamParams(
                            learning_rate=step_lr,
                            **adam_kwargs,
                        ),
                        grad_accumulation_normalization=(
                            cfg.grad_accumulation_normalization
                        ),
                    )
                step = next_step

                # 4. Publish this policy before the next rollout batch.
                with elapsed_timer("weight_sync"):
                    saved = policy.save_weights_for_sampler(f"step-{step}")
                    service.hotload_sampler_snapshot(saved.path)

                for index in row_indices:
                    row_loader.mark_resolved(index)

                loop_stats["train_wall_time"] = time.monotonic() - train_started
                loop_stats["scheduler_step_wall_time"] = (
                    loop_stats["rollout_batch_wall_time"]
                    + loop_stats["train_wall_time"]
                )
                metrics = compute_step_metrics(
                    prompt_groups=prompt_groups,
                    fwd_bwd_results=[fwd_bwd_result],
                    optim_result=optim_result,
                    n_accum=1,
                    timing_metrics=flush_timing(),
                    loop_stats=loop_stats,
                )
                metrics["train/step"] = step
                metrics["train/learning_rate"] = step_lr
                reward = metrics.get("rollout/filtered_reward", 0.0)
                ref_kl = metrics.get("train/ref_kl", 0.0)
                logger.info(
                    "Step %d | reward %.3f | RefKL %.4f",
                    step,
                    reward,
                    ref_kl,
                )
                log_metrics_json(step, reward=reward, ref_kl=ref_kl)
                log_metrics(metrics, step=step)

                if (
                    cfg.dcp_save_interval > 0
                    and (step - step_offset) % cfg.dcp_save_interval == 0
                ):
                    with elapsed_timer("dcp_save") as span:
                        checkpoint.save(
                            f"step-{step}",
                            resumable=True,
                            promotable=False,
                            data_consumed=row_loader.data_consumed,
                        )
                    logger.info(
                        "[step %d] checkpoint saved (%.1fs)",
                        step,
                        span.elapsed,
                    )

            return step

        global_step = asyncio.run(run_training())

        has_trained_steps = global_step > step_offset
        has_advanced_dataset = row_loader.data_consumed > prior_rows_consumed
        if cfg.save_final_checkpoint and (
            has_trained_steps or has_advanced_dataset
        ):
            checkpoint.save(
                f"step-{global_step}",
                resumable=True,
                promotable=has_trained_steps,
                data_consumed=row_loader.data_consumed,
            )
            if cfg.output_model_id and has_trained_steps:
                checkpoint.promote_latest(cfg.output_model_id, cfg.base_model)

        logger.info(
            "Synchronous RL training complete: %d steps (%d new)",
            global_step,
            global_step - step_offset,
        )
        wandb_finish(metrics_file=os.environ.get("COOKBOOK_METRICS_FILE"))

        return {
            "steps": global_step,
            "policy_job_id": service.trainer_job_id,
            "reference_job_id": service.reference_trainer_job_id,
            "deployment_id": service.deployment_id,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main(
        Config(
            log_path="./rl_logs",
            dataset=(
                "https://raw.githubusercontent.com/eval-protocol/python-sdk/"
                "main/development/gsm8k_sample.jsonl"
            ),
            deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
        )
    )
