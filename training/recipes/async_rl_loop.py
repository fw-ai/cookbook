#!/usr/bin/env python3
"""Async RL recipe with per-run rollouts and recipe-owned training.

EXPERIMENTAL -- under active development.  API surface (``Config`` field
names, ``RolloutSetup`` shape, gate semantics) may change.  The recipe is intentionally minimal-surface: the
only thing most users need to write is the rollout function; everything
else (gate, advantage, optional reference KL, weight sync, TIS, pipeline chunking,
checkpoints) is handled by ``main()``.  See
``skills/fireworks-training/references/rl-async.md`` for the full contract.

Acknowledgements -- prior art referenced while designing this loop:

* AReaL  (https://github.com/inclusionAI/AReaL)
* slime  (https://github.com/THUDM/slime)
* Miles  (https://github.com/radixark/miles)

Users write ``rollout_fn(sample_prompt) -> RolloutRun | None`` -- one
trajectory per call.  ``sample_prompt`` is the dataset row's dict re-named
once it crosses the dataset/sampling seam.  The recipe fans each dataset
row out to ``completions_per_prompt`` parallel calls and assembles the
resulting runs into a PromptGroup inside the loop.

Rollout dependencies (tokenizer, sampler, request gate, etc.) flow
through :class:`RolloutSetup`.  The user supplies a
``rollout_fn_factory(setup) -> rollout_fn`` callable that closes over
the setup; this matches AReaL's workflow construction pattern.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import signal
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

import tinker
from fireworks.training.sdk.training_spec import (
    LRSchedulerSpec,
    compute_lr,
    default_constant_schedule,
    normalize_lr_scheduler_spec,
)

from training.utils.client import GradAccNormalization
from training.utils import (
    CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO,
    DEFAULT_ADAM,
    DeployConfig,
    TrainerConfig,
    WandBConfig,
    ReconnectableClient,
    build_service_client,
    log_metrics,
    load_deployment_tokenizer,
    load_jsonl_dataset,
    read_api_extra_headers_env,
    setup_wandb,
    validate_config,
    wandb_finish,
)
from training.utils.checkpoints import TrainingCheckpoints
from training.utils.dataloader import CursorDataLoader
from training.utils.logging import ASYNC_RL_WANDB_METRIC_STEPS
from training.utils.rl import PromptGroup
from training.utils.rl.async_rl import (
    AsyncRLCoordinator,
    AsyncRLTelemetry,
    PostStepMetricsFn,
    TrainingChunk,
    RolloutRow,
    run_async_rl_lifecycle,
)
from training.utils.rl.grpo import make_grpo_loss_fn, validate_grpo_config
from training.utils.rl.losses import combine_prompt_groups
from training.utils.rl.tis import TISConfig
from training.train_loop import DynamicFilterFn
from training.utils.rl.rollout import RolloutRun
from training.utils.timer import elapsed_timer, flush_timing

logger = logging.getLogger(__name__)

__all__ = [
    "Config",
    "RolloutFn",
    "RolloutFnFactory",
    "RolloutSetup",
    "main",
]


@dataclass
class Config:
    log_path: str
    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str | None = None
    """JSONL path/URL; optional when passing ``rows=`` to ``main()``."""

    learning_rate: float = 1e-5
    lr_scheduler: LRSchedulerSpec = field(default_factory=default_constant_schedule)
    """Per-step LR scheduler spec for managed and local async RL runs."""

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

    prompt_groups_per_step: int = 1
    max_head_offpolicy_versions: int = 0
    """Staleness budget in weight-sync versions; ``0`` is fully on-policy.
    See ``skills/fireworks-training/references/rl-async.md`` (gate semantics)."""
    max_concurrency_rollout_sample: int | None = None
    """In-flight LLM-call cap (same unit as ``deployment.max_batch_size``);
    must be ``>= completions_per_prompt`` or the gate stalls."""
    min_group_size: int = 1
    """Minimum surviving rollout runs per row to emit a PromptGroup."""
    grad_accumulation_normalization: GradAccNormalization | str | None = None
    """Optional server-side normalization for accumulated gradients.
    ``None`` leaves accumulated gradients unchanged."""

    grad_clip_norm: float = 0.0
    """Max gradient norm for clipping. 0 disables clipping."""

    eps_clip: float = 0.2
    """Lower/upper PPO clip epsilon used by the client-side GRPO update."""
    eps_clip_high: float | None = None
    """Optional asymmetric upper clip epsilon; defaults to ``eps_clip``."""
    pipeline_chunks_per_step: int = 1
    """Scheduler chunk cap per global optimizer batch.

    The loop sends ready prompt groups without waiting to fill a chunk; trainer
    continuous batching owns execution-level coalescing/microbatching.
    """
    tis: TISConfig = field(default_factory=TISConfig)
    """TIS (Train-Inference IS) weight correction config."""
    anchor_logp: Literal["old_policy", "rollout"] = "old_policy"
    """PPO anchor source.

    ``"old_policy"`` snapshots trainer logprobs and applies TIS against the
    rollout behavior policy. ``"rollout"`` skips that forward, anchors PPO on
    rollout logprobs, and makes the TIS ratio identity.
    """

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync_before_training: bool = False
    """Deprecated compatibility field; async RL always performs initial sync."""
    dcp_save_interval: int = 0
    weight_sync_timeout: int = 600
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="rl-async"))
    cleanup_on_exit: bool = True
    """Clean up SDK-created trainer/deployment resources on close."""

    init_from_checkpoint: str | None = None
    """Resume from prior checkpoint; bare name = this job, ``"job:name"``
    = cross-job."""
    save_final_checkpoint: bool = True
    """Save a resumable+promotable checkpoint at the end of training."""
    output_model_id: str | None = None
    """Promote the final checkpoint to this 4-segment model id on clean exit."""


@dataclass
class RolloutSetup:
    """Dependencies the recipe hands the rollout factory once at startup.

    Inference endpoint, tokenizer, sampling kwargs, plus an ``extras`` dict
    for caller state. See
    ``skills/fireworks-training/references/rl-async.md``.
    """

    tokenizer: Any
    tokenizer_id: str
    sample_kwargs: dict[str, Any]
    inference_base_url: str
    api_key: str
    model: str
    completions_per_prompt: int
    extras: dict[str, Any] = field(default_factory=dict)


RolloutFn = Callable[..., Awaitable[RolloutRun | None]]
RolloutFnFactory = Callable[[RolloutSetup], RolloutFn]


_ROLLOUT_CONTEXT_KWARGS = frozenset(
    {
        "cursor_index",
        "row_index",
        "epoch",
        "rollout_idx",
        "sample_index",
        "end_of_epoch",
    }
)


def _rollout_fn_accepts_any_context_kwargs(rollout_fn: RolloutFn) -> bool:
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in inspect.signature(rollout_fn).parameters.values()
    )


def _rollout_fn_context_param_names(rollout_fn: RolloutFn) -> frozenset[str]:
    if _rollout_fn_accepts_any_context_kwargs(rollout_fn):
        return _ROLLOUT_CONTEXT_KWARGS
    return _ROLLOUT_CONTEXT_KWARGS & inspect.signature(rollout_fn).parameters.keys()


def _save_checkpoint(
    ckpt: TrainingCheckpoints,
    *,
    name: str,
    data_consumed: int,
    resumable: bool = True,
    promotable: bool = False,
) -> None:
    logger.info("[%s] dcp_save...", name)
    with elapsed_timer("dcp_save") as span:
        ckpt.save(
            name,
            resumable=resumable,
            promotable=promotable,
            data_consumed=data_consumed,
        )
    logger.info("[%s] dcp_save: done (%.1fs)", name, span.elapsed)


def main(
    config: Config,
    *,
    rollout_fn_factory: RolloutFnFactory,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    rows: list[dict] | None = None,
    rollout_extras: dict[str, Any] | None = None,
) -> None:
    """Run the async RL loop with a user-supplied rollout factory.

    ``rollout_fn_factory(setup) -> rollout_fn`` is called once at startup
    with the assembled :class:`RolloutSetup`.  The returned
    ``rollout_fn(sample_prompt) -> RolloutRun | None`` is invoked
    ``completions_per_prompt`` times per dataset row (each invocation is
    one trajectory draw against the inference deployment).

    Remote trainer and sampler setup is owned by the SDK-managed Tinker path.
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
    logger.warning(
        "async_rl_loop is EXPERIMENTAL and under active development; "
        "the Config / RolloutSetup API may change. See "
        "skills/fireworks-training/references/rl-async.md.",
    )

    def _signal_handler(signum, _):
        name = signal.Signals(signum).name
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    if rows is None and not cfg.dataset:
        raise ValueError("Provide either cfg.dataset or rows= to main().")
    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required.")
    validate_config(
        cfg.base_model,
        cfg.dataset or None,
        deploy=cfg.deployment,
        output_model_id=cfg.output_model_id,
        require_dataset=(rows is None),
    )
    if cfg.completions_per_prompt < 2:
        raise ValueError(
            "async_rl_loop requires cfg.completions_per_prompt >= 2: the "
            "default GRPO-style advantage normalizer (z-score by "
            "torch.std(rewards)) is undefined on length-1 reward tensors "
            "and would drop every group, silently consuming the dataset "
            "without ever training.  Set completions_per_prompt >= 2 (the "
            f"default is 4); got {cfg.completions_per_prompt}."
        )
    if cfg.pipeline_chunks_per_step < 1:
        raise ValueError(
            "pipeline_chunks_per_step must be >= 1; got "
            f"{cfg.pipeline_chunks_per_step}."
        )
    lr_scheduler = normalize_lr_scheduler_spec(cfg.lr_scheduler)
    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": cfg.completions_per_prompt,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "max_head_offpolicy_versions": cfg.max_head_offpolicy_versions,
            "max_concurrency_rollout_sample": cfg.max_concurrency_rollout_sample,
            "pipeline_chunks_per_step": cfg.pipeline_chunks_per_step,
            "max_completion_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "tokenizer_id": cfg.deployment.tokenizer_model,
            "shuffle": cfg.shuffle,
            "seed": cfg.seed,
            "algorithm": "grpo",
            "trainer_loss": "client",
            "kl_beta": cfg.kl_beta,
            "anchor_logp": cfg.anchor_logp,
            "lr": cfg.learning_rate,
            "lr_schedule": lr_scheduler.type,
        },
        metric_steps=ASYNC_RL_WANDB_METRIC_STEPS,
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
        training_client = service.create_training_client(
            cfg.base_model, lora_rank=cfg.lora_rank
        )
        sampler = service.create_deployment_sampler(tokenizer=tokenizer)
        rollout_model = sampler.model
        training_profile = getattr(service, "training_profile", None)
        accelerator_type = getattr(service, "accelerator_type", None)
        if accelerator_type is None:
            accelerator_type = getattr(training_profile, "accelerator_type", None)
        accelerator_count = getattr(service, "accelerator_count", None)
        if accelerator_count is None:
            accelerator_count = getattr(training_profile, "accelerator_count", None)
        log_metrics({"rollout/step": 0}, step=0)

        policy = ReconnectableClient.from_training_client(
            training_client,
            base_model=cfg.base_model,
            lora_rank=cfg.lora_rank,
            job_id=service.trainer_job_id,
            service=service,
        )
        reference = None
        if cfg.kl_beta > 0:
            reference_training_client = service.create_reference_client(
                cfg.base_model,
                lora_rank=cfg.lora_rank,
            )
            reference = ReconnectableClient.from_training_client(
                reference_training_client,
                base_model=cfg.base_model,
                lora_rank=0,
                job_id=service.reference_client_job_id,
                service=service,
                base_only=True,
            )

        ckpt = TrainingCheckpoints(
            policy,
            service,
            trainer_id=service.trainer_job_id,
            log_path=cfg.log_path,
            lora_rank=cfg.lora_rank,
        )

        resume_info = ckpt.resume(
            init_from_checkpoint=cfg.init_from_checkpoint,
        )
        step_offset = resume_info.step if resume_info else 0
        if step_offset:
            logger.info("Resuming from step %d", step_offset)
            log_metrics(
                {"train/step": step_offset, "rollout/step": step_offset},
                step=step_offset,
            )

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

        prior_rows_consumed = resume_info.data_consumed if resume_info else 0
        row_loader = CursorDataLoader(
            rows,
            start_cursor=prior_rows_consumed,
            epochs=cfg.epochs,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
        )

        remaining_rows = max(0, row_loader.total_items - prior_rows_consumed)
        total_steps_estimate = step_offset + math.ceil(
            remaining_rows / max(1, cfg.prompt_groups_per_step)
        )

        logger.info("algorithm=grpo trainer_loss=client kl_beta=%g", cfg.kl_beta)

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            # Full-distribution on-policy sampling. Without explicit top_p/top_k
            # the serving stack applies the model's generation_config.json
            # defaults (e.g. Qwen3.5: top_k=20/top_p=0.95), which truncate
            # rollouts and bias the policy-gradient estimator.
            top_p=1.0,
            top_k=0,
            max_seq_len=service.max_context_length,
            http_timeout=cfg.deployment.sample_timeout,
            logprobs=True,
        )

        rollout_setup = RolloutSetup(
            tokenizer=tokenizer,
            tokenizer_id=cfg.deployment.tokenizer_model,
            sample_kwargs=sample_kwargs,
            inference_base_url="" if sampler is None else sampler.base_url,
            api_key=api_key,
            model=rollout_model,
            completions_per_prompt=cfg.completions_per_prompt,
            extras=dict(rollout_extras or {}),
        )
        rollout_fn = rollout_fn_factory(rollout_setup)
        rollout_context_param_names = _rollout_fn_context_param_names(rollout_fn)

        def make_row_requests():
            rows_per_epoch = len(rows)
            for item in row_loader:
                row = item.value
                idx = item.index
                epoch = idx // rows_per_epoch if rows_per_epoch else 0
                row_index = idx % rows_per_epoch if rows_per_epoch else idx
                end_of_epoch = (
                    row_index == rows_per_epoch - 1 if rows_per_epoch else True
                )
                source_row_id = row.get("id") if isinstance(row, dict) else None

                def run_one_rollout(
                    sub_index: int,
                    sample_prompt=row,
                    cursor_index=idx,
                    row_index=row_index,
                    epoch=epoch,
                    end_of_epoch=end_of_epoch,
                ):
                    context = {
                        "cursor_index": cursor_index,
                        "row_index": row_index,
                        "epoch": epoch,
                        "rollout_idx": sub_index,
                        "sample_index": sub_index,
                        "end_of_epoch": end_of_epoch,
                    }
                    if rollout_context_param_names:
                        return rollout_fn(
                            sample_prompt,
                            **{
                                key: context[key] for key in rollout_context_param_names
                            },
                        )
                    return rollout_fn(sample_prompt)

                yield RolloutRow(
                    row_id=idx,
                    run_factory=run_one_rollout,
                    row_meta={"row_id": source_row_id},
                    on_resolved=lambda _reason, idx=idx: row_loader.mark_resolved(idx),
                )

        def ref_forward(groups: list[PromptGroup]) -> None:
            if reference is None:
                return
            all_ref_data = [d for pg in groups for d in pg.ref_data]
            ref_fwd = reference.forward(all_ref_data, "cross_entropy")
            idx = 0
            for pg in groups:
                n = len(pg.ref_data)
                pg.ref_logprobs = [
                    ref_fwd.loss_fn_outputs[idx + i]["logprobs"].data for i in range(n)
                ]
                idx += n

        def fwd_bwd_batch(
            data,
            adv,
            ref_lp,
            prompt_lens,
            inf_lp,
            raw_inf_lp,
            old_policy_logprobs,
        ):
            """Run client-side GRPO with PPO clipping, TIS, and optional reference KL.

            To switch to built-in PPO or another loss, replace this call rather
            than adding dispatch. See
            ``skills/fireworks-training/references/rl-custom-loss.md``.
            """
            return policy.forward_backward_custom(
                data,
                make_grpo_loss_fn(
                    advantages=adv,
                    ref_logprobs=ref_lp,
                    prompt_len=prompt_lens,
                    inf_logprobs=inf_lp,
                    old_policy_logprobs=old_policy_logprobs,
                    kl_beta=cfg.kl_beta,
                    eps_clip=cfg.eps_clip,
                    eps_clip_high=cfg.eps_clip_high,
                    tis_config=cfg.tis,
                    raw_inf_logprobs=raw_inf_lp,
                ),
            )

        def train_chunk(chunk: TrainingChunk) -> dict[str, Any]:
            """Run the visible GRPO forward/backward phase for one chunk."""

            prompt_groups = list(chunk.groups)
            with elapsed_timer("ref_forward"):
                ref_forward(prompt_groups)

            data, adv, ref_lp, prompt_lens, inf_lp, raw_inf_lp = combine_prompt_groups(
                prompt_groups,
                include_raw=True,
            )
            if cfg.anchor_logp == "old_policy":
                with elapsed_timer("old_policy_forward"):
                    old_policy_fwd = policy.forward(data, "cross_entropy")
                    old_policy_logprobs = [
                        old_policy_fwd.loss_fn_outputs[i]["logprobs"].data
                        for i in range(len(data))
                    ]
            else:
                if len(inf_lp) != len(data):
                    raise ValueError(
                        "anchor_logp='rollout' requires one rollout_logprobs "
                        f"row per datum; got {len(inf_lp)} rows for {len(data)} datums."
                    )
                if any(not row for row in inf_lp):
                    raise ValueError(
                        "anchor_logp='rollout' requires non-empty rollout_logprobs."
                    )
                old_policy_logprobs = inf_lp

            with elapsed_timer("fwd_bwd"):
                fwd_bwd_result = fwd_bwd_batch(
                    data,
                    adv,
                    ref_lp,
                    prompt_lens,
                    inf_lp,
                    raw_inf_lp,
                    old_policy_logprobs,
                )
            return {
                "prompt_groups": prompt_groups,
                "fwd_bwd_result": fwd_bwd_result,
            }

        def optimizer_step(step: int) -> dict[str, Any]:
            """Apply exactly one optimizer mutation for one rollout batch."""

            step_lr = compute_lr(
                lr_scheduler,
                step=step,
                base_lr=cfg.learning_rate,
                total_steps=total_steps_estimate,
            )
            adam_kwargs = dict(DEFAULT_ADAM)
            adam_kwargs["grad_clip_norm"] = cfg.grad_clip_norm
            adam_params = tinker.AdamParams(learning_rate=step_lr, **adam_kwargs)
            with elapsed_timer("optim_step"):
                result = policy.optim_step(
                    adam_params,
                    grad_accumulation_normalization=cfg.grad_accumulation_normalization,
                )
            return {
                "result": result,
                "learning_rate": step_lr,
            }

        def sync_weights(step: int) -> float:
            with elapsed_timer("weight_sync") as sync_span:
                saved = policy.save_weights_for_sampler(f"step-{step}")
                service.hotload_sampler_snapshot(saved.path)
            return sync_span.elapsed

        async def run_training(
            post_step_metrics_fn: PostStepMetricsFn | None,
        ) -> tuple[int, dict[str, Any]]:
            telemetry = AsyncRLTelemetry(
                producer_metrics_fn=lambda metrics: log_metrics(
                    metrics,
                    step=int(metrics["producer/event"]),
                ),
                step_metrics_fn=lambda metrics, step: log_metrics(
                    metrics,
                    step=step,
                ),
                post_step_metrics_fn=post_step_metrics_fn,
            )
            coordinator = AsyncRLCoordinator(
                rows=make_row_requests(),
                completions_per_prompt=cfg.completions_per_prompt,
                prompt_groups_per_step=cfg.prompt_groups_per_step,
                training_chunks_per_step=cfg.pipeline_chunks_per_step,
                max_head_off_policy_versions=cfg.max_head_offpolicy_versions,
                max_concurrent_rollouts=cfg.max_concurrency_rollout_sample,
                with_reference=(reference is not None),
                min_group_size=cfg.min_group_size,
                dynamic_filter_fn=dynamic_filter_fn,
                global_step=step_offset,
                resolved_rows_offset=prior_rows_consumed,
                resolved_rows_fn=lambda: row_loader.data_consumed,
            )
            async with coordinator:
                telemetry.start(coordinator.snapshot)
                try:
                    while (batch := await coordinator.next_batch()) is not None:
                        chunk_outputs: list[dict[str, Any]] = []

                        async for chunk in batch.chunks():
                            coordinator.raise_if_failed(batch)
                            output = await coordinator.run_blocking(
                                "train_chunk",
                                train_chunk,
                                chunk,
                                optimizer_batch=batch,
                            )
                            coordinator.raise_if_failed(batch)
                            chunk_outputs.append(output)

                        coordinator.raise_if_failed(batch)
                        optimizer = await coordinator.run_blocking(
                            "optimizer",
                            optimizer_step,
                            batch.batch_id,
                            optimizer_batch=batch,
                        )

                        # A producer failure here can only affect a future batch.
                        # Finish this optimizer's hotload and publication so trainer
                        # and sampler versions cannot diverge on shutdown.
                        sync_wall_time = await coordinator.run_blocking(
                            "weight_sync",
                            sync_weights,
                            batch.batch_id,
                            optimizer_batch=batch,
                        )
                        published = coordinator.publish(batch)

                        telemetry.finish_step(
                            batch=batch,
                            trained_against_version=(published.trained_against_version),
                            prompt_groups=[
                                group
                                for output in chunk_outputs
                                for group in output["prompt_groups"]
                            ],
                            fwd_bwd_results=[
                                output["fwd_bwd_result"] for output in chunk_outputs
                            ],
                            optim_result=optimizer["result"],
                            timing_metrics=flush_timing(),
                            weight_sync_time=sync_wall_time,
                            learning_rate=optimizer["learning_rate"],
                        )

                        rollouts_completed = batch.batch_id - step_offset
                        interval = cfg.dcp_save_interval
                        if (
                            interval > 0
                            and rollouts_completed > 0
                            and rollouts_completed % interval == 0
                        ):
                            try:
                                await coordinator.run_blocking(
                                    "checkpoint",
                                    _save_checkpoint,
                                    ckpt,
                                    name=f"step-{batch.batch_id}",
                                    data_consumed=published.resolved_rows,
                                )
                            except (OSError, RuntimeError) as error:
                                logger.warning(
                                    "[step %d] dcp_save failed: %s",
                                    batch.batch_id,
                                    error,
                                )
                finally:
                    await telemetry.aclose()

                return coordinator.global_step, telemetry.final_stats()

        global_step, final_stats = asyncio.run(run_async_rl_lifecycle(run_training))
        # Save resume progress even if all remaining rows were dropped.
        # Promotion still requires at least one optimizer step.
        resume_row_cursor = int(final_stats["resolved_rows"])
        has_trained_steps = global_step > step_offset
        has_advanced_dataset = resume_row_cursor > prior_rows_consumed
        if cfg.save_final_checkpoint and (has_trained_steps or has_advanced_dataset):
            cp_name = f"step-{global_step}"
            ckpt.save(
                cp_name,
                resumable=True,
                promotable=has_trained_steps,
                data_consumed=resume_row_cursor,
            )
            if cfg.output_model_id and has_trained_steps:
                ckpt.promote_latest(cfg.output_model_id, cfg.base_model)

        logger.info(
            "Async RL training complete: %d steps (%d new in this run)",
            global_step,
            global_step - step_offset,
        )
        wandb_finish(metrics_file=os.environ.get("COOKBOOK_METRICS_FILE"))
        return {
            "steps": global_step,
            "policy_job_id": service.trainer_job_id,
            "reference_job_id": service.reference_trainer_job_id,
            "deployment_id": service.deployment_id,
            "accelerator_type": accelerator_type,
            "accelerator_count": accelerator_count,
        }
