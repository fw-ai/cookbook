#!/usr/bin/env python3
"""Async RL recipe with per-run rollouts and server-side GRPO.

EXPERIMENTAL -- under active development.  API surface (``Config`` field
names, ``RolloutSetup`` shape, gate semantics) may change.  The recipe is
intentionally minimal-surface: the only thing most users need to write is the
rollout function; everything else (gate, advantage, weight sync, pipeline
chunking, checkpoints) is handled by ``main()``.  Each optimizer batch trains
with the trainer's built-in GRPO loss, clipped against the rollout logprobs.
To change the objective, fork ``train_chunk``; the modules in
``training.utils.rl.algorithm`` share one interface.  See
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

Rollout dependencies (tokenizer, sampler, sampling kwargs) flow through
:class:`RolloutSetup`.  The user supplies a ``rollout_fn_factory(setup) ->
rollout_fn`` callable that closes over the setup and its own typed task
config; this matches AReaL's workflow construction pattern.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

import tinker
from fireworks.training.sdk.training_spec import (
    LRSchedulerSpec,
    compute_lr,
    default_constant_schedule,
    normalize_lr_scheduler_spec,
)

from training.utils.termination import TerminatedBySignal
from training.utils import (
    CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO,
    DEFAULT_ADAM,
    DeployConfig,
    TrainerConfig,
    WandBConfig,
    ReconnectableClient,
    build_service_client,
    make_weight_sync,
    flush_phase_trace,
    log_metrics,
    load_deployment_tokenizer,
    load_jsonl_dataset,
    read_api_extra_headers_env,
    resolve_router_replay_enabled,
    setup_wandb,
    validate_config,
    wandb_finish,
)
from training.utils.checkpoints import TrainingCheckpoints, validate_warm_start_config
from training.utils.dataloader import CursorDataLoader
from training.utils.data import compute_advantages
from training.utils.logging import ASYNC_RL_WANDB_METRIC_STEPS, log_run_resources
from training.utils.metrics import sampler_metrics, weight_sync_metrics
from training.utils.rl.async_rl import (
    AsyncRLCoordinator,
    AsyncRLTelemetry,
    OverlappedEvaluation,
    TrainingChunk,
    RolloutRow,
)
from training.utils.rl.algorithm import grpo as objective
from training.utils.rl.common import inference_drift_metrics
from training.utils.rl.heldout import RolloutEvaluationFn, run_evaluation as evaluate_rollouts
from training.utils.rl.losses import combine_prompt_groups
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
from training.utils.timer import elapsed_timer, flush_timing, wall_timer

logger = logging.getLogger(__name__)

__all__ = [
    "Config",
    "RolloutFn",
    "RolloutEvaluationFn",
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
    """Per-step LR scheduler spec."""

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
    """LoRA alpha scaling factor. Ignored when ``lora_rank == 0``.

    Defaults to ``32`` to match Tinker and the Training API SDK client
    ``DEFAULT_LORA_ALPHA``. Override when you need a different scaling factor."""

    prompt_groups_per_step: int = 1
    max_head_offpolicy_versions: int = 0
    """Staleness budget in weight-sync versions; ``0`` is fully on-policy.
    See ``skills/fireworks-training/references/rl-async.md`` (gate semantics)."""
    max_concurrency_rollout_sample: int | None = None
    """In-flight LLM-call cap (same unit as ``deployment.max_batch_size``);
    must be ``>= completions_per_prompt`` or the gate stalls."""
    min_group_size: int = 1
    """Minimum surviving rollout runs per row to emit a PromptGroup."""
    max_incomplete_group_retries: int = 0
    """Rebuild a row this many times when fewer than ``min_group_size`` runs land."""
    pipeline_chunks_per_step: int = 1
    """Scheduler chunk cap per global optimizer batch.

    The scheduler creates balanced chunk targets and exposes the batch once its
    first target is full. Later chunks can fill while the trainer is active.
    """

    grad_clip_norm: float = 0.0
    """Max gradient norm for clipping. 0 disables clipping."""
    eps_clip: float = 0.2
    """GRPO lower clip epsilon; also the upper one unless ``eps_clip_high`` is set."""
    eps_clip_high: float | None = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    dcp_save_interval: int = 0
    weight_sync_timeout: int = 600
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="rl-async"))
    cleanup_on_exit: bool = True
    """Clean up SDK-created trainer/deployment resources on close."""

    init_from_checkpoint: str | None = None
    """Resume from prior checkpoint; bare name = this job, ``"job:name"``
    = cross-job."""
    warm_start_from_adapter: str | None = None
    """Initialize LoRA weights from a PEFT adapter with fresh optimizer state.

    Mutually exclusive with ``init_from_checkpoint`` and requires
    ``lora_rank > 0``.
    """
    save_final_checkpoint: bool = True
    """Save a resumable+promotable checkpoint at the end of training."""
    output_model_id: str | None = None
    """Promote the final checkpoint to this 4-segment model id on clean exit."""


def _save_checkpoint(
    ckpt: TrainingCheckpoints,
    *,
    name: str,
    data_consumed: int,
    resumable: bool = True,
    promotable: bool = False,
) -> None:
    logger.info("[%s] dcp_save...", name)
    with wall_timer(
        "checkpoint_save",
        category="checkpoint",
        attributes={"name": name},
    ) as span:
        ckpt.save(
            name,
            resumable=resumable,
            promotable=promotable,
            data_consumed=data_consumed,
        )
    logger.info("[%s] dcp_save: done (%.1fs)", name, span.elapsed)


def _run_optimizer_step(
    policy: ReconnectableClient,
    config: Config,
    *,
    learning_rate: float,
):
    adam_kwargs = dict(DEFAULT_ADAM)
    adam_kwargs["grad_clip_norm"] = config.grad_clip_norm
    adam_params = tinker.AdamParams(learning_rate=learning_rate, **adam_kwargs)
    return policy.optim_step(
        adam_params,
        grad_accumulation_normalization=objective.NORMALIZATION,
        emit_grad_norm_metrics=True,
    )


def main(
    config: Config,
    *,
    rollout_fn_factory: RolloutFnFactory,
    advantage_fn: Callable[[list[float]], list[float]] = compute_advantages,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    evaluation_fn: RolloutEvaluationFn | None = None,
    evaluation_interval: int = 1,
    rows: list[dict] | None = None,
) -> dict[str, Any]:
    """Run the async RL loop with a user-supplied rollout factory.

    ``rollout_fn_factory(setup) -> rollout_fn`` is called once at startup
    with the assembled :class:`RolloutSetup`.  The returned
    ``rollout_fn(sample_prompt) -> RolloutRun | None`` is invoked
    ``completions_per_prompt`` times per dataset row (each invocation is
    one trajectory draw against the inference deployment).

    ``advantage_fn`` maps one prompt group's rewards to per-rollout advantages.
    The default subtracts the group mean and divides by its standard deviation.
    Supply a mean-only function for REINFORCE with group-centered rewards.

    Remote trainer and sampler setup and lifecycle are owned by the SDK-managed
    Tinker path.
    """
    cfg = config
    if evaluation_interval < 1:
        raise ValueError("evaluation_interval must be >= 1")
    policy_loss = objective.GRPOConfig(eps_clip=cfg.eps_clip, eps_clip_high=cfg.eps_clip_high)
    if cfg.trainer.reference_training_shape_id or cfg.trainer.reference_job_id:
        raise ValueError(
            "async_rl_loop trains without a reference model; unset "
            "trainer.reference_training_shape_id and trainer.reference_job_id."
        )
    logger.warning(
        "async_rl_loop is EXPERIMENTAL and under active development; "
        "the Config / RolloutSetup API may change. See "
        "skills/fireworks-training/references/rl-async.md.",
    )

    def _signal_handler(signum, _):
        name = signal.Signals(signum).name
        raise TerminatedBySignal(name)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    if rows is None and not cfg.dataset:
        raise ValueError("Provide either cfg.dataset or rows= to main().")
    validate_config(
        cfg.base_model,
        cfg.dataset or None,
        deploy=cfg.deployment,
        output_model_id=cfg.output_model_id,
        require_dataset=(rows is None),
    )
    validate_warm_start_config(
        warm_start_from_adapter=cfg.warm_start_from_adapter,
        init_from_checkpoint=cfg.init_from_checkpoint,
        lora_rank=cfg.lora_rank,
    )
    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required.")
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
            "trainer_loss": f"server_{objective.LOSS_FN}",
            "eps_clip": policy_loss.eps_clip,
            "eps_clip_high": policy_loss.upper_eps,
            "grad_clip_norm": cfg.grad_clip_norm,
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
            max_lora_rank=cfg.lora_rank,
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
        )
        stack.callback(service.close)
        training_client = service.create_training_client(
            cfg.base_model,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
        )
        # Decide R3 after the trainer is bound: its create_model response is
        # the authoritative MoE signal, so no GET model is needed (private
        # early-access base models reject that with 403).
        router_replay_enabled = resolve_router_replay_enabled(
            requested=True,
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
            base_model=cfg.base_model,
            training_client=training_client,
        )
        log_metrics({"rollout/step": 0}, step=0)

        policy = ReconnectableClient.from_training_client(
            training_client,
            base_model=cfg.base_model,
            lora_rank=cfg.lora_rank,
            job_id=service.trainer_job_id,
            service=service,
        )
        publish_weights = make_weight_sync(policy, service, cfg.deployment)

        ckpt = TrainingCheckpoints(
            policy,
            service,
            trainer_id=service.trainer_job_id,
            log_path=cfg.log_path,
            lora_rank=cfg.lora_rank,
        )

        resume_info = ckpt.resume(
            init_from_checkpoint=cfg.init_from_checkpoint,
            warm_start_from_adapter=cfg.warm_start_from_adapter,
        )
        step_offset = resume_info.step if resume_info else 0
        if step_offset:
            logger.info("Resuming from step %d", step_offset)
            log_metrics(
                {"train/step": step_offset, "rollout/step": step_offset},
                step=step_offset,
            )

        with elapsed_timer("weight_sync") as span:
            publish_weights(f"step-{step_offset}", checkpoint_type="base")
        logger.info(
            "[step %d] initial weight sync (%.1fs)",
            step_offset,
            span.elapsed,
        )
        flush_timing()
        log_run_resources(service)

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

        sample_kwargs = default_sample_kwargs(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=service.max_context_length,
            http_timeout=cfg.deployment.sample_timeout,
            router_replay=router_replay_enabled,
        )
        sampler = service.create_deployment_sampler(tokenizer=tokenizer)
        stack.callback(sampler.close)
        # One rollout object serves training and evaluation, so stateful
        # rollouts (sandbox pools, judges) keep a single concurrency budget.
        # Evaluation overlaps training here, so sampler/* includes its requests.
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
        rollout_context_param_names = rollout_context_names(rollout_fn)

        async def run_evaluation(step: int) -> None:
            metrics = await evaluate_rollouts(step, evaluation_fn, rollout_fn)
            log_metrics({"rollout/step": step, **metrics}, step=step)

        evaluations = OverlappedEvaluation(
            run_evaluation if evaluation_fn is not None else None,
            interval=evaluation_interval,
        )

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
                source_row_id = row.get("id")

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
                        "evaluation": False,
                    }
                    return call_rollout_fn(
                        rollout_fn, sample_prompt, context, rollout_context_param_names
                    )

                yield RolloutRow(
                    row_id=idx,
                    run_factory=run_one_rollout,
                    row_meta={"row_id": source_row_id},
                    on_resolved=lambda _reason, idx=idx: row_loader.mark_resolved(idx),
                )

        def train_chunk(chunk: TrainingChunk) -> dict[str, Any]:
            """One server-side GRPO forward/backward, clipped against the rollout logprobs."""
            batch = combine_prompt_groups(list(chunk.groups))
            # Fork recipes (pseudo code; utils in training.utils.rl):
            #   Recompute the anchor on the trainer (a proximal snapshot when off-policy):
            #     batch, _ = anchor.recompute_anchor(policy, batch)
            #     extra = anchor.anchor_drift_metrics(batch)
            #   Then correct the sampler/trainer mismatch with truncated IS:
            #     datums, tis_metrics = tis.apply_tis(datums, batch, cap=2.0)  # after build_datums
            #   Another objective (CISPO, DAPO, GSPO, DRO, IS, ...): import it as `objective`
            #   and build its Config; client side: policy.forward_backward_custom(
            #     batch.data, objective.make_loss_fn(batch, policy_loss)).
            datums = objective.build_datums(batch, policy_loss)
            with elapsed_timer("fwd_bwd"):
                result = policy.forward_backward(
                    datums, objective.LOSS_FN, loss_fn_config=objective.loss_fn_config(policy_loss)
                )
            result.metrics.update(objective.metrics(batch, result, policy_loss))
            result.metrics.update(inference_drift_metrics(batch, result))
            return {"prompt_groups": batch.groups, "fwd_bwd_result": result}

        def optimizer_step(step: int) -> dict[str, Any]:
            """Apply exactly one optimizer mutation for one rollout batch."""

            step_lr = compute_lr(
                lr_scheduler,
                step=step,
                base_lr=cfg.learning_rate,
                total_steps=total_steps_estimate,
            )
            with elapsed_timer("optim_step"):
                result = _run_optimizer_step(policy, cfg, learning_rate=step_lr)
            return {
                "result": result,
                "learning_rate": step_lr,
            }

        def sync_weights(step: int) -> tuple[float, dict[str, float]]:
            with wall_timer() as span:
                result = publish_weights(f"step-{step}")
            return span.elapsed, weight_sync_metrics(result)

        async def run_training() -> tuple[int, dict[str, Any]]:
            def _step_metrics(metrics: dict[str, Any], step: int) -> None:
                log_metrics(metrics, step=step)
                # Flush the client phase trace at each optimizer-step boundary
                # so the trace file is inspectable while the run is in flight.
                flush_phase_trace()

            telemetry = AsyncRLTelemetry(
                producer_metrics_fn=lambda metrics: log_metrics(
                    metrics,
                    step=int(metrics["producer/event"]),
                ),
                step_metrics_fn=_step_metrics,
            )
            coordinator = AsyncRLCoordinator(
                rows=make_row_requests(),
                advantage_fn=advantage_fn,
                completions_per_prompt=cfg.completions_per_prompt,
                prompt_groups_per_step=cfg.prompt_groups_per_step,
                training_chunks_per_step=cfg.pipeline_chunks_per_step,
                max_head_off_policy_versions=cfg.max_head_offpolicy_versions,
                max_concurrent_rollouts=cfg.max_concurrency_rollout_sample,
                min_group_size=cfg.min_group_size,
                max_incomplete_group_retries=cfg.max_incomplete_group_retries,
                dynamic_filter_fn=dynamic_filter_fn,
                global_step=step_offset,
                resolved_rows_offset=prior_rows_consumed,
                resolved_rows_fn=lambda: row_loader.data_consumed,
            )
            async with coordinator:
                telemetry.start(coordinator.snapshot)
                evaluations.start(step_offset, force=True)
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

                        evaluation_step = evaluations.active_step
                        if evaluation_step is not None:
                            with wall_timer(
                                "evaluation_join",
                                category="evaluation",
                                attributes={"step": evaluation_step},
                            ) as span:
                                await evaluations.join()
                            log_metrics(
                                {
                                    "rollout/step": evaluation_step,
                                    "eval/weight_sync_wait_time": span.elapsed,
                                },
                                step=evaluation_step,
                            )

                        # A producer failure here can only affect a future batch.
                        # Finish this optimizer's hotload and publication so trainer
                        # and sampler versions cannot diverge on shutdown.
                        sync_wall_time, sync_metrics = await coordinator.run_blocking(
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
                            step_time=published.step_time,
                            weight_update_time=sync_wall_time,
                            learning_rate=optimizer["learning_rate"],
                            extra_metrics={
                                **sync_metrics,
                                **sampler_metrics(sampler.drain_metrics()),
                            },
                        )
                        evaluations.start(batch.batch_id)

                        rollouts_completed = batch.batch_id - step_offset
                        interval = cfg.dcp_save_interval
                        if (
                            interval > 0
                            and rollouts_completed > 0
                            and rollouts_completed % interval == 0
                        ):
                            try:
                                with wall_timer() as span:
                                    await coordinator.run_blocking(
                                        "checkpoint",
                                        _save_checkpoint,
                                        ckpt,
                                        name=f"step-{batch.batch_id}",
                                        data_consumed=published.resolved_rows,
                                    )
                                log_metrics(
                                    {
                                        "rollout/step": batch.batch_id,
                                        "checkpoint/wall_time": span.elapsed,
                                    },
                                    step=batch.batch_id,
                                )
                            except (OSError, RuntimeError) as error:
                                logger.warning(
                                    "[step %d] dcp_save failed: %s",
                                    batch.batch_id,
                                    error,
                                )
                    await evaluations.join()
                    evaluations.start(coordinator.global_step, force=True)
                    await evaluations.join()
                finally:
                    try:
                        await evaluations.cancel()
                    finally:
                        try:
                            await telemetry.aclose()
                        finally:
                            await close_rollout_fn(rollout_fn)

                return coordinator.global_step, telemetry.final_stats()

        global_step, final_stats = asyncio.run(run_training())
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
            "deployment_id": service.deployment_id,
        }
