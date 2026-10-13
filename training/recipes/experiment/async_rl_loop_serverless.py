#!/usr/bin/env python3
"""Async RL on the Fireworks serverless training and sampling pool.

The scheduler, rollout contract, batching, telemetry, one-update-per-batch
semantics and objective match :mod:`training.recipes.async_rl_loop`: each group
trains with the trainer's built-in GRPO loss, clipped against the rollout
logprobs. The resource lifecycle is serverless: no trainer job or inference
deployment is provisioned. A training session publishes each policy version with
``save_weights_for_sampler`` and the next rollout version samples through a
session-bound sampling client.
"""

# TODO: Fold this experiment into ``training.recipes.async_rl_loop`` once
# serverless snapshot publication/client replacement and dedicated deployment
# hotload share one weight-sync contract. Keep serverless lifecycle details
# local to this experiment until then.

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import hashlib
import logging
import math
import os
import time
import re
from dataclasses import dataclass, field
from typing import Any

import tinker
from fireworks.training.sdk import FiretitanServiceClient
from fireworks.training.sdk.training_spec import (
    LRSchedulerSpec,
    compute_lr,
    default_constant_schedule,
    normalize_lr_scheduler_spec,
)

from training.utils.rl.heldout import RolloutEvaluationFn, evaluation_rollout_fn
from training.utils.rl.heldout import run_evaluation as evaluate_rollouts
from training.utils.rl.rollout.setup import (
    DynamicFilterFn,
    RolloutFnFactory,
    RolloutSetup,
    call_rollout_fn,
    default_sample_kwargs,
    rollout_context_names,
)
from training.utils.phase_tracing import phase_span
from training.utils.termination import close_in_thread, run_in_thread, with_signals
from training.utils import (
    DEFAULT_ADAM,
    WandBConfig,
    load_jsonl_dataset,
    load_tokenizer,
    log_metrics,
    read_api_extra_headers_env,
    resolve_router_replay_enabled,
    setup_wandb,
    wandb_finish,
)
from training.utils.dataloader import CursorDataLoader
from training.utils.logging import ASYNC_RL_WANDB_METRIC_STEPS
from training.utils.serverless import serverless_base_url
from training.utils.rl.async_rl import (
    AsyncRLCoordinator,
    AsyncRLTelemetry,
    OverlappedEvaluation,
    RolloutRow,
    TrainingGroup,
)
from training.utils.rl.algorithm import grpo as objective
from training.utils.rl.streaming import (
    SubmittedGroup,
    track_group,
    retain_submission,
    submission_window,
    drain_groups,
    add_batch_diagnostics,
    streaming_metrics,
)
from training.utils.rl.losses import combine_prompt_groups
from training.utils.rl.rollout.lifecycle import close_rollout_fn
from training.utils.timer import elapsed_timer, flush_timing, wall_timer

logger = logging.getLogger(__name__)

__all__ = [
    "Config",
    "ServerlessSampler",
    "main",
    "run_sampling_preflight",
]

_MAX_SNAPSHOT_NAME_LENGTH = 54


@dataclass
class Config:
    """Serverless async-RL configuration with no dedicated resource fields."""

    base_model: str = "accounts/fireworks/models/kimi-k3"
    tokenizer_model: str = "moonshotai/Kimi-K3"
    tokenizer_revision: str | None = None
    dataset: str | None = None

    learning_rate: float = 3e-5
    lr_scheduler: LRSchedulerSpec = field(default_factory=default_constant_schedule)
    completions_per_prompt: int = 8
    prompt_groups_per_step: int = 8
    max_completion_tokens: int = 8192
    max_seq_len: int = 524288
    temperature: float = 1.0
    epochs: int = 1
    shuffle: bool = False
    seed: int = 0
    max_rows: int = 320
    lora_rank: int = 64
    lora_alpha: int = 32
    """LoRA scaling factor; ``32`` matches the SDK's ``DEFAULT_LORA_ALPHA``."""

    adam_beta2: float = DEFAULT_ADAM["beta2"]
    adam_epsilon: float = DEFAULT_ADAM["eps"]
    weight_decay: float = DEFAULT_ADAM["weight_decay"]

    max_head_offpolicy_versions: int = 0
    max_concurrency_rollout_sample: int | None = None
    min_group_size: int = 8
    max_incomplete_group_retries: int = 2

    grad_clip_norm: float = 0.0
    eps_clip: float = 0.2
    """GRPO lower clip epsilon; also the upper one unless ``eps_clip_high`` is set."""
    eps_clip_high: float | None = None
    sample_timeout: float = 2400.0
    snapshot_prefix: str = "async-rl"
    metrics_file: str | None = None
    init_from_checkpoint: str | None = None
    step_offset: int = 0
    resolved_rows_offset: int = 0
    dcp_save_interval: int = 0
    save_final_checkpoint: bool = True
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="serverless-rl-async"))


class ServerlessSampler:
    """Stable rollout-facing sampler whose session snapshot can be replaced."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._in_flight: dict[int, int] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def model(self) -> str:
        return str(self._require_client().deployment_sampler.model)

    @property
    def base_url(self) -> str:
        return str(self._require_client().deployment_sampler.base_url)

    @property
    def tokenizer(self) -> Any:
        return self._require_client().deployment_sampler.tokenizer

    async def sample_with_prompt_tokens(self, *args: Any, **kwargs: Any) -> Any:
        async with self._lock:
            client = self._require_client()
            key = id(client)
            self._in_flight[key] = self._in_flight.get(key, 0) + 1
        try:
            return await client.deployment_sampler.sample_with_prompt_tokens(
                *args,
                **kwargs,
            )
        finally:
            async with self._lock:
                remaining = self._in_flight[key] - 1
                if remaining:
                    self._in_flight[key] = remaining
                else:
                    self._in_flight.pop(key)
                    self._condition.notify_all()

    async def replace(self, client: Any) -> None:
        """Publish a new snapshot, then retire the prior client after its calls."""

        async with self._condition:
            old_client = self._require_client()
            self._client = client
            while self._in_flight.get(id(old_client), 0):
                await self._condition.wait()
        await asyncio.to_thread(old_client.close)

    async def aclose(self) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closed = True
            client = self._client
            self._client = None
            while client is not None and self._in_flight.get(id(client), 0):
                await self._condition.wait()
        if client is not None:
            await asyncio.to_thread(client.close)

    def _require_client(self) -> Any:
        if self._closed or self._client is None:
            raise RuntimeError("serverless sampler is closed")
        return self._client


def _validate_config(cfg: Config) -> None:
    positive = {
        "completions_per_prompt": cfg.completions_per_prompt,
        "prompt_groups_per_step": cfg.prompt_groups_per_step,
        "max_completion_tokens": cfg.max_completion_tokens,
        "max_seq_len": cfg.max_seq_len,
        "max_rows": cfg.max_rows,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
    if cfg.completions_per_prompt < 2:
        raise ValueError("GRPO requires completions_per_prompt >= 2")
    if cfg.min_group_size < 1 or cfg.min_group_size > cfg.completions_per_prompt:
        raise ValueError("min_group_size must be in [1, completions_per_prompt]")
    if cfg.max_incomplete_group_retries < 0:
        raise ValueError("max_incomplete_group_retries must be >= 0")
    if cfg.max_head_offpolicy_versions < 0:
        raise ValueError("max_head_offpolicy_versions must be >= 0")
    if cfg.step_offset < 0:
        raise ValueError("step_offset must be >= 0")
    if cfg.resolved_rows_offset < 0:
        raise ValueError("resolved_rows_offset must be >= 0")
    if (cfg.step_offset or cfg.resolved_rows_offset) and not cfg.init_from_checkpoint:
        raise ValueError("serverless resume offsets require init_from_checkpoint")
    minimum_resolved = cfg.step_offset * cfg.prompt_groups_per_step
    if cfg.resolved_rows_offset < minimum_resolved:
        raise ValueError("resolved_rows_offset cannot be smaller than the prompt groups " "represented by step_offset")
    if (
        cfg.max_concurrency_rollout_sample is not None
        and cfg.max_concurrency_rollout_sample < cfg.completions_per_prompt
    ):
        raise ValueError("max_concurrency_rollout_sample must be at least completions_per_prompt")
    if cfg.max_completion_tokens >= cfg.max_seq_len:
        raise ValueError("max_completion_tokens must be smaller than max_seq_len")
    if cfg.lora_rank < 1:
        raise ValueError("serverless async RL requires lora_rank >= 1")
    if cfg.lora_alpha < 1:
        raise ValueError("lora_alpha must be >= 1")
    if not 0.0 <= cfg.adam_beta2 < 1.0:
        raise ValueError("adam_beta2 must be in [0, 1)")
    if cfg.adam_epsilon <= 0.0:
        raise ValueError("adam_epsilon must be > 0")
    if cfg.weight_decay < 0.0:
        raise ValueError("weight_decay must be >= 0")
    if cfg.dcp_save_interval < 0:
        raise ValueError("dcp_save_interval must be >= 0")


def _make_service(api_key: str, base_url: str) -> FiretitanServiceClient:
    headers = read_api_extra_headers_env()
    return FiretitanServiceClient(
        api_key=api_key,
        base_url=serverless_base_url(base_url),
        default_headers=headers,
    )


def _snapshot_name(prefix: str, suffix: str) -> str:
    """Build a DNS label with room for the SDK's 8-character session suffix."""

    prefix_slug = re.sub(r"[^a-z0-9]+", "-", prefix.lower()).strip("-")
    suffix_slug = re.sub(r"[^a-z0-9]+", "-", suffix.lower()).strip("-")
    candidate = "-".join(part for part in (prefix_slug, suffix_slug) if part)
    if not candidate:
        raise ValueError("snapshot prefix and suffix must contain an alphanumeric character")
    if len(candidate) <= _MAX_SNAPSHOT_NAME_LENGTH:
        return candidate

    digest = hashlib.sha256(candidate.encode()).hexdigest()[:8]
    tail = suffix_slug[-20:]
    head_length = _MAX_SNAPSHOT_NAME_LENGTH - len(digest) - len(tail) - 2
    head = prefix_slug[:head_length].rstrip("-")
    return "-".join(part for part in (head, tail, digest) if part)


def _create_snapshot_sampling_client(
    *,
    service: FiretitanServiceClient,
    training_client: Any,
    tokenizer: Any,
    name: str,
) -> tuple[Any, str]:
    saved = training_client.save_weights_for_sampler(name).result()
    snapshot = getattr(saved, "path", None)
    if not snapshot:
        raise RuntimeError(f"serverless sampler snapshot {name!r} has no path")
    client = service.create_sampling_client(
        model_path=snapshot,
        tokenizer=tokenizer,
    )
    return client, snapshot


def _wandb_config(cfg: Config, *, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "algorithm": "grpo",
        "trainer_loss": f"server_{objective.LOSS_FN}",
        "base_model": cfg.base_model,
        "tokenizer_model": cfg.tokenizer_model,
        "tokenizer_revision": cfg.tokenizer_revision,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "learning_rate": cfg.learning_rate,
        "adam_beta2": cfg.adam_beta2,
        "adam_epsilon": cfg.adam_epsilon,
        "weight_decay": cfg.weight_decay,
        "completions_per_prompt": cfg.completions_per_prompt,
        "prompt_groups_per_step": cfg.prompt_groups_per_step,
        "max_rows": cfg.max_rows,
        "max_seq_len": cfg.max_seq_len,
        "max_completion_tokens": cfg.max_completion_tokens,
        "max_head_offpolicy_versions": cfg.max_head_offpolicy_versions,
        "max_concurrency_rollout_sample": cfg.max_concurrency_rollout_sample,
        "step_offset": cfg.step_offset,
        "resolved_rows_offset": cfg.resolved_rows_offset,
        "dcp_save_interval": cfg.dcp_save_interval,
        "eps_clip": cfg.eps_clip,
        "eps_clip_high": cfg.eps_clip_high,
    }


def _rollout_setup(
    cfg: Config,
    *,
    tokenizer: Any,
    sampler: ServerlessSampler,
    api_key: str,
    router_replay_enabled: bool,
) -> RolloutSetup:
    return RolloutSetup(
        sampler=sampler,
        tokenizer=tokenizer,
        tokenizer_id=cfg.tokenizer_model,
        sample_kwargs=default_sample_kwargs(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=cfg.max_seq_len,
            http_timeout=cfg.sample_timeout,
            router_replay=router_replay_enabled,
        ),
        completions_per_prompt=cfg.completions_per_prompt,
        inference_base_url=sampler.base_url,
        api_key=api_key,
        model=sampler.model,
    )


def _router_replay_enabled(cfg: Config, *, api_key: str, base_url: str, training_client: Any) -> bool:
    """R3 for MoE policies, decided from the bound trainer's create response."""
    return resolve_router_replay_enabled(
        requested=True,
        api_key=api_key,
        base_url=base_url,
        additional_headers=read_api_extra_headers_env(),
        base_model=cfg.base_model,
        training_client=training_client,
    )


def run_sampling_preflight(
    config: Config,
    *,
    rollout_fn_factory: RolloutFnFactory,
    evaluation_fn: RolloutEvaluationFn,
) -> dict[str, Any]:
    """Evaluate a zero-update LoRA snapshot without mutating training state."""

    cfg = config
    _validate_config(cfg)
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    setup_wandb(
        cfg.wandb,
        _wandb_config(cfg, mode="serverless-sampling-preflight"),
        metric_steps=ASYNC_RL_WANDB_METRIC_STEPS,
    )
    service: FiretitanServiceClient | None = None
    sampler: ServerlessSampler | None = None
    try:
        service = _make_service(api_key, base_url)
        tokenizer = load_tokenizer(cfg.tokenizer_model, cfg.tokenizer_revision)
        training_client = service.create_lora_training_client(
            base_model=cfg.base_model,
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
        )
        sampling_client, _snapshot = _create_snapshot_sampling_client(
            service=service,
            training_client=training_client,
            tokenizer=tokenizer,
            name=_snapshot_name(cfg.snapshot_prefix, "preflight-step-0"),
        )
        sampler = ServerlessSampler(sampling_client)
        setup = _rollout_setup(
            cfg,
            tokenizer=tokenizer,
            sampler=sampler,
            api_key=api_key,
            router_replay_enabled=_router_replay_enabled(
                cfg,
                api_key=api_key,
                base_url=base_url,
                training_client=training_client,
            ),
        )
        rollout_fn = rollout_fn_factory(setup)

        async def evaluate() -> dict[str, Any]:
            try:
                metrics = await evaluation_fn(0, evaluation_rollout_fn(rollout_fn))
                result = dict(metrics or {})
                log_metrics(
                    {"rollout/step": 0, **result},
                    step=0,
                    metrics_file=cfg.metrics_file,
                )
                return result
            finally:
                try:
                    await close_rollout_fn(rollout_fn)
                finally:
                    await sampler.aclose()

        return asyncio.run(evaluate())
    finally:
        try:
            if sampler is not None and not sampler.closed:
                asyncio.run(sampler.aclose())
            if service is not None:
                service.close()
        finally:
            wandb_finish(metrics_file=cfg.metrics_file)


@with_signals
async def main(
    config: Config,
    *,
    rollout_fn_factory: RolloutFnFactory,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    evaluation_fn: RolloutEvaluationFn | None = None,
    evaluation_interval: int = 1,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train with shared async scheduling and serverless weight publication."""

    cfg = config
    _validate_config(cfg)
    policy_loss = objective.GRPOConfig(eps_clip=cfg.eps_clip, eps_clip_high=cfg.eps_clip_high)
    if evaluation_interval < 1:
        raise ValueError("evaluation_interval must be >= 1")

    if rows is None and not cfg.dataset:
        raise ValueError("Provide either cfg.dataset or rows= to main().")
    lr_scheduler = normalize_lr_scheduler_spec(cfg.lr_scheduler)
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    setup_wandb(
        cfg.wandb,
        _wandb_config(cfg, mode="serverless-training"),
        metric_steps=ASYNC_RL_WANDB_METRIC_STEPS,
    )
    service: FiretitanServiceClient | None = None
    sampler: ServerlessSampler | None = None
    stack = AsyncExitStack()
    try:
        service = _make_service(api_key, base_url)
        stack.push_async_callback(close_in_thread, service.close)
        tokenizer = await run_in_thread(load_tokenizer, cfg.tokenizer_model, cfg.tokenizer_revision)
        training_client = await run_in_thread(
            service.create_lora_training_client,
            base_model=cfg.base_model,
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
        )
        if cfg.init_from_checkpoint:
            await run_in_thread(lambda: training_client.load_state_with_optimizer(cfg.init_from_checkpoint).result())
        initial_client, initial_snapshot = await run_in_thread(
            _create_snapshot_sampling_client,
            service=service,
            training_client=training_client,
            tokenizer=tokenizer,
            name=_snapshot_name(cfg.snapshot_prefix, f"step-{cfg.step_offset}"),
        )
        sampler = ServerlessSampler(initial_client)
        stack.push_async_callback(sampler.aclose)
        router_replay_enabled = await run_in_thread(
            _router_replay_enabled,
            cfg,
            api_key=api_key,
            base_url=base_url,
            training_client=training_client,
        )
        rollout_setup = _rollout_setup(
            cfg,
            tokenizer=tokenizer,
            sampler=sampler,
            api_key=api_key,
            router_replay_enabled=router_replay_enabled,
        )
        rollout_fn = rollout_fn_factory(rollout_setup)
        stack.push_async_callback(close_rollout_fn, rollout_fn)
        context_names = rollout_context_names(rollout_fn)
        if rows is None:
            rows = await run_in_thread(load_jsonl_dataset, cfg.dataset, cfg.max_rows)
        else:
            rows = list(rows)
        if not rows:
            raise ValueError("serverless async RL dataset is empty")
        row_loader = CursorDataLoader(
            rows,
            start_cursor=cfg.resolved_rows_offset,
            epochs=cfg.epochs,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
        )
        remaining_rows = max(
            0,
            row_loader.total_items - cfg.resolved_rows_offset,
        )
        total_steps = cfg.step_offset + math.ceil(remaining_rows / cfg.prompt_groups_per_step)

        def make_row_requests():
            rows_per_epoch = len(rows)
            for item in row_loader:
                row = item.value
                index = item.index
                epoch = index // rows_per_epoch
                row_index = index % rows_per_epoch

                def run_one_rollout(
                    sample_index: int,
                    sample_prompt=row,
                    cursor_index=index,
                    row_index=row_index,
                    epoch=epoch,
                ):
                    context = {
                        "cursor_index": cursor_index,
                        "row_index": row_index,
                        "epoch": epoch,
                        "rollout_idx": sample_index,
                        "sample_index": sample_index,
                        "end_of_epoch": row_index == rows_per_epoch - 1,
                        "evaluation": False,
                    }
                    return call_rollout_fn(rollout_fn, sample_prompt, context, context_names)

                yield RolloutRow(
                    row_id=index,
                    run_factory=run_one_rollout,
                    row_meta={"row_id": row.get("id")},
                    on_resolved=lambda _reason, index=index: row_loader.mark_resolved(index),
                )

        def train_group(groups: list[TrainingGroup], owner: asyncio.Task) -> SubmittedGroup:
            """Build and submit in the trainer worker; polling stays on the event loop."""
            batch = combine_prompt_groups([group.group for group in groups])
            # Fork recipes (utils in training.utils.rl): recompute anchors before
            # build_datums; apply tis.apply_tis(datums, batch, cap=2.0) afterwards.
            # Client-side loss: training_client.forward_backward_custom(
            #     batch.data, objective.make_loss_fn(batch, policy_loss)).
            datums = objective.build_datums(batch, policy_loss)
            started = time.monotonic()
            with phase_span("fwd_bwd_submit", category="training", attributes={"groups": len(groups)}):
                future = training_client.forward_backward(
                    datums, objective.LOSS_FN, loss_fn_config=objective.loss_fn_config(policy_loss)
                )
            return track_group(
                future,
                batch,
                ready_at=min(group.ready_at for group in groups),
                ready_utc=min(group.ready_utc for group in groups),
                submit_started_at=started,
                timeout=None,
                abort_task=owner,
                context={
                    "step": groups[0].batch_id,
                    "rows": [group.source_token for group in groups],
                    "versions": [group.submit_version for group in groups],
                },
                group_ready_times=[group.ready_at for group in groups],
            )

        def optimizer_step(step: int) -> dict[str, Any]:
            learning_rate = compute_lr(
                lr_scheduler,
                step=step,
                base_lr=cfg.learning_rate,
                total_steps=total_steps,
            )
            adam_kwargs = dict(DEFAULT_ADAM)
            adam_kwargs["grad_clip_norm"] = cfg.grad_clip_norm
            adam_kwargs["beta2"] = cfg.adam_beta2
            adam_kwargs["eps"] = cfg.adam_epsilon
            adam_kwargs["weight_decay"] = cfg.weight_decay
            params = tinker.AdamParams(learning_rate=learning_rate, **adam_kwargs)
            with elapsed_timer("optim_step"):
                # Groups add token-summed losses; normalize the step to a
                # per-token mean. Global/RMS grad norms only.
                result = training_client.optim_step(
                    params,
                    grad_accumulation_normalization=objective.NORMALIZATION,
                    emit_grad_norm_metrics=True,
                ).result()
            return {"result": result, "learning_rate": learning_rate}

        def create_sampler_version(step: int) -> tuple[Any, str]:
            return _create_snapshot_sampling_client(
                service=service,
                training_client=training_client,
                tokenizer=tokenizer,
                name=_snapshot_name(cfg.snapshot_prefix, f"step-{step}"),
            )

        def save_training_state(step: int) -> str:
            saved = training_client.save_state(_snapshot_name(cfg.snapshot_prefix, f"state-step-{step}")).result()
            path = getattr(saved, "path", None)
            if not path:
                raise RuntimeError(f"serverless DCP checkpoint at step {step} has no path")
            logger.info("[step %d] saved serverless DCP checkpoint: %s", step, path)
            return str(path)

        async def run_evaluation(step: int) -> None:
            metrics = await evaluate_rollouts(step, evaluation_fn, rollout_fn)
            log_metrics({"rollout/step": step, **metrics}, step=step, metrics_file=cfg.metrics_file)

        evaluations = OverlappedEvaluation(
            run_evaluation if evaluation_fn is not None else None,
            interval=evaluation_interval,
        )

        async def run_training() -> tuple[int, dict[str, Any], str, list[str]]:
            periodic_checkpoints: list[str] = []
            telemetry = AsyncRLTelemetry(
                producer_metrics_fn=lambda metrics: log_metrics(
                    metrics,
                    step=int(metrics["producer/event"]),
                    metrics_file=cfg.metrics_file,
                ),
                step_metrics_fn=lambda metrics, step: log_metrics(
                    metrics,
                    step=step,
                    metrics_file=cfg.metrics_file,
                ),
            )
            coordinator = AsyncRLCoordinator(
                rows=make_row_requests(),
                completions_per_prompt=cfg.completions_per_prompt,
                prompt_groups_per_step=cfg.prompt_groups_per_step,
                max_head_off_policy_versions=cfg.max_head_offpolicy_versions,
                max_concurrent_rollouts=cfg.max_concurrency_rollout_sample,
                min_group_size=cfg.min_group_size,
                max_incomplete_group_retries=(cfg.max_incomplete_group_retries),
                dynamic_filter_fn=dynamic_filter_fn,
                global_step=cfg.step_offset,
                resolved_rows_offset=cfg.resolved_rows_offset,
                resolved_rows_fn=lambda: row_loader.data_consumed,
            )
            latest_snapshot = initial_snapshot
            async with coordinator:
                telemetry.start(coordinator.snapshot)
                evaluations.start(cfg.step_offset, force=True)
                try:
                    while (batch := await coordinator.next_batch()) is not None:
                        async with submission_window() as submissions:
                            async for groups in batch.ready_group_batches():
                                coordinator.raise_if_failed(batch)
                                await retain_submission(
                                    submissions,
                                    coordinator.run_blocking(
                                        "train_group",
                                        train_group,
                                        groups,
                                        asyncio.current_task(),
                                        optimizer_batch=batch,
                                    ),
                                )
                                batch.start_training(at=submissions[-1].submitted_at)
                            coordinator.raise_if_failed(batch)
                            results = await drain_groups(submissions)
                        add_batch_diagnostics(submissions, results, policy_loss, objective)
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
                            ) as evaluation_wait_span:
                                await evaluations.join()
                            log_metrics(
                                {
                                    "rollout/step": evaluation_step,
                                    "eval/weight_sync_wait_time": (evaluation_wait_span.elapsed),
                                },
                                step=evaluation_step,
                                metrics_file=cfg.metrics_file,
                            )
                        with wall_timer(
                            "weight_update",
                            attributes={"step": batch.batch_id},
                        ) as update_span:
                            (
                                next_client,
                                latest_snapshot,
                            ) = await coordinator.run_blocking(
                                "weight_sync",
                                create_sampler_version,
                                batch.batch_id,
                                optimizer_batch=batch,
                            )
                            await sampler.replace(next_client)
                            # Sidecar harnesses (Harbor TITO) read setup.model when
                            # each trial launches, not the live sampler.
                            rollout_setup.model = sampler.model
                        published = coordinator.publish(batch)
                        telemetry.finish_step(
                            batch=batch,
                            trained_against_version=(published.trained_against_version),
                            prompt_groups=[group for submission in submissions for group in submission.batch.groups],
                            fwd_bwd_results=results,
                            optim_result=optimizer["result"],
                            timing_metrics={
                                **flush_timing(),
                                **streaming_metrics(submissions),
                            },
                            step_time=published.step_time,
                            weight_update_time=update_span.elapsed,
                            learning_rate=optimizer["learning_rate"],
                        )
                        evaluations.start(batch.batch_id)
                        interval = cfg.dcp_save_interval
                        completed_steps = batch.batch_id - cfg.step_offset
                        if interval > 0 and completed_steps > 0 and completed_steps % interval == 0:
                            with wall_timer(
                                "checkpoint",
                                category="checkpoint",
                                attributes={"step": batch.batch_id},
                            ) as checkpoint_span:
                                checkpoint = await coordinator.run_blocking(
                                    "checkpoint",
                                    save_training_state,
                                    batch.batch_id,
                                )
                            periodic_checkpoints.append(checkpoint)
                            log_metrics(
                                {
                                    "rollout/step": batch.batch_id,
                                    "checkpoint/wall_time": checkpoint_span.elapsed,
                                },
                                step=batch.batch_id,
                                metrics_file=cfg.metrics_file,
                            )
                    await evaluations.join()
                    evaluations.start(coordinator.global_step, force=True)
                    await evaluations.join()
                finally:
                    try:
                        await evaluations.cancel()
                    finally:
                        await telemetry.aclose()
            return (
                coordinator.global_step,
                telemetry.final_stats(),
                latest_snapshot,
                periodic_checkpoints,
            )

        steps, final_stats, final_snapshot, periodic_checkpoints = await run_training()
        resolved_rows = int(final_stats["resolved_rows"])
        final_checkpoint = None
        if cfg.save_final_checkpoint and (steps > cfg.step_offset or resolved_rows > cfg.resolved_rows_offset):
            if (
                periodic_checkpoints
                and cfg.dcp_save_interval > 0
                and (steps - cfg.step_offset) % cfg.dcp_save_interval == 0
            ):
                final_checkpoint = periodic_checkpoints[-1]
            else:
                final_checkpoint = await run_in_thread(save_training_state, steps)
        session = getattr(service, "training_session_name", None) or getattr(
            service,
            "training_session_id",
            None,
        )
        return {
            "steps": steps,
            "accepted_groups": (cfg.step_offset * cfg.prompt_groups_per_step + int(final_stats["total_accepted"])),
            "resolved_rows": resolved_rows,
            "training_session": session,
            "training_run": getattr(training_client, "run_name", None) or getattr(training_client, "run_id", None),
            "final_sampler_snapshot": final_snapshot,
            "periodic_training_checkpoints": periodic_checkpoints,
            "final_training_checkpoint": final_checkpoint,
        }
    finally:
        try:
            await stack.aclose()
        finally:
            wandb_finish(metrics_file=cfg.metrics_file)
