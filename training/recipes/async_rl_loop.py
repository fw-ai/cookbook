#!/usr/bin/env python3
"""Async RL recipe with per-run rollouts and recipe-owned training.

EXPERIMENTAL -- under active development.  API surface (``Config`` field
names, ``RolloutSetup`` shape, gate semantics) may change.  The recipe is intentionally minimal-surface: the
only thing most users need to write is the rollout function; everything
else (gate, advantage, ref forward, weight sync, KL/TIS, pipeline chunking,
checkpoints) is handled by ``main()``.  See
``skills/dev/references/rl/async-rl.md`` for the full contract.

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
import os
import signal
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

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
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    ReconnectableClient,
    build_service_client,
    load_deployment_tokenizer,
    load_jsonl_dataset,
    read_api_extra_headers_env,
    setup_wandb,
    validate_config,
    wandb_finish,
    wandb_log,
)
from training.utils.checkpoints import TrainingCheckpoints
from training.utils.dataloader import CursorDataLoader
from training.utils.logging import ASYNC_RL_WANDB_METRIC_STEPS
from training.utils.rl import PromptGroup
from training.utils.rl.async_train import RowRequest, run_async_rl_loop
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.dro import DROConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.losses import (
    LossPath,
    PolicyLoss,
    build_builtin_loss_datums,
    build_loss_fn,
    combine_prompt_groups,
    get_builtin_loss_config,
    validate_loss_path,
)
from training.utils.rl.metrics import (
    build_accumulated_async_loop_stats,
    build_train_chunk_metrics,
    compute_step_metrics,
    total_target_tokens,
)
from training.utils.rl.tis import TISConfig
from training.train_loop import DynamicFilterFn
from training.utils.rl.rollout import RolloutRun
from training.utils.runner_state import (
    estimate_async_total_steps,
    start_running,
    write_completed,
    write_running_step,
)
from training.utils.timer import elapsed_timer, flush_timing

logger = logging.getLogger(__name__)

__all__ = [
    "Config",
    "RolloutFn",
    "RolloutFnFactory",
    "RolloutSetup",
    "main",
]

# Pinned: raising weight-sync interval trades rollout staleness for weight-sync wall-time,
# almost never worth it in fully-async RL.
_WEIGHT_SYNC_INTERVAL = 1


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
    """Staleness budget in weight-sync versions; ``0`` = strict on-policy.
    See ``skills/dev/references/rl/async-rl.md`` (gate semantics)."""
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

    policy_loss: PolicyLoss = "grpo"
    """One of the registered RL policy losses (see :data:`PolicyLoss`)."""

    loss_path: LossPath = "client"
    """Which forward/backward path to use.

    - ``"builtin"`` -- server-side ``forward_backward(...)`` with a fused
      kernel. Faster, but cannot apply KL (``kl_beta`` must be 0).
    - ``"client"`` -- client-side ``forward_backward_custom(...)``. Always
      works; slower because the client evaluates the Python loss closure.

    Validated at startup by :func:`validate_loss_path`; mismatches raise
    instead of silently falling back.
    """

    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    dro: DROConfig = field(default_factory=DROConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    ratio_log_cap: float = 20.0
    pipeline_chunks_per_step: int = 1
    """Scheduler chunk cap per global optimizer batch.

    The loop sends ready prompt groups without waiting to fill a chunk; trainer
    continuous batching owns execution-level coalescing/microbatching.
    """
    tis: TISConfig = field(default_factory=TISConfig)

    use_rollout_logprobs: bool = False
    """Use rollout-time logprobs as the PPO/IS old-policy anchor.

    Mirrors Slime's ``use_rollout_logprobs`` flag. ``False`` preserves the
    historical async behavior by recomputing old-policy logprobs on the trainer.
    Set ``True`` when the sampler/policy gap is known to be negligible and you
    want to skip the old-policy forward pass.
    """

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync_before_training: bool = False
    dcp_save_interval: int = 0
    weight_sync_timeout: int = 600
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="rl-async"))
    cleanup_on_exit: bool = True
    """Clean up SDK-created trainer/deployment resources on close."""

    runner: RunnerConfig = field(default_factory=RunnerConfig)
    """Optional orchestration outputs (status / metadata / metrics / output
    model). When unset the recipe still runs; the orchestration layer just
    won't receive progress updates. See ``training.utils.runner``."""

    init_from_checkpoint: str | dict | None = None
    """Resume from prior checkpoint; bare name = this job, ``"job:name"``
    = cross-job. Rows returned by ``list_checkpoints`` are also accepted."""
    dataloader_cursor: int | None = None
    """Explicit raw-row cursor. When set, local cursor resolution is skipped."""
    save_final_checkpoint: bool = True
    """Save a resumable+promotable checkpoint at the end of training."""
    output_model_id: str | None = None
    """Promote the final checkpoint to this 4-segment model id on clean exit."""


@dataclass
class RolloutSetup:
    """Dependencies the recipe hands the rollout factory once at startup.

    Inference endpoint, tokenizer, sampling kwargs, plus an ``extras`` dict
    for caller state.  See ``skills/dev/references/rl/async-rl.md``.
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
    step: int,
    row_cursor: int,
    resumable: bool = True,
    promotable: bool = False,
) -> None:
    name = f"step-{step}"
    logger.info("[%s] dcp_save...", name)
    with elapsed_timer("dcp_save") as span:
        ckpt.save(
            step,
            resumable=resumable,
            promotable=promotable,
            row_cursor=row_cursor,
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
    runner = RunnerIO(cfg.runner)

    logger.warning(
        "async_rl_loop is EXPERIMENTAL and under active development; "
        "the Config / RolloutSetup API may change. See "
        "skills/dev/references/rl/async-rl.md.",
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
            "pipeline_chunks_per_step": cfg.pipeline_chunks_per_step,
            "tokenizer_id": cfg.deployment.tokenizer_model,
            "shuffle": cfg.shuffle,
            "seed": cfg.seed,
            "kl_beta": cfg.kl_beta,
            "loss_path": cfg.loss_path,
            "use_rollout_logprobs": int(cfg.use_rollout_logprobs),
            "lr": cfg.learning_rate,
            "lr_schedule": lr_scheduler.type,
        },
        metric_steps=ASYNC_RL_WANDB_METRIC_STEPS,
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    runner.write_status(RunStatus.PENDING, message="provisioning")

    with runner, ExitStack() as stack:
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
        training_client = service.create_training_client(cfg.base_model, lora_rank=cfg.lora_rank)
        sampler = service.create_deployment_sampler(tokenizer=tokenizer)
        rollout_model = sampler.model
        runner.set_accelerator_info(
            service.accelerator_type,
            service.accelerator_count,
            profile=service.training_profile,
        )

        runner.write_metadata()

        wandb_log({"rollout/step": 0})

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
            dataloader_cursor=cfg.dataloader_cursor,
        )
        step_offset = resume_info.step
        if step_offset:
            logger.info("Resuming from step %d", step_offset)
            wandb_log(
                {"train/step": step_offset, "rollout/step": step_offset},
            )

        if cfg.weight_sync_before_training or service.requires_initial_sampler_sync():
            with elapsed_timer("weight_sync") as span:
                ckpt.sync_weights(step_offset, service.hotload_sampler_snapshot)
            logger.info("[step %d] weight sync (%.1fs)", step_offset, span.elapsed)

        if rows is None:
            rows = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        else:
            rows = list(rows)

        prior_rows_consumed = resume_info.row_cursor
        row_loader = CursorDataLoader(
            rows,
            start_cursor=prior_rows_consumed,
            epochs=cfg.epochs,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
        )

        total_steps_estimate = estimate_async_total_steps(
            step_offset=step_offset,
            total_items=row_loader.total_items,
            prior_rows_consumed=prior_rows_consumed,
            prompt_groups_per_step=cfg.prompt_groups_per_step,
        )

        validate_loss_path(cfg, service.training_profile)
        if cfg.loss_path == "builtin":
            builtin_loss = get_builtin_loss_config(cfg)
            client_loss_builder = None
            logger.info(
                "policy_loss=%s loss_path=builtin (server-side loss=%s)",
                cfg.policy_loss,
                builtin_loss[0],
            )
        else:
            builtin_loss = None
            client_loss_builder = build_loss_fn(cfg)
            logger.info(
                "policy_loss=%s loss_path=client (forward_backward_custom)",
                cfg.policy_loss,
            )
        logger.info(
            "use_rollout_logprobs=%s",
            cfg.use_rollout_logprobs,
        )

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

        ctx_metadata: dict[str, Any] = {
            "completions_per_prompt": cfg.completions_per_prompt,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "max_head_offpolicy_versions": cfg.max_head_offpolicy_versions,
            "pipeline_chunks_per_step": cfg.pipeline_chunks_per_step,
            "max_concurrency_rollout_sample": cfg.max_concurrency_rollout_sample,
            "weight_sync_interval": _WEIGHT_SYNC_INTERVAL,
            "max_completion_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "shuffle": cfg.shuffle,
            "seed": cfg.seed,
            "loss_path": cfg.loss_path,
            "use_rollout_logprobs": cfg.use_rollout_logprobs,
            "tokenizer_id": cfg.deployment.tokenizer_model,
            "model": rollout_model,
        }

        def make_row_requests():
            rows_per_epoch = len(rows)
            for item in row_loader:
                row = item.value
                idx = item.index
                epoch = idx // rows_per_epoch if rows_per_epoch else 0
                row_index = idx % rows_per_epoch if rows_per_epoch else idx
                end_of_epoch = row_index == rows_per_epoch - 1 if rows_per_epoch else True
                source_row_id = row.get("id") if isinstance(row, dict) else None

                def factory(
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
                                key: context[key]
                                for key in rollout_context_param_names
                            },
                        )
                    return rollout_fn(sample_prompt)

                yield RowRequest(
                    row_id=idx,
                    run_factory=factory,
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
            if builtin_loss is not None:
                loss_name, loss_cfg = builtin_loss
                rl_datums = build_builtin_loss_datums(
                    data,
                    adv,
                    old_policy_logprobs,
                    inf_lp,
                    prompt_lens,
                    cfg.tis,
                    policy_loss=cfg.policy_loss,
                )
                return policy.forward_backward(
                    rl_datums,
                    loss_name,
                    loss_fn_config=loss_cfg,
                )

            assert client_loss_builder is not None
            return policy.forward_backward_custom(
                data,
                client_loss_builder(
                    adv,
                    ref_lp,
                    prompt_lens,
                    inf_lp,
                    old_policy_logprobs,
                    raw_inf_lp,
                ),
            )

        train_accum: dict[str, Any] = {
            "prompt_groups": [],
            "fwd_bwd_results": [],
            "latest_loop_stats": None,
            "train_start": None,
            "trainer_wait_for_sampler_time": 0.0,
            "sampler_wait_for_trainer_time": 0.0,
        }

        def train_step(
            step: int,
            prompt_groups: list[PromptGroup],
            loop_stats: dict | None,
            run_optimizer_step: bool,
        ) -> tuple[int, dict]:
            """ref_forward + old_policy_logprobs snapshot + train chunks + metrics.

            Each chunk runs fwd/bwd. ``run_optimizer_step`` controls
            whether the chunk also runs ``optim_step``. In default mode the
            chunk is the full rollout batch. In pipeline mode only the final
            accumulated chunk steps it.

            Slime-aligned dual-axis logging: ``train/*`` lands on the
            ``train/step`` axis (one point per optimizer step); ``rollout/*`` /
            ``perf/*`` / ``async/*`` / ``version/*`` land on ``rollout/step``
            (one point per outer rollout batch).
            checkpoint identities remain optimizer-step labels (resume math is
            in optim steps); the off-policy budget is accounted in
            weight-sync versions inside ``_StalenessController`` and is
            independent of those labels.
            """
            train_start = time.monotonic()
            num_chunks = int(
                (loop_stats or {}).get(
                    "pipeline/chunks_per_step",
                    max(1, cfg.pipeline_chunks_per_step),
                )
            )
            rollout_id = step + 1
            if train_accum["train_start"] is None:
                train_accum["train_start"] = train_start
            train_accum["prompt_groups"].extend(prompt_groups)
            if loop_stats is not None:
                train_accum["latest_loop_stats"] = dict(loop_stats)
                train_accum["trainer_wait_for_sampler_time"] += float(
                    loop_stats.get("trainer_wait_for_sampler_time", 0.0)
                )
                train_accum["sampler_wait_for_trainer_time"] += float(
                    loop_stats.get("sampler_wait_for_trainer_time", 0.0)
                )

            with elapsed_timer("ref_forward") as span:
                ref_forward(prompt_groups)
            logger.info(
                "[rollout %d] ref_forward (%.1fs)", rollout_id, span.elapsed,
            )

            data, adv, ref_lp, prompt_lens, inf_lp, raw_inf_lp = combine_prompt_groups(
                prompt_groups,
                include_raw=True,
            )
            if cfg.use_rollout_logprobs:
                if len(inf_lp) != len(data):
                    raise ValueError(
                        "use_rollout_logprobs=True requires one rollout_logprobs "
                        f"row per datum; got {len(inf_lp)} rows for {len(data)} datums."
                    )
                if any(not row for row in inf_lp):
                    raise ValueError(
                        "use_rollout_logprobs=True requires non-empty rollout_logprobs."
                    )
                old_policy_logprobs = inf_lp
                logger.info(
                    "[rollout %d] old_policy_logprobs: using rollout_logprobs",
                    rollout_id,
                )
            else:
                with elapsed_timer("old_policy_forward") as span:
                    old_policy_fwd = policy.forward(data, "cross_entropy")
                    old_policy_logprobs = [
                        old_policy_fwd.loss_fn_outputs[i]["logprobs"].data
                        for i in range(len(data))
                    ]
                logger.info(
                    "[rollout %d] old_policy_forward (%.1fs)",
                    rollout_id, span.elapsed,
                )

            optim_result: Any = None
            chunk_idx = int((loop_stats or {}).get("pipeline/chunk_idx", 1))

            with elapsed_timer("fwd_bwd") as span:
                fwd_bwd_result = fwd_bwd_batch(
                    data,
                    adv,
                    ref_lp,
                    prompt_lens,
                    inf_lp,
                    raw_inf_lp,
                    old_policy_logprobs,
                )
                train_accum["fwd_bwd_results"].append(fwd_bwd_result)
            logger.info(
                "[rollout %d step %d] fwd_bwd (chunk %d/%d) (%.1fs)",
                rollout_id, step + 1, chunk_idx, num_chunks, span.elapsed,
            )

            step_lr = compute_lr(
                lr_scheduler,
                step=step + 1,
                base_lr=cfg.learning_rate,
                total_steps=total_steps_estimate,
            )
            if not run_optimizer_step:
                wandb_log(
                    build_train_chunk_metrics(
                        fwd_bwd_result,
                        None,
                        step=step + 1,
                        chunk_idx=chunk_idx,
                        num_chunks=num_chunks,
                        learning_rate=step_lr,
                        run_optimizer_step=False,
                    )
                )
                return step, {}

            adam_kwargs = dict(DEFAULT_ADAM)
            adam_kwargs["grad_clip_norm"] = cfg.grad_clip_norm
            adam_params = tinker.AdamParams(learning_rate=step_lr, **adam_kwargs)
            with elapsed_timer("optim_step") as span:
                optim_result = policy.optim_step(
                    adam_params,
                    grad_accumulation_normalization=cfg.grad_accumulation_normalization,
                )
            step += 1
            logger.info(
                "[rollout %d step %d] optim_step (chunk %d/%d) (%.1fs)",
                rollout_id, step, chunk_idx, num_chunks, span.elapsed,
            )

            wandb_log(
                build_train_chunk_metrics(
                    fwd_bwd_result,
                    optim_result,
                    step=step,
                    chunk_idx=chunk_idx,
                    num_chunks=num_chunks,
                    learning_rate=step_lr,
                    run_optimizer_step=True,
                )
            )

            prompt_groups = list(train_accum["prompt_groups"])
            fwd_bwd_results = list(train_accum["fwd_bwd_results"])
            train_start = train_accum["train_start"] or train_start
            loop_stats = build_accumulated_async_loop_stats(
                prompt_groups=prompt_groups,
                latest_loop_stats=train_accum["latest_loop_stats"],
                trainer_wait_for_sampler_time=train_accum[
                    "trainer_wait_for_sampler_time"
                ],
                sampler_wait_for_trainer_time=train_accum[
                    "sampler_wait_for_trainer_time"
                ],
                train_wall_time=time.monotonic() - train_start,
            )
            metrics = compute_step_metrics(
                prompt_groups=prompt_groups,
                fwd_bwd_results=fwd_bwd_results,
                optim_result=optim_result,
                n_accum=len(fwd_bwd_results),
                timing_metrics=flush_timing(),
                loop_stats=loop_stats,
                completions_per_prompt=cfg.completions_per_prompt,
            )
            # Per-rollout reference KL for the human summary line below.
            ref_kl = metrics.get("train/ref_kl", 0.0)
            metrics["rollout/step"] = rollout_id
            metrics["train/step"] = step  # monotonic fallback for the wandb global step
            for k, v in ctx_metadata.items():
                if isinstance(v, bool):
                    metrics[f"ctx/{k}"] = int(v)
                elif isinstance(v, (int, float)) and v is not None:
                    metrics[f"ctx/{k}"] = v

            logger.info(
                "Rollout %d (step %d) | Reward %.3f | Acc %.1f%% | RefKL %.4f",
                rollout_id,
                step,
                metrics.get("rollout/reward", 0.0),
                metrics.get("rollout/accuracy", 0.0) * 100,
                ref_kl,
            )
            wandb_metrics = {k: v for k, v in metrics.items() if not k.startswith("train/")}
            wandb_metrics["train/step"] = step
            wandb_log(wandb_metrics)
            # Report the number of trained target tokens, not raw rollout length.
            write_running_step(
                runner,
                step=step,
                total_steps=total_steps_estimate,
                metrics=metrics,
                tokens=total_target_tokens(prompt_groups),
            )
            # DCP cadence is in rollout batches. Async has one optimizer step
            # per rollout batch; pipeline chunks do not multiply it.
            rollouts_completed = step - step_offset
            interval = cfg.dcp_save_interval
            if (
                loop_stats is not None
                and interval > 0
                and rollouts_completed > 0
                and rollouts_completed % interval == 0
            ):
                try:
                    _save_checkpoint(
                        ckpt,
                        step=step,
                        row_cursor=int(loop_stats["resolved_rows"]),
                    )
                except (OSError, RuntimeError) as e:
                    # Periodic save: surface the failure but keep training.
                    # The final save (after the loop) is allowed to propagate
                    # so orchestration sees terminal save problems.
                    logger.warning("[step %d] dcp_save failed: %s", step, e)
            train_accum["prompt_groups"] = []
            train_accum["fwd_bwd_results"] = []
            train_accum["latest_loop_stats"] = None
            train_accum["train_start"] = None
            train_accum["trainer_wait_for_sampler_time"] = 0.0
            train_accum["sampler_wait_for_trainer_time"] = 0.0
            return step, metrics

        def log_post_train_metrics(metrics: dict[str, Any]) -> None:
            wandb_log(metrics)
            train_step_value = int(metrics.get("train/step", 0))
            runner.append_metrics(train_step_value, metrics)

        start_running(
            runner,
            step=step_offset,
            total_steps=total_steps_estimate,
        )

        global_step, final_stats = asyncio.run(
            run_async_rl_loop(
                rows=make_row_requests(),
                train_step_fn=train_step,
                completions_per_prompt=cfg.completions_per_prompt,
                prompt_groups_per_step=cfg.prompt_groups_per_step,
                max_head_offpolicy_versions=cfg.max_head_offpolicy_versions,
                with_reference=(reference is not None),
                min_group_size=cfg.min_group_size,
                weight_sync_fn=lambda step: ckpt.sync_weights(
                    step, service.hotload_sampler_snapshot
                ),
                weight_sync_interval=_WEIGHT_SYNC_INTERVAL,
                max_concurrent=cfg.max_concurrency_rollout_sample,
                dynamic_filter_fn=dynamic_filter_fn,
                pipeline_chunks_per_step=cfg.pipeline_chunks_per_step,
                post_train_metrics_fn=log_post_train_metrics,
                global_step=step_offset,
                resolved_rows_fn=lambda: row_loader.row_cursor,
                return_final_stats=True,
            )
        )

        # Save resume progress even if all remaining rows were dropped.
        # Promotion still requires at least one optimizer step.
        resume_row_cursor = int(final_stats["resolved_rows"])
        has_trained_steps = global_step > step_offset
        has_advanced_dataset = resume_row_cursor > prior_rows_consumed
        promoted_checkpoint: str | None = None
        if cfg.save_final_checkpoint and (has_trained_steps or has_advanced_dataset):
            ckpt.save(
                global_step,
                resumable=True,
                promotable=has_trained_steps,
                row_cursor=resume_row_cursor,
            )
            if cfg.output_model_id and has_trained_steps:
                ckpt.promote_latest(cfg.output_model_id, cfg.base_model)
                promoted_checkpoint = cp_name

        if promoted_checkpoint is not None and cfg.output_model_id:
            runner.write_output_model(
                model_id=cfg.output_model_id,
                checkpoint=promoted_checkpoint,
                job_id=service.trainer_job_id,
            )

        # Clamp progress at 100% when filtering/partial final batches shorten the run.
        final_step = max(global_step, total_steps_estimate)
        write_completed(
            runner,
            step=final_step,
            total_steps=final_step,
        )

        logger.info(
            "Async RL training complete: %d steps (%d new in this run)",
            global_step, global_step - step_offset,
        )
        wandb_finish()
        return {
            "steps": global_step,
            "policy_job_id": service.trainer_job_id,
            "reference_job_id": service.reference_trainer_job_id,
            "deployment_id": service.deployment_id,
        }
