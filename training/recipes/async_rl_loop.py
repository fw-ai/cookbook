#!/usr/bin/env python3
"""Async RL recipe with per-run rollouts and recipe-owned training.

EXPERIMENTAL -- under active development.  API surface (``Config`` field
names, ``RolloutSetup`` shape, gate semantics) may change.  The recipe is intentionally minimal-surface: the
only thing most users need to write is the rollout function; everything
else (gate, advantage, ref forward, weight sync, KL/TIS, PPO inner loop,
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
import math
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
from training.utils.rl import PromptGroup
from training.utils.rl.async_train import RowRequest, run_async_rl_loop
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.dro import DROConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.losses import (
    LossConfig,
    PolicyLoss,
    build_loss_fn,
    combine_prompt_groups,
    validate_loss_path,
)
from training.utils.rl.metrics import compute_minibatch_metrics, compute_step_metrics, total_target_tokens
from training.utils.rl.tis import TISConfig
from training.utils.rl.train import DynamicFilterFn, TrainStepFns
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
    """LR scheduler spec."""

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
    grad_accumulation_normalization: GradAccNormalization | str | None = (
        GradAccNormalization.NUM_LOSS_TOKENS
    )

    policy_loss: PolicyLoss = "grpo"
    """One of the registered RL policy losses (see :data:`PolicyLoss`).
    Client-side only -- ``loss_path`` is not exposed, see skill doc."""

    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    dro: DROConfig = field(default_factory=DROConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    ratio_log_cap: float = 20.0
    ppo_n_minibatches: int = 1
    """Inner PPO steps per rollout batch sharing one ``old_policy_logprobs``
    snapshot; ``1`` uses the default 1:1 behavior."""
    synchronous_training: bool = False
    """Drain rollouts before each train step (no overlap); baseline knob
    for measuring async savings."""
    tis: TISConfig = field(default_factory=TISConfig)

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync_before_training: bool = False
    dcp_save_interval: int = 0
    weight_sync_timeout: int = 600
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="rl-async"))
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    """Optional orchestration outputs (status / metadata / metrics / output
    model). When unset the recipe still runs; the orchestration layer just
    won't receive progress updates. See ``training.utils.runner``."""

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
    if cfg.ppo_n_minibatches < 1:
        raise ValueError(
            f"ppo_n_minibatches must be >= 1; got {cfg.ppo_n_minibatches}."
        )
    lr_scheduler = normalize_lr_scheduler_spec(cfg.lr_scheduler)
    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": cfg.completions_per_prompt,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "max_head_offpolicy_versions": cfg.max_head_offpolicy_versions,
            "ppo_n_minibatches": cfg.ppo_n_minibatches,
            "tokenizer_id": cfg.deployment.tokenizer_model,
            "shuffle": cfg.shuffle,
            "seed": cfg.seed,
            "kl_beta": cfg.kl_beta,
            "lr": cfg.learning_rate,
            "lr_schedule": lr_scheduler.type,
        },
    )
    # Dual axis: train/* per inner PPO minibatch, rollout/* per outer batch.
    try:
        import wandb as _wandb  # noqa: WPS433 (deliberate local import)

        if _wandb.run is not None:
            _wandb.define_metric("rollout/step")
            _wandb.define_metric("rollout/*", step_metric="rollout/step")
            _wandb.define_metric("perf/*", step_metric="rollout/step")
            _wandb.define_metric("infra/*", step_metric="rollout/step")
            _wandb.define_metric("ctx/*", step_metric="rollout/step")
            _wandb.define_metric("batch/*", step_metric="rollout/step")
            _wandb.define_metric("async/*", step_metric="rollout/step")
            _wandb.define_metric("version/*", step_metric="rollout/step")
    except ImportError:
        pass

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
            reference_required=cfg.kl_beta > 0,
        )
        stack.callback(service.close)
        training_client = service.create_training_client(cfg.base_model, lora_rank=cfg.lora_rank)
        sampler = service.create_deployment_sampler(tokenizer=tokenizer)
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
        )
        step_offset = resume_info.step if resume_info else 0
        if step_offset:
            logger.info("Resuming from step %d", step_offset)
            rollout_offset = step_offset // max(1, cfg.ppo_n_minibatches)
            wandb_log(
                {"train/step": step_offset, "rollout/step": rollout_offset},
            )

        if cfg.weight_sync_before_training or service.requires_initial_sampler_sync():
            with elapsed_timer("weight_sync") as span:
                saved = policy.save_weights_for_sampler(
                    f"step-{step_offset}",
                    checkpoint_type="base",
                )
                service.hotload_sampler_snapshot(saved.path)
            logger.info("[step %d] weight sync (%.1fs)", step_offset, span.elapsed)

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

        total_steps_estimate = estimate_async_total_steps(
            step_offset=step_offset,
            total_items=row_loader.total_items,
            prior_rows_consumed=prior_rows_consumed,
            prompt_groups_per_step=cfg.prompt_groups_per_step,
            ppo_n_minibatches=cfg.ppo_n_minibatches,
        )

        # This recipe is client-side only.  ``LossConfig`` adapts the cfg
        # fields to the ``LossArgs`` Protocol that ``build_loss_fn`` reads;
        # ``loss_path="client"`` is fixed (no builtin server-side path).
        loss_args = LossConfig(
            policy_loss=cfg.policy_loss,
            loss_path="client",
            kl_beta=cfg.kl_beta,
            eps_clip=cfg.eps_clip,
            eps_clip_high=cfg.eps_clip_high,
            ratio_log_cap=cfg.ratio_log_cap,
            dapo=cfg.dapo,
            dro=cfg.dro,
            gspo=cfg.gspo,
            cispo=cfg.cispo,
            tis=cfg.tis,
        )
        validate_loss_path(loss_args)
        client_loss_builder = build_loss_fn(loss_args)

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=service.max_context_length,
            http_timeout=cfg.deployment.sample_timeout,
            logprobs=True,
        )

        rollout_setup = RolloutSetup(
            tokenizer=tokenizer,
            tokenizer_id=cfg.deployment.tokenizer_model,
            sample_kwargs=sample_kwargs,
            inference_base_url=sampler.base_url,
            api_key=api_key,
            model=sampler.model,
            completions_per_prompt=cfg.completions_per_prompt,
            extras=dict(rollout_extras or {}),
        )
        rollout_fn = rollout_fn_factory(rollout_setup)
        rollout_context_param_names = _rollout_fn_context_param_names(rollout_fn)

        ctx_metadata: dict[str, Any] = {
            "completions_per_prompt": cfg.completions_per_prompt,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "max_head_offpolicy_versions": cfg.max_head_offpolicy_versions,
            "ppo_n_minibatches": cfg.ppo_n_minibatches,
            "max_concurrency_rollout_sample": cfg.max_concurrency_rollout_sample,
            "weight_sync_interval": _WEIGHT_SYNC_INTERVAL,
            "max_completion_tokens": cfg.max_completion_tokens,
            "temperature": cfg.temperature,
            "shuffle": cfg.shuffle,
            "seed": cfg.seed,
            "tokenizer_id": cfg.deployment.tokenizer_model,
            "model": sampler.model,
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

        def fwd_bwd_minibatch(
            data, adv, ref_lp, prompt_lens, inf_lp, old_policy_logprobs,
        ):
            """One inner PPO minibatch.

            Callers pre-compute ``old_policy_logprobs`` once per rollout
            batch and pass a slice of the flattened rollout tensors for
            this minibatch.
            """
            return policy.forward_backward_custom(
                data, client_loss_builder(adv, ref_lp, prompt_lens, inf_lp, old_policy_logprobs),
            )

        def train_step(
            step: int, prompt_groups: list[PromptGroup], loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            """ref_forward + old_policy_logprobs snapshot + num_minibatches x (fwd_bwd + optim_step) + metrics.

            ``num_minibatches = cfg.ppo_n_minibatches``.  ``old_policy_logprobs``
            is snapshotted once per rollout batch and reused across every
            inner optim step so the PPO ratio measures genuine policy drift.
            DCP checkpoints fire only at rollout boundaries (cadence in
            rollout batches, not optim steps) so resume accounting is
            independent of the minibatch count.

            Slime-aligned dual-axis logging: per-minibatch ``train/*`` lands
            on the ``train/step`` axis (one point per inner PPO step);
            per-batch ``rollout/*`` / ``perf/*`` / ``async/*`` / ``version/*``
            land on ``rollout/step`` (one point per outer rollout batch).
            checkpoint identities remain optimizer-step labels (resume math is
            in optim steps); the off-policy budget is accounted in
            weight-sync versions inside ``_StalenessController`` and is
            independent of those labels.
            """
            train_start = time.monotonic()
            num_minibatches = max(1, cfg.ppo_n_minibatches)
            # 1-indexed outer-batch counter.  ``step`` here is the optim-step
            # count carried over from prior batches, which is always a
            # multiple of num_minibatches at batch boundaries.
            rollout_id = step // num_minibatches + 1

            with elapsed_timer("ref_forward") as span:
                ref_forward(prompt_groups)
            logger.info(
                "[rollout %d] ref_forward (%.1fs)", rollout_id, span.elapsed,
            )

            data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(prompt_groups)
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

            n = len(data)
            minibatch_size = max(1, math.ceil(n / num_minibatches))
            fwd_bwd_results: list = []
            optim_result: Any = None
            for minibatch_idx in range(num_minibatches):
                mb_start = minibatch_idx * minibatch_size
                mb_end = min(mb_start + minibatch_size, n)
                if mb_start >= mb_end:
                    break

                with elapsed_timer("fwd_bwd") as span:
                    fwd_bwd_result = fwd_bwd_minibatch(
                        data[mb_start:mb_end],
                        adv[mb_start:mb_end],
                        ref_lp[mb_start:mb_end],
                        prompt_lens[mb_start:mb_end],
                        inf_lp[mb_start:mb_end],
                        old_policy_logprobs[mb_start:mb_end],
                    )
                    fwd_bwd_results.append(fwd_bwd_result)
                logger.info(
                    "[rollout %d step %d] fwd_bwd (mb %d/%d) (%.1fs)",
                    rollout_id, step + 1, minibatch_idx + 1, num_minibatches,
                    span.elapsed,
                )

                step_lr = compute_lr(
                    lr_scheduler,
                    step=step + 1,
                    base_lr=cfg.learning_rate,
                    total_steps=total_steps_estimate,
                )
                adam_params = tinker.AdamParams(learning_rate=step_lr, **DEFAULT_ADAM)
                with elapsed_timer("optim_step") as span:
                    optim_result = policy.optim_step(
                        adam_params,
                        grad_accumulation_normalization=cfg.grad_accumulation_normalization,
                    )
                step += 1
                logger.info(
                    "[rollout %d step %d] optim_step (mb %d/%d) (%.1fs)",
                    rollout_id, step, minibatch_idx + 1, num_minibatches,
                    span.elapsed,
                )

                # Per-minibatch train/* on train/step axis (each inner step
                # is genuinely distinct data, not an average).
                mb_metrics = compute_minibatch_metrics(fwd_bwd_result, optim_result)
                if mb_metrics:
                    mb_metrics["train/step"] = step
                    mb_metrics["train/minibatch_idx"] = minibatch_idx + 1
                    mb_metrics["train/num_minibatches"] = num_minibatches
                    wandb_log(mb_metrics)

            # ``step_wall_time`` covers the full step (queue wait + all K
            # minibatches), so ``perf/wait_time_ratio`` = wait / (wait + train).
            if loop_stats is not None:
                train_wall = time.monotonic() - train_start
                loop_stats["step_wall_time"] = (
                    loop_stats.get("trainer_wait_for_sampler_time", 0.0) + train_wall
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
            # Capture the per-rollout reference KL for the human summary line
            # before the train/* stripping below removes it.
            ref_kl = metrics.get("train/ref_kl", 0.0)
            # train/* already logged per-minibatch above; strip the averages.
            metrics = {k: v for k, v in metrics.items() if not k.startswith("train/")}
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
            wandb_log(metrics)
            # Report the number of trained target tokens, not raw rollout length.
            write_running_step(
                runner,
                step=step,
                total_steps=total_steps_estimate,
                metrics=metrics,
                tokens=total_target_tokens(prompt_groups),
            )
            # DCP cadence is in rollout batches, not optim steps, so
            # ppo_n_minibatches doesn't change save frequency.
            rollouts_completed = (step - step_offset) // num_minibatches
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
                        name=f"step-{step}",
                        data_consumed=int(loop_stats["resolved_rows"]),
                    )
                except (OSError, RuntimeError) as e:
                    # Periodic save: surface the failure but keep training.
                    # The final save (after the loop) is allowed to propagate
                    # so orchestration sees terminal save problems.
                    logger.warning("[step %d] dcp_save failed: %s", step, e)
            return step, metrics

        start_running(
            runner,
            step=step_offset,
            total_steps=total_steps_estimate,
        )

        global_step, final_stats = asyncio.run(
            run_async_rl_loop(
                rows=make_row_requests(),
                train_fns=TrainStepFns(train_step=train_step),
                completions_per_prompt=cfg.completions_per_prompt,
                prompt_groups_per_step=cfg.prompt_groups_per_step,
                max_head_offpolicy_versions=cfg.max_head_offpolicy_versions,
                with_reference=(reference is not None),
                min_group_size=cfg.min_group_size,
                weight_sync_fn=lambda step: service.hotload_sampler_snapshot(
                    policy.save_weights_for_sampler(f"step-{step}").path
                ),
                weight_sync_interval=_WEIGHT_SYNC_INTERVAL,
                max_concurrent=cfg.max_concurrency_rollout_sample,
                dynamic_filter_fn=dynamic_filter_fn,
                synchronous_training=cfg.synchronous_training,
                global_step=step_offset,
                resolved_rows_fn=lambda: row_loader.data_consumed,
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
            cp_name = f"step-{global_step}"
            ckpt.save(
                cp_name,
                resumable=True,
                promotable=has_trained_steps,
                data_consumed=resume_row_cursor,
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

        # Clamp progress at 100% when dynamic filtering shortens the run.
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
