#!/usr/bin/env python3
"""Async RL recipe -- training mechanics only.

One extension point: ``rollout_fn(row, ctx) -> Rollout | None``.  The user
owns everything about the rollout (sampling, grading, multi-turn, remote
agents, per-turn logging).  The recipe owns everything about the training
side (infra provisioning, loss, optimizer, weight sync, gate-native async
off-policy scheduling).

The recipe always uses the client-side ``forward_backward_custom`` loss
path so the same code works for every supported loss (GRPO, DAPO, GSPO,
CISPO, REINFORCE, etc.), multi-turn masking, and custom corrections.
Users who want the server-side builtin kernel for a supported loss can
replace the ``fwd_bwd_one`` body with ``resolve_builtin_loss`` +
``build_builtin_loss_datums`` + ``policy.forward_backward``; see
``training/recipes/rl_loop.py`` for the dual-path pattern.

Resume + checkpoint-ladder semantics mirror :mod:`training.recipes.rl_loop`:
``init_from_checkpoint`` / ``warm_start_from_adapter`` for cold start,
``dcp_save_interval`` for periodic resumable saves, plus a final
resumable + promotable save (with optional ``output_model_id`` promote)
on clean exit.  See :class:`training.utils.checkpoints.TrainingCheckpoints`
for the semantic axes (resumable vs promotable).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time as _time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import tinker
import transformers

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from training.utils.client import GradAccNormalization
from fireworks.training.sdk.deployment import (
    AdaptiveConcurrencyController,
    DeploymentSampler,
)
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils import (
    DEFAULT_ADAM,
    ConcurrencyConfig,
    DeployConfig,
    InfraConfig,
    ResourceCleanup,
    WandBConfig,
    WeightSyncConfig,
    load_jsonl_dataset,
    read_api_extra_headers_env,
    replicate_rows_for_epochs,
    setup_wandb,
    validate_config,
    wandb_finish,
    wandb_log,
)
from training.utils.checkpoints import TrainingCheckpoints, validate_warm_start_config
from training.utils.rl import PromptGroup, setup_infra
from training.utils.rl.async_train import run_async_rl_loop
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.losses import build_loss_fn, combine_prompt_groups
from training.utils.rl.metrics import compute_step_metrics
from training.utils.rl.tis import TISConfig
from training.utils.rl.train import DynamicFilterFn, TrainStepFns
from training.utils.rl.rollout import Rollout, rollout_to_prompt_group
from training.utils.timer import flush_timing, timer

logger = logging.getLogger(__name__)

__all__ = ["Config", "RolloutContext", "RolloutFn", "main"]


@dataclass
class Config:
    log_path: str
    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str | None = None
    """JSONL dataset path or URL.  Optional -- ``main(..., rows=...)`` also
    accepts rows directly for users building their dataset in Python."""

    learning_rate: float = 1e-5
    kl_beta: float = 0.001
    completions_per_prompt: int = 4
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    max_rows: int = 100
    max_seq_len: int | None = None
    lora_rank: int = 0

    prompt_groups_per_step: int = 1
    max_head_offpolicy_versions: int = 0
    """Staleness budget in optimizer-step versions.  ``0`` = strict on-policy."""
    sample_max_concurrency: int | None = None
    # NOTE: hotload cadence comes from ``cfg.weight_sync.weight_sync_interval``
    # (the standard nested ``WeightSyncConfig`` surface).  Earlier rounds
    # exposed a duplicate top-level ``weight_sync_interval`` field that
    # silently overrode the nested one; that surface has been removed so
    # ``WeightSyncConfig(weight_sync_interval=N)`` is the single source
    # of truth and existing knobs (``dcp_save_interval`` etc.) compose
    # the same way they do in the sync recipe.

    grad_accumulation_normalization: GradAccNormalization | str | None = (
        GradAccNormalization.NUM_LOSS_TOKENS
    )

    policy_loss: str = "grpo"
    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    ratio_log_cap: float = 20.0
    tis: TISConfig = field(default_factory=TISConfig)

    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="rl-async"))

    init_from_checkpoint: str | None = None
    """Resume from a prior checkpoint.  Bare name resumes from this trainer's
    own history.  ``"job_id:checkpoint_name"`` resumes across trainer jobs."""
    warm_start_from_adapter: str | None = None
    """LoRA-only cold start from a HuggingFace adapter (no resume state).
    Mutually exclusive with ``init_from_checkpoint``.  Requires ``lora_rank > 0``."""
    save_final_checkpoint: bool = True
    """Save a resumable+promotable checkpoint at the end of training."""
    output_model_id: str | None = None
    """When set on a clean final save, promote the latest checkpoint to this
    model id (4-segment ``accounts/<acct>/models/<id>`` form)."""
    policy_job_id: str | None = None
    """Reuse an existing policy trainer job (e.g. when resuming)."""


@dataclass
class RolloutContext:
    """What a ``rollout_fn`` receives to do its job.

    The context keeps the public surface narrow: direct rollouts call
    ``sample_with_tokens(...)`` and custom HTTP integrations can use
    ``inference_base_url`` with ``api_key`` / ``model``.  The underlying
    SDK sampler and concurrency controller stay recipe internals.

    All fields are live: ``current_version()`` returns the up-to-date version
    counter at call time, so multi-turn rollouts that span a hotload see the
    new version on later segments.
    """

    tokenizer: Any
    tokenizer_id: str
    completions_per_prompt: int
    sample_kwargs: dict[str, Any]
    sample_with_tokens: Callable[..., Awaitable[Any]]
    inference_base_url: str
    api_key: str
    model: str
    current_version: Callable[[], int]


RolloutFn = Callable[[dict, RolloutContext], Awaitable[Rollout | None]]


def main(
    config: Config,
    *,
    rollout_fn: RolloutFn,
    dynamic_filter_fn: DynamicFilterFn | None = None,
    rows: list[dict] | None = None,
    cancel_on_exit: bool = False,
    ctx_extras: dict[str, Any] | None = None,
) -> None:
    """Run the async RL loop with a user-supplied rollout function.

    ``cancel_on_exit=True`` registers the policy + reference trainers
    and the inference deployment with a ``ResourceCleanup`` scope so
    they are cancelled / scaled down if ``main()`` exits via an
    exception (rollout-fn failure, SIGINT/SIGTERM, checkpoint save
    error).  The default ``False`` preserves the long-running-job
    semantics callers may rely on (interactive resume, manual
    teardown).
    """
    cfg = config

    def _signal_handler(signum, _):
        name = signal.Signals(signum).name
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    if rows is None and not cfg.dataset:
        raise ValueError("Provide either cfg.dataset or rows= to main().")
    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required.")

    # Always run the base_model / output_model_id preflight, even when
    # the caller passes ``rows=`` (no JSONL dataset).  Skipping these
    # checks would let a malformed base_model or invalid
    # output_model_id slip through provisioning and only fail at
    # trainer creation or final promotion — wasting an entire run.
    validate_config(
        cfg.base_model,
        cfg.dataset or None,
        cfg.weight_sync,
        cfg.deployment,
        output_model_id=cfg.output_model_id,
        require_dataset=(rows is None),
    )
    # The default ``rollout_to_prompt_group`` advantage_fn is the
    # GRPO-style ``compute_advantages`` z-score normalizer, which
    # divides by ``torch.std(rewards)``.  On length-1 reward tensors
    # ``std`` is undefined (NaN); ``rollout_to_prompt_group`` then
    # drops the group (R49 finite-advantage guard) and the loop counts
    # the row as ``sample_fail``.  With ``completions_per_prompt=1``
    # every row hits this path, so a misconfigured run silently
    # consumes and checkpoints the whole dataset without ever
    # training — far worse than a hard error at startup.  Reject
    # upfront so the user gets a clear diagnostic instead of a
    # zero-step "successful" run.  (REINFORCE-style single-sample
    # objectives are documented in the recipe header but require
    # plumbing a custom advantage_fn; until that's wired through the
    # config, a single-sample run is unsupported.)
    if cfg.completions_per_prompt < 2:
        raise ValueError(
            "async_rl_loop requires cfg.completions_per_prompt >= 2: the "
            "default GRPO-style advantage normalizer (z-score by "
            "torch.std(rewards)) is undefined on length-1 reward tensors "
            "and would drop every group, silently consuming the dataset "
            "without ever training.  Set completions_per_prompt >= 2 (the "
            f"default is 4); got {cfg.completions_per_prompt}."
        )
    validate_warm_start_config(
        warm_start_from_adapter=cfg.warm_start_from_adapter,
        init_from_checkpoint=cfg.init_from_checkpoint,
        lora_rank=cfg.lora_rank,
        # ``setup_infra`` runs the precise post-resolve check — it
        # knows whether the resolved ref shape is LoRA-capable.
        has_separate_lora_reference=False,
    )
    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": cfg.completions_per_prompt,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "max_head_offpolicy_versions": cfg.max_head_offpolicy_versions,
            "kl_beta": cfg.kl_beta,
            "lr": cfg.learning_rate,
        },
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    rlor_mgr = TrainerJobManager(
        api_key=api_key, base_url=base_url, additional_headers=additional_headers,
    )
    deploy_mgr = DeploymentManager(
        api_key=api_key, base_url=base_url, additional_headers=additional_headers,
    )

    # ``ResourceCleanup`` cancels remote trainers and scales the
    # deployment to zero on scope exit.  Without it, a recipe that
    # exits via a rollout-fn exception, checkpoint failure, or
    # SIGINT/SIGTERM after ``setup_infra`` returned would leave the
    # policy/reference trainers and the inference deployment running,
    # leaking expensive GPU resources until manual teardown.
    with ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup, ExitStack() as stack:
        infra = setup_infra(
            rlor_mgr=rlor_mgr,
            deploy_mgr=deploy_mgr,
            base_model=cfg.base_model,
            infra_cfg=cfg.infra,
            deploy_cfg=cfg.deployment,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            policy_job_id=cfg.policy_job_id,
            needs_reference=(cfg.kl_beta > 0),
            needs_inference=True,
            role_prefix="rl-async",
            api_key=api_key,
            cleanup=cleanup if cancel_on_exit else None,
            warm_start_from_adapter=cfg.warm_start_from_adapter,
            init_from_checkpoint=cfg.init_from_checkpoint,
        )
        policy_job_id = infra.policy_job_id
        for closeable in infra.closeables:
            stack.callback(closeable.close)

        wandb_log(infra.boot_metrics, step=0)

        policy = infra.policy
        reference = infra.reference

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.deployment.tokenizer_model, trust_remote_code=True,
        )
        initial_window = cfg.concurrency.initial_window or (8 * infra.deployment_gpu_count)
        concurrency_controller = AdaptiveConcurrencyController(
            initial_window=initial_window,
            min_window=cfg.concurrency.min_window,
            max_window=cfg.concurrency.max_window,
            prefill_queue_target=cfg.concurrency.prefill_queue_target,
        )
        sampler = DeploymentSampler(
            inference_url=deploy_mgr.inference_url,
            model=infra.inference_model,
            api_key=api_key,
            tokenizer=tokenizer,
            concurrency_controller=concurrency_controller,
        )
        weight_syncer = WeightSyncer(
            policy_client=policy.inner,
            deploy_mgr=deploy_mgr,
            deployment_id=infra.deployment_id,
            base_model=cfg.base_model,
            hotload_timeout=cfg.weight_sync.weight_sync_timeout,
            first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
            lora_rank=cfg.lora_rank,
        )

        ckpt = TrainingCheckpoints(
            policy,
            rlor_mgr,
            trainer_id=policy_job_id,
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
            wandb_log({"train/step": step_offset}, step_offset)

        if cfg.weight_sync.weight_sync_before_training and infra.deployment_id:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            weight_syncer.save_and_hotload(name, checkpoint_type="base")
            ckpt.invalidate_promotable_snapshot_cache()

        current_version = step_offset

        if rows is None:
            raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
            rows = replicate_rows_for_epochs(raw_dataset, cfg.epochs)
        elif cfg.epochs > 1:
            # ``cfg.epochs`` must apply uniformly regardless of how the
            # caller supplied ``rows``.  When the recipe loads from
            # ``cfg.dataset`` we already replicated; when the caller built
            # rows in Python and passed ``rows=...``, we replicate here.
            # Without this, ``epochs > 1`` runs trained on a single pass
            # of the supplied rows and the persisted raw-row cursor could
            # not resume into later epochs.
            rows = replicate_rows_for_epochs(list(rows), cfg.epochs)

        # On resume, slice ``rows`` from the persisted raw-row cursor (the
        # number of rows actually pulled from the iterator across the prior
        # run, NOT ``step_offset * prompt_groups_per_step`` which undercounts
        # whenever ``rollout_fn`` returned None or ``dynamic_filter_fn``
        # rejected a sampled group).  The cursor is maintained by
        # ``_RawRowCursor`` below (see ``_data_consumed_at``).  Older
        # checkpoints written before this fix carry a step-derived
        # ``data_consumed`` value; they will under-skip on resume but never
        # over-skip, so existing checkpoints are still safe to load.
        prior_rows_consumed = resume_info.data_consumed if resume_info else 0
        if prior_rows_consumed > 0:
            rows = rows[prior_rows_consumed:]

        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)
        client_loss_builder = build_loss_fn(
            policy_loss=cfg.policy_loss,
            kl_beta=cfg.kl_beta,
            dapo_config=cfg.dapo,
            gspo_config=cfg.gspo,
            cispo_config=cfg.cispo,
            ratio_log_cap=cfg.ratio_log_cap,
            tis_config=cfg.tis,
            eps_clip=cfg.eps_clip,
            eps_clip_high=cfg.eps_clip_high,
        )

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=infra.max_seq_len,
            http_timeout=cfg.deployment.sample_timeout,
            logprobs=True,
        )

        ctx = RolloutContext(
            tokenizer=tokenizer,
            tokenizer_id=cfg.deployment.tokenizer_model,
            completions_per_prompt=cfg.completions_per_prompt,
            sample_kwargs=sample_kwargs,
            sample_with_tokens=sampler.sample_with_tokens,
            inference_base_url=deploy_mgr.inference_url,
            api_key=api_key,
            model=infra.inference_model,
            current_version=lambda: current_version,
        )
        # Attach caller-supplied extras (e.g. ``renderer``,
        # ``sample_with_prompt_tokens``, ``build_env`` for the
        # shipped multi-turn example rollouts) so example
        # ``rollout_fn`` callables run unmodified through this hook.
        if ctx_extras:
            for k, v in ctx_extras.items():
                setattr(ctx, k, v)

        async def sample_one_prompt(row: dict) -> PromptGroup | None:
            # Don't catch exceptions here.  Deterministic integration
            # bugs from the user's rollout_fn (e.g.
            # ``ExtensionPropertyError`` from the multi-turn helpers,
            # ``PrefixMismatch`` from the trajectory assembler, schema
            # errors) MUST fail loud — converting them to ``None``
            # makes them count as ``sample_fail`` in the loop, folds
            # them into ``data_consumed`` via ``_on_finalize``, and
            # the run finishes "successfully" while persisting a
            # resume cursor that skips the broken rows on the next
            # run.  Transient / recoverable errors (rollout-service
            # network blips, etc.) are the user's responsibility to
            # absorb inside their own rollout_fn — and the canonical
            # ``make_text_rollout_fn`` already does that, returning
            # ``None`` on service failure.
            rollout = await rollout_fn(row, ctx)
            if rollout is None:
                return None
            return rollout_to_prompt_group(
                rollout, with_reference=(reference is not None),
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

        def fwd_bwd_one(prompt_groups: list[PromptGroup]):
            # Always use forward_backward_custom -- supports every registered
            # policy_loss, kl_beta > 0, and per-token loss_mask for multi-turn.
            # For the server-side builtin kernel pattern see rl_loop.py.
            data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(prompt_groups)
            prox_fwd = policy.forward(data, "cross_entropy")
            prox_lp = [prox_fwd.loss_fn_outputs[i]["logprobs"].data for i in range(len(data))]
            return policy.forward_backward_custom(
                data, client_loss_builder(adv, ref_lp, prompt_lens, inf_lp, prox_lp),
            )

        def train_step(
            step: int, prompt_groups: list[PromptGroup], loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            t0 = _time.time()
            ref_forward(prompt_groups)
            logger.info("[step %d] ref_forward (%.1fs)", step + 1, _time.time() - t0)

            t0 = _time.time()
            fwd_bwd_result = fwd_bwd_one(prompt_groups)
            logger.info("[step %d] fwd_bwd (%.1fs)", step + 1, _time.time() - t0)

            t0 = _time.time()
            optim_result = policy.optim_step(
                adam_params,
                grad_accumulation_normalization=cfg.grad_accumulation_normalization,
            )
            step += 1
            logger.info("[step %d] optim_step (%.1fs)", step, _time.time() - t0)

            metrics = compute_step_metrics(
                prompt_groups=prompt_groups,
                fwd_bwd_results=[fwd_bwd_result],
                optim_result=optim_result,
                n_accum=1,
                timing_metrics=flush_timing(),
                loop_stats=loop_stats,
                completions_per_prompt=cfg.completions_per_prompt,
            )
            metrics["train/step"] = step
            logger.info(
                "Step %d | Reward %.3f | Acc %.1f%% | KL %.4f",
                step,
                metrics.get("rollout/reward", 0.0),
                metrics.get("rollout/accuracy", 0.0) * 100,
                metrics.get("train/mean_kl", 0.0),
            )
            wandb_log(metrics, step)
            return step, metrics

        def _weight_sync(step: int) -> None:
            nonlocal current_version
            with timer("weight_sync"):
                weight_syncer.save_and_hotload(f"step-{step}")
            ckpt.invalidate_promotable_snapshot_cache()
            current_version = step

        # ---- Durable-row resume cursor ----------------------------------
        # The async loop intentionally pulls rows from the iterator faster
        # than they're trained: prefetch / off-policy overlap can keep
        # rows in flight or buffered ahead of the current optim step.
        # Counting "rows pulled" therefore over-reports — a periodic
        # checkpoint that fires while N rows are buffered would persist
        # ``data_consumed = pulled``, and on resume those buffered rows
        # would be skipped permanently (silent data loss).
        #
        # Instead, count rows that have been *durably resolved*:
        #
        #   durable_consumed
        #     = prior_rows_consumed
        #     + cumulative trained groups (one per row that produced a
        #       trained sample)
        #     + cumulative ``sample_fail`` rows (rollout_fn returned None
        #       or sample_fn raised)
        #     + cumulative ``filter_drops`` rows (dynamic_filter_fn rejected)
        #
        # The metrics callback fires after each train_step (and after the
        # R18 final partial-batch flush), so this cell is the up-to-date
        # truth at every checkpoint boundary.
        _durable_consumed_cell = [int(prior_rows_consumed)]
        _trained_groups_in_run = [0]

        def _data_consumed_at(step: int) -> int:
            """Return the durably-resolved row cursor — the resume anchor.

            Excludes rows whose rollout is still in flight or whose
            PromptGroup is buffered ahead of the trainer at checkpoint
            time.  Persisted to ``dataloader.json`` so resume slices
            ``rows`` exactly past the rows the prior run actually
            finished with."""
            return _durable_consumed_cell[0]

        def _maybe_periodic_save(step: int) -> None:
            interval = cfg.weight_sync.dcp_save_interval
            if interval <= 0 or step <= step_offset:
                return
            rollouts_completed = step - step_offset
            if rollouts_completed % interval != 0:
                return
            logger.info("[step %d] dcp_save...", step)
            t0 = _time.time()
            try:
                ckpt.save(
                    f"step-{step}",
                    resumable=True,
                    promotable=False,
                    data_consumed=_data_consumed_at(step),
                )
                logger.info("[step %d] dcp_save: done (%.1fs)", step, _time.time() - t0)
            except Exception as e:
                logger.warning("[step %d] dcp_save failed: %s", step, e)

        def _metrics_cb(loop_metrics: dict) -> None:
            for k, v in concurrency_controller.step_completed().items():
                loop_metrics[f"concurrency/{k}"] = v
            # Update the durable-consumed cursor before persisting:
            # ``valid_prompt_groups`` is this step's contribution
            # (delta), while ``sample_fails`` and ``filter_drops`` are
            # cumulative within this run.
            _trained_groups_in_run[0] += int(
                loop_metrics.get("valid_prompt_groups", 0)
            )
            _durable_consumed_cell[0] = (
                int(prior_rows_consumed)
                + _trained_groups_in_run[0]
                + int(loop_metrics.get("sample_fails", 0))
                + int(loop_metrics.get("filter_drops", 0))
            )
            wandb_log(loop_metrics, step=loop_metrics.get("train/step", 0))
            _maybe_periodic_save(loop_metrics.get("train/step", 0))

        def _on_finalize(stats: dict) -> None:
            # The loop has exited.  ``_metrics_cb`` only fires after a
            # successful train_step, so any tail rows whose rollout
            # returned None (sample_fails) or whose group was rejected
            # by ``dynamic_filter_fn`` after the LAST train_step never
            # made it into ``_durable_consumed_cell``.  This callback
            # delivers the final cumulative counts so the absolute
            # cursor advances over those tail rows before the final
            # save fires — otherwise a rerun would replay the
            # already-exhausted suffix.
            _durable_consumed_cell[0] = (
                int(prior_rows_consumed)
                + _trained_groups_in_run[0]
                + int(stats.get("sample_fails", 0))
                + int(stats.get("filter_drops", 0))
            )

        global_step = asyncio.run(
            run_async_rl_loop(
                sample_fns=(sample_one_prompt(row) for row in rows),
                train_fns=TrainStepFns(train_step=train_step),
                prompt_groups_per_step=cfg.prompt_groups_per_step,
                max_head_offpolicy_versions=cfg.max_head_offpolicy_versions,
                weight_sync_fn=_weight_sync,
                weight_sync_interval=cfg.weight_sync.weight_sync_interval,
                max_concurrent=cfg.sample_max_concurrency,
                dynamic_filter_fn=dynamic_filter_fn,
                global_step=step_offset,
                metrics_callback=_metrics_cb,
                on_finalize=_on_finalize,
            )
        )

        # Save a resumable checkpoint whenever this run either trained
        # at least one step OR consumed rows that didn't reach a
        # successful train_step (e.g. ``rollout_fn`` returned None for
        # every row, or ``dynamic_filter_fn`` rejected every group).
        # Without the second branch a deterministically-failing
        # dataset suffix would never advance ``data_consumed`` past
        # zero, and a rerun against the same ``log_path`` would
        # repeatedly reprocess the same already-exhausted rows.
        # Promotion still requires actual training progress — there's
        # no point publishing a model when no fwd_bwd ran.
        durable_consumed = _data_consumed_at(global_step)
        trained_progressed = global_step > step_offset
        consumed_progressed = durable_consumed > prior_rows_consumed
        if cfg.save_final_checkpoint and (trained_progressed or consumed_progressed):
            cp_name = f"step-{global_step}"
            # Let any failure propagate.  Other recipes do this too — the
            # safer default is to fail hard so orchestration sees the
            # incomplete run and operators can investigate / re-run.
            # Silently swallowing here meant a transient control-plane
            # error or invalid ``output_model_id`` would leave no
            # resumable checkpoint and no promoted model while the job
            # reported success.
            ckpt.save(
                cp_name,
                resumable=True,
                promotable=trained_progressed,
                data_consumed=durable_consumed,
                step=global_step,
            )
            if cfg.output_model_id and trained_progressed:
                ckpt.promote_latest(cfg.output_model_id, cfg.base_model)

        # Successful-completion path.  Without this block ``main()``
        # would fall off the end of the file: the function returns
        # ``None``, ``wandb_finish`` never fires (active run leaks until
        # process exit), and the orchestration layer never sees a
        # success signal.  The branch above is gated on
        # ``save_final_checkpoint`` and ``global_step > step_offset``,
        # so on early-exit / no-train paths nothing reported success at
        # all.  Always run the completion path.
        logger.info(
            "Async RL training complete: %d steps (%d new in this run)",
            global_step, global_step - step_offset,
        )
        wandb_finish()
        # Surface ``reference_job_id`` so callers running with
        # ``cancel_on_exit=False`` can reuse or explicitly clean up the
        # auto-provisioned reference trainer (``setup_infra(...,
        # needs_reference=cfg.kl_beta > 0)`` may bring one up when KL
        # is enabled).  Without it the programmatic API leaks the
        # reference job — callers know the policy job id but have no
        # handle on the reference job to ``cancel_trainer_job`` or
        # reattach it.  ``None`` when no separate reference was
        # provisioned (e.g. ``kl_beta == 0``, or the recipe used the
        # in-process ``policy.create_base_reference()`` path).
        return {
            "steps": global_step,
            "policy_job_id": policy_job_id,
            "reference_job_id": infra.reference_job_id,
        }
