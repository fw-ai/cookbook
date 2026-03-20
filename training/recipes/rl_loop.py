#!/usr/bin/env python3
"""GRPO training loop with concurrent rollout.

A readable, modifiable RL training loop using the Fireworks RLOR API.

Each optimizer step samples ``prompt_groups_per_step`` prompts concurrently,
then runs a single training update + ``optim_step`` (1:1 ratio).

RL losses can execute in two places:

- Server-side builtin path: ``forward_backward(...)`` with a builtin kernel
  resolved by :func:`training.utils.rl.losses.resolve_builtin_loss`.
- Client-side custom path: ``forward_backward_custom(...)`` with a Python
  loss closure built by :func:`training.utils.rl.losses.build_loss_fn`.

Customisation:

- ``Config.reward_fn`` -- plug in your own reward function.
- ``Config.filter_fn`` -- plug in your own rollout filter.
- ``Config.policy_loss`` -- select a registered loss algorithm.

Usage::

    export FIREWORKS_API_KEY=...
    python -m recipes.rl_loop
"""

from __future__ import annotations

import os
import signal
import asyncio
import logging
import itertools
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import tinker

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.deployment import DeploymentSampler
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils import (
    DEFAULT_ADAM,
    ResourceCleanup,
    ReconnectableClient,
    RLPromptDataset,
    wandb_log,
    setup_wandb,
    wandb_finish,
    validate_config,
    log_metrics_json,
    setup_deployment,
    create_trainer_job,
    load_jsonl_dataset,
    prepare_sampling_messages,
)
from training.utils.checkpoint_utils import (
    resolve_resume,
    save_checkpoint,
    CheckpointKind,
)
from training.utils.config import DeployConfig, InfraConfig, WeightSyncConfig  # noqa: F401 (tests access via module)
from training.utils.rl.config import Config
from training.utils.rl.datum import build_prompt_group
from training.utils.rl.losses import build_loss_fn, resolve_builtin_loss
import time as _time

from training.utils.rl.train import TrainContext, train_one_step, train_one_group, finish_step
from training.utils.rl.rollout import AsyncRolloutScheduler, collect_sync_batch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Infrastructure setup (users don't touch this)
# ---------------------------------------------------------------------------


@dataclass
class _Infra:
    """Everything the training loop needs from infrastructure setup."""

    ctx: TrainContext
    sampler: Any  # DeploymentSampler
    use_reference: bool
    step_offset: int
    resume_info: Any
    remaining_rows: list
    resume_data_consumed: int
    reference_job_id: str | None


def _setup_infra(
    cfg: Config,
    rlor_mgr: TrainerJobManager,
    deploy_mgr: DeploymentManager,
    cleanup: ResourceCleanup,
    cleanup_on_exit: bool,
    api_key: str,
    base_url: str,
) -> _Infra:
    """Create deployment, trainers, clients, dataset, and build TrainContext.

    Called once at the start of ``main()``; the return value feeds the loop.
    """
    import time as _time

    # -- Resolve training shapes ----------------------------------------

    profile = None
    if cfg.infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)
        dep_shape = getattr(profile, "deployment_shape", None) or getattr(
            profile, "deployment_shape_version", None
        )
        if dep_shape and not cfg.deployment.deployment_shape:
            cfg.deployment.deployment_shape = dep_shape

    if profile and cfg.max_seq_len is None:
        cfg.max_seq_len = profile.max_supported_context_length
        logger.info("max_seq_len from training shape: %d", cfg.max_seq_len)
    if cfg.max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )

    ref_profile = None
    if cfg.infra.ref_training_shape_id:
        ref_profile = rlor_mgr.resolve_training_profile(cfg.infra.ref_training_shape_id)

    use_reference = ref_profile is not None
    if not use_reference:
        logger.info("No ref_training_shape_id set, skipping reference model")

    if cfg.async_rollout:
        if not cfg.deployment.deployment_shape:
            extra = cfg.deployment.deployment_extra_args or []
            if "--hot-load-async-transition" not in extra:
                cfg.deployment.deployment_extra_args = extra + [
                    "--hot-load-async-transition"
                ]
                logger.info(
                    "Auto-injected --hot-load-async-transition for async rollout"
                )
        else:
            logger.info(
                "Async rollout: deployment shape is set, ensure it includes "
                "--hot-load-async-transition in its extra_args"
            )

    # -- Deployment + trainers ------------------------------------------

    _infra_start = _time.time()

    dep_info = setup_deployment(deploy_mgr, cfg.deployment, cfg.base_model, cfg.infra)
    if cleanup_on_exit:
        cleanup.deployment(cfg.deployment.deployment_id, action="scale_to_zero")

    prompt_groups_per_step = cfg.prompt_groups_per_step
    completions_per_prompt = cfg.completions_per_prompt
    logger.info(
        "Training: prompt_groups_per_step=%d | completions_per_prompt=%d",
        prompt_groups_per_step,
        completions_per_prompt,
    )

    if use_reference:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pol_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.base_model,
                infra=cfg.infra,
                profile=profile,
                lora_rank=cfg.lora_rank,
                max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name="grpo-policy",
                hot_load_deployment_id=cfg.deployment.deployment_id,
                job_id=cfg.policy_job_id,
                base_url_override=cfg.policy_base_url,
            )
            ref_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.base_model,
                infra=cfg.infra,
                profile=ref_profile,
                lora_rank=cfg.lora_rank,
                max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name="grpo-reference",
                forward_only=True,
                job_id=cfg.reference_job_id,
                base_url_override=cfg.reference_base_url,
            )
            policy_ep = pol_fut.result()
            policy_job_id = policy_ep.job_id
            if not cfg.policy_job_id:
                cleanup.trainer(policy_job_id)
            reference_ep = ref_fut.result()
            reference_job_id = reference_ep.job_id
            if not cfg.reference_job_id:
                cleanup.trainer(reference_job_id)
    else:
        policy_ep = create_trainer_job(
            rlor_mgr,
            base_model=cfg.base_model,
            infra=cfg.infra,
            profile=profile,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            display_name="grpo-policy",
            hot_load_deployment_id=cfg.deployment.deployment_id,
            job_id=cfg.policy_job_id,
            base_url_override=cfg.policy_base_url,
        )
        policy_job_id = policy_ep.job_id
        if not cfg.policy_job_id:
            cleanup.trainer(policy_job_id)
        reference_ep = None
        reference_job_id = None

    # -- Clients, sampler, weight syncer --------------------------------

    policy = ReconnectableClient(
        rlor_mgr,
        policy_ep.job_id,
        cfg.base_model,
        cfg.lora_rank,
        fw_api_key=api_key,
        endpoint=policy_ep if cfg.policy_base_url else None,
    )
    reference = (
        ReconnectableClient(
            rlor_mgr,
            reference_ep.job_id,
            cfg.base_model,
            cfg.lora_rank,
            fw_api_key=api_key,
            endpoint=reference_ep if cfg.reference_base_url else None,
        )
        if reference_ep
        else None
    )

    import transformers

    inference_model = dep_info.inference_model if dep_info else cfg.base_model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.deployment.tokenizer_model, trust_remote_code=True
    )
    sampler = DeploymentSampler(
        inference_url=deploy_mgr.inference_url,
        model=inference_model,
        api_key=api_key,
        tokenizer=tokenizer,
    )
    weight_syncer = WeightSyncer(
        policy_client=policy.inner,
        deploy_mgr=deploy_mgr,
        deployment_id=cfg.deployment.deployment_id,
        base_model=cfg.base_model,
        hotload_timeout=cfg.weight_sync.weight_sync_timeout,
        first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
        dcp_timeout=cfg.weight_sync.dcp_timeout,
    )

    infra_boot_time = _time.time() - _infra_start
    boot_metrics: dict = {
        "train/step": 0,
        "infra/total_boot_time": infra_boot_time,
    }
    if deploy_mgr.boot_time_s is not None:
        boot_metrics["infra/deploy_boot_time"] = deploy_mgr.boot_time_s
    wandb_log(boot_metrics, step=0)

    # -- Resume ---------------------------------------------------------

    resume_info = resolve_resume(policy, cfg.log_path, cfg.init_from_checkpoint)
    step_offset = resume_info.step if resume_info else 0
    wandb_log({"train/step": step_offset}, step_offset)

    if cfg.weight_sync.weight_sync_before_training and cfg.deployment.deployment_id:
        name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
        weight_syncer.save_and_hotload(name, checkpoint_type="base")

    # -- Dataset + loss -------------------------------------------------

    raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
    all_rows = raw_dataset * cfg.epochs
    rl_dataset = RLPromptDataset(all_rows, prompts_per_step=prompt_groups_per_step)
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

    builtin_server_loss = resolve_builtin_loss(
        cfg.policy_loss,
        profile,
        dapo_config=cfg.dapo,
        gspo_config=cfg.gspo,
        cispo_config=cfg.cispo,
        ratio_log_cap=cfg.ratio_log_cap,
        eps_clip=cfg.eps_clip,
        eps_clip_high=cfg.eps_clip_high,
    )

    # -- TrainContext ----------------------------------------------------

    ctx = TrainContext(
        policy=policy,
        reference=reference,
        weight_syncer=weight_syncer,
        adam_params=adam_params,
        grad_accumulation_normalization=cfg.grad_accumulation_normalization,
        builtin_server_loss=builtin_server_loss,
        client_loss_builder=client_loss_builder,
        tis_config=cfg.tis,
        policy_loss=cfg.policy_loss,
        log_path=cfg.log_path,
        policy_job_id=policy_job_id,
        completions_per_prompt=completions_per_prompt,
        trajectory_dir=cfg.trajectory_dir,
        weight_sync_interval=cfg.weight_sync.weight_sync_interval,
        dcp_save_interval=cfg.weight_sync.dcp_save_interval,
        wandb_log=wandb_log,
        log_metrics_json=log_metrics_json,
    )

    # -- Remaining rows -------------------------------------------------

    remaining_rows = []
    for i_step in range(step_offset, len(rl_dataset)):
        remaining_rows.extend(rl_dataset.get_batch(i_step))

    resume_data_consumed = resume_info.data_consumed if resume_info else 0

    return _Infra(
        ctx=ctx,
        sampler=sampler,
        use_reference=use_reference,
        step_offset=step_offset,
        resume_info=resume_info,
        remaining_rows=remaining_rows,
        resume_data_consumed=resume_data_consumed,
        reference_job_id=reference_job_id,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
    deploy_mgr: DeploymentManager | None = None,
    cleanup_on_exit: bool = False,
):
    cfg = config

    # -- Signal handling ------------------------------------------------

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # -- Validation -----------------------------------------------------

    validate_config(
        cfg.base_model,
        cfg.dataset,
        cfg.weight_sync,
        cfg.deployment,
        output_model_id=cfg.output_model_id,
    )
    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step

    step_target = prompt_groups_per_step
    if cfg.async_rollout:
        step_target = (
            cfg.valid_prompt_groups_per_step
            if cfg.valid_prompt_groups_per_step is not None
            else prompt_groups_per_step
        )
        if step_target < 1:
            raise ValueError("valid_prompt_groups_per_step must be >= 1")
        if cfg.max_head_offpolicy_versions < 0:
            raise ValueError("max_head_offpolicy_versions must be >= 0")
    if not cfg.deployment.tokenizer_model:
        raise ValueError(
            "deployment.tokenizer_model is required for client-side tokenization. "
            "Set it to the HuggingFace model name (e.g. 'Qwen/Qwen3-1.7B')."
        )
    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": completions_per_prompt,
            "prompt_groups_per_step": prompt_groups_per_step,
            "kl_beta": cfg.kl_beta,
            "lr": cfg.learning_rate,
        },
    )

    # -- Setup infrastructure -------------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)

    with ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup:
        infra = _setup_infra(
            cfg, rlor_mgr, deploy_mgr, cleanup, cleanup_on_exit, api_key, base_url
        )
        ctx = infra.ctx

        # -- Sampling closure (customize via Config.reward_fn) ----------

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=cfg.max_seq_len,
            http_timeout=cfg.deployment.sample_timeout,
        )
        if cfg.router_replay:
            sample_kwargs.update(
                include_routing_matrix=True, echo=True, logprobs=True
            )
        sample_kwargs["logprobs"] = True

        # Concurrency gate: limits actual HTTP requests to the deployment.
        # sample_with_tokens(n=K) fans out into K individual HTTP requests
        # via asyncio.gather, so the semaphore must gate each HTTP request,
        # not each prompt.  We monkey-patch the sampler's _do_one_completion
        # to acquire the semaphore before each request.
        if cfg.sample_max_concurrency is not None:
            _http_semaphore = asyncio.Semaphore(cfg.sample_max_concurrency)
            _orig_do_one = infra.sampler._do_one_completion

            async def _gated_do_one(*args, **kwargs):
                async with _http_semaphore:
                    return await _orig_do_one(*args, **kwargs)

            infra.sampler._do_one_completion = _gated_do_one
            logger.info(
                "Sample concurrency gate: max %d concurrent HTTP requests",
                cfg.sample_max_concurrency,
            )

        async def sample_one_prompt(row: dict):
            """Sample completions for one prompt and return a PromptGroup."""
            messages = row.get("messages", [])
            input_messages = prepare_sampling_messages(messages)
            if not input_messages:
                return None

            try:
                sampled = await infra.sampler.sample_with_tokens(
                    messages=input_messages,
                    n=completions_per_prompt,
                    **sample_kwargs,
                )
            except Exception as e:
                logger.warning("Sampling failed: %s", e)
                return None

            return build_prompt_group(
                sampled,
                row,
                reward_fn=cfg.reward_fn,
                completions_per_prompt=completions_per_prompt,
                use_reference=infra.use_reference,
                router_replay=cfg.router_replay,
                router_replay_completion_only=cfg.router_replay_completion_only,
                trajectory_dir=cfg.trajectory_dir,
                input_messages=input_messages,
            )

        # -- Checkpoint helper ------------------------------------------

        def _save_ckpt(name: str, extra: dict):
            save_checkpoint(
                ctx.policy, name, cfg.log_path, extra, kind=CheckpointKind.STATE
            )

        # ==============================================================
        # Training loop
        # ==============================================================

        if cfg.async_rollout:
            # --- Async path: overlapped rollout + training ---
            ctx.weight_sync_interval = 1

            async_state = (
                (infra.resume_info.async_state if infra.resume_info else None) or {}
            )

            async def _async_loop() -> int:
                scheduler = AsyncRolloutScheduler(
                    step_target=step_target,
                    max_head_offpolicy_versions=cfg.max_head_offpolicy_versions,
                    filter_fn=cfg.filter_fn,
                    global_step=infra.step_offset,
                    total_accepted=async_state.get("total_accepted", 0),
                    total_rejected=async_state.get("total_rejected", 0),
                    rows_submitted=async_state.get("rows_submitted", 0),
                    max_concurrent=cfg.sample_max_concurrency,
                )
                row_iter = iter(infra.remaining_rows)
                if async_state.get("rows_submitted", 0) > 0:
                    skip = async_state["rows_submitted"]
                    for _ in range(min(skip, len(infra.remaining_rows))):
                        next(row_iter, None)
                    logger.info("Async resume: skipped %d rows", skip)

                step = infra.step_offset

                while not scheduler.data_exhausted:
                    # -- Stream groups: train each immediately, server accumulates --
                    groups: list = []
                    fwd_bwd_results: list = []
                    version_offsets: list[int] = []
                    raw_rewards: list[float] = []
                    t0 = _time.time()

                    async for group, version in scheduler.stream_groups(
                        sample_one_prompt, row_iter
                    ):
                        groups.append(group)
                        version_offsets.append(scheduler.current_version - version)
                        raw_rewards.extend(group.rewards)

                        # Train this group now — server accumulates gradients
                        result = train_one_group(ctx, group)
                        fwd_bwd_results.append(result)
                        logger.info(
                            "[async step %d] trained group %d/%d",
                            step + 1, len(groups), step_target,
                        )

                    if not groups:
                        break

                    wall_time = _time.time() - t0
                    logger.info(
                        "[async step %d] streamed %d groups (%.1fs)",
                        step + 1, len(groups), wall_time,
                    )

                    loop_stats = {
                        "valid_prompt_groups": len(groups),
                        "total_sampled": len(groups),
                        "filter_drops": 0,
                        "sample_fails": 0,
                        "sample_wait_time": wall_time,
                        "step_wall_time": wall_time,
                        "all_raw_rewards": raw_rewards,
                        "version_offsets": version_offsets,
                    }

                    step, metrics = finish_step(
                        ctx,
                        step,
                        groups,
                        fwd_bwd_results,
                        loop_stats,
                        save_checkpoint_fn=lambda name, extra: save_checkpoint(
                            ctx.policy,
                            name,
                            cfg.log_path,
                            {**extra, "async_state": scheduler.get_state()},
                            kind=CheckpointKind.STATE,
                        ),
                        step_target=step_target,
                        resume_data_consumed=infra.resume_data_consumed,
                        step_offset=infra.step_offset,
                    )
                    metrics["async/version"] = scheduler.current_version
                    scheduler.bump_version()

                return step

            global_step = asyncio.run(_async_loop())

        else:
            # --- Sync path: collect batch → train → repeat ---

            async def _sync_loop() -> int:
                step = infra.step_offset
                coro_iter = iter(
                    sample_one_prompt(row) for row in infra.remaining_rows
                )

                while True:
                    batch = list(
                        itertools.islice(coro_iter, prompt_groups_per_step)
                    )
                    if not batch:
                        break

                    groups, stats = await collect_sync_batch(
                        batch,
                        filter_fn=cfg.filter_fn,
                        target=prompt_groups_per_step,
                    )

                    if not groups:
                        logger.warning(
                            "[step %d] no valid prompt groups after filtering, "
                            "skipping",
                            step + 1,
                        )
                        continue

                    logger.info(
                        "Sampling complete: %d/%d groups in %.1fs "
                        "(failed=%d, filtered=%d)",
                        stats.valid_groups,
                        prompt_groups_per_step,
                        stats.wall_time,
                        stats.sample_fails,
                        stats.filter_drops,
                    )

                    loop_stats = {
                        "valid_prompt_groups": stats.valid_groups,
                        "total_sampled": stats.total_sampled,
                        "filter_drops": stats.filter_drops,
                        "sample_fails": stats.sample_fails,
                        "sample_wait_time": stats.wall_time,
                        "step_wall_time": stats.wall_time,
                        "all_raw_rewards": list(stats.raw_rewards),
                    }

                    step, _ = train_one_step(
                        ctx,
                        step,
                        groups,
                        loop_stats,
                        save_checkpoint_fn=_save_ckpt,
                        step_target=prompt_groups_per_step,
                        resume_data_consumed=infra.resume_data_consumed,
                        step_offset=infra.step_offset,
                    )

                return step

            global_step = asyncio.run(_sync_loop())

        # -- Final checkpoint -------------------------------------------

        if global_step > infra.step_offset:
            try:
                _data_consumed = infra.resume_data_consumed + (
                    global_step - infra.step_offset
                ) * prompt_groups_per_step
                cp_name = f"step-{global_step}"
                paths = save_checkpoint(
                    ctx.policy,
                    cp_name,
                    cfg.log_path,
                    {
                        "step": global_step,
                        "data_consumed": _data_consumed,
                        "source_job_id": ctx.policy_job_id,
                    },
                    kind=CheckpointKind.BOTH,
                )

                if getattr(cfg, "output_model_id", None):
                    rlor_mgr.promote_checkpoint(
                        ctx.policy_job_id,
                        paths["sampler_path"],
                        cfg.output_model_id,
                    )
            except Exception as e:
                logger.warning("Failed to save final checkpoint: %s", e)

            logger.info("Training complete: %d steps", global_step)
            wandb_finish()
            return {
                "steps": global_step,
                "policy_job_id": ctx.policy_job_id,
                "reference_job_id": infra.reference_job_id,
            }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main(Config(log_path="./rl_logs"))
