#!/usr/bin/env python3
"""Async GRPO training loop with streaming rollout-training overlap.

A sibling to ``rl_loop.py`` that streams prompt groups to the trainer
as they arrive from sampling rather than collecting a full batch first.
The trainer accumulates gradients across ``ref_fwd_bwd`` calls;
``finish_step`` fires ``optim_step`` + weight sync after all groups in a
step have been processed.

Key differences from ``rl_loop.py``:

* **Streaming pipeline.** ``AsyncRolloutScheduler.stream_groups()`` yields
  groups one at a time; each is immediately sent to the trainer for
  ``ref_forward + fwd_bwd``. Sampling for the next group continues in
  the background.
* **Two-level capacity gating.** ``max_head_offpolicy_versions`` controls
  staleness; ``sample_max_concurrency`` controls in-flight HTTP requests
  to the deployment.
* **Forced 1:1 weight sync.** The streaming pipeline requires
  ``weight_sync_interval=1`` (any other value is overridden).

Acknowledgements: ``AsyncRolloutScheduler`` and the two-level capacity
gating are inspired by AReaL's ``BatchTaskDispatcher`` and
``StalenessManager`` (https://github.com/inclusionAI/AReaL).

Usage::

    export FIREWORKS_API_KEY=...
    python -m recipes.async_rl_loop
"""

from __future__ import annotations

import os
import re
import signal
import asyncio
import logging
import warnings
import time as _time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import List, Optional

import tinker

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.client import GradAccNormalization
from training.utils import (
    DEFAULT_ADAM,
    ConcurrencyConfig,
    DeployConfig,
    InfraConfig,
    ResourceCleanup,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    WeightSyncConfig,
    compute_advantages,
    log_metrics_json,
    load_jsonl_dataset,
    prepare_sampling_messages,
    read_api_extra_headers_env,
    setup_wandb,
    validate_config,
    wandb_finish,
    wandb_log,
)
from training.utils.checkpoint_utils import (
    CheckpointKind,
    resolve_resume,
    save_checkpoint,
)
from training.utils.rl import (
    AsyncRolloutScheduler,
    PromptGroup,
    TrainContext,
    align_inference_logprobs,
    finish_step,
    make_policy_datum,
    make_reference_datum,
    ref_fwd_bwd,
    setup_infra,
)
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.losses import build_loss_fn, resolve_builtin_loss
from training.utils.rl.router_replay import build_r3_routing_matrices
from training.utils.rl.tis import TISConfig

logger = logging.getLogger(__name__)


warnings.warn(
    "async_rl_loop is EXPERIMENTAL — APIs may change without notice. "
    "For production training, use rl_loop.py.",
    stacklevel=2,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Async GRPO config. Knobs particular to the async path are flagged."""

    log_path: str
    """Directory for checkpoints and logs. Required."""

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    rollout_base_model: str | None = None
    dataset: str = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"

    learning_rate: float = 1e-5
    kl_beta: float = 0.001
    completions_per_prompt: int = 4
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    max_rows: int = 100
    max_seq_len: int | None = None
    lora_rank: int = 0

    # -- Sync sizing (shared with rl_loop) ----------------------------------

    prompt_groups_per_step: int = 1
    """Default number of accepted groups per optimizer step."""

    # -- Async-specific knobs -----------------------------------------------

    valid_prompt_groups_per_step: int | None = None
    """Target accepted groups per step (defaults to ``prompt_groups_per_step``)."""

    max_head_offpolicy_versions: int = 2
    """Maximum staleness: how many optimizer steps a rollout may lag the
    current weight version. 0 = strict on-policy."""

    sample_max_concurrency: int | None = None
    """Max in-flight HTTP requests to the deployment. ``None`` = capped
    only by the staleness window. Independent of the policy window."""

    # -- Router replay (R3) -------------------------------------------------

    router_replay: bool = False
    router_replay_completion_only: bool = True

    # -- Training -----------------------------------------------------------

    grad_accumulation_normalization: GradAccNormalization | str | None = (
        GradAccNormalization.NUM_LOSS_TOKENS
    )

    # -- Loss ---------------------------------------------------------------

    policy_loss: str = "grpo"
    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    ratio_log_cap: float = 20.0
    tis: TISConfig = field(default_factory=TISConfig)

    # -- Trajectory / resume ------------------------------------------------

    trajectory_dir: str | None = None
    policy_job_id: str | None = None
    policy_base_url: str | None = None
    reference_job_id: str | None = None
    reference_base_url: str | None = None
    init_from_checkpoint: str | None = None
    output_model_id: str | None = None
    save_final_checkpoint: bool = True
    step_timeout: int = 0

    # -- Sub-configs --------------------------------------------------------

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="grpo-tinker"))
    runner: RunnerConfig = field(default_factory=RunnerConfig)


# ---------------------------------------------------------------------------
# Reward function -- customise this for your task
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> Optional[str]:
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


# ---------------------------------------------------------------------------
# Rollout filter -- customise this for your task
# ---------------------------------------------------------------------------


def should_accept(pg: PromptGroup) -> bool:
    """Reject groups where all rewards are identical (zero-variance)."""
    return len(set(pg.rewards)) > 1


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
    runner = RunnerIO(cfg.runner)

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(
        cfg.base_model,
        cfg.dataset,
        cfg.weight_sync,
        cfg.deployment,
        output_model_id=cfg.output_model_id,
    )
    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step
    step_target = cfg.valid_prompt_groups_per_step or prompt_groups_per_step

    if step_target < 1:
        raise ValueError("valid_prompt_groups_per_step must be >= 1")
    if cfg.max_head_offpolicy_versions < 0:
        raise ValueError("max_head_offpolicy_versions must be >= 0")
    if not cfg.deployment.tokenizer_model:
        raise ValueError(
            "deployment.tokenizer_model is required for client-side tokenization."
        )

    # Streaming pipeline requires 1:1 weight sync.
    if cfg.weight_sync.weight_sync_interval != 1:
        logger.warning(
            "Async streaming pipeline forces weight_sync_interval=1 "
            "(was %d)", cfg.weight_sync.weight_sync_interval,
        )
        cfg.weight_sync.weight_sync_interval = 1

    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": completions_per_prompt,
            "prompt_groups_per_step": prompt_groups_per_step,
            "step_target": step_target,
            "max_head_offpolicy_versions": cfg.max_head_offpolicy_versions,
            "kl_beta": cfg.kl_beta,
            "lr": cfg.learning_rate,
        },
    )

    # -- Setup infrastructure -----------------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(
            api_key=api_key, base_url=base_url, additional_headers=additional_headers,
        )
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(
            api_key=api_key, base_url=base_url, additional_headers=additional_headers,
        )

    runner.write_status(RunStatus.PENDING, message="provisioning")

    def _on_trainer_status(msg: str) -> None:
        runner.write_status(RunStatus.PENDING, message=msg)

    with runner, ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup, ExitStack() as stack:
        infra = setup_infra(
            cfg,
            rlor_mgr=rlor_mgr,
            deploy_mgr=deploy_mgr,
            api_key=api_key,
            cleanup=cleanup,
            cleanup_on_exit=cleanup_on_exit,
            on_status=_on_trainer_status,
        )
        for closeable in infra.closeables:
            stack.callback(closeable.close)

        runner.set_accelerator_info(profile=infra.policy_profile)
        wandb_log(infra.boot_metrics, step=0)

        policy = infra.policy
        reference = infra.reference
        sampler = infra.sampler
        weight_syncer = infra.weight_syncer

        # -- Resume ---------------------------------------------------------

        resume_info = resolve_resume(policy, cfg.log_path, cfg.init_from_checkpoint)
        step_offset = resume_info.step if resume_info else 0
        wandb_log({"train/step": step_offset}, step_offset)

        if cfg.weight_sync.weight_sync_before_training and cfg.deployment.deployment_id:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            weight_syncer.save_and_hotload(name, checkpoint_type="base")

        # -- Loss + adam -----------------------------------------------------

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
            infra.policy_profile,
            dapo_config=cfg.dapo,
            gspo_config=cfg.gspo,
            cispo_config=cfg.cispo,
            ratio_log_cap=cfg.ratio_log_cap,
            eps_clip=cfg.eps_clip,
            eps_clip_high=cfg.eps_clip_high,
        )

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
            policy_job_id=infra.policy_job_id,
            completions_per_prompt=completions_per_prompt,
            trajectory_dir=cfg.trajectory_dir,
            weight_sync_interval=1,
            dcp_save_interval=cfg.weight_sync.dcp_save_interval,
            wandb_log=wandb_log,
            log_metrics_json=log_metrics_json,
        )

        # -- Sampling closure (VISIBLE — customise this) --------------------

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=cfg.max_seq_len,
            http_timeout=cfg.deployment.sample_timeout,
            logprobs=True,
        )
        if cfg.router_replay:
            sample_kwargs.update(include_routing_matrix=True, echo=True)

        async def sample_one_prompt(row: dict) -> PromptGroup | None:
            messages = row.get("messages", [])
            input_messages = prepare_sampling_messages(messages)
            if not input_messages:
                return None

            try:
                sampled = await sampler.sample_with_tokens(
                    messages=input_messages,
                    n=completions_per_prompt,
                    **sample_kwargs,
                )
            except Exception as e:
                logger.warning("Sampling failed (%s): %s", type(e).__name__, e or repr(e))
                return None

            if not sampled or len(sampled) < completions_per_prompt:
                return None

            rewards = [reward_fn(s.text, row) for s in sampled]
            advantages = compute_advantages(rewards)

            prompt_len = sampled[0].prompt_len
            policy_data: List[tinker.Datum] = []
            reference_data: List[tinker.Datum] = []
            adv_filtered: List[float] = []
            inf_logprobs_aligned: List[List[float]] = []

            for idx, s in enumerate(sampled):
                tokens = s.full_tokens
                if len(tokens) < 2:
                    continue
                model_input_len = len(tokens) - 1

                rm = None
                if cfg.router_replay:
                    rm = build_r3_routing_matrices(
                        s.routing_matrices,
                        s.prompt_len,
                        model_input_len,
                        completion_only=cfg.router_replay_completion_only,
                    )

                policy_data.append(make_policy_datum(tokens, routing_matrices=rm))
                if reference is not None:
                    reference_data.append(make_reference_datum(tokens))

                adv_filtered.append(advantages[idx])
                inf_logprobs_aligned.append(
                    align_inference_logprobs(
                        s.inference_logprobs,
                        prompt_len=prompt_len,
                        total_len=model_input_len,
                        echoed=getattr(s, "logprobs_echoed", False),
                    )
                )

            if not policy_data:
                return None

            comp_lens = [len(s.full_tokens) - s.prompt_len for s in sampled]
            trunc = [s.finish_reason == "length" for s in sampled]

            return PromptGroup(
                data=policy_data,
                ref_data=reference_data,
                advantages=adv_filtered,
                ref_logprobs=None,
                prompt_len=prompt_len,
                rewards=rewards,
                inf_logprobs=inf_logprobs_aligned,
                completion_lens=comp_lens,
                truncated=trunc,
                prompt=input_messages if cfg.trajectory_dir else None,
                completions=[s.text for s in sampled] if cfg.trajectory_dir else None,
                row_meta={"ground_truth": row.get("ground_truth", "")} if cfg.trajectory_dir else None,
            )

        # -- Streaming loop body (VISIBLE — recipe owns this) ---------------

        raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        all_rows = raw_dataset * cfg.epochs
        async_state = (resume_info.async_state if resume_info else None) or {}

        async def _async_loop() -> int:
            scheduler = AsyncRolloutScheduler(
                step_target=step_target,
                max_head_offpolicy_versions=cfg.max_head_offpolicy_versions,
                filter_fn=should_accept,
                global_step=step_offset,
                total_accepted=async_state.get("total_accepted", 0),
                total_rejected=async_state.get("total_rejected", 0),
                rows_submitted=async_state.get("rows_submitted", 0),
                max_concurrent=cfg.sample_max_concurrency,
            )
            row_iter = iter(all_rows)
            # Skip rows that resumed checkpoints already submitted.
            skip = async_state.get("rows_submitted", 0)
            for _ in range(min(skip, len(all_rows))):
                next(row_iter, None)

            step = step_offset
            total_steps = max(0, len(all_rows) // step_target)
            runner.start_training()
            runner.write_status(
                RunStatus.RUNNING, total_steps=total_steps, message="training",
            )

            while not scheduler.data_exhausted:
                groups: list[PromptGroup] = []
                fwd_results: list = []
                version_offsets: list[int] = []
                raw_rewards: list[float] = []
                t0 = _time.time()

                async for group, version in scheduler.stream_groups(sample_one_prompt, row_iter):
                    groups.append(group)
                    version_offsets.append(scheduler.current_version - version)
                    raw_rewards.extend(group.rewards)

                    # Train this group now — server accumulates gradients.
                    fwd_results.append(ref_fwd_bwd(ctx, group))
                    logger.info(
                        "[async step %d] trained group %d/%d",
                        step + 1, len(groups), step_target,
                    )

                if not groups:
                    break

                wall_time = _time.time() - t0
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
                    fwd_results,
                    loop_stats=loop_stats,
                    save_checkpoint_fn=lambda name, extra: save_checkpoint(
                        ctx.policy,
                        name,
                        cfg.log_path,
                        {**extra, "async_state": scheduler.get_state()},
                        kind=CheckpointKind.STATE,
                    ),
                    step_target=step_target,
                    resume_data_consumed=resume_info.data_consumed if resume_info else 0,
                    step_offset=step_offset,
                )
                metrics["async/version"] = scheduler.current_version
                metrics["async/staleness_max"] = max(version_offsets, default=0)
                runner.append_metrics(step, metrics)
                runner.write_status(
                    RunStatus.RUNNING,
                    step=step,
                    total_steps=total_steps,
                    message="training",
                )
                scheduler.bump_version()

            return step

        global_step = asyncio.run(_async_loop())

        # -- Final checkpoint ----------------------------------------------

        if cfg.save_final_checkpoint and global_step > step_offset:
            try:
                data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    global_step - step_offset
                ) * step_target
                cp_name = f"step-{global_step}"
                paths = save_checkpoint(
                    policy,
                    cp_name,
                    cfg.log_path,
                    {
                        "step": global_step,
                        "data_consumed": data_consumed,
                        "source_job_id": ctx.policy_job_id,
                    },
                    kind=CheckpointKind.BOTH,
                    base_model=cfg.base_model,
                    training_shape=cfg.infra.training_shape_id,
                )
                if cfg.output_model_id:
                    rlor_mgr.promote_checkpoint(
                        ctx.policy_job_id,
                        paths["sampler_path"],
                        cfg.output_model_id,
                        cfg.base_model,
                    )
                    runner.write_output_model(
                        model_id=cfg.output_model_id,
                        checkpoint=cp_name,
                        job_id=ctx.policy_job_id,
                    )
            except Exception as e:
                logger.warning("Failed to save final checkpoint: %s", e)

        runner.write_status(
            RunStatus.COMPLETED, step=global_step, message="done",
        )
        runner.write_metadata()
        logger.info("Async training complete: %d steps", global_step)
        wandb_finish()
        return {
            "steps": global_step,
            "policy_job_id": ctx.policy_job_id,
            "reference_job_id": infra.reference_job_id,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = Config(
        log_path="./async_rl_logs",
        infra=InfraConfig(
            training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200",
        ),
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
    )
    main(cfg)
