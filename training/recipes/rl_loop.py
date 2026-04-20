#!/usr/bin/env python3
"""GRPO training loop with concurrent rollout — library-style recipe.

A self-contained training script. Reads top-to-bottom: Config → reward
→ filter → main(). The cookbook contributes only pure functions and
data shapes; this file owns the loop body (sampling orchestration,
forward/backward/optim_step sequence, weight sync). Fork it and edit.

Each optimizer step samples ``prompt_groups_per_step`` prompts
concurrently, accumulates valid groups (rejecting zero-variance), then
runs a single ``forward_backward_custom`` (or builtin
``forward_backward``) + ``optim_step`` (1:1 ratio). Weight sync fires
every ``weight_sync_interval`` steps.

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.rl_loop
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
import signal
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import List, Optional

import tinker
import transformers

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.client import GradAccNormalization
from fireworks.training.sdk.deployment import DeploymentSampler
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils import (
    DEFAULT_ADAM,
    ConcurrencyConfig,
    DeployConfig,
    InfraConfig,
    ReconnectableClient,
    ResourceCleanup,
    RLPromptDataset,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    WeightSyncConfig,
    compute_advantages,
    load_jsonl_dataset,
    log_metrics_json,
    prepare_sampling_messages,
    read_api_extra_headers_env,
    setup_or_reattach_deployment,
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
from training.utils.client import DEFAULT_TIMEOUT_S
from training.utils.rl import (
    PromptGroup,
    TrainContext,
    align_inference_logprobs,
    finish_step,
    make_concurrency_controller,
    make_policy_datum,
    make_reference_datum,
    provision_trainer_pair,
    ref_fwd_bwd,
    resolve_policy_profile,
    resolve_reference_profile,
)
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.losses import build_loss_fn, resolve_builtin_loss
from training.utils.rl.router_replay import build_r3_routing_matrices
from training.utils.rl.tis import TISConfig
from training.utils.timer import timer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

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
    """Auto-populated from the training shape's max_supported_context_length
    if left as None."""
    lora_rank: int = 0

    prompt_groups_per_step: int = 1
    """Number of accepted (non-filtered) prompt groups per optimizer step."""

    router_replay: bool = False
    router_replay_completion_only: bool = True

    grad_accumulation_normalization: GradAccNormalization | str | None = (
        GradAccNormalization.NUM_LOSS_TOKENS
    )

    policy_loss: str = "grpo"
    """``"grpo"``, ``"importance_sampling"``, ``"dapo"``, ``"dro"``,
    ``"gspo"``, ``"reinforce"``, or ``"cispo"``."""

    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    ratio_log_cap: float = 20.0
    tis: TISConfig = field(default_factory=TISConfig)

    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    trajectory_dir: str | None = None

    policy_job_id: str | None = None
    policy_base_url: str | None = None
    reference_job_id: str | None = None
    reference_base_url: str | None = None
    init_from_checkpoint: str | None = None

    output_model_id: str | None = None
    save_final_checkpoint: bool = True

    step_timeout: int = 0

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
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
        cfg.base_model, cfg.dataset, cfg.weight_sync, cfg.deployment,
        output_model_id=cfg.output_model_id,
    )
    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step
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

    # -- Setup infrastructure (direct one-shot builder calls) ---------------

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

    # Resolve training shapes — direct one-shot calls.
    cfg.infra.training_shape_id, policy_profile = resolve_policy_profile(
        rlor_mgr,
        shape_id=cfg.infra.training_shape_id,
        base_model=cfg.base_model,
        lora_rank=cfg.lora_rank,
        max_seq_len=cfg.max_seq_len,
    )
    if not cfg.deployment.deployment_shape and policy_profile.deployment_shape_version:
        cfg.deployment.deployment_shape = policy_profile.deployment_shape_version
    if cfg.max_seq_len is None:
        cfg.max_seq_len = policy_profile.max_supported_context_length
    if cfg.max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )
    cfg.infra.ref_training_shape_id, ref_profile = resolve_reference_profile(
        rlor_mgr,
        shape_id=cfg.infra.ref_training_shape_id,
        base_model=cfg.base_model,
        lora_rank=cfg.lora_rank,
        max_seq_len=cfg.max_seq_len,
        kl_beta=cfg.kl_beta,
    )
    logger.info(
        "Policy shape=%s  ref shape=%s  deployment_shape=%s",
        cfg.infra.training_shape_id, cfg.infra.ref_training_shape_id,
        cfg.deployment.deployment_shape,
    )

    runner.set_accelerator_info(profile=policy_profile)
    runner.write_status(RunStatus.PENDING, message="provisioning")

    def _on_trainer_status(msg: str) -> None:
        runner.write_status(RunStatus.PENDING, message=msg)

    boot_start = time.time()

    with runner, ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup, ExitStack() as stack:
        _cleanup = cleanup if cleanup_on_exit else None

        # Provision trainers (parallel when both needed).
        policy_ep, reference_ep = provision_trainer_pair(
            rlor_mgr,
            base_model=cfg.base_model,
            infra_config=cfg.infra,
            policy_profile=policy_profile,
            ref_profile=ref_profile,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            policy_job_id=cfg.policy_job_id,
            reference_job_id=cfg.reference_job_id,
            policy_base_url=cfg.policy_base_url,
            reference_base_url=cfg.reference_base_url,
            cleanup=_cleanup,
            on_status=_on_trainer_status,
        )
        policy_job_id = policy_ep.job_id
        reference_job_id = reference_ep.job_id if reference_ep else policy_ep.job_id

        # Connect deployment to the policy trainer's hot-load bucket.
        dep_info = setup_or_reattach_deployment(
            deploy_mgr, cfg.deployment, cfg.base_model, cfg.infra, policy_ep.job_name,
        )
        if cleanup_on_exit:
            cleanup.deployment(cfg.deployment.deployment_id, action="scale_to_zero")

        # Trainer clients (close registered onto stack for shutdown).
        timeout = cfg.step_timeout or DEFAULT_TIMEOUT_S

        def _make_client(ep):
            c = ReconnectableClient(
                rlor_mgr, ep.job_id, cfg.base_model,
                lora_rank=cfg.lora_rank,
                fw_api_key=api_key,
                default_timeout=timeout,
                endpoint=ep if (cfg.policy_base_url or cfg.reference_base_url) else None,
            )
            if hasattr(c, "close"):
                stack.callback(c.close)
            return c

        policy = _make_client(policy_ep)
        if reference_ep is not None:
            reference = _make_client(reference_ep)
        elif cfg.lora_rank > 0 and cfg.kl_beta > 0:
            # Share the policy trainer's session: base-only model handle.
            reference = policy.create_base_reference()
            stack.callback(reference.close)
        else:
            reference = None

        inference_model = dep_info.inference_model if dep_info else cfg.base_model
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.deployment.tokenizer_model, trust_remote_code=True,
        )

        # Concurrency controller and sampler.
        concurrency_controller = make_concurrency_controller(
            cfg.concurrency, deploy_mgr, cfg.deployment,
        )
        sampler = DeploymentSampler(
            inference_url=deploy_mgr.inference_url,
            model=inference_model,
            api_key=api_key,
            tokenizer=tokenizer,
            concurrency_controller=concurrency_controller,
        )
        weight_syncer = WeightSyncer(
            policy_client=policy.inner,
            deploy_mgr=deploy_mgr,
            deployment_id=cfg.deployment.deployment_id,
            base_model=cfg.rollout_base_model or cfg.base_model,
            hotload_timeout=cfg.weight_sync.weight_sync_timeout,
            first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
            lora_rank=cfg.lora_rank,
        )

        boot_metrics: dict = {
            "train/step": 0,
            "infra/total_boot_time": time.time() - boot_start,
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

        # -- Loss + dataset + adam -----------------------------------------

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
            cfg.policy_loss, policy_profile,
            dapo_config=cfg.dapo, gspo_config=cfg.gspo, cispo_config=cfg.cispo,
            ratio_log_cap=cfg.ratio_log_cap,
            eps_clip=cfg.eps_clip, eps_clip_high=cfg.eps_clip_high,
        )

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=cfg.max_seq_len,
            http_timeout=cfg.deployment.sample_timeout,
            logprobs=True,
        )
        if cfg.router_replay:
            sample_kwargs.update(include_routing_matrix=True, echo=True)

        # -- Sample one prompt (VISIBLE -- customise this) -----------------

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
                logger.warning("Sampling failed: %s", e)
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
                        s.routing_matrices, s.prompt_len, model_input_len,
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

        # -- TrainContext: bundle handed to ref_fwd_bwd / finish_step ------

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
            # Hotload from finish_step at the configured interval (1:1 by default).
            weight_sync_interval=cfg.weight_sync.weight_sync_interval,
            dcp_save_interval=cfg.weight_sync.dcp_save_interval,
            wandb_log=wandb_log,
            log_metrics_json=log_metrics_json,
        )

        # -- Training step body (VISIBLE — direct primitive calls) ----------

        def _train_one_step(
            step: int,
            groups: list[PromptGroup],
            stats: dict,
        ) -> int:
            """One optimizer step.

            For each group: ``ref_fwd_bwd(ctx, group)`` runs ref_forward +
            policy_forward + forward_backward (server accumulates gradients).
            Then ``finish_step`` fires optim_step + (optional) hotload +
            metrics + (optional) DCP save.
            """
            fwd_results = [ref_fwd_bwd(ctx, g) for g in groups]

            def _save(name: str, extra: dict) -> object:
                return save_checkpoint(
                    policy, name, cfg.log_path, extra,
                    kind=CheckpointKind.STATE,
                    base_model=cfg.base_model,
                    training_shape=cfg.infra.training_shape_id,
                )

            step, metrics = finish_step(
                ctx, step, groups, fwd_results, loop_stats=stats,
                save_checkpoint_fn=_save,
                step_target=prompt_groups_per_step,
                resume_data_consumed=resume_info.data_consumed if resume_info else 0,
                step_offset=step_offset,
            )

            # Recipe-specific telemetry that finish_step doesn't know about.
            if concurrency_controller is not None:
                cc_summary = concurrency_controller.step_completed()
                for k, v in cc_summary.items():
                    metrics[f"concurrency/{k}"] = v
            step_tokens = sum(
                len(d.loss_fn_inputs["target_tokens"].data)
                for pg in groups for d in pg.data
            )
            total_steps_target = len(rl_dataset) - step_offset
            runner.append_metrics(step, metrics, tokens=step_tokens)
            runner.write_status(
                RunStatus.RUNNING, step=step, total_steps=total_steps_target,
                message="training",
            )
            runner.write_metadata()
            return step

        # -- Outer loop: window-bounded sampling/training, hotload between -

        remaining_rows: list[dict] = []
        for i_step in range(step_offset, len(rl_dataset)):
            remaining_rows.extend(rl_dataset.get_batch(i_step))

        total_rl_steps = len(rl_dataset) - step_offset
        runner.start_training()
        runner.write_status(
            RunStatus.RUNNING, total_steps=total_rl_steps, message="training",
        )

        async def _run() -> int:
            """Outer loop: iterate weight-sync windows, sample+train inside each."""
            sync_interval = cfg.weight_sync.weight_sync_interval
            window_size = (
                sync_interval * prompt_groups_per_step
                if sync_interval > 0
                else len(remaining_rows)
            )
            row_iter = iter(remaining_rows)
            step = step_offset
            cumulative_fails = 0
            cumulative_drops = 0

            while True:
                window_rows = list(itertools.islice(row_iter, window_size))
                if not window_rows:
                    break

                # Fire all sampling tasks in this window in parallel.
                tasks = [asyncio.create_task(sample_one_prompt(r)) for r in window_rows]
                pending: list[PromptGroup] = []
                fails = 0
                drops = 0
                rewards_seen: list[float] = []
                step_before = step
                window_start = time.time()

                # Drain results in completion order; train as soon as a batch fills.
                for fut in asyncio.as_completed(tasks):
                    pg = await fut
                    if pg is None:
                        fails += 1
                        continue
                    rewards_seen.extend(pg.rewards)
                    if not should_accept(pg):
                        drops += 1
                        continue
                    pending.append(pg)

                    if len(pending) >= prompt_groups_per_step:
                        batch = pending[:prompt_groups_per_step]
                        pending = pending[prompt_groups_per_step:]
                        wall = time.time() - window_start
                        stats = {
                            "valid_prompt_groups": len(batch),
                            "total_sampled": fails + drops + len(batch),
                            "filter_drops": drops,
                            "sample_fails": fails,
                            "sample_wait_time": 0.0,
                            "step_wall_time": wall,
                            "all_raw_rewards": list(rewards_seen),
                        }
                        # Train off-thread so we keep awaiting remaining tasks.
                        step = await asyncio.to_thread(_train_one_step, step, batch, stats)
                        cumulative_fails += fails
                        cumulative_drops += drops
                        fails = drops = 0
                        rewards_seen = []
                        window_start = time.time()

                # Partial batch at end of window (e.g. dataset exhausted mid-window).
                if pending:
                    wall = time.time() - window_start
                    stats = {
                        "valid_prompt_groups": len(pending),
                        "total_sampled": fails + drops + len(pending),
                        "filter_drops": drops,
                        "sample_fails": fails,
                        "sample_wait_time": 0.0,
                        "step_wall_time": wall,
                        "all_raw_rewards": list(rewards_seen),
                    }
                    step = await asyncio.to_thread(_train_one_step, step, pending, stats)
                    cumulative_fails += fails
                    cumulative_drops += drops

                # No window-end hotload — finish_step already fires
                # save_and_hotload at ctx.weight_sync_interval cadence.
                _ = step_before  # retained for future window-level diagnostics

            # End-of-run summary.
            total_prompts = len(remaining_rows)
            if total_prompts > 0:
                rejected = cumulative_fails + cumulative_drops
                logger.info(
                    "RL loop complete: %d steps, %d/%d prompts sampled "
                    "(%d filtered, %d failed)",
                    step, total_prompts - rejected, total_prompts,
                    cumulative_drops, cumulative_fails,
                )
            return step

        global_step = asyncio.run(_run())

        # -- Final checkpoint ----------------------------------------------

        if cfg.save_final_checkpoint and global_step > step_offset:
            try:
                data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    global_step - step_offset
                ) * prompt_groups_per_step
                cp_name = f"step-{global_step}"
                paths = save_checkpoint(
                    policy, cp_name, cfg.log_path,
                    {
                        "step": global_step,
                        "data_consumed": data_consumed,
                        "source_job_id": policy_job_id,
                    },
                    kind=CheckpointKind.BOTH,
                    base_model=cfg.base_model,
                    training_shape=cfg.infra.training_shape_id,
                )
                if getattr(cfg, "output_model_id", None):
                    rlor_mgr.promote_checkpoint(
                        policy_job_id,
                        paths["sampler_path"],
                        cfg.output_model_id,
                        cfg.base_model,
                    )
                    runner.write_output_model(
                        model_id=cfg.output_model_id,
                        checkpoint=cp_name, job_id=policy_job_id,
                    )
            except Exception as e:
                logger.warning("Failed to save final checkpoint: %s", e)

            runner.write_status(
                RunStatus.COMPLETED, step=global_step, total_steps=total_rl_steps,
                message="done",
            )
            runner.write_metadata()
            logger.info("Training complete: %d steps", global_step)
            wandb_finish()
            return {
                "steps": global_step,
                "policy_job_id": policy_job_id,
                "reference_job_id": reference_job_id,
            }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = Config(
        log_path="./rl_logs",
        base_model="accounts/fireworks/models/qwen3-8b",
        infra=InfraConfig(
            training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200",
        ),
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
    )
    main(cfg)
