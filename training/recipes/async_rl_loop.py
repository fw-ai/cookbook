#!/usr/bin/env python3
"""Async RL recipe — gate-native off-policy, single sample-fn extension point.

End-to-end demonstration of :func:`run_async_rl_loop`: single-turn GRPO
sampling against a Fireworks deployment, packaged into a
:class:`PromptGroup` through the :class:`Trajectory` adapter, with explicit
off-policy budgeting via ``max_head_offpolicy_versions``.

Users customize one closure -- ``rollout_one_prompt`` -- to change what a
single rollout looks like.  Single-turn recipes write it with one call to
``sampler.sample_with_tokens``; multi-turn / agent / remote-grader setups
write their own loop inside that closure and hand back a
:class:`Trajectory` whose segments carry per-turn versions.

Usage::

    export FIREWORKS_API_KEY=...
    python -m training.recipes.async_rl_loop
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import time as _time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Optional

import tinker
import transformers

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.client import GradAccNormalization
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
    RLPromptDataset,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    WeightSyncConfig,
    load_jsonl_dataset,
    log_metrics_json,
    prepare_sampling_messages,
    read_api_extra_headers_env,
    setup_wandb,
    validate_config,
    wandb_log,
)
from training.utils.checkpoint_utils import (
    CheckpointKind,
    resolve_resume,
    save_checkpoint,
)
from training.utils.rl import PromptGroup, setup_infra
from training.utils.rl.async_train import run_async_rl_loop
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.losses import (
    build_builtin_loss_datums,
    build_loss_fn,
    combine_prompt_groups,
    resolve_builtin_loss,
)
from training.utils.rl.metrics import compute_step_metrics
from training.utils.rl.tis import TISConfig
from training.utils.rl.train import TrainStepFns
from training.utils.rl.trajectory import (
    CompletionSegment,
    Trajectory,
    trajectory_to_prompt_group,
)
from training.utils.timer import flush_timing, timer

logger = logging.getLogger(__name__)


@dataclass
class Config:
    log_path: str

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    rollout_base_model: str | None = None
    dataset: str = (
        "https://raw.githubusercontent.com/eval-protocol/python-sdk/"
        "main/development/gsm8k_sample.jsonl"
    )

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
    """Staleness budget, in optimizer-step versions.  ``0`` = strict on-policy.

    The async loop lets rollouts run up to this many versions behind the
    current policy before the gate blocks further submissions.  Raising this
    increases throughput at the cost of off-policy drift (bounded by TIS
    correction in the loss)."""

    sample_max_concurrency: int | None = None
    """Optional hard cap on in-flight rollouts (``None`` = gate-bounded only)."""

    weight_sync_interval: int = 1
    """Fire a weight-sync hotload every N optimizer steps."""

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
    trajectory_dir: str | None = None

    policy_job_id: str | None = None
    reference_job_id: str | None = None
    init_from_checkpoint: str | None = None
    output_model_id: str | None = None
    save_final_checkpoint: bool = True
    step_timeout: int = 0

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="rl-async"))
    runner: RunnerConfig = field(default_factory=RunnerConfig)


def _extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    digits = re.search(r"(-?\d+)", match.group(1))
    return digits.group(1) if digits else None


def reward_fn(completion: str, row: dict) -> float:
    predicted = _extract_answer(completion)
    truth = _extract_answer(str(row.get("ground_truth", "")))
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


def should_accept(pg: PromptGroup) -> bool:
    """Reject zero-variance groups (all rewards identical)."""
    return len(set(pg.rewards)) > 1


def _dump_trajectory(
    trajectory_dir: str, step: int, prompt_groups: list[PromptGroup],
) -> None:
    os.makedirs(trajectory_dir, exist_ok=True)
    path = os.path.join(trajectory_dir, f"step_{step:04d}.jsonl")
    n_records = 0
    with open(path, "w") as f:
        for pg_idx, pg in enumerate(prompt_groups):
            completions = pg.completions or []
            for comp_idx, comp_text in enumerate(completions):
                record = {
                    "step": step,
                    "prompt_group": pg_idx,
                    "completion_index": comp_idx,
                    "prompt": pg.prompt,
                    "completion": comp_text,
                    "reward": pg.rewards[comp_idx] if comp_idx < len(pg.rewards) else None,
                    "advantage": pg.advantages[comp_idx] if comp_idx < len(pg.advantages) else None,
                    "completion_len": (
                        pg.completion_lens[comp_idx]
                        if comp_idx < len(pg.completion_lens) else None
                    ),
                    "truncated": (
                        pg.truncated[comp_idx] if comp_idx < len(pg.truncated) else None
                    ),
                    "ground_truth": pg.row_meta.get("ground_truth") if pg.row_meta else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_records += 1


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
    deploy_mgr: DeploymentManager | None = None,
    cancel_on_exit: bool = False,
) -> None:
    cfg = config
    runner = RunnerIO(cfg.runner)

    def _signal_handler(signum, _):
        name = signal.Signals(signum).name
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    validate_config(
        cfg.base_model, cfg.dataset, cfg.weight_sync, cfg.deployment,
        output_model_id=cfg.output_model_id,
    )
    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required.")

    completions_per_prompt = cfg.completions_per_prompt
    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": completions_per_prompt,
            "prompt_groups_per_step": cfg.prompt_groups_per_step,
            "max_head_offpolicy_versions": cfg.max_head_offpolicy_versions,
            "kl_beta": cfg.kl_beta,
            "lr": cfg.learning_rate,
        },
    )

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
            rlor_mgr=rlor_mgr,
            deploy_mgr=deploy_mgr,
            base_model=cfg.base_model,
            infra_cfg=cfg.infra,
            deploy_cfg=cfg.deployment,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            step_timeout=cfg.step_timeout,
            policy_job_id=cfg.policy_job_id,
            reference_job_id=cfg.reference_job_id,
            needs_reference=(cfg.kl_beta > 0),
            needs_inference=True,
            role_prefix="rl-async",
            api_key=api_key,
            cleanup=cleanup if cancel_on_exit else None,
            on_status=_on_trainer_status,
        )
        for closeable in infra.closeables:
            stack.callback(closeable.close)

        runner.set_accelerator_info(profile=infra.policy_profile)
        wandb_log(infra.boot_metrics, step=0)

        policy = infra.policy
        reference = infra.reference
        policy_profile = infra.policy_profile
        policy_job_id = infra.policy_job_id

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
            base_model=cfg.rollout_base_model or cfg.base_model,
            hotload_timeout=cfg.weight_sync.weight_sync_timeout,
            first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
            lora_rank=cfg.lora_rank,
        )

        resume_info = resolve_resume(policy, cfg.log_path, cfg.init_from_checkpoint)
        step_offset = resume_info.step if resume_info else 0
        wandb_log({"train/step": step_offset}, step_offset)

        current_version = step_offset

        if cfg.weight_sync.weight_sync_before_training and infra.deployment_id:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            weight_syncer.save_and_hotload(name, checkpoint_type="base")

        raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        all_rows = raw_dataset * cfg.epochs
        rl_dataset = RLPromptDataset(all_rows, prompts_per_step=cfg.prompt_groups_per_step)

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

        # -- Rollout: one trajectory per row (single-turn) ---------------------
        async def rollout_one_prompt(row: dict) -> Trajectory | None:
            messages = row.get("messages", [])
            input_messages = prepare_sampling_messages(messages)
            if not input_messages:
                return None

            try:
                sampled = await sampler.sample_with_tokens(
                    messages=input_messages, n=completions_per_prompt, **sample_kwargs,
                )
            except Exception as e:
                logger.warning("Sampling failed: %s", e)
                return None

            if not sampled or len(sampled) < completions_per_prompt:
                return None

            prompt_len = sampled[0].prompt_len
            prompt_tokens = list(sampled[0].full_tokens[:prompt_len])

            completions: list[list[CompletionSegment]] = []
            rewards: list[float] = []
            version_at_submit = current_version
            for s in sampled:
                completion_tokens = list(s.full_tokens[prompt_len:])
                if not s.inference_logprobs:
                    raise RuntimeError(
                        "Deployment did not return logprobs. "
                        "Ensure sample_kwargs['logprobs']=True and the "
                        "deployment supports logprobs."
                    )
                inf_lp = list(s.inference_logprobs)
                # DeploymentSampler returns logprobs aligned with generated tokens
                # when echo=False (the default used here).
                if len(inf_lp) != len(completion_tokens):
                    raise RuntimeError(
                        f"Logprob length {len(inf_lp)} != completion length "
                        f"{len(completion_tokens)}. Cookbook requires pre-aligned "
                        "logprobs from the rollout source."
                    )
                completions.append([
                    CompletionSegment(
                        tokens=completion_tokens,
                        inference_logprobs=inf_lp,
                        version=version_at_submit,
                        finish_reason=s.finish_reason,
                        text=s.text,
                    )
                ])
                rewards.append(reward_fn(s.text, row))

            return Trajectory(
                prompt_tokens=prompt_tokens,
                completions=completions,
                rewards=rewards,
                prompt_messages=input_messages if cfg.trajectory_dir else None,
                row_meta={"ground_truth": row.get("ground_truth", "")}
                if cfg.trajectory_dir else None,
            )

        async def sample_one_prompt(row: dict) -> PromptGroup | None:
            traj = await rollout_one_prompt(row)
            if traj is None:
                return None
            return trajectory_to_prompt_group(
                traj,
                with_reference=(reference is not None),
                persist_raw=bool(cfg.trajectory_dir),
            )

        # -- Training callbacks ------------------------------------------------
        builtin_server_loss = resolve_builtin_loss(
            cfg.policy_loss, policy_profile,
            kl_beta=cfg.kl_beta,
            dapo_config=cfg.dapo, gspo_config=cfg.gspo, cispo_config=cfg.cispo,
            ratio_log_cap=cfg.ratio_log_cap,
            eps_clip=cfg.eps_clip, eps_clip_high=cfg.eps_clip_high,
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
            data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(prompt_groups)
            prox_fwd = policy.forward(data, "cross_entropy")
            prox_lp = [prox_fwd.loss_fn_outputs[i]["logprobs"].data for i in range(len(data))]
            if builtin_server_loss is not None:
                kernel_loss, kernel_config = builtin_server_loss
                rl_datums = build_builtin_loss_datums(
                    data, adv, prox_lp, inf_lp, prompt_lens,
                    cfg.tis, policy_loss=cfg.policy_loss,
                )
                return policy.forward_backward(
                    rl_datums, kernel_loss, loss_fn_config=kernel_config,
                )
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

            if cfg.weight_sync.dcp_save_interval > 0 and step % cfg.weight_sync.dcp_save_interval == 0:
                _dc = (resume_info.data_consumed if resume_info else 0) + (
                    step - step_offset
                ) * cfg.prompt_groups_per_step
                save_checkpoint(
                    policy, f"step-{step}", cfg.log_path,
                    {
                        "step": step, "data_consumed": _dc,
                        "source_job_id": policy_job_id,
                    },
                    kind=CheckpointKind.STATE,
                    base_model=cfg.base_model,
                    training_shape=infra.training_shape_id,
                )

            metrics = compute_step_metrics(
                prompt_groups=prompt_groups,
                fwd_bwd_results=[fwd_bwd_result],
                optim_result=optim_result,
                n_accum=1,
                timing_metrics=flush_timing(),
                loop_stats=loop_stats,
                completions_per_prompt=completions_per_prompt,
            )
            metrics["train/step"] = step
            avg_reward = metrics.get("rollout/reward", 0.0)
            avg_acc = metrics.get("rollout/accuracy", 0.0)
            avg_kl = metrics.get("train/mean_kl", 0.0)
            logger.info(
                "Step %d | Reward %.3f | Acc %.1f%% | KL %.4f",
                step, avg_reward, avg_acc * 100, avg_kl,
            )
            log_metrics_json(step, reward=avg_reward, accuracy=avg_acc, kl=avg_kl)
            wandb_log(metrics, step)

            step_tokens = sum(
                len(d.loss_fn_inputs["target_tokens"].data)
                for pg in prompt_groups for d in pg.data
            )
            total_rl_steps = len(rl_dataset) - step_offset
            runner.append_metrics(step, metrics, tokens=step_tokens)
            runner.write_status(
                RunStatus.RUNNING, step=step, total_steps=total_rl_steps, message="training",
            )
            runner.write_metadata()

            if cfg.trajectory_dir:
                _dump_trajectory(cfg.trajectory_dir, step, prompt_groups)
            return step, metrics

        def _weight_sync(step: int) -> None:
            nonlocal current_version
            with timer("weight_sync"):
                weight_syncer.save_and_hotload(f"step-{step}")
            current_version = step

        def _metrics_cb(loop_metrics: dict) -> None:
            if concurrency_controller is not None:
                cc = concurrency_controller.step_completed()
                for k, v in cc.items():
                    loop_metrics[f"concurrency/{k}"] = v
            wandb_log(loop_metrics, step=loop_metrics.get("train/step", 0))

        train_fns = TrainStepFns(train_step=train_step)

        remaining_rows: list[dict] = []
        for i_step in range(step_offset, len(rl_dataset)):
            remaining_rows.extend(rl_dataset.get_batch(i_step))

        total_rl_steps = len(rl_dataset) - step_offset
        runner.start_training()
        runner.write_status(RunStatus.RUNNING, total_steps=total_rl_steps, message="training")

        global_step = asyncio.run(
            run_async_rl_loop(
                sample_fns=(sample_one_prompt(row) for row in remaining_rows),
                train_fns=train_fns,
                prompt_groups_per_step=cfg.prompt_groups_per_step,
                max_head_offpolicy_versions=cfg.max_head_offpolicy_versions,
                weight_sync_fn=(
                    _weight_sync if cfg.weight_sync.weight_sync_interval > 0 else None
                ),
                weight_sync_interval=cfg.weight_sync_interval,
                max_concurrent=cfg.sample_max_concurrency,
                dynamic_filter_fn=should_accept,
                global_step=step_offset,
                metrics_callback=_metrics_cb,
            )
        )

        if cfg.save_final_checkpoint and global_step > step_offset:
            try:
                _dc = (resume_info.data_consumed if resume_info else 0) + (
                    global_step - step_offset
                ) * cfg.prompt_groups_per_step
                cp_name = f"step-{global_step}"
                paths = save_checkpoint(
                    policy, cp_name, cfg.log_path,
                    {
                        "step": global_step, "data_consumed": _dc,
                        "source_job_id": policy_job_id,
                    },
                    kind=CheckpointKind.BOTH,
                    base_model=cfg.base_model,
                    training_shape=infra.training_shape_id,
                )
                if cfg.output_model_id:
                    rlor_mgr.promote_checkpoint(
                        policy_job_id, paths["sampler_path"],
                        cfg.output_model_id, cfg.base_model,
                    )
                    runner.write_output_model(
                        model_id=cfg.output_model_id, checkpoint=cp_name,
                        job_id=policy_job_id,
                    )
            except Exception as e:
                logger.warning("Failed to save final checkpoint: %s", e)


if __name__ == "__main__":
    main(Config(log_path="/tmp/rl-async"))
