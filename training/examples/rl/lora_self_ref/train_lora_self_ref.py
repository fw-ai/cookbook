#!/usr/bin/env python3
"""LoRA RL training with self-reference: no separate reference trainer needed.

Demonstrates RL (GRPO) with LoRA where the **same** policy trainer provides
both the LoRA-adapted policy model and the frozen base model used for KL
divergence reference logprobs.  This eliminates the need for a separate
FORWARD_ONLY reference trainer job, halving GPU cost for reference forward
passes.

How it works
------------
When training with LoRA, the base model weights are frozen — only the LoRA
adapter is trained.  The "reference model" for KL divergence is just the
base model *without* the LoRA adapter.  The RLOR trainer service supports
this via ``create_base_training_client()``, which creates a ``base-<hex>``
model handle.  Forward passes through this handle run with all LoRA adapters
disabled, giving reference logprobs from the frozen base weights.

Compared to the standard ``rl_loop.py`` recipe
-----------------------------------------------
- **One trainer job** instead of two (policy + reference).
- **No reference training shape** needed.
- ``ReconnectableClient(base_only=True)`` replaces the separate reference
  ``ReconnectableClient`` backed by a FORWARD_ONLY trainer.
- Everything else (sampling, rewards, loss computation, weight sync) stays
  identical.

Usage:
    export FIREWORKS_API_KEY=...
    python -m training.examples.rl.lora_self_ref.train_lora_self_ref
"""

from __future__ import annotations

import os
import re
import json
import signal
import asyncio
import logging
from contextlib import ExitStack
from typing import List, Optional
from dataclasses import field, dataclass

import tinker

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.client import GradAccNormalization
from fireworks.training.sdk.deployment import DeploymentSampler
from fireworks.training.sdk.deployment import AdaptiveConcurrencyController
from fireworks.training.sdk.weight_syncer import WeightSyncer

from training.utils import (
    DEFAULT_ADAM,
    InfraConfig,
    ResourceCleanup,
    RunnerConfig,
    RunnerIO,
    RunStatus,
    WandBConfig,
    DeployConfig,
    WeightSyncConfig,
    ReconnectableClient,
    RLPromptDataset,
    wandb_log,
    setup_wandb,
    wandb_finish,
    log_metrics_json,
    get_deployment_gpu_count,
    setup_deployment,
    compute_advantages,
    create_trainer_job,
    load_jsonl_dataset,
    prepare_sampling_messages,
    ShapeSelectionRequest,
    materialize_profile_infra,
    select_validated_launch_shapes,
)
from training.utils.checkpoint_utils import (
    resolve_resume,
    save_checkpoint,
    CheckpointKind,
)
from training.utils.rl import PromptGroup
from training.utils.rl.tis import TISConfig
from training.utils.rl.train import TrainStepFns, run_rl_loop
from training.utils.rl.losses import (
    build_builtin_loss_datums,
    build_loss_fn,
    combine_prompt_groups,
    resolve_builtin_loss,
)
from training.utils.rl.metrics import compute_step_metrics
from training.utils.timer import timer, flush_timing

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs. Required, no default."""

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    dataset: str = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"

    learning_rate: float = 1e-5
    kl_beta: float = 0.001
    completions_per_prompt: int = 4
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    epochs: int = 1
    max_rows: int = 100
    max_seq_len: int | None = None
    lora_rank: int = 32
    """LoRA rank.  Must be > 0 for self-reference to work."""

    prompt_groups_per_step: int = 1
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    ratio_log_cap: float = 20.0
    tis: TISConfig = field(default_factory=TISConfig)

    grad_accumulation_normalization: GradAccNormalization | str | None = GradAccNormalization.NUM_LOSS_TOKENS
    policy_loss: str = "grpo"

    policy_job_id: str | None = None
    """Pre-created RLOR policy trainer job ID (skip creation if set)."""
    policy_base_url: str | None = None

    init_from_checkpoint: str | None = None
    output_model_id: str | None = None
    save_final_checkpoint: bool = True

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="grpo-lora-self-ref"))
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
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


def should_accept(pg: PromptGroup) -> bool:
    return len(set(pg.rewards)) > 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
    deploy_mgr: DeploymentManager | None = None,
):
    cfg = config
    runner = RunnerIO(cfg.runner)

    if cfg.lora_rank <= 0:
        raise ValueError(
            "lora_rank must be > 0 for self-reference RL.  The base model "
            "(without LoRA adapter) serves as the KL reference."
        )

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name
        logger.warning("Received %s — raising SystemExit for cleanup", name)
        raise SystemExit(f"Terminated by {name}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step

    if not cfg.deployment.tokenizer_model:
        raise ValueError("deployment.tokenizer_model is required.")

    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": completions_per_prompt,
            "prompt_groups_per_step": prompt_groups_per_step,
            "kl_beta": cfg.kl_beta,
            "lr": cfg.learning_rate,
            "lora_rank": cfg.lora_rank,
            "self_reference": True,
        },
    )

    # -- Setup infrastructure -----------------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)

    # -- Resolve training shapes (policy only -- no reference shape needed) --

    policy_selection = select_validated_launch_shapes(
        rlor_mgr,
        deploy_mgr=deploy_mgr,
        request=ShapeSelectionRequest(
            base_model=cfg.base_model,
            max_seq_len=cfg.max_seq_len,
            trainer_role="policy",
            needs_deployment=True,
            lora_rank=cfg.lora_rank,
            explicit_training_shape_id=cfg.infra.training_shape_id,
            explicit_deployment_shape=cfg.deployment.deployment_shape,
        ),
    )
    if policy_selection.training_shape_id:
        cfg.infra.training_shape_id = policy_selection.training_shape_id
    if policy_selection.deployment_shape:
        cfg.deployment.deployment_shape = policy_selection.deployment_shape

    profile = policy_selection.training_profile
    policy_infra = materialize_profile_infra(cfg.infra, profile) if profile else cfg.infra

    if profile and cfg.max_seq_len is None:
        cfg.max_seq_len = profile.max_supported_context_length
        logger.info("max_seq_len from training shape: %d", cfg.max_seq_len)
    if cfg.max_seq_len is None:
        raise ValueError("max_seq_len is required.")

    # NOTE: No reference shape selection — the policy trainer handles both
    # roles via create_base_training_client().
    use_reference = cfg.kl_beta > 0
    logger.info(
        "LoRA self-reference mode: lora_rank=%d, kl_beta=%.4f, "
        "reference handled by same trainer (no FORWARD_ONLY job)",
        cfg.lora_rank,
        cfg.kl_beta,
    )

    import time as _time

    runner.set_accelerator_info(
        policy_infra.accelerator_type,
        policy_infra.accelerator_count,
        profile=profile,
    )
    runner.write_status(RunStatus.PENDING, message="provisioning")

    def _on_trainer_status(msg: str) -> None:
        runner.write_status(RunStatus.PENDING, message=msg)

    _infra_start = _time.time()

    with runner, ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup, ExitStack() as stack:
        # -- Create ONE trainer job (policy + self-reference) ------------------

        policy_ep = create_trainer_job(
            rlor_mgr,
            base_model=cfg.base_model,
            infra=policy_infra,
            profile=profile,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            display_name="grpo-lora-policy",
            job_id=cfg.policy_job_id,
            base_url_override=cfg.policy_base_url,
            cleanup=cleanup,
            on_status=_on_trainer_status,
        )
        policy_job_id = policy_ep.job_id

        # -- Create deployment ------------------------------------------------

        cfg.deployment.hot_load_trainer_job = policy_ep.job_name
        dep_info = setup_deployment(deploy_mgr, cfg.deployment, cfg.base_model, cfg.infra)

        # -- Create policy client (LoRA adapter) ------------------------------

        policy = ReconnectableClient(
            rlor_mgr,
            policy_ep.job_id,
            cfg.base_model,
            cfg.lora_rank,
            fw_api_key=api_key,
            endpoint=policy_ep if cfg.policy_base_url else None,
        )
        if hasattr(policy, "close"):
            stack.callback(policy.close)

        # -- Create reference client (base model, no LoRA adapter) ------------
        # Reuses the SAME service session as the policy client.  The
        # base-only model handle runs forward() with LoRA adapters disabled,
        # giving frozen-base-weight logprobs for KL divergence.

        reference = policy.create_base_reference() if use_reference else None

        import transformers

        inference_model = dep_info.inference_model if dep_info else cfg.base_model
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.deployment.tokenizer_model, trust_remote_code=True
        )

        # -- Concurrency controller -------------------------------------------

        gpu_count = get_deployment_gpu_count(deploy_mgr, cfg.deployment)
        _SLOTS_PER_GPU = 8
        initial_window = _SLOTS_PER_GPU * gpu_count
        concurrency_controller = AdaptiveConcurrencyController(
            initial_window=initial_window,
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
            base_model=cfg.base_model,
            hotload_timeout=cfg.weight_sync.weight_sync_timeout,
            first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
            lora_rank=cfg.lora_rank,
        )

        infra_boot_time = _time.time() - _infra_start
        wandb_log({"train/step": 0, "infra/total_boot_time": infra_boot_time}, step=0)

        # -- Resume -----------------------------------------------------------

        resume_info = resolve_resume(policy, cfg.log_path, cfg.init_from_checkpoint)
        step_offset = resume_info.step if resume_info else 0
        wandb_log({"train/step": step_offset}, step_offset)

        if cfg.weight_sync.weight_sync_before_training and cfg.deployment.deployment_id:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            weight_syncer.save_and_hotload(name, checkpoint_type="base")

        # -- Prepare sampling and training ------------------------------------

        raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        all_rows = raw_dataset * cfg.epochs
        rl_dataset = RLPromptDataset(all_rows, prompts_per_step=prompt_groups_per_step)
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)

        client_loss_builder = build_loss_fn(
            policy_loss=cfg.policy_loss,
            kl_beta=cfg.kl_beta,
            ratio_log_cap=cfg.ratio_log_cap,
            tis_config=cfg.tis,
            eps_clip=cfg.eps_clip,
            eps_clip_high=cfg.eps_clip_high,
        )

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=cfg.max_seq_len,
            logprobs=True,
        )

        # -- Sample one prompt ------------------------------------------------

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

                policy_datum = tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData(
                            data=tokens[1:], dtype="int64", shape=[model_input_len]
                        ),
                    },
                )
                policy_data.append(policy_datum)

                if use_reference:
                    reference_datum = tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData(
                                data=tokens[1:], dtype="int64", shape=[model_input_len]
                            ),
                        },
                    )
                    reference_data.append(reference_datum)

                adv_filtered.append(advantages[idx])

                if not s.inference_logprobs:
                    raise RuntimeError(
                        f"Inference logprobs required but sample {idx} has none."
                    )
                response_start = max(0, prompt_len - 1)
                echoed = getattr(s, "logprobs_echoed", False)
                aligned = (
                    list(s.inference_logprobs)
                    if echoed
                    else [0.0] * response_start + list(s.inference_logprobs)
                )
                inf_logprobs_aligned.append(aligned)

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
            )

        # -- Training callbacks ------------------------------------------------

        def ref_forward(groups: list[PromptGroup]) -> None:
            """Compute reference logprobs via the base model (same trainer)."""
            if not use_reference or reference is None:
                return
            all_ref_data = [d for pg in groups for d in pg.ref_data]
            if not all_ref_data:
                return
            ref_fwd = reference.forward(all_ref_data, "cross_entropy")
            idx = 0
            for pg in groups:
                n = len(pg.ref_data)
                pg.ref_logprobs = [
                    ref_fwd.loss_fn_outputs[idx + i]["logprobs"].data
                    for i in range(n)
                ]
                idx += n

        builtin_server_loss = resolve_builtin_loss(
            cfg.policy_loss,
            profile,
            ratio_log_cap=cfg.ratio_log_cap,
            eps_clip=cfg.eps_clip,
            eps_clip_high=cfg.eps_clip_high,
        )

        def fwd_bwd_one(prompt_groups: list[PromptGroup]):
            if not prompt_groups:
                raise ValueError("fwd_bwd_one requires at least one prompt group")

            data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(prompt_groups)

            t0 = _time.time()
            prox_fwd = policy.forward(data, "cross_entropy")
            prox_lp = [prox_fwd.loss_fn_outputs[i]["logprobs"].data for i in range(len(data))]
            logger.info("policy_forward: done (%.1fs)", _time.time() - t0)

            t0 = _time.time()
            if builtin_server_loss is not None:
                kernel_loss, kernel_config = builtin_server_loss
                rl_datums = build_builtin_loss_datums(
                    data, adv, prox_lp, inf_lp, prompt_lens,
                    cfg.tis, policy_loss=cfg.policy_loss,
                )
                fwd_bwd_result = policy.forward_backward(
                    rl_datums, kernel_loss, loss_fn_config=kernel_config,
                )
            else:
                fwd_bwd_result = policy.forward_backward_custom(
                    data,
                    client_loss_builder(adv, ref_lp, prompt_lens, inf_lp, prox_lp),
                )
            logger.info("fwd_bwd: done (%.1fs)", _time.time() - t0)
            return fwd_bwd_result

        def train_step(
            step: int,
            prompt_groups: list[PromptGroup],
            loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            t0 = _time.time()
            ref_forward(prompt_groups)
            logger.info("[step %d] ref_forward (self-ref): done (%.1fs)", step + 1, _time.time() - t0)

            t0 = _time.time()
            fwd_bwd_result = fwd_bwd_one(prompt_groups)
            logger.info("[step %d] fwd_bwd: done (%.1fs)", step + 1, _time.time() - t0)

            t0 = _time.time()
            optim_result = policy.optim_step(
                adam_params,
                grad_accumulation_normalization=cfg.grad_accumulation_normalization,
            )
            step += 1
            logger.info("[step %d] optim_step: done (%.1fs)", step, _time.time() - t0)

            if cfg.weight_sync.dcp_save_interval > 0 and step % cfg.weight_sync.dcp_save_interval == 0:
                _data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    step - step_offset
                ) * prompt_groups_per_step
                save_checkpoint(
                    policy, f"step-{step}", cfg.log_path,
                    {"step": step, "data_consumed": _data_consumed, "source_job_id": policy_job_id},
                    kind=CheckpointKind.STATE,
                    base_model=cfg.base_model,
                    training_shape=cfg.infra.training_shape_id,
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
                "Step %d | Reward: %.3f | Acc: %.1f%% | KL: %.4f",
                step, avg_reward, avg_acc * 100, avg_kl,
            )
            log_metrics_json(step, reward=avg_reward, accuracy=avg_acc, kl=avg_kl)
            wandb_log(metrics, step)

            total_rl_steps = len(rl_dataset) - step_offset
            runner.append_metrics(step, metrics, tokens=0)
            runner.write_status(
                RunStatus.RUNNING, step=step, total_steps=total_rl_steps, message="training",
            )
            runner.write_metadata()

            return step, metrics

        # -- Run ---------------------------------------------------------------

        def _weight_sync(step: int) -> None:
            logger.info("[step %d] weight_sync...", step)
            t0 = _time.time()
            with timer("weight_sync"):
                weight_syncer.save_and_hotload(f"step-{step}")
            logger.info("[step %d] weight_sync: done (%.1fs)", step, _time.time() - t0)

        def _loop_metrics_callback(loop_metrics: dict) -> None:
            if concurrency_controller is not None:
                cc_summary = concurrency_controller.step_completed()
                for k, v in cc_summary.items():
                    loop_metrics[f"concurrency/{k}"] = v
            wandb_log(loop_metrics, step=loop_metrics.get("train/step", 0))

        train_fns = TrainStepFns(train_step=train_step)

        remaining_rows = []
        for i_step in range(step_offset, len(rl_dataset)):
            remaining_rows.extend(rl_dataset.get_batch(i_step))

        total_rl_steps = len(rl_dataset) - step_offset
        runner.start_training()
        runner.write_status(RunStatus.RUNNING, total_steps=total_rl_steps, message="training")

        global_step = asyncio.run(
            run_rl_loop(
                sample_fns=(sample_one_prompt(row) for row in remaining_rows),
                train_fns=train_fns,
                prompt_groups_per_step=prompt_groups_per_step,
                dynamic_filter_fn=should_accept,
                global_step=step_offset,
                metrics_callback=_loop_metrics_callback,
                weight_sync_fn=_weight_sync if cfg.weight_sync.weight_sync_interval > 0 else None,
                weight_sync_interval=cfg.weight_sync.weight_sync_interval,
            )
        )

        # -- Final checkpoint --------------------------------------------------

        if cfg.save_final_checkpoint and global_step > step_offset:
            try:
                _data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    global_step - step_offset
                ) * prompt_groups_per_step
                cp_name = f"step-{global_step}"
                paths = save_checkpoint(
                    policy, cp_name, cfg.log_path,
                    {"step": global_step, "data_consumed": _data_consumed, "source_job_id": policy_job_id},
                    kind=CheckpointKind.BOTH,
                    base_model=cfg.base_model,
                    training_shape=cfg.infra.training_shape_id,
                )
                if getattr(cfg, "output_model_id", None):
                    rlor_mgr.promote_checkpoint(
                        policy_job_id, paths["sampler_path"], cfg.output_model_id, cfg.base_model,
                    )
                    runner.write_output_model(
                        model_id=cfg.output_model_id, checkpoint=cp_name, job_id=policy_job_id,
                    )
            except Exception as e:
                logger.warning("Failed to save final checkpoint: %s", e)

        runner.write_status(
            RunStatus.COMPLETED, step=global_step, total_steps=total_rl_steps, message="done",
        )
        runner.write_metadata()
        logger.info("Training complete: %d steps", global_step)
        wandb_finish()
        return {"steps": global_step, "policy_job_id": policy_job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        log_path="./rl_lora_self_ref_logs",
        base_model="accounts/fireworks/models/qwen3-8b",
        lora_rank=32,
        infra=InfraConfig(
            training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200",
        ),
        deployment=DeployConfig(
            tokenizer_model="Qwen/Qwen3-8B",
        ),
    )
    main(cfg)
