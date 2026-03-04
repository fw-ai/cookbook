#!/usr/bin/env python3
"""GRPO training loop with concurrent rollout.

A readable, modifiable RL training loop using the Fireworks RLOR API.
Fork this script and customise the reward function, loss, or sampling
strategy to fit your task.

Each optimizer step samples ``prompt_groups_per_step`` prompts concurrently,
then runs a single ``forward_backward_custom`` + ``optim_step`` (1:1 ratio).

Usage:
    export FIREWORKS_API_KEY=...
    export FIREWORKS_ACCOUNT_ID=...
    python -m recipes.rl_loop
"""

from __future__ import annotations

import os
import re
import asyncio
import logging
from typing import List, Optional
from dataclasses import field, dataclass
from concurrent.futures import ThreadPoolExecutor

import tinker

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from training.utils import (
    DEFAULT_ADAM,
    InfraConfig,
    WandBConfig,
    DeployConfig,
    ResumeConfig,
    HotloadConfig,
    ReconnectableClient,
    wandb_log,
    setup_wandb,
    setup_resume,
    wandb_finish,
    validate_config,
    log_metrics_json,
    setup_deployment,
    compute_advantages,
    create_trainer_job,
    load_jsonl_dataset,
)
from fireworks.training.sdk.deployment import DeploymentSampler
from training.utils.rl import ISConfig, PromptGroup
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils.rl.pp import compute_pp_recommendation
from training.utils.timer import timer, flush_timing
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.train import TrainStepFns, run_rl_loop
from training.utils.rl.losses import build_loss_fn, combine_prompt_groups
from training.utils.rl.metrics import compute_step_metrics
from training.utils.rl.router_replay import build_r3_routing_matrices

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
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
    """Max sequence length for sampling and training.  When using training
    shapes, this is auto-populated from the shape's
    ``max_supported_context_length``.  Can be set manually when not using
    shapes, or as an override with ``skip_validations=True``."""
    lora_rank: int = 0

    prompt_groups_per_step: int = 1
    """Number of prompt groups per optimizer step.

    All groups are collected before a single ``forward_backward_custom`` +
    ``optim_step`` pair fires (1:1 ratio)."""

    max_concurrent: int = 32
    """Cap on concurrent in-flight sampling requests to the inference server."""

    router_replay: bool = False
    router_replay_completion_only: bool = True

    policy_loss: str = "grpo"
    """``"grpo"``, ``"dapo"``, ``"gspo"``, or ``"cispo"``."""

    tis_enabled: bool = False
    tis: ISConfig = field(default_factory=ISConfig)
    dapo: DAPOConfig = field(default_factory=DAPOConfig)
    gspo: GSPOConfig = field(default_factory=GSPOConfig)
    cispo: CISPOConfig = field(default_factory=CISPOConfig)

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    hotload: HotloadConfig = field(default_factory=HotloadConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="grpo-tinker"))
    resume: ResumeConfig = field(default_factory=ResumeConfig)


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
    """Reject groups where all rewards are identical (zero-variance).

    Passed to ``run_rl_loop`` as a pluggable filter.  Replace with your
    own logic (e.g. minimum reward threshold, response length filter).
    """
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

    validate_config(cfg.base_model, cfg.dataset, cfg.hotload, cfg.deployment, cfg.infra, cfg.resume)
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

    # -- Setup infrastructure -----------------------------------------------

    api_key = os.environ["FIREWORKS_API_KEY"]
    account = os.environ.get("FIREWORKS_ACCOUNT_ID", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(api_key=api_key, account_id=account, base_url=base_url)
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(api_key=api_key, account_id=account, base_url=base_url)

    # -- Resolve training shapes -----------------------------------------------

    use_reference = cfg.kl_beta != 0
    if not use_reference:
        logger.info("kl_beta=0: skipping reference model creation")

    profile = None
    if cfg.infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)
        if profile.deployment_shape and not cfg.deployment.deployment_shape:
            cfg.deployment.deployment_shape = profile.deployment_shape

    if profile and profile.pipeline_parallelism > 1:
        pp_rec = compute_pp_recommendation(profile, completions_per_prompt)
        logger.info(
            "PP recommendation: set prompt_groups_per_step=%d for optimal efficiency (current=%d)",
            pp_rec.recommended_prompts_per_step,
            prompt_groups_per_step,
        )

    if profile and cfg.max_seq_len is None:
        cfg.max_seq_len = profile.max_supported_context_length
        logger.info("max_seq_len from training shape: %d", cfg.max_seq_len)
    if cfg.max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )

    ref_profile = profile
    if use_reference and cfg.infra.ref_training_shape_id:
        logger.info("Using separate ref training shape: %s", cfg.infra.ref_training_shape_id)
        ref_profile = rlor_mgr.resolve_training_profile(cfg.infra.ref_training_shape_id)

    import time as _time
    _infra_start = _time.time()

    policy_job_id: str | None = None
    reference_job_id: str | None = None

    try:
        dep_info = setup_deployment(deploy_mgr, cfg.deployment, cfg.base_model, cfg.infra)

        logger.info(
            "Training: prompt_groups_per_step=%d | completions_per_prompt=%d",
            prompt_groups_per_step,
            completions_per_prompt,
        )

        if use_reference:
            with ThreadPoolExecutor(max_workers=2) as pool:
                pol_fut = pool.submit(
                    create_trainer_job, rlor_mgr,
                    base_model=cfg.base_model, infra=cfg.infra, profile=profile,
                    lora_rank=cfg.lora_rank, max_seq_len=cfg.max_seq_len,
                    learning_rate=cfg.learning_rate,
                    display_name="grpo-policy",
                    hot_load_deployment_id=cfg.deployment.deployment_id,
                )
                ref_fut = pool.submit(
                    create_trainer_job, rlor_mgr,
                    base_model=cfg.base_model, infra=cfg.infra, profile=ref_profile,
                    lora_rank=cfg.lora_rank, max_seq_len=cfg.max_seq_len,
                    learning_rate=cfg.learning_rate,
                    display_name="grpo-reference", forward_only=True,
                )
                policy_ep = pol_fut.result()
                policy_job_id = policy_ep.job_id
                reference_ep = ref_fut.result()
                reference_job_id = reference_ep.job_id
        else:
            policy_ep = create_trainer_job(
                rlor_mgr,
                base_model=cfg.base_model, infra=cfg.infra, profile=profile,
                lora_rank=cfg.lora_rank, max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name="grpo-policy",
                hot_load_deployment_id=cfg.deployment.deployment_id,
            )
            policy_job_id = policy_ep.job_id
            reference_ep = None

        policy = ReconnectableClient(rlor_mgr, policy_ep.job_id, cfg.base_model, cfg.lora_rank)
        reference = (
            ReconnectableClient(rlor_mgr, reference_ep.job_id, cfg.base_model, cfg.lora_rank)
            if reference_ep else None
        )

        import transformers

        inference_model = dep_info.inference_model if dep_info else cfg.base_model
        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.deployment.tokenizer_model, trust_remote_code=True)
        sampler = DeploymentSampler(
            inference_url=deploy_mgr.inference_url,
            model=inference_model, api_key=api_key, tokenizer=tokenizer,
        )
        weight_syncer = WeightSyncer(
            policy_client=policy.inner, deploy_mgr=deploy_mgr,
            deployment_id=cfg.deployment.deployment_id, base_model=cfg.base_model,
            hotload_timeout=cfg.hotload.hot_load_timeout,
            first_checkpoint_type=cfg.hotload.first_checkpoint_type,
            dcp_timeout=cfg.hotload.dcp_timeout,
        )

        infra_boot_time = _time.time() - _infra_start
        boot_metrics: dict = {
            "train/step": 0,
            "infra/total_boot_time": infra_boot_time,
        }
        if deploy_mgr.boot_time_s is not None:
            boot_metrics["infra/deploy_boot_time"] = deploy_mgr.boot_time_s
        wandb_log(boot_metrics, step=0)

        step_offset, _ = setup_resume(policy, cfg.resume)
        if cfg.hotload.hot_load_before_training and cfg.deployment.deployment_id:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            weight_syncer.save_and_hotload(name, checkpoint_type="base")

        # -- Prepare sampling and training --------------------------------------

        dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)
        loss_builder = build_loss_fn(
            policy_loss=cfg.policy_loss, kl_beta=cfg.kl_beta,
            tis_enabled=cfg.tis_enabled, tis_config=cfg.tis,
            dapo_config=cfg.dapo, gspo_config=cfg.gspo,
            cispo_config=cfg.cispo,
        )

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens, temperature=cfg.temperature,
            max_seq_len=cfg.max_seq_len,
            http_timeout=cfg.deployment.sample_timeout,
        )
        if cfg.router_replay:
            sample_kwargs.update(include_routing_matrix=True, echo=True, logprobs=True)
        sample_kwargs["logprobs"] = True

        # -- Sample one prompt (VISIBLE -- customise this) ----------------------

        async def sample_one_prompt(row: dict) -> PromptGroup | None:
            """Sample completions for one prompt and return a PromptGroup."""
            messages = row.get("messages", [])
            input_messages = [m for m in messages if m.get("role") != "assistant"]
            if not input_messages:
                return None

            try:
                sampled = await asyncio.to_thread(
                    sampler.sample_with_tokens,
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

                policy_datum = tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(tokens[:-1], routing_matrices=rm),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData(data=tokens[1:], dtype="int64", shape=[model_input_len]),
                    },
                )
                policy_data.append(policy_datum)

                if use_reference:
                    reference_datum = tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData(data=tokens[1:], dtype="int64", shape=[model_input_len]),
                        },
                    )
                    reference_data.append(reference_datum)

                adv_filtered.append(advantages[idx])

                if not s.inference_logprobs:
                    raise RuntimeError(
                        f"Inference logprobs required but sample {idx} has none. "
                        f"Ensure the deployment returns logprobs."
                    )
                response_start = max(0, prompt_len - 1)
                echoed = getattr(s, "logprobs_echoed", False)
                aligned = (
                    list(s.inference_logprobs) if echoed else [0.0] * response_start + list(s.inference_logprobs)
                )
                inf_logprobs_aligned.append(aligned)

            if not policy_data:
                return None

            comp_lens = [len(s.full_tokens) - s.prompt_len for s in sampled]
            trunc = [s.finish_reason == "length" for s in sampled]

            return PromptGroup(
                data=policy_data, ref_data=reference_data,
                advantages=adv_filtered, ref_logprobs=[],
                prompt_len=prompt_len, rewards=rewards, inf_logprobs=inf_logprobs_aligned,
                completion_lens=comp_lens, truncated=trunc,
            )

        # -- Training callbacks ----------------------------------------------------

        def ref_forward(groups: list[PromptGroup]) -> None:
            """Compute reference logprobs for all prompt groups (one call)."""
            if not use_reference:
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

        def train_step(
            step: int,
            prompt_groups: list[PromptGroup],
            loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            """ref_forward + fwd_bwd + optim_step + hotload + metrics (1:1)."""
            import time as _t

            t0 = _t.time()
            ref_forward(prompt_groups)
            logger.info("[step %d] ref_forward: done (%.1fs)", step + 1, _t.time() - t0)

            data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(prompt_groups)
            t0 = _t.time()
            fwd_bwd_result = policy.forward_backward_custom(
                data, loss_builder(adv, ref_lp, prompt_lens, inf_lp),
            )
            logger.info("[step %d] fwd_bwd: done (%.1fs)", step + 1, _t.time() - t0)

            t0 = _t.time()
            optim_result = policy.optim_step(adam_params)
            step += 1
            logger.info("[step %d] optim_step: done (%.1fs)", step, _t.time() - t0)

            if cfg.hotload.hot_load_interval > 0 and step % cfg.hotload.hot_load_interval == 0:
                logger.info("[step %d] hotload: saving + loading...", step)
                t0 = _t.time()
                with timer("weight_sync"):
                    weight_syncer.save_and_hotload(f"step-{step}")
                logger.info("[step %d] hotload: done (%.1fs)", step, _t.time() - t0)
            if cfg.hotload.dcp_save_interval > 0 and step % cfg.hotload.dcp_save_interval == 0:
                logger.info("[step %d] dcp_save...", step)
                t0 = _t.time()
                with timer("dcp_save"):
                    weight_syncer.save_dcp(f"step-{step}")
                logger.info("[step %d] dcp_save: done (%.1fs)", step, _t.time() - t0)

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
            return step, metrics

        # -- Run ----------------------------------------------------------------

        def _loop_metrics_callback(loop_metrics: dict) -> None:
            """Called by run_rl_loop after each train step with loop-level metrics."""
            wandb_log(loop_metrics, step=loop_metrics.get("train/step", 0))

        train_fns = TrainStepFns(train_step=train_step)

        all_rows = dataset * cfg.epochs

        global_step = asyncio.run(run_rl_loop(
            sample_fns=(sample_one_prompt(row) for row in all_rows),
            train_fns=train_fns,
            prompt_groups_per_step=prompt_groups_per_step,
            max_concurrent=cfg.max_concurrent,
            dynamic_filter_fn=should_accept,
            global_step=step_offset,
            metrics_callback=_loop_metrics_callback,
        ))

        # -- Final checkpoint ----------------------------------------------------

        if global_step > step_offset:
            try:
                policy.save_state(f"step-{global_step}", timeout=cfg.hotload.dcp_timeout)
            except Exception as e:
                logger.warning("Failed to save final checkpoint: %s", e)

            logger.info("Training complete: %d steps", global_step)
            return {
                "steps": global_step,
                "policy_job_id": policy_job_id,
                "reference_job_id": reference_job_id,
            }
    finally:
        wandb_finish()
        if cleanup_on_exit:
            if policy_job_id:
                try:
                    logger.info("Cleanup: deleting policy trainer job %s", policy_job_id)
                    rlor_mgr.delete(policy_job_id)
                except Exception as e:
                    logger.warning("Cleanup: failed to delete policy job %s: %s", policy_job_id, e)
            if reference_job_id:
                try:
                    logger.info("Cleanup: deleting reference trainer job %s", reference_job_id)
                    rlor_mgr.delete(reference_job_id)
                except Exception as e:
                    logger.warning("Cleanup: failed to delete reference job %s: %s", reference_job_id, e)
            if cfg.deployment.deployment_id:
                try:
                    logger.info("Cleanup: scaling deployment to zero %s", cfg.deployment.deployment_id)
                    deploy_mgr.scale_to_zero(cfg.deployment.deployment_id)
                except Exception as e:
                    logger.warning("Cleanup: failed to scale deployment %s: %s", cfg.deployment.deployment_id, e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(Config())
