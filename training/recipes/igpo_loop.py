#!/usr/bin/env python3
"""IGPO training loop: Information Gain-based Policy Optimization.

Extends the GRPO training loop with turn-level Information Gain rewards
for multi-turn agent trajectories. Based on Wang et al., "Information
Gain-based Policy Optimization" (ICLR 2026, arXiv:2510.14967).

Key differences from rl_loop.py (GRPO):
  - After sampling, T+1 scoring forward passes (1 baseline + T turns)
    compute per-turn IG rewards on the policy trainer.
  - Advantages are per-turn: all tokens in turn t share A_{i,t}.
  - Scoring passes run async via ThreadPoolExecutor.

Fork this script to customise reward_fn, turn boundary detection,
or IG scoring.

Usage:
    export FIREWORKS_API_KEY=...
    python -m recipes.igpo_loop
"""

from __future__ import annotations

import os
import re
import json
import math
import signal
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import field, dataclass
from concurrent.futures import ThreadPoolExecutor

import tinker

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.client import GradAccNormalization
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
    validate_config,
    log_metrics_json,
    setup_deployment,
    create_trainer_job,
    read_api_extra_headers_env,
    load_jsonl_dataset,
    prepare_sampling_messages,
)
from training.utils.client import DEFAULT_TIMEOUT_S
from training.utils.checkpoint_utils import (
    resolve_resume,
    save_checkpoint,
    CheckpointKind,
)
from fireworks.training.sdk.deployment import DeploymentSampler
from training.utils.rl import PromptGroup
from training.utils.rl.tis import TISConfig
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils.timer import timer, flush_timing
from training.utils.rl.train import TrainStepFns, run_rl_loop
from training.utils.rl.losses import (
    build_loss_fn,
    combine_prompt_groups,
    resolve_builtin_loss,
)
from training.utils.rl.metrics import compute_step_metrics
from training.utils.rl.router_replay import build_r3_routing_matrices

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    log_path: str
    """Directory for checkpoints and logs."""

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
    lora_rank: int = 0

    prompt_groups_per_step: int = 1
    router_replay: bool = False
    router_replay_completion_only: bool = True

    grad_accumulation_normalization: GradAccNormalization | str | None = (
        GradAccNormalization.NUM_LOSS_TOKENS
    )

    policy_loss: str = "grpo"
    tis: TISConfig = field(default_factory=TISConfig)
    eps_clip: float = 0.2
    eps_clip_high: float | None = None

    # IGPO-specific
    gamma: float = 0.99
    """Discount factor for turn-level return accumulation."""
    ig_weight: float = 1.0
    """Weight for information gain reward vs outcome reward."""
    scoring_workers: int = 8
    """ThreadPoolExecutor max_workers for async IG scoring."""

    policy_job_id: str | None = None
    reference_job_id: str | None = None
    init_from_checkpoint: str | None = None
    output_model_id: str | None = None

    step_timeout: int = 0
    """Timeout in seconds for forward_backward / optim_step calls.
    0 = use DEFAULT_TIMEOUT_S from training.utils.client."""

    infra: InfraConfig = field(default_factory=InfraConfig)
    deployment: DeployConfig = field(default_factory=DeployConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="igpo-tinker")
    )
    runner: RunnerConfig = field(default_factory=RunnerConfig)


# ---------------------------------------------------------------------------
# Turn boundary detection
# ---------------------------------------------------------------------------

TURN_BOUNDARY_PATTERNS = [
    r"</?tool_call>",
    r"</?tool_response>",
    r"</?function_call>",
    r"</?search>",
    r"</?search_results?>",
    r"</?observation>",
]
_TURN_BOUNDARY_RE = re.compile("|".join(TURN_BOUNDARY_PATTERNS), re.IGNORECASE)


def detect_turn_boundaries(
    text: str, prompt_len: int, total_tokens: int
) -> List[Tuple[int, int]]:
    """Split a trajectory's response tokens into turn ranges.

    Returns (start, end) token-index pairs. Turns are delimited by
    tool-call/tool-response tags. If none found, one turn spans all.
    """
    boundaries = [m.start() for m in _TURN_BOUNDARY_RE.finditer(text)]
    if not boundaries:
        return [(prompt_len, total_tokens)]

    total_chars = len(text) or 1
    total_resp_tokens = total_tokens - prompt_len

    tb = [prompt_len]
    for co in boundaries:
        tp = prompt_len + int(co / total_chars * total_resp_tokens)
        tp = max(prompt_len, min(tp, total_tokens))
        if tp != tb[-1]:
            tb.append(tp)
    tb.append(total_tokens)

    turns = [(tb[i], tb[i + 1]) for i in range(len(tb) - 1) if tb[i + 1] > tb[i]]
    return turns if turns else [(prompt_len, total_tokens)]


# ---------------------------------------------------------------------------
# IG scoring, turn advantages, and loss function — imported from shared utils
# ---------------------------------------------------------------------------

from training.utils.rl.igpo import (
    build_score_datum as _build_score_datum,
    score_prefix as _score_prefix,
    compute_turn_advantages,
    expand_turn_advantages,
    make_igpo_loss_fn,
)


# ---------------------------------------------------------------------------
# Reward function -- customise for your task
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    digits = re.search(r"(-?\d+)", match.group(1))
    return digits.group(1) if digits else None


def reward_fn(completion: str, row: dict) -> float:
    """Return 1.0 if the model's numeric answer matches ground truth."""
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


# ---------------------------------------------------------------------------
# Rollout filter
# ---------------------------------------------------------------------------


def should_accept(pg: PromptGroup) -> bool:
    """Reject groups where all rewards are identical."""
    return len(set(pg.rewards)) > 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: Config,
    rlor_mgr: TrainerJobManager | None = None,
    deploy_mgr: DeploymentManager | None = None,
    cancel_on_exit: bool = False,
    cleanup_on_exit: bool | None = None,
):
    if cleanup_on_exit is not None:
        import warnings
        warnings.warn(
            "igpo_loop.main(cleanup_on_exit=...) is deprecated; use cancel_on_exit=...",
            DeprecationWarning, stacklevel=2,
        )
        cancel_on_exit = cleanup_on_exit

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
        raise ValueError("deployment.tokenizer_model is required.")
    setup_wandb(
        cfg.wandb,
        {
            "completions_per_prompt": completions_per_prompt,
            "prompt_groups_per_step": prompt_groups_per_step,
            "kl_beta": cfg.kl_beta,
            "lr": cfg.learning_rate,
            "gamma": cfg.gamma,
            "ig_weight": cfg.ig_weight,
        },
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    additional_headers = read_api_extra_headers_env()

    if rlor_mgr is None:
        rlor_mgr = TrainerJobManager(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
        )
    if deploy_mgr is None:
        deploy_mgr = DeploymentManager(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
        )

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
    if cfg.max_seq_len is None:
        raise ValueError("max_seq_len is required.")

    ref_profile = None
    if cfg.infra.ref_training_shape_id:
        ref_profile = rlor_mgr.resolve_training_profile(cfg.infra.ref_training_shape_id)
    use_reference = ref_profile is not None

    import time as _time

    runner.set_accelerator_info(cfg.infra.accelerator_type, cfg.infra.accelerator_count)
    runner.write_status(RunStatus.RUNNING, message="provisioning")

    with ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup:
        # Create trainer jobs first (trainer owns the hot-load bucket)
        if use_reference:
            with ThreadPoolExecutor(max_workers=2) as pool:
                pol_fut = pool.submit(
                    create_trainer_job, rlor_mgr, base_model=cfg.base_model,
                    infra=cfg.infra, profile=profile, lora_rank=cfg.lora_rank,
                    max_seq_len=cfg.max_seq_len, learning_rate=cfg.learning_rate,
                    display_name="igpo-policy",
                    job_id=cfg.policy_job_id,
                    cleanup=cleanup if not cfg.policy_job_id else None,
                )
                ref_fut = pool.submit(
                    create_trainer_job, rlor_mgr, base_model=cfg.base_model,
                    infra=cfg.infra, profile=ref_profile, lora_rank=cfg.lora_rank,
                    max_seq_len=cfg.max_seq_len, learning_rate=cfg.learning_rate,
                    display_name="igpo-reference", forward_only=True,
                    job_id=cfg.reference_job_id,
                    cleanup=cleanup if not cfg.reference_job_id else None,
                )
                policy_ep = pol_fut.result()
                reference_ep = ref_fut.result()
        else:
            policy_ep = create_trainer_job(
                rlor_mgr, base_model=cfg.base_model, infra=cfg.infra,
                profile=profile, lora_rank=cfg.lora_rank,
                max_seq_len=cfg.max_seq_len, learning_rate=cfg.learning_rate,
                display_name="igpo-policy",
                job_id=cfg.policy_job_id,
                cleanup=cleanup if not cfg.policy_job_id else None,
            )
            reference_ep = None

        # Create deployment referencing the trainer's hot-load bucket
        cfg.deployment.hot_load_trainer_job = policy_ep.job_name
        dep_info = setup_deployment(deploy_mgr, cfg.deployment, cfg.base_model, cfg.infra)
        if cancel_on_exit:
            cleanup.deployment(cfg.deployment.deployment_id, action="scale_to_zero")

        policy_job_id = policy_ep.job_id
        reference_job_id = reference_ep.job_id if reference_ep else None

        _timeout = cfg.step_timeout or DEFAULT_TIMEOUT_S
        policy = ReconnectableClient(
            rlor_mgr, policy_ep.job_id, cfg.base_model, cfg.lora_rank,
            fw_api_key=api_key,
            default_timeout=_timeout,
        )
        reference = (
            ReconnectableClient(
                rlor_mgr, reference_ep.job_id, cfg.base_model, cfg.lora_rank,
                fw_api_key=api_key,
                default_timeout=_timeout,
            )
            if reference_ep else None
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

        # Resume
        resume_info = resolve_resume(policy, cfg.log_path, cfg.init_from_checkpoint)
        step_offset = resume_info.step if resume_info else 0

        if cfg.weight_sync.weight_sync_before_training and cfg.deployment.deployment_id:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            weight_syncer.save_and_hotload(name, checkpoint_type="base")

        # Dataset
        raw_dataset = load_jsonl_dataset(cfg.dataset, cfg.max_rows)
        all_rows = raw_dataset * cfg.epochs
        rl_dataset = RLPromptDataset(all_rows, prompts_per_step=prompt_groups_per_step)
        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)

        sample_kwargs: dict = dict(
            max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature,
            max_seq_len=cfg.max_seq_len,
            http_timeout=cfg.deployment.sample_timeout,
        )
        if cfg.router_replay:
            sample_kwargs.update(include_routing_matrix=True, echo=True, logprobs=True)
        sample_kwargs["logprobs"] = True

        # Scoring executor for async IG scoring
        scoring_executor = ThreadPoolExecutor(max_workers=cfg.scoring_workers)

        # -- Sample one prompt (same as rl_loop, but stores tokens + text) --

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

            # Detect turn boundaries for each completion
            turn_boundaries = [
                detect_turn_boundaries(s.text, s.prompt_len, len(s.full_tokens))
                for s in sampled
            ]

            prompt_len = sampled[0].prompt_len
            policy_data: List[tinker.Datum] = []
            reference_data: List[tinker.Datum] = []
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

                policy_data.append(tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(tokens[:-1], routing_matrices=rm),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData(
                            data=tokens[1:], dtype="int64", shape=[model_input_len]
                        ),
                    },
                ))
                if use_reference:
                    reference_data.append(tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData(
                                data=tokens[1:], dtype="int64", shape=[model_input_len]
                            ),
                        },
                    ))

                if not s.inference_logprobs:
                    raise RuntimeError(f"Inference logprobs required but sample {idx} has none.")
                response_start = max(0, prompt_len - 1)
                echoed = getattr(s, "logprobs_echoed", False)
                aligned = list(s.inference_logprobs) if echoed else [0.0] * response_start + list(s.inference_logprobs)
                inf_logprobs_aligned.append(aligned)

            if not policy_data:
                return None

            comp_lens = [len(s.full_tokens) - s.prompt_len for s in sampled]
            trunc = [s.finish_reason == "length" for s in sampled]

            # Placeholder advantages (will be recomputed with IG in train_step)
            advantages = [0.0] * len(policy_data)

            return PromptGroup(
                data=policy_data,
                ref_data=reference_data,
                advantages=advantages,
                ref_logprobs=None,
                prompt_len=prompt_len,
                rewards=rewards,
                inf_logprobs=inf_logprobs_aligned,
                completion_lens=comp_lens,
                truncated=trunc,
                row_meta={
                    "ground_truth": row.get("ground_truth", ""),
                    "turn_boundaries": turn_boundaries,
                    "full_tokens": [list(s.full_tokens) for s in sampled],
                },
            )

        # -- Training callbacks -------------------------------------------------

        def ref_forward(groups: list[PromptGroup]) -> None:
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

        def ig_score_and_recompute_advantages(groups: list[PromptGroup]) -> dict:
            """Run IG scoring and replace placeholder advantages with per-token IGPO advantages."""
            ig_metrics: Dict[str, float] = {}
            total_turns = 0
            total_ig = 0.0
            total_baseline = 0.0
            n_groups = 0

            for pg in groups:
                meta = pg.row_meta or {}
                gt = meta.get("ground_truth", "")
                turn_boundaries = meta.get("turn_boundaries", [])
                full_tokens_list = meta.get("full_tokens", [])

                if not gt or not turn_boundaries or not full_tokens_list:
                    continue

                answer_tokens = tokenizer.encode(gt, add_special_tokens=False)
                if not answer_tokens:
                    continue

                prompt_tokens = full_tokens_list[0][: pg.prompt_len]

                # Async IG scoring: baseline + per-turn prefixes
                baseline_future = scoring_executor.submit(
                    _score_prefix, policy.inner, prompt_tokens, answer_tokens,
                )
                scoring_futures = []
                for i, ft in enumerate(full_tokens_list):
                    rollout_futures = []
                    for _, t_end in turn_boundaries[i]:
                        prefix = ft[:t_end]
                        rollout_futures.append(
                            scoring_executor.submit(
                                _score_prefix, policy.inner, prefix, answer_tokens,
                            )
                        )
                    scoring_futures.append(rollout_futures)

                baseline_logp = baseline_future.result(timeout=300)

                # Compute IG rewards
                all_turn_rewards: List[List[float]] = []
                for i in range(len(full_tokens_list)):
                    prev_logp = baseline_logp
                    turn_rewards: List[float] = []
                    for t_idx, future in enumerate(scoring_futures[i]):
                        score_logp = future.result(timeout=300)
                        r_t = cfg.ig_weight * (score_logp - prev_logp)
                        if t_idx == len(scoring_futures[i]) - 1:
                            r_t += pg.rewards[i]
                        turn_rewards.append(r_t)
                        prev_logp = score_logp
                    all_turn_rewards.append(turn_rewards)

                # Turn-level advantages
                turn_adv = compute_turn_advantages(all_turn_rewards, gamma=cfg.gamma)

                # Expand to per-token advantages
                n_data = len(pg.data)
                per_token_adv_list = []
                for i in range(n_data):
                    total_lp = len(pg.data[i].model_input.input_ids.data) if hasattr(pg.data[i].model_input, 'input_ids') else len(pg.data[i].loss_fn_inputs["target_tokens"].data)
                    adv_expanded = expand_turn_advantages(
                        turn_adv[i] if i < len(turn_adv) else [],
                        turn_boundaries[i] if i < len(turn_boundaries) else [],
                        pg.prompt_len,
                        total_lp,
                    )
                    per_token_adv_list.append(adv_expanded)

                # Store per-token advantages in row_meta for loss function
                pg.row_meta["per_token_advantages"] = per_token_adv_list

                # Update summary advantages (for metrics/logging)
                for i in range(n_data):
                    if i < len(turn_adv) and turn_adv[i]:
                        pg.advantages[i] = sum(turn_adv[i]) / len(turn_adv[i])

                total_turns += sum(len(r) for r in all_turn_rewards)
                total_ig += sum(r for rlist in all_turn_rewards for r in rlist)
                total_baseline += baseline_logp
                n_groups += 1

            if n_groups > 0:
                ig_metrics["igpo/avg_turns"] = total_turns / n_groups / max(len(groups[0].rewards), 1)
                ig_metrics["igpo/mean_ig"] = total_ig / max(total_turns, 1)
                ig_metrics["igpo/baseline_logp"] = total_baseline / n_groups

            return ig_metrics

        def fwd_bwd_one(prompt_groups: list[PromptGroup]):
            """Forward/backward with IGPO per-token advantages."""
            if not prompt_groups:
                raise ValueError("fwd_bwd_one requires at least one prompt group")

            data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(prompt_groups)

            t0 = _time.time()
            prox_fwd = policy.forward(data, "cross_entropy")
            prox_lp = [prox_fwd.loss_fn_outputs[i]["logprobs"].data for i in range(len(data))]
            logger.info("policy_forward: done (%.1fs)", _time.time() - t0)

            # Collect per-token advantages from all groups
            all_per_token_adv: List[List[float]] = []
            for pg in prompt_groups:
                meta = pg.row_meta or {}
                pta = meta.get("per_token_advantages")
                if pta:
                    all_per_token_adv.extend(pta)
                else:
                    for i in range(len(pg.data)):
                        n_lp = len(pg.data[i].loss_fn_inputs["target_tokens"].data)
                        all_per_token_adv.append([pg.advantages[i]] * n_lp)

            t0 = _time.time()
            loss_fn = make_igpo_loss_fn(
                per_token_advantages=all_per_token_adv,
                ref_logprobs=ref_lp,
                prompt_lens=prompt_lens,
                inf_logprobs=inf_lp,
                prox_logprobs=prox_lp,
                kl_beta=cfg.kl_beta,
                eps_clip=cfg.eps_clip,
            )
            fwd_bwd_result = policy.forward_backward_custom(data, loss_fn)
            logger.info("fwd_bwd: done (%.1fs)", _time.time() - t0)
            return fwd_bwd_result

        def train_step(
            step: int,
            prompt_groups: list[PromptGroup],
            loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            # 1. Reference forward (KL penalty)
            t0 = _time.time()
            ref_forward(prompt_groups)
            logger.info("[step %d] ref_forward: done (%.1fs)", step + 1, _time.time() - t0)

            # 2. IG scoring + turn-level advantage computation
            t0 = _time.time()
            ig_metrics = ig_score_and_recompute_advantages(prompt_groups)
            ig_time = _time.time() - t0
            logger.info("[step %d] ig_scoring: done (%.1fs)", step + 1, ig_time)

            # 3. Forward/backward with IGPO loss
            t0 = _time.time()
            fwd_bwd_result = fwd_bwd_one(prompt_groups)
            logger.info("[step %d] fwd_bwd: done (%.1fs)", step + 1, _time.time() - t0)

            # 4. Optimizer step
            t0 = _time.time()
            optim_result = policy.optim_step(
                adam_params,
                grad_accumulation_normalization=cfg.grad_accumulation_normalization,
            )
            step += 1
            logger.info("[step %d] optim_step: done (%.1fs)", step, _time.time() - t0)

            # 5. Weight sync
            if cfg.weight_sync.weight_sync_interval > 0 and step % cfg.weight_sync.weight_sync_interval == 0:
                with timer("weight_sync"):
                    weight_syncer.save_and_hotload(f"step-{step}")
            if cfg.weight_sync.dcp_save_interval > 0 and step % cfg.weight_sync.dcp_save_interval == 0:
                _data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    step - step_offset
                ) * prompt_groups_per_step
                save_checkpoint(
                    policy, f"step-{step}", cfg.log_path,
                    {"step": step, "data_consumed": _data_consumed, "source_job_id": policy_job_id},
                    kind=CheckpointKind.STATE,
                )

            # 6. Metrics
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
            metrics["igpo/scoring_time"] = ig_time
            metrics.update(ig_metrics)

            avg_reward = metrics.get("rollout/reward", 0.0)
            avg_acc = metrics.get("rollout/accuracy", 0.0)
            avg_kl = metrics.get("train/mean_kl", 0.0)
            logger.info(
                "Step %d | Reward: %.3f | Acc: %.1f%% | KL: %.4f | IG: %.4f | Turns: %.1f",
                step, avg_reward, avg_acc * 100, avg_kl,
                ig_metrics.get("igpo/mean_ig", 0.0),
                ig_metrics.get("igpo/avg_turns", 0.0),
            )
            log_metrics_json(step, reward=avg_reward, accuracy=avg_acc, kl=avg_kl)
            wandb_log(metrics, step)

            total_rl_steps = len(rl_dataset) - step_offset
            step_tokens = sum(
                len(d.loss_fn_inputs["target_tokens"].data) for pg in prompt_groups for d in pg.data
            )
            runner.append_metrics(step, metrics, tokens=step_tokens)
            runner.write_status(RunStatus.RUNNING, step=step, total_steps=total_rl_steps, message="training")
            runner.write_metadata()

            return step, metrics

        # -- Run ----------------------------------------------------------------

        train_fns = TrainStepFns(train_step=train_step)

        remaining_rows = []
        for i_step in range(step_offset, len(rl_dataset)):
            remaining_rows.extend(rl_dataset.get_batch(i_step))

        total_rl_steps = len(rl_dataset) - step_offset
        runner.start_training()
        runner.write_status(RunStatus.RUNNING, total_steps=total_rl_steps, message="training")

        with runner:
            global_step = asyncio.run(
                run_rl_loop(
                    sample_fns=(sample_one_prompt(row) for row in remaining_rows),
                    train_fns=train_fns,
                    prompt_groups_per_step=prompt_groups_per_step,
                    dynamic_filter_fn=should_accept,
                    global_step=step_offset,
                )
            )

        # Final checkpoint
        if global_step > step_offset:
            try:
                _data_consumed = (resume_info.data_consumed if resume_info else 0) + (
                    global_step - step_offset
                ) * prompt_groups_per_step
                cp_name = f"step-{global_step}"
                paths = save_checkpoint(
                    policy, cp_name, cfg.log_path,
                    {"step": global_step, "data_consumed": _data_consumed, "source_job_id": policy_job_id},
                    kind=CheckpointKind.BOTH,
                )
                if getattr(cfg, "output_model_id", None):
                    rlor_mgr.promote_checkpoint(policy_job_id, paths["sampler_path"], cfg.output_model_id)
                    runner.write_output_model(model_id=cfg.output_model_id, checkpoint=cp_name, job_id=policy_job_id)
            except Exception as e:
                logger.warning("Final checkpoint failed: %s", e)

            runner.write_status(RunStatus.COMPLETED, step=global_step, total_steps=total_rl_steps, message="done")
            runner.write_metadata()
            logger.info("IGPO training complete: %d steps", global_step)
            wandb_finish()
            scoring_executor.shutdown(wait=False)
            return {"steps": global_step, "policy_job_id": policy_job_id, "reference_job_id": reference_job_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(Config(log_path="./igpo_logs"))
