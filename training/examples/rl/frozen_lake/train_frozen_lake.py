#!/usr/bin/env python3
"""GRPO training on FrozenLake multi-turn tool calls with eval-protocol rollouts.

Demonstrates reinforcement learning with multi-turn tool-calling:
  - eval-protocol handles the data plane (token-ID-based rollout with environment)
  - cookbook handles the training plane (GRPO loss, weight sync, reference model)
  - training uses the server-side builtin loss path when available, otherwise
    it falls back to the client-side custom loss path

Usage:
    Follow the setup instructions in ../../../README.md.
    export FIREWORKS_API_KEY=...
    python train_frozen_lake.py --training-shape <shape_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Dict, List, cast

import tinker

from eval_protocol.models import EvaluationRow, InputMetadata
from eval_protocol.pytest import evaluation_test

from training.examples.rl.frozen_lake.frozen_lake_rollout import (
    DEFAULT_SYSTEM_PROMPT_INSTRUCTIONS,
    FrozenLakeToolRolloutProcessor,
)
from training.examples.rl.frozen_lake.masking import (
    compute_model_output_spans,
    build_ui_token_mask,
)

from fireworks.training.sdk.client import GradAccNormalization

from training.utils import (
    DEFAULT_ADAM,
    TrainerConfig,
    WandBConfig,
    DeployConfig,
    ReconnectableClient,
    wandb_log,
    setup_wandb,
    wandb_finish,
    log_metrics_json,
    build_service_client,
    read_api_extra_headers_env,
    build_datum_from_token_mask,
    validate_config,
)
from training.utils.rl import PromptGroup
from training.utils.rl.async_train import RowRequest, run_async_rl_loop
from training.utils.rl.rollout import (
    RolloutSample,
    load_eval_protocol_input_rows,
    make_eval_protocol_rollout_fn_factory,
)
from training.utils.rl.grpo import make_grpo_loss_fn
from training.utils.rl.losses import combine_prompt_groups
from training.utils.rl.tis import TISConfig
from training.utils.rl.metrics import compute_step_metrics
from training.utils.checkpoints import TrainingCheckpoints
from training.utils.timer import timer, flush_timing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_USER_PROMPT_TEMPLATE = "FrozenLake grid observation:\n{observation}"
DEFAULT_VISUAL_PROMPT_TEMPLATE = (
    "You are playing FrozenLake. The image shows the current grid. "
    "The current textual observation is below.\n"
    "{observation}\n\n"
    "Tiles are labeled S, F, H, and G. The agent is marked with a red dot. "
    "Use exactly one lake_move tool call with action LEFT, DOWN, RIGHT, or UP."
)

DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_INSTRUCTIONS
WEIGHT_SYNC_INTERVAL = 1
DCP_SAVE_INTERVAL = 20
WEIGHT_SYNC_TIMEOUT_S = 900


@dataclass
class FrozenLakeConfig:
    log_path: str = "./frozen_lake_logs"

    base_model: str = "accounts/fireworks/models/qwen3-8b"
    tokenizer_model: str = "Qwen/Qwen3-8B"

    learning_rate: float = 1e-5
    kl_beta: float = 0.001
    completions_per_prompt: int = 4
    max_completion_tokens: int = 128
    temperature: float = 1.0
    epochs: int = 3
    max_seeds: int = 20
    max_steps: int = 12
    lora_rank: int = 0
    max_seq_len: int | None = None

    prompt_groups_per_step: int = 4
    max_concurrent: int = 16

    eps_clip: float = 0.2
    """PPO clip epsilon for the client-side GRPO objective."""
    eps_clip_high: float | None = None
    """Asymmetric upper clip bound for the client-side GRPO objective."""
    tis: TISConfig = field(default_factory=TISConfig)
    """TIS (Train-Inference IS) weight correction config."""

    seed_jsonl_path: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "seeds.jsonl")
    )
    map_name: str = "4x4"
    use_random_map: bool = True
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE
    visual_prompt_template: str = DEFAULT_VISUAL_PROMPT_TEMPLATE
    observation_mode: str = "text"
    allow_plaintext_action_fallback: bool = False

    training_shape: str = ""
    deployment_shape: str = ""
    deployment_id: str | None = None
    deployment_replica_count: int | None = None

    wandb_entity: str = field(default_factory=lambda: os.environ.get("WANDB_ENTITY", ""))
    wandb_project: str = field(
        default_factory=lambda: os.environ.get("WANDB_PROJECT", "frozen-lake-grpo")
    )

    policy_job_id: str | None = None
    reference_job_id: str | None = None
    inference_base_url: str | None = None
    output_model_id: str | None = None


def parse_args() -> FrozenLakeConfig:
    parser = argparse.ArgumentParser(description="GRPO training on FrozenLake tool calls")
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3-8b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--training-shape", default=os.environ.get("TRAINING_SHAPE", ""))
    parser.add_argument("--deployment-shape", default="")
    parser.add_argument("--deployment-id", default=None)
    parser.add_argument("--deployment-replica-count", type=int, default=None)

    parser.add_argument("--seed-jsonl-path",
                        default=os.path.join(os.path.dirname(__file__), "seeds.jsonl"))
    parser.add_argument("--max-seeds", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--map-name", default="4x4")
    parser.add_argument("--use-random-map", action="store_true", default=True)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--completions-per-prompt", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-beta", type=float, default=0.001)
    parser.add_argument("--ratio-log-cap", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--prompt-groups-per-step", type=int, default=4)
    parser.add_argument("--min-samples-per-fwd-bwd", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=0)

    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "frozen-lake-grpo"))
    parser.add_argument("--visual-prompt-template", default=DEFAULT_VISUAL_PROMPT_TEMPLATE)
    parser.add_argument("--observation-mode", choices=("text", "image"), default="text")
    parser.add_argument("--allow-plaintext-action-fallback", action="store_true")

    parser.add_argument("--policy-job-id", default=None,
                        help="Pre-created policy trainer job ID (skip creation)")
    parser.add_argument("--reference-job-id", default=None,
                        help="Pre-created reference trainer job ID (skip creation)")
    parser.add_argument("--inference-base-url", default=None,
                        help="Direct base URL for inference deployment (skip gateway)")
    parser.add_argument("--output-model-id", type=str, required=True,
                        help="Promote final checkpoint to this model ID")

    return cast(FrozenLakeConfig, parser.parse_args(namespace=FrozenLakeConfig()))


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------


def load_seed_contexts(path: str, max_seeds: int) -> List[Dict[str, Any]]:
    """Load environment contexts from a seed JSONL file."""
    contexts: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            contexts.append({
                "map_name": entry.get("map_name", "4x4"),
                "use_random_map": True,
                "seed": int(entry["seed"]),
            })
            if len(contexts) >= max_seeds:
                break
    return contexts


# ---------------------------------------------------------------------------
# EvaluationRow -> PromptGroup conversion
# ---------------------------------------------------------------------------


def evaluation_row_to_training_data(
    row: EvaluationRow,
) -> tuple[list[tinker.Datum], int, list[float], list[float]]:
    """Convert a completed rollout EvaluationRow into tinker training datums.

    For a multi-turn episode, the last turn's prompt_ids contain the full
    conversation context.  We build one datum per episode using the full
    token sequence (all turns concatenated by the rollout processor).

    Returns (datums, prompt_len, inf_logprobs_per_datum, [episode_reward]).
    """
    extra = row.execution_metadata.extra or {}
    token_turn_traces = extra.get("token_turn_traces") or []
    step_rewards = extra.get("step_rewards") or []

    if not token_turn_traces:
        return [], 0, [], []

    # The last turn's prompt_ids = full context; its completion_ids = final action.
    # But for training, we want the complete trajectory as one sequence.
    # Build the full sequence: first turn prompt + all completions + turn suffixes.
    # The rollout processor already tracks prompt_ids per turn where each turn's
    # prompt_ids = previous prompt_ids + assistant_turn_ids + tool_suffix_ids.
    # So the last turn's prompt_ids + completion_ids = full episode tokens.
    last_trace = token_turn_traces[-1]
    last_prompt_ids = [int(x) for x in (last_trace.get("prompt_ids") or [])]
    last_completion_ids = [int(x) for x in (last_trace.get("completion_ids") or [])]
    full_tokens = last_prompt_ids + last_completion_ids

    if len(full_tokens) < 2:
        return [], 0, [], []

    # prompt_len = first turn's prompt length (system + user message, before any model output)
    first_prompt_len = len([int(x) for x in (token_turn_traces[0].get("prompt_ids") or [])])

    model_request_traces = extra.get("model_request_traces") or []
    spans = compute_model_output_spans(token_turn_traces, model_request_traces)
    token_mask = build_ui_token_mask(spans, len(full_tokens))
    rendered = build_datum_from_token_mask(
        full_tokens,
        token_mask,
        include_loss_mask=True,
    )
    datum = rendered.datum
    model_input_len = len(rendered.token_ids) - 1

    # Reconstruct rollout_logprobs aligned to the full sequence.
    # Pad prompt positions with 0.0, then fill in per-turn completion logprobs.
    inf_logprobs = [0.0] * model_input_len

    for trace in token_turn_traces:
        turn_prompt_len = len(trace.get("prompt_ids") or [])
        turn_completion_logprobs = trace.get("completion_logprobs") or []
        start_pos = max(0, turn_prompt_len - 1)
        for i, lp in enumerate(turn_completion_logprobs):
            pos = start_pos + i
            if pos < model_input_len:
                inf_logprobs[pos] = float(lp)

    episode_reward = 1.0 if step_rewards and float(step_rewards[-1]) > 0 else 0.0

    return [datum], first_prompt_len, [inf_logprobs], [episode_reward]


def evaluation_row_to_rollout_sample(row: EvaluationRow) -> RolloutSample | None:
    """Convert a completed FrozenLake EvaluationRow into one RolloutSample."""
    extra = row.execution_metadata.extra or {}
    token_turn_traces = extra.get("token_turn_traces") or []
    step_rewards = extra.get("step_rewards") or []

    if not token_turn_traces:
        return None

    last_trace = token_turn_traces[-1]
    last_prompt_ids = [int(x) for x in (last_trace.get("prompt_ids") or [])]
    last_completion_ids = [int(x) for x in (last_trace.get("completion_ids") or [])]
    full_tokens = last_prompt_ids + last_completion_ids
    if len(full_tokens) < 2:
        return None

    model_request_traces = extra.get("model_request_traces") or []
    spans = compute_model_output_spans(token_turn_traces, model_request_traces)
    ui_token_mask = build_ui_token_mask(spans, len(full_tokens))
    loss_mask = [1 if int(m) > 0 else 0 for m in ui_token_mask]

    # RolloutSample.logprobs are token-aligned rollout_logprobs.
    # Non-generated tokens keep 0.0; generated token logprobs are copied from
    # each turn's completion payload.
    rollout_logprobs = [0.0] * len(full_tokens)
    for trace in token_turn_traces:
        turn_prompt_len = len(trace.get("prompt_ids") or [])
        turn_completion_logprobs = trace.get("completion_logprobs") or []
        for i, lp in enumerate(turn_completion_logprobs):
            pos = turn_prompt_len + i
            if pos < len(rollout_logprobs):
                rollout_logprobs[pos] = float(lp)

    episode_reward = 1.0 if step_rewards and float(step_rewards[-1]) > 0 else 0.0
    return RolloutSample(
        tokens=full_tokens,
        logprobs=rollout_logprobs,
        loss_mask=loss_mask,
        reward=episode_reward,
        finish_reason=str(row.execution_metadata.finish_reason or "stop"),
    )


def frozen_lake_eval_row_factory(row: Dict[str, Any]) -> EvaluationRow:
    """Build one per-run EvaluationRow from an eval3 input row."""
    base_row = row.get("evaluation_row")
    rollout_idx = int(row.get("rollout_idx", 0))
    if not isinstance(base_row, EvaluationRow):
        raise TypeError("FrozenLake row must include an 'evaluation_row' EvaluationRow.")

    eval_row = base_row.model_copy(deep=True)
    base_row_id = eval_row.input_metadata.row_id or "row"
    eval_row.input_metadata.row_id = f"{base_row_id}_{rollout_idx}"
    return eval_row


def make_frozen_lake_eval3_evaluator(
    *,
    input_rows: List[EvaluationRow],
    rollout_processor: FrozenLakeToolRolloutProcessor,
    completion_params: Dict[str, Any],
    steps: int,
):
    """Create the eval3 expression that managed RFT can adapt into rollout_fn.

    FrozenLake's scalar reward and token trace are produced by the rollout
    processor, so the pointwise evaluator body is intentionally identity.
    """

    def frozen_lake_eval(row):
        return row

    # eval-protocol validates annotations by object identity, while this module
    # uses postponed annotations. Set the runtime annotations explicitly.
    frozen_lake_eval.__annotations__ = {"row": EvaluationRow, "return": EvaluationRow}

    return evaluation_test(
        input_rows=[input_rows],
        completion_params=[completion_params],
        rollout_processor=rollout_processor,
        mcp_config_path="",
        steps=steps,
    )(frozen_lake_eval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: FrozenLakeConfig | None = None) -> dict:
    """Run FrozenLake GRPO training. Returns a dict with 'steps' and 'rewards'."""
    if cfg is None:
        cfg = parse_args()

    logger.info("FrozenLake GRPO training: %s", cfg.base_model)

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    completions_per_prompt = cfg.completions_per_prompt
    prompt_groups_per_step = cfg.prompt_groups_per_step

    deploy_cfg = DeployConfig(
        deployment_id=cfg.deployment_id,
        deployment_shape=cfg.deployment_shape or None,
        replica_count=cfg.deployment_replica_count,
        tokenizer_model=cfg.tokenizer_model,
        sample_timeout=1200,
    )
    wandb_cfg = WandBConfig(
        entity=cfg.wandb_entity or None,
        project=cfg.wandb_project,
        run_name=cfg.deployment_id or f"frozen-lake-{int(time.time()) % 100000}",
    )

    validate_config(
        cfg.base_model,
        cfg.seed_jsonl_path,
        deploy=deploy_cfg,
        output_model_id=cfg.output_model_id,
    )

    setup_wandb(wandb_cfg, {
        "completions_per_prompt": completions_per_prompt,
        "prompt_groups_per_step": prompt_groups_per_step,
        "kl_beta": cfg.kl_beta,
        "lr": cfg.learning_rate,
        "temperature": cfg.temperature,
        "max_steps": cfg.max_steps,
        "max_seeds": cfg.max_seeds,
    })

    # -- Load seed contexts --------------------------------------------------

    seed_contexts = load_seed_contexts(cfg.seed_jsonl_path, cfg.max_seeds)
    logger.info(
        "Loaded %d seed contexts from %s",
        len(seed_contexts), os.path.abspath(cfg.seed_jsonl_path),
    )

    use_reference = cfg.kl_beta > 0 or cfg.reference_job_id is not None

    # -- Infrastructure setup -----------------------------------------------

    _infra_start = time.time()
    policy_job_id: str | None = None
    trajectory_log = None
    global_step = step_offset = 0

    reward_history: list[float] = []
    _shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal _shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.warning("Received %s — initiating graceful shutdown...", sig_name)
        _shutdown_requested = True
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    service = build_service_client(
        api_key=api_key,
        base_url=base_url,
        inference_url=cfg.inference_base_url,
        additional_headers=read_api_extra_headers_env(),
        base_model=cfg.base_model,
        tokenizer_model=cfg.tokenizer_model,
        lora_rank=cfg.lora_rank,
        max_context_length=cfg.max_seq_len,
        learning_rate=cfg.learning_rate,
        trainer=TrainerConfig(
            job_id=cfg.policy_job_id or None,
            training_shape_id=cfg.training_shape or None,
            reference_job_id=(cfg.reference_job_id if use_reference else None),
        ),
        deployment=deploy_cfg,
        hotload_timeout_s=WEIGHT_SYNC_TIMEOUT_S,
        cleanup_trainer_on_close=not cfg.policy_job_id,
        reference_required=use_reference,
    )

    with closing(service):
        policy = ReconnectableClient.from_training_client(
            service.create_training_client(cfg.base_model, lora_rank=cfg.lora_rank),
            base_model=cfg.base_model,
            lora_rank=cfg.lora_rank,
            job_id=service.trainer_job_id,
            service=service,
        )
        policy_job_id = service.trainer_job_id
        sampler = service.create_deployment_sampler()
        reference = None
        if use_reference:
            reference = ReconnectableClient.from_training_client(
                service.create_reference_client(cfg.base_model, lora_rank=cfg.lora_rank),
                base_model=cfg.base_model,
                lora_rank=0,
                job_id=service.reference_client_job_id,
                service=service,
                base_only=True,
            )

        inference_model = sampler.model

        infra_boot_time = time.time() - _infra_start
        wandb_log({"train/step": 0, "infra/total_boot_time": infra_boot_time}, step=0)

        ckpt = TrainingCheckpoints(
            policy,
            service,
            trainer_id=policy_job_id,
            log_path=cfg.log_path,
            lora_rank=cfg.lora_rank,
        )
        resume_info = ckpt.resume()
        step_offset = resume_info.step if resume_info else 0
        prior_rows_consumed = resume_info.data_consumed if resume_info else 0
        if cfg.deployment_id:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            saved = policy.save_weights_for_sampler_ext(name, checkpoint_type="base")
            service.hotload_sampler_snapshot(saved.snapshot_name)

        # -- Build rollout processor ----------------------------------------
        rollout_base_url = sampler.base_url.rstrip("/") + ("" if cfg.inference_base_url else "/inference")
        rollout_processor = FrozenLakeToolRolloutProcessor(
            model_id=inference_model,
            tokenizer_name_or_path=cfg.tokenizer_model,
            api_key=api_key,
            base_url=rollout_base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_completion_tokens,
            system_prompt=cfg.system_prompt,
            user_prompt_template=cfg.user_prompt_template,
            logprobs=True,
            observation_mode=cfg.observation_mode,
            allow_plaintext_action_fallback=cfg.allow_plaintext_action_fallback,
        )
        all_prompts = seed_contexts * cfg.epochs
        frozen_lake_input_rows = [
            EvaluationRow(
                input_metadata=InputMetadata(
                    row_id=f"seed_{env_context.get('seed', row_idx)}_{row_idx}",
                    dataset_info={
                        "environment_context": dict(env_context),
                        "user_prompt_template": cfg.user_prompt_template,
                        "visual_prompt_template": cfg.visual_prompt_template,
                    },
                ),
            )
            for row_idx, env_context in enumerate(all_prompts)
        ]
        frozen_lake_evaluator = make_frozen_lake_eval3_evaluator(
            input_rows=frozen_lake_input_rows,
            rollout_processor=rollout_processor,
            completion_params={"model": inference_model},
            steps=cfg.max_steps,
        )
        eval3_input_rows = load_eval_protocol_input_rows(frozen_lake_evaluator)[prior_rows_consumed:]

        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)
        # -- Trajectory logging -----------------------------------------------
        trajectory_path = f"/tmp/frozen_lake_trajectories_{int(time.time())}.jsonl"
        trajectory_log = open(trajectory_path, "a")
        logger.info("Logging trajectories to %s", trajectory_path)
        try:
            # -- Per-sample rollout function ------------------------------------

            def convert_and_log_sample(result: EvaluationRow) -> RolloutSample | None:
                extra = result.execution_metadata.extra or {}
                dataset_info = result.input_metadata.dataset_info or {}
                env_context = dict(dataset_info.get("environment_context") or {})
                rollout_idx = str(result.input_metadata.row_id or "").rsplit("_", 1)[-1]
                if extra.get("rollout_error"):
                    logger.warning(
                        "Rollout error for seed %s sample %s: %s",
                        env_context.get("seed"),
                        rollout_idx,
                        extra["rollout_error"],
                    )
                    return None

                sample = evaluation_row_to_rollout_sample(result)
                if sample is None:
                    return None

                if trajectory_log:
                    entry = {
                        "seed": env_context.get("seed"),
                        "rollout_idx": rollout_idx,
                        "messages": [
                            m.model_dump() if hasattr(m, "model_dump") else m
                            for m in (result.messages or [])
                        ],
                        "step_rewards": extra.get("step_rewards", []),
                        "reward": sample.reward,
                        "rollout_error": extra.get("rollout_error"),
                    }
                    trajectory_log.write(json.dumps(entry) + "\n")
                    trajectory_log.flush()

                return sample

            frozen_lake_rollout_fn = make_eval_protocol_rollout_fn_factory(
                frozen_lake_evaluator,
                row_factory=frozen_lake_eval_row_factory,
                sample_converter=convert_and_log_sample,
            )(None)

            # -- Training callbacks ---------------------------------------------

            def ref_forward_batch(groups: list[PromptGroup]) -> None:
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
                        ref_fwd.loss_fn_outputs[idx + i]["logprobs"].data for i in range(n)
                    ]
                    idx += n

            logger.info("algorithm=grpo trainer_loss=client")

            def fwd_bwd_one(sub: list[PromptGroup]):
                data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(sub)
                old_policy_fwd = policy.forward(data, "cross_entropy")
                old_policy_lp = [old_policy_fwd.loss_fn_outputs[i]["logprobs"].data for i in range(len(data))]
                return policy.forward_backward_custom(
                    data,
                    make_grpo_loss_fn(
                        adv,
                        ref_lp,
                        prompt_lens,
                        inf_logprobs=inf_lp,
                        old_policy_logprobs=old_policy_lp,
                        kl_beta=cfg.kl_beta,
                        eps_clip=cfg.eps_clip,
                        eps_clip_high=cfg.eps_clip_high,
                        tis_config=cfg.tis,
                    ),
                )

            def train_step(
                step: int,
                prompt_groups: list[PromptGroup],
                loop_stats: dict | None,
                run_optimizer_step: bool,
            ) -> tuple[int, dict]:
                """ref_forward + fwd_bwd + optim_step + metrics (1:1)."""
                if not run_optimizer_step:
                    raise ValueError("frozen_lake async train_step only supports optimizer steps")
                t0 = time.time()
                ref_forward_batch(prompt_groups)
                logger.info("[step %d] ref_forward: done (%.1fs)", step + 1, time.time() - t0)

                t0 = time.time()
                fwd_bwd_result = fwd_bwd_one(prompt_groups)
                logger.info("[step %d] fwd_bwd: done (%.1fs)", step + 1, time.time() - t0)

                t0 = time.time()
                optim_result = policy.optim_step(
                    adam_params,
                    grad_accumulation_normalization=GradAccNormalization.NUM_LOSS_TOKENS,
                )
                step += 1
                logger.info("[step %d] optim_step: done (%.1fs)", step, time.time() - t0)

                if DCP_SAVE_INTERVAL > 0 and step % DCP_SAVE_INTERVAL == 0:
                    logger.info("[step %d] dcp_save...", step)
                    t0 = time.time()
                    with timer("dcp_save"):
                        ckpt.save(
                            f"step-{step}",
                            resumable=True,
                            promotable=False,
                            data_consumed=step - step_offset,
                        )
                    logger.info("[step %d] dcp_save: done (%.1fs)", step, time.time() - t0)

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
                if loop_stats:
                    metrics["rollout/sample_fail_count"] = loop_stats.get("sample_fails", 0)
                    metrics["rollout/filter_drops"] = loop_stats.get("filter_drops", 0)

                avg_reward = metrics.get("rollout/reward", 0.0)
                avg_ref_kl = metrics.get("train/ref_kl", 0.0)
                mean_loss = metrics.get("train/mean_loss", 0.0)
                adv_loss = metrics.get("train/mean_adv_loss", 0.0)
                kl_pen = metrics.get("train/mean_kl_penalty", 0.0)
                mask_r = metrics.get("train/mask_ratio", 0.0)
                inf_kld = metrics.get("train/inference_kld", 0.0)
                logger.info(
                    "Step %d | Reward: %.3f | RefKL: %.4f | Loss: %.4f "
                    "(adv=%.4f kl_pen=%.4f) | InfKLD: %.4f | MaskRatio: %.2f",
                    step, avg_reward, avg_ref_kl, mean_loss, adv_loss, kl_pen, inf_kld, mask_r,
                )
                reward_history.append(avg_reward)
                log_metrics_json(step, reward=avg_reward, ref_kl=avg_ref_kl)
                _wandb_step[0] = max(_wandb_step[0] + 1, step)
                wandb_log(metrics, _wandb_step[0])
                return step, metrics

            def _weight_sync(step: int) -> None:
                logger.info("[step %d] weight_sync: saving + loading...", step)
                t0 = time.time()
                with timer("weight_sync"):
                    saved = policy.save_weights_for_sampler_ext(f"step-{step}")
                    service.hotload_sampler_snapshot(saved.snapshot_name)
                logger.info("[step %d] weight_sync: done (%.1fs)", step, time.time() - t0)

            def should_accept(pg: PromptGroup) -> bool:
                return len(set(pg.rewards)) > 1

            max_concurrent_samples = cfg.max_concurrent if cfg.max_concurrent > 0 else None
            min_group_size = 2 if completions_per_prompt > 1 else 1
            async_weight_sync_interval = (
                max(1, WEIGHT_SYNC_INTERVAL)
                if WEIGHT_SYNC_INTERVAL > 0
                else 1
            )
            max_head_offpolicy_versions = (
                max(0, async_weight_sync_interval - 1)
                if WEIGHT_SYNC_INTERVAL > 0
                else max(0, (len(eval3_input_rows) + prompt_groups_per_step - 1) // prompt_groups_per_step)
            )
            logger.info(
                "Training: %d seeds x %d epochs = %d prompt groups, "
                "%d completions/prompt, %d groups/step, max_concurrent_samples=%s",
                len(seed_contexts), cfg.epochs, len(eval3_input_rows),
                completions_per_prompt, prompt_groups_per_step, max_concurrent_samples,
            )

            _wandb_step = [step_offset]

            def make_row_requests():
                for row_idx, eval_row in enumerate(eval3_input_rows, start=prior_rows_consumed):
                    dataset_info = eval_row.input_metadata.dataset_info or {}
                    env_context_dict = dict(dataset_info.get("environment_context") or {})

                    def factory(
                        rollout_idx: int,
                        *,
                        row_idx: int = row_idx,
                        eval_row: EvaluationRow = eval_row,
                    ):
                        return frozen_lake_rollout_fn({
                            "row_id": row_idx,
                            "rollout_idx": rollout_idx,
                            "evaluation_row": eval_row,
                        })

                    yield RowRequest(
                        row_id=row_idx,
                        run_factory=factory,
                        row_meta={
                            "seed": env_context_dict.get("seed"),
                            "environment_context": env_context_dict,
                        },
                    )

            global_step, final_stats = asyncio.run(run_async_rl_loop(
                rows=make_row_requests(),
                train_step_fn=train_step,
                completions_per_prompt=completions_per_prompt,
                prompt_groups_per_step=prompt_groups_per_step,
                max_head_offpolicy_versions=max_head_offpolicy_versions,
                with_reference=use_reference,
                min_group_size=min_group_size,
                weight_sync_fn=_weight_sync if WEIGHT_SYNC_INTERVAL > 0 else None,
                weight_sync_interval=async_weight_sync_interval,
                max_concurrent=max_concurrent_samples,
                dynamic_filter_fn=should_accept,
                global_step=step_offset,
                resolved_rows_offset=prior_rows_consumed,
                return_final_stats=True,
            ))

            # -- Final checkpoint -----------------------------------------------

            if global_step > step_offset:
                try:
                    cp_name = f"step-{global_step}"
                    _data_consumed = int(final_stats["resolved_rows"])
                    ckpt.save(
                        cp_name,
                        resumable=True,
                        promotable=True,
                        data_consumed=_data_consumed,
                    )
                    if getattr(cfg, "output_model_id", None):
                        ckpt.promote_latest(cfg.output_model_id, cfg.base_model)
                except Exception as e:
                    logger.warning("Failed to save final checkpoint: %s", e)
                logger.info("Training complete: %d steps", global_step)

        finally:
            if trajectory_log and not trajectory_log.closed:
                trajectory_log.close()
            wandb_finish()

    return {"steps": global_step - step_offset, "rewards": reward_history}


if __name__ == "__main__":
    main()
