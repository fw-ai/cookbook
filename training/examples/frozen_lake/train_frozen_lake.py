#!/usr/bin/env python3
"""GRPO training on FrozenLake multi-turn tool calls with eval-protocol rollouts.

Demonstrates reinforcement learning with multi-turn tool-calling:
  - eval-protocol handles the data plane (token-ID-based rollout with environment)
  - cookbook handles the training plane (GRPO loss, weight sync, reference model)
  - training uses the server-side builtin loss path when available, otherwise
    it falls back to the client-side custom loss path

Usage:
    pip install --pre "fireworks-ai>=1.0.0a36" tinker-cookbook eval-protocol
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
import sys
import time
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, cast

import tinker

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from eval_protocol.models import EvaluationRow, InputMetadata
from eval_protocol.pytest.types import RolloutProcessorConfig

from training.examples.frozen_lake.frozen_lake_rollout import (
    DEFAULT_SYSTEM_PROMPT_INSTRUCTIONS,
    FrozenLakeToolRolloutProcessor,
)
from training.examples.frozen_lake.masking import (
    compute_model_output_spans,
    build_ui_token_mask,
)

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.client import GradAccNormalization
from fireworks.training.sdk.weight_syncer import WeightSyncer

from training.utils import (
    DEFAULT_ADAM,
    InfraConfig,
    ResourceCleanup,
    WandBConfig,
    DeployConfig,
    WeightSyncConfig,
    ReconnectableClient,
    wandb_log,
    setup_wandb,
    wandb_finish,
    log_metrics_json,
    setup_deployment,
    create_trainer_job,
    compute_advantages,
    build_datum_from_token_mask,
    validate_config,
)
from training.utils.rl import PromptGroup
from training.utils.rl.train import TrainStepFns, run_rl_loop
from training.utils.rl.losses import build_builtin_loss_datums, build_loss_fn, combine_prompt_groups, resolve_builtin_loss
from training.utils.rl.tis import TISConfig
from training.utils.rl.metrics import compute_step_metrics
from training.utils.rl.pp import compute_pp_recommendation
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

    policy_loss: str = "grpo"
    """``"grpo"``, ``"importance_sampling"``, ``"dapo"``, ``"dro"``, ``"gspo"``, or ``"cispo"``.

    If an eligible builtin kernel exists for the selected loss, training uses
    the server-side ``forward_backward(...)`` path. Otherwise it falls back to
    the client-side ``forward_backward_custom(...)`` path.
    """
    ratio_log_cap: float = 20.0
    tis_enabled: bool = False

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
    accelerator_type: str = ""
    deployment_id: str | None = None
    region: str | None = None
    deployment_region: str | None = None
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
    parser.add_argument("--accelerator-type", default="")
    parser.add_argument("--deployment-id", default=None)
    parser.add_argument("--region", default="US_VIRGINIA_1")
    parser.add_argument("--deployment-region", default=None)
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

    # Reconstruct inference logprobs aligned to the full sequence.
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
    total_samples_per_step = prompt_groups_per_step * completions_per_prompt

    infra = InfraConfig(
        training_shape_id=cfg.training_shape or None,
        region=cfg.region,
    )
    deploy_cfg = DeployConfig(
        deployment_id=cfg.deployment_id,
        deployment_shape=cfg.deployment_shape or None,
        deployment_accelerator_type=cfg.accelerator_type or None,
        deployment_region=cfg.deployment_region,
        replica_count=cfg.deployment_replica_count,
        tokenizer_model=cfg.tokenizer_model,
        sample_timeout=1200,
    )
    weight_sync_cfg = WeightSyncConfig(
        weight_sync_interval=1,
        dcp_save_interval=20,
        dcp_timeout=2700,
        first_checkpoint_type="base",
        weight_sync_before_training=bool(cfg.deployment_id),
        weight_sync_timeout=900,
    )
    wandb_cfg = WandBConfig(
        entity=cfg.wandb_entity or None,
        project=cfg.wandb_project,
        run_name=cfg.deployment_id or f"frozen-lake-{int(time.time()) % 100000}",
    )

    validate_config(
        cfg.base_model,
        cfg.seed_jsonl_path,
        weight_sync_cfg,
        deploy_cfg,
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

    rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    deploy_mgr = DeploymentManager(
        api_key=api_key,
        base_url=base_url,
        hotload_api_url=base_url,
        inference_url=cfg.inference_base_url or base_url,
    )

    # -- Load seed contexts --------------------------------------------------

    seed_contexts = load_seed_contexts(cfg.seed_jsonl_path, cfg.max_seeds)
    logger.info(
        "Loaded %d seed contexts from %s",
        len(seed_contexts), os.path.abspath(cfg.seed_jsonl_path),
    )

    # -- Resolve training shapes --------------------------------------------

    profile = None
    if infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(infra.training_shape_id)
        dsv = profile.deployment_shape_version or ""
        if dsv and not deploy_cfg.deployment_shape:
            # Strip /versions/... suffix to get the parent deployment shape
            idx = dsv.find("/versions/")
            deploy_cfg.deployment_shape = dsv[:idx] if idx >= 0 else dsv
            logger.info("Deployment shape from training shape: %s", deploy_cfg.deployment_shape)

    if profile and profile.pipeline_parallelism > 1:
        pp_rec = compute_pp_recommendation(profile, completions_per_prompt)
        logger.info(
            "PP recommendation: prompt_groups_per_step=%d (current=%d)",
            pp_rec.recommended_prompts_per_step, prompt_groups_per_step,
        )

    if profile and cfg.max_seq_len is None:
        cfg.max_seq_len = profile.max_supported_context_length
        logger.info("max_seq_len from training shape: %d", cfg.max_seq_len)
    if cfg.max_seq_len is None:
        cfg.max_seq_len = 4096
        logger.info("max_seq_len defaulting to %d (no training shape)", cfg.max_seq_len)

    ref_profile = None
    if infra.ref_training_shape_id:
        ref_profile = rlor_mgr.resolve_training_profile(infra.ref_training_shape_id)
    use_reference = ref_profile is not None
    if not use_reference:
        logger.info("No ref_training_shape_id set, skipping reference model")

    # -- Infrastructure setup -----------------------------------------------

    _infra_start = time.time()
    policy_job_id: str | None = None
    reference_job_id: str | None = None
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

    with ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup, ExitStack() as stack:
        if cfg.policy_job_id and cfg.deployment_id:
            dep_info = None
            logger.info("Skipping deployment setup — using pre-created resources")
        else:
            dep_info = setup_deployment(deploy_mgr, deploy_cfg, cfg.base_model, infra)
            if not cfg.deployment_id and deploy_cfg.deployment_id and os.environ.get("KEEP_DEPLOYMENT", "0") != "1":
                cleanup.deployment(deploy_cfg.deployment_id)
            elif deploy_cfg.deployment_id and os.environ.get("KEEP_DEPLOYMENT", "0") == "1":
                logger.info("Keeping deployment %s (KEEP_DEPLOYMENT=1)", deploy_cfg.deployment_id)

        # -- Create or reuse trainer jobs ------------------------------------
        # Pre-created job IDs let CI scripts manage jobs externally (e.g. via
        # firectl-admin for regions the main gateway doesn't support yet).

        def _make_job(label: str, precreated_id: str | None, job_profile=None, **extra_kw):
            if precreated_id:
                ep = create_trainer_job(
                    rlor_mgr, base_model=cfg.base_model, infra=infra,
                    job_id=precreated_id,
                )
                return ep, precreated_id, True
            ep = create_trainer_job(
                rlor_mgr, base_model=cfg.base_model, infra=infra, profile=job_profile,
                lora_rank=cfg.lora_rank, max_seq_len=cfg.max_seq_len,
                learning_rate=cfg.learning_rate,
                display_name=f"frozen-lake-{label}",
                hot_load_deployment_id=deploy_cfg.deployment_id if label == "policy" else None,  # weight sync target deployment
                **extra_kw,
            )
            return ep, ep.job_id, False

        precreated_policy = False
        precreated_reference = False
        with ThreadPoolExecutor(max_workers=2) as pool:
            pol_fut = pool.submit(_make_job, "policy", cfg.policy_job_id, job_profile=profile)
            ref_fut = (
                pool.submit(_make_job, "reference", cfg.reference_job_id, job_profile=ref_profile, forward_only=True)
                if use_reference else None
            )

            policy_ep, policy_job_id, precreated_policy = pol_fut.result()
            if ref_fut:
                reference_ep, reference_job_id, precreated_reference = ref_fut.result()
            else:
                reference_ep, reference_job_id, precreated_reference = None, None, False

            if not precreated_policy:
                cleanup.trainer(policy_job_id)
            if not precreated_reference and reference_job_id:
                cleanup.trainer(reference_job_id)

        policy = ReconnectableClient(
            rlor_mgr, policy_ep.job_id, cfg.base_model, cfg.lora_rank,
            fw_api_key=api_key,
            endpoint=policy_ep,
        )
        if hasattr(policy, "close"):
            stack.callback(policy.close)
        reference = (
            ReconnectableClient(
                rlor_mgr, reference_ep.job_id, cfg.base_model, cfg.lora_rank,
                fw_api_key=api_key,
                endpoint=reference_ep,
            )
            if reference_ep else None
        )
        if reference is not None and hasattr(reference, "close"):
            stack.callback(reference.close)

        if dep_info:
            inference_model = dep_info.inference_model
        elif deploy_cfg.deployment_id:
            inference_model = f"{cfg.base_model}#accounts/{deploy_mgr.account_id}/deployments/{deploy_cfg.deployment_id}"
        else:
            inference_model = cfg.base_model
        weight_syncer = WeightSyncer(
            policy_client=policy.inner, deploy_mgr=deploy_mgr,
            deployment_id=deploy_cfg.deployment_id, base_model=cfg.base_model,
            hotload_timeout=weight_sync_cfg.weight_sync_timeout,
            first_checkpoint_type=weight_sync_cfg.first_checkpoint_type,
            dcp_timeout=weight_sync_cfg.dcp_timeout,
        )

        infra_boot_time = time.time() - _infra_start
        wandb_log({"train/step": 0, "infra/total_boot_time": infra_boot_time}, step=0)

        from training.utils.checkpoint_utils import resolve_resume
        resume_info = resolve_resume(policy, cfg.log_path)
        step_offset = resume_info.step if resume_info else 0
        if weight_sync_cfg.weight_sync_before_training and deploy_cfg.deployment_id:
            name = f"resume-{step_offset}-base" if step_offset > 0 else "step-0-base"
            weight_syncer.save_and_hotload(name, checkpoint_type="base")

        # -- Wait for deployment readiness -----------------------------------

        inference_url = deploy_mgr.inference_url
        logger.info("Waiting for deployment to be ready for inference...")
        import httpx
        _inference_prefix = "/v1" if cfg.inference_base_url else "/inference/v1"
        _readiness_url = inference_url.rstrip("/") + _inference_prefix + "/completions"
        for _ready_attempt in range(600):
            try:
                _resp = httpx.post(
                    _readiness_url,
                    json={"model": inference_model, "prompt": "test", "max_tokens": 1},
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=15,
                )
                if _resp.status_code == 200:
                    logger.info("Deployment is ready for inference")
                    break
                logger.info("Deployment not ready yet (status=%d), waiting...", _resp.status_code)
                time.sleep(5)
                continue
            except Exception as e:
                logger.info("Readiness check failed (%s), waiting...", e)
                time.sleep(5)
        else:
            logger.warning("Deployment readiness timeout, proceeding anyway")

        # -- Build rollout processor ----------------------------------------
        rollout_base_url = inference_url.rstrip("/") + ("" if cfg.inference_base_url else "/inference")
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
        rollout_config = RolloutProcessorConfig(
            completion_params={"model": inference_model},
            mcp_config_path="",
            steps=cfg.max_steps,
            semaphore=asyncio.Semaphore(cfg.max_concurrent),
        )

        adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, **DEFAULT_ADAM)
        # Client-side fallback: build the Python loss closure used by
        # forward_backward_custom(...) when no eligible builtin kernel exists.
        client_loss_builder = build_loss_fn(
            policy_loss=cfg.policy_loss, kl_beta=cfg.kl_beta,
            ratio_log_cap=cfg.ratio_log_cap,
            tis_config=TISConfig(),
        )

        # -- Trajectory logging -----------------------------------------------
        trajectory_path = f"/tmp/frozen_lake_trajectories_{int(time.time())}.jsonl"
        trajectory_log = open(trajectory_path, "a")
        logger.info("Logging trajectories to %s", trajectory_path)
        try:
            # -- Sample one prompt group ----------------------------------------

            async def sample_one_prompt(env_context: Dict[str, Any]) -> PromptGroup | None:
                """Run completions_per_prompt rollouts for one seed, return PromptGroup."""
                rows: List[EvaluationRow] = []
                for rollout_idx in range(completions_per_prompt):
                    rows.append(EvaluationRow(
                        input_metadata=InputMetadata(
                            row_id=f"seed_{env_context.get('seed', 0)}_{rollout_idx}",
                            dataset_info={
                                "environment_context": dict(env_context),
                                "user_prompt_template": cfg.user_prompt_template,
                                "visual_prompt_template": cfg.visual_prompt_template,
                            },
                        ),
                    ))

                tasks = rollout_processor(rows, rollout_config)
                completed_rows: List[EvaluationRow] = []
                for task in tasks:
                    try:
                        result = await task
                        extra = result.execution_metadata.extra or {}
                        if extra.get("rollout_error"):
                            logger.warning(
                                "Rollout error for seed %s: %s",
                                env_context.get("seed"), extra["rollout_error"],
                            )
                            continue
                        completed_rows.append(result)
                    except Exception as e:
                        logger.warning("Rollout task failed for seed %s: %s", env_context.get("seed"), e)

                if trajectory_log:
                    for row in completed_rows:
                        extra = row.execution_metadata.extra or {}
                        entry = {
                            "seed": env_context.get("seed"),
                            "messages": [m.model_dump() if hasattr(m, "model_dump") else m for m in (row.messages or [])],
                            "step_rewards": extra.get("step_rewards", []),
                            "reward": 1.0 if extra.get("step_rewards") and float(extra["step_rewards"][-1]) > 0 else 0.0,
                            "rollout_error": extra.get("rollout_error"),
                        }
                        trajectory_log.write(json.dumps(entry) + "\n")
                        trajectory_log.flush()

                if len(completed_rows) < 2:
                    return None

                all_datums: List[tinker.Datum] = []
                all_ref_datums: List[tinker.Datum] = []
                all_rewards: List[float] = []
                all_inf_logprobs: List[List[float]] = []
                first_prompt_len = 0

                for row in completed_rows:
                    datums, prompt_len, inf_lps, rewards = evaluation_row_to_training_data(row)
                    if not datums:
                        continue
                    all_datums.extend(datums)
                    all_rewards.extend(rewards)
                    all_inf_logprobs.extend(inf_lps)
                    if first_prompt_len == 0:
                        first_prompt_len = prompt_len

                    if use_reference:
                        for d in datums:
                            ref_datum = tinker.Datum(
                                model_input=d.model_input,
                                loss_fn_inputs=d.loss_fn_inputs,
                            )
                            all_ref_datums.append(ref_datum)

                if not all_datums or len(all_rewards) < 2:
                    return None

                advantages = compute_advantages(all_rewards)

                return PromptGroup(
                    data=all_datums,
                    ref_data=all_ref_datums,
                    advantages=advantages,
                    ref_logprobs=[],
                    prompt_len=first_prompt_len,
                    rewards=all_rewards,
                    inf_logprobs=all_inf_logprobs,
                )

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

            # Server-side fast path: resolve the builtin kernel/config used by
            # forward_backward(...). Returns None when this loss has no builtin
            # implementation, and raises when the current profile is ineligible.
            builtin_server_loss = resolve_builtin_loss(
                cfg.policy_loss,
                profile,
                ratio_log_cap=cfg.ratio_log_cap,
            )

            def fwd_bwd_one(sub: list[PromptGroup]):
                data, adv, ref_lp, prompt_lens, inf_lp = combine_prompt_groups(sub)
                prox_fwd = policy.forward(data, "cross_entropy")
                prox_lp = [prox_fwd.loss_fn_outputs[i]["logprobs"].data for i in range(len(data))]
                if builtin_server_loss is not None:
                    # Server-side builtin path: pre-pack the rollout tensors
                    # into datums the trainer kernel understands, then call
                    # forward_backward(...).
                    kernel_loss, kernel_config = builtin_server_loss
                    rl_datums = build_builtin_loss_datums(
                        data,
                        adv,
                        prox_lp,
                        inf_lp,
                        prompt_lens,
                        policy_loss=cfg.policy_loss,
                    )
                    return policy.forward_backward(
                        rl_datums, kernel_loss, loss_fn_config=kernel_config,
                    )
                # Client-side custom path: execute the Python loss closure
                # returned by build_loss_fn(...) via forward_backward_custom(...).
                return policy.forward_backward_custom(
                    data, client_loss_builder(adv, ref_lp, prompt_lens, inf_lp, prox_lp),
                )

            def train_step(
                step: int,
                prompt_groups: list[PromptGroup],
                loop_stats: dict | None = None,
            ) -> tuple[int, dict]:
                """ref_forward + fwd_bwd + optim_step + metrics (1:1)."""
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

                if weight_sync_cfg.dcp_save_interval > 0 and step % weight_sync_cfg.dcp_save_interval == 0:
                    logger.info("[step %d] dcp_save...", step)
                    t0 = time.time()
                    with timer("dcp_save"):
                        weight_syncer.save_dcp(f"step-{step}")
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
                avg_kl = metrics.get("train/mean_kl", 0.0)
                mean_loss = metrics.get("train/mean_loss", 0.0)
                adv_loss = metrics.get("train/mean_adv_loss", 0.0)
                kl_pen = metrics.get("train/mean_kl_penalty", 0.0)
                mask_r = metrics.get("train/mask_ratio", 0.0)
                inf_kld = metrics.get("train/inference_kld", 0.0)
                logger.info(
                    "Step %d | Reward: %.3f | KL: %.4f | Loss: %.4f "
                    "(adv=%.4f kl_pen=%.4f) | InfKLD: %.4f | MaskRatio: %.2f",
                    step, avg_reward, avg_kl, mean_loss, adv_loss, kl_pen, inf_kld, mask_r,
                )
                reward_history.append(avg_reward)
                log_metrics_json(step, reward=avg_reward, kl=avg_kl)
                _wandb_step[0] = max(_wandb_step[0] + 1, step)
                wandb_log(metrics, _wandb_step[0])
                return step, metrics

            def _weight_sync(step: int) -> None:
                logger.info("[step %d] weight_sync: saving + loading...", step)
                t0 = time.time()
                with timer("weight_sync"):
                    weight_syncer.save_and_hotload(f"step-{step}")
                logger.info("[step %d] weight_sync: done (%.1fs)", step, time.time() - t0)

            train_fns = TrainStepFns(train_step=train_step)

            def should_accept(pg: PromptGroup) -> bool:
                return len(set(pg.rewards)) > 1

            all_prompts = seed_contexts * cfg.epochs
            logger.info(
                "Training: %d seeds x %d epochs = %d prompt groups, "
                "%d completions/prompt, %d groups/step",
                len(seed_contexts), cfg.epochs, len(all_prompts),
                completions_per_prompt, prompt_groups_per_step,
            )

            _wandb_step = [step_offset]

            def _filtered_step_callback(loop_metrics: dict) -> None:
                _wandb_step[0] += 1
                wandb_log(loop_metrics, step=_wandb_step[0])

            global_step = asyncio.run(run_rl_loop(
                sample_fns=(sample_one_prompt(ctx) for ctx in all_prompts),
                train_fns=train_fns,
                prompt_groups_per_step=prompt_groups_per_step,
                dynamic_filter_fn=should_accept,
                global_step=step_offset,
                metrics_callback=_filtered_step_callback,
                weight_sync_fn=_weight_sync if weight_sync_cfg.weight_sync_interval > 0 else None,
                weight_sync_interval=weight_sync_cfg.weight_sync_interval,
            ))

            # -- Final checkpoint -----------------------------------------------

            if global_step > step_offset:
                try:
                    cp_name = f"step-{global_step}"
                    _data_consumed = (resume_info.data_consumed if resume_info else 0) + (global_step - step_offset) * prompt_groups_per_step
                    from training.utils.checkpoint_utils import save_checkpoint
                    paths = save_checkpoint(policy, cp_name, cfg.log_path, {
                        "step": global_step,
                        "data_consumed": _data_consumed,
                        "source_job_id": policy_job_id,
                    }, kind="both")

                    if getattr(cfg, "output_model_id", None):
                        rlor_mgr.promote_checkpoint(
                            policy_job_id,
                            paths["sampler_path"],
                            cfg.output_model_id,
                        )
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
