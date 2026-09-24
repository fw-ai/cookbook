#!/usr/bin/env python3
"""Evaluate or train Pi on the calibrated TextWorld cooking suite."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path

from training.examples.rl.harbor.pi.constants import (
    PI_HARBOR_IMPORT_PATH,
    PINNED_PI_VERSION,
)
from training.examples.rl.harbor.pi.prepare_tasks import prepare as prepare_pi_tasks
from training.examples.rl.harbor.pi.rollout import make_rollout_fn
from training.examples.rl.harbor.recipes.textworld.dataset import (
    load_frozen_dataset,
)
from training.examples.rl.harbor.tito.e2b_templates import prebuild_e2b_templates
from training.examples.rl.harbor.tito.evaluate import evaluate_rows, make_fixed_evaluation
from training.examples.rl.harbor.tito.trial import (
    DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS,
    load_harbor_rows,
)
from training.recipes.async_rl_loop import (
    Config,
    RolloutSetup,
    main,
    policy_loss_metadata,
)
from training.utils import (
    DeployConfig,
    TrainerConfig,
    WandBConfig,
    read_api_extra_headers_env,
    resolve_router_replay_enabled,
)
from training.utils.rl.rollout.lifecycle import close_rollout_fn
from training.utils.rl.cispo import CISPOConfig
from training.utils.rl.dapo import DAPOConfig
from training.utils.rl.dppo import DPPOConfig
from training.utils.rl.dro import DROConfig
from training.utils.rl.gspo import GSPOConfig
from training.utils.rl.score_centering import ScoreCenteringConfig
from training.utils.tokenizers import load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

COMPLETIONS_PER_PROMPT = 8
EVALUATION_COMPLETIONS_PER_PROMPT = 3
PROMPT_GROUPS_PER_STEP = 8
PIPELINE_CHUNKS_PER_STEP = 2
MAX_HEAD_OFFPOLICY_VERSIONS = 2
MAX_COMPLETION_TOKENS = 8_192
MAX_SEQUENCE_TOKENS = 24_576
TEXTWORLD_TOOL_TIMEOUT_SECONDS = 120


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Required in both modes: the sampling-only check must resolve the same
    # Router Replay decision the training run will make.
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--renderer-name", required=True)
    parser.add_argument("--textworld-dataset", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--shuffle-seed", default=None, type=int)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional training-row cap for controlled ablations",
    )
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        help=(
            "Score the evaluation games against --deployment-id without "
            "creating a trainer; use it to check headroom before training"
        ),
    )
    parser.add_argument(
        "--deployment-id",
        default=None,
        help=(
            "Existing inference deployment to score in sampling-only mode or "
            "reattach for training"
        ),
    )
    parser.add_argument(
        "--trainer-job-id",
        default=None,
        help="Existing managed trainer job to reattach for training",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQUENCE_TOKENS)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=MAX_COMPLETION_TOKENS,
    )
    parser.add_argument(
        "--eval-completions-per-prompt",
        type=int,
        default=EVALUATION_COMPLETIONS_PER_PROMPT,
    )
    parser.add_argument("--eval-concurrency", type=int, default=32)
    parser.add_argument("--harbor-trial-config", default=None)
    parser.add_argument("--training-shape-id", default=None)
    parser.add_argument("--replica-count", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="Global gradient clipping threshold; 0 disables clipping",
    )
    parser.add_argument(
        "--grad-norm-metrics",
        choices=("off", "basic", "detailed"),
        default="basic",
        help="Trainer-side optimizer gradient-norm telemetry",
    )
    parser.add_argument(
        "--policy-loss",
        choices=(
            "grpo",
            "gspo",
            "dapo",
            "dro",
            "cispo",
            "dppo",
            "score_centering",
        ),
        default="grpo",
        help="Policy surrogate used for group-relative advantages",
    )
    parser.add_argument("--gspo-clip-ratio-low", type=float, default=3e-4)
    parser.add_argument("--gspo-clip-ratio-high", type=float, default=4e-4)
    parser.add_argument("--gspo-seq-ratio-log-cap", type=float, default=10.0)
    parser.add_argument(
        "--gspo-token-reduction",
        choices=("mean", "sum"),
        default="mean",
        help="Reduce active response-token losses by mean (paper) or sum (ablation)",
    )
    parser.add_argument("--dapo-eps-clip", type=float, default=0.2)
    parser.add_argument("--dapo-eps-clip-high", type=float, default=0.28)
    parser.add_argument("--dapo-eps-clip-c", type=float, default=None)
    parser.add_argument("--dapo-ratio-log-cap", type=float, default=20.0)
    parser.add_argument("--dro-beta", type=float, default=0.05)
    parser.add_argument("--cispo-eps-low", type=float, default=0.2)
    parser.add_argument("--cispo-eps-high", type=float, default=0.28)
    parser.add_argument("--cispo-ratio-log-cap", type=float, default=20.0)
    parser.add_argument(
        "--dppo-divergence",
        choices=("binary_tv", "binary_kl"),
        default="binary_tv",
    )
    parser.add_argument("--dppo-threshold", type=float, default=None)
    parser.add_argument("--dppo-ratio-log-cap", type=float, default=20.0)
    parser.add_argument(
        "--score-centering-top-k",
        type=int,
        default=ScoreCenteringConfig().top_k,
    )
    parser.add_argument(
        "--score-centering-tail-mass-epsilon",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--completions-per-prompt",
        type=int,
        default=COMPLETIONS_PER_PROMPT,
    )
    parser.add_argument(
        "--prompt-groups-per-step",
        type=int,
        default=PROMPT_GROUPS_PER_STEP,
    )
    parser.add_argument(
        "--pipeline-chunks-per-step",
        type=int,
        default=PIPELINE_CHUNKS_PER_STEP,
    )
    parser.add_argument(
        "--full-sync",
        action="store_true",
        help=(
            "Use fully on-policy batches and synchronous hot-load transitions "
            "(max_head_offpolicy_versions=0)"
        ),
    )
    parser.add_argument(
        "--max-head-offpolicy-versions",
        type=int,
        default=MAX_HEAD_OFFPOLICY_VERSIONS,
        help=(
            "Maximum number of optimizer versions that rollout production may "
            "run ahead; ignored when --full-sync is set"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of passes over the frozen training task order",
    )
    parser.add_argument("--max-concurrent-trials", type=int, default=128)
    parser.add_argument("--template-concurrency", type=int, default=8)
    parser.add_argument("--e2b-request-timeout", type=float, default=900.0)
    parser.add_argument("--sample-timeout", type=int, default=1800)
    parser.add_argument("--weight-sync-timeout", type=int, default=1800)
    parser.add_argument(
        "--harness-tool-timeout-seconds",
        type=int,
        default=TEXTWORLD_TOOL_TIMEOUT_SECONDS,
    )
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=40)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "harbor-rl-textworld"),
    )
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--tito-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "replica_count",
        "max_concurrent_trials",
        "template_concurrency",
        "sample_timeout",
        "weight_sync_timeout",
        "harness_tool_timeout_seconds",
        "evaluation_interval",
        "checkpoint_interval",
        "eval_completions_per_prompt",
        "eval_concurrency",
        "max_seq_len",
        "max_completion_tokens",
        "completions_per_prompt",
        "prompt_groups_per_step",
        "pipeline_chunks_per_step",
        "epochs",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.e2b_request_timeout <= 0:
        raise ValueError("--e2b-request-timeout must be positive")
    if args.max_head_offpolicy_versions < 0:
        raise ValueError("--max-head-offpolicy-versions must be non-negative")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive")
    if args.learning_rate < 0:
        raise ValueError("--learning-rate must be non-negative")
    if args.grad_clip_norm < 0:
        raise ValueError("--grad-clip-norm must be non-negative")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.max_completion_tokens >= args.max_seq_len:
        raise ValueError("--max-completion-tokens must be less than --max-seq-len")
    if args.sampling_only:
        if not args.deployment_id:
            raise ValueError("--sampling-only requires --deployment-id")
        if args.trainer_job_id:
            raise ValueError("--sampling-only does not accept --trainer-job-id")
    else:
        if args.shuffle_seed is None:
            raise ValueError("--shuffle-seed is required for training")
    for name in ("FIREWORKS_API_KEY", "E2B_API_KEY"):
        if not os.environ.get(name):
            raise ValueError(f"{name} must be set")
    if args.wandb_entity and not os.environ.get("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY must be set when --wandb-entity is used")


def _rows_by_id(rows: list[dict]) -> dict[str, dict]:
    result = {str(row["task_name"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("prepared TextWorld rows contain duplicate task IDs")
    return result


def _build_config(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    row_count: int,
) -> Config:
    return Config(
        log_path=str(run_dir / "logs"),
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        kl_beta=0.0,
        completions_per_prompt=args.completions_per_prompt,
        prompt_groups_per_step=args.prompt_groups_per_step,
        pipeline_chunks_per_step=args.pipeline_chunks_per_step,
        min_group_size=1,
        max_incomplete_group_retries=0,
        max_completion_tokens=args.max_completion_tokens,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=row_count,
        shuffle=False,
        seed=0,
        lora_rank=0,
        max_head_offpolicy_versions=(
            0 if args.full_sync else args.max_head_offpolicy_versions
        ),
        max_concurrency_rollout_sample=None,
        router_replay=True,
        router_replay_completion_only=True,
        grad_clip_norm=args.grad_clip_norm,
        grad_norm_metrics=args.grad_norm_metrics,
        eps_clip=0.2,
        anchor_logp="rollout",
        server_side_grpo=args.policy_loss == "grpo",
        policy_loss=args.policy_loss,
        gspo_execution=(
            "two_pass" if args.policy_loss == "gspo" else "builtin"
        ),
        gspo=GSPOConfig(
            clip_ratio_low=args.gspo_clip_ratio_low,
            clip_ratio_high=args.gspo_clip_ratio_high,
            seq_ratio_log_cap=args.gspo_seq_ratio_log_cap,
            token_reduction=args.gspo_token_reduction,
        ),
        dapo=DAPOConfig(
            eps_clip=args.dapo_eps_clip,
            eps_clip_high=args.dapo_eps_clip_high,
            eps_clip_c=args.dapo_eps_clip_c,
            ratio_log_cap=args.dapo_ratio_log_cap,
        ),
        dro=DROConfig(beta=args.dro_beta),
        cispo=CISPOConfig(
            eps_low=args.cispo_eps_low,
            eps_high=args.cispo_eps_high,
            ratio_log_cap=args.cispo_ratio_log_cap,
        ),
        dppo=DPPOConfig(
            divergence=args.dppo_divergence,
            threshold=args.dppo_threshold,
            ratio_log_cap=args.dppo_ratio_log_cap,
        ),
        score_centering=ScoreCenteringConfig(
            top_k=args.score_centering_top_k,
            tail_mass_epsilon=args.score_centering_tail_mass_epsilon,
        ),
        dcp_save_interval=args.checkpoint_interval,
        weight_sync_timeout=args.weight_sync_timeout,
        cleanup_on_exit=True,
        init_from_checkpoint=args.init_from_checkpoint,
        save_final_checkpoint=True,
        trainer=TrainerConfig(
            job_id=args.trainer_job_id,
            training_shape_id=args.training_shape_id,
        ),
        deployment=DeployConfig(
            deployment_id=args.deployment_id,
            tokenizer_model=args.tokenizer_model,
            tokenizer_revision=args.tokenizer_revision,
            replica_count=args.replica_count,
            sample_timeout=args.sample_timeout,
            hot_load_transition_type="SYNC" if args.full_sync else None,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name
            or f"harbor-textworld-pi-{int(time.time()) % 100000}",
        ),
    )


def _rollout_extras(args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    return {
        "renderer_name": args.renderer_name,
        "max_concurrent_trials": args.max_concurrent_trials,
        "terminal_failure_reward": 0.0,
        "harbor_reward_key": "reward",
        "retry_include_exceptions": sorted(
            DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS | {"AgentSetupTimeoutError"}
        ),
        "harness_tool_timeout_seconds": args.harness_tool_timeout_seconds,
        "tool_profile": "textworld",
        "harbor_trial_config": args.harbor_trial_config,
        "harbor_environment": "e2b",
        "harbor_trials_dir": str(run_dir / "trials"),
        "tito_sidecar_bundle_root": str(run_dir / "sidecar-bundles"),
        "tito_debug_enabled": args.tito_debug,
        "tito_debug_run_id": run_dir.name,
        "tito_debug_redact_text": True,
    }


async def _prebuild_templates(
    rows: list[dict],
    *,
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    await prebuild_e2b_templates(
        rows,
        trials_dir=run_dir / "trials",
        agent_import_path=PI_HARBOR_IMPORT_PATH,
        agent_version=PINNED_PI_VERSION,
        agent_provider="fireworks-rl",
        context_limit=args.max_seq_len,
        output_limit=args.max_completion_tokens,
        trial_config=args.harbor_trial_config,
        max_concurrency=args.template_concurrency,
        tool_timeout_seconds=args.harness_tool_timeout_seconds,
    )


def _run_sampling_only(
    args: argparse.Namespace,
    *,
    rows: list[dict],
    run_dir: Path,
    dataset_manifest_sha256: str,
) -> None:
    """Score the frozen evaluation games against an existing deployment."""
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    sample_kwargs: dict[str, object] = {
        "max_tokens": args.max_completion_tokens,
        "temperature": args.temperature,
        "top_p": 1.0,
        "top_k": 0,
        "max_seq_len": args.max_seq_len,
        "http_timeout": args.sample_timeout,
        "logprobs": True,
    }
    # Mirror async_rl_loop: routing data exists only on MoE base models, and a
    # dense deployment rejects the request outright.
    if resolve_router_replay_enabled(
        requested=True,
        api_key=api_key,
        base_url=base_url,
        additional_headers=read_api_extra_headers_env(),
        base_model=args.base_model,
    ):
        sample_kwargs.update(include_routing_matrix=True, echo=False)
    else:
        logger.info("Router Replay skipped for dense model %s", args.base_model)
    setup = RolloutSetup(
        tokenizer=load_tokenizer(args.tokenizer_model, args.tokenizer_revision),
        tokenizer_id=args.tokenizer_model,
        sample_kwargs=sample_kwargs,
        inference_base_url=base_url,
        api_key=api_key,
        model=args.deployment_id,
        completions_per_prompt=args.eval_completions_per_prompt,
        extras=_rollout_extras(args, run_dir),
    )
    rollout_fn = make_rollout_fn(setup)

    async def evaluate() -> dict[str, float | int]:
        try:
            return await evaluate_rows(
                rollout_fn,
                rows,
                completions_per_prompt=args.eval_completions_per_prompt,
                metric_prefix="sampling",
                step=0,
                max_concurrency=args.eval_concurrency,
            )
        finally:
            await close_rollout_fn(rollout_fn)

    metrics = asyncio.run(evaluate())
    result = {
        "mode": "sampling_only",
        "deployment_id": args.deployment_id,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "task_ids": [str(row["task_name"]) for row in rows],
        "completions_per_prompt": args.eval_completions_per_prompt,
        "temperature": args.temperature,
        "max_seq_len": args.max_seq_len,
        "max_completion_tokens": args.max_completion_tokens,
        "metrics": metrics,
        "trials_dir": str(run_dir / "trials"),
    }
    result_path = run_dir / "sampling-result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "TextWorld sampling-only reward %.4f over %d games x %d completions",
        float(metrics.get("sampling/reward", 0.0)),
        len(rows),
        args.eval_completions_per_prompt,
    )
    print(json.dumps({**result, "result_path": str(result_path)}, sort_keys=True))


def run() -> None:
    args = parse_args()
    _validate_args(args)

    import e2b.connection_config as e2b_connection_config

    e2b_connection_config.REQUEST_TIMEOUT = args.e2b_request_timeout
    dataset = load_frozen_dataset(args.textworld_dataset)
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    prepared_root = run_dir / "prepared"
    prepare_pi_tasks(dataset.root, prepared_root, PINNED_PI_VERSION)

    all_rows = load_harbor_rows(prepared_root)
    by_id = _rows_by_id(all_rows)
    missing = [
        task_id
        for task_id in (*dataset.train_task_ids, *dataset.evaluation_task_ids)
        if task_id not in by_id
    ]
    if missing:
        raise ValueError(f"prepared TextWorld tasks are missing IDs: {missing}")
    train_rows = [by_id[task_id] for task_id in dataset.train_task_ids]
    evaluation_rows = [by_id[task_id] for task_id in dataset.evaluation_task_ids]

    if args.sampling_only:
        asyncio.run(
            _prebuild_templates(evaluation_rows, args=args, run_dir=run_dir)
        )
        _run_sampling_only(
            args,
            rows=evaluation_rows,
            run_dir=run_dir,
            dataset_manifest_sha256=dataset.manifest_sha256,
        )
        return

    random.Random(args.shuffle_seed).shuffle(train_rows)
    if args.max_rows is not None:
        train_rows = train_rows[: args.max_rows]
    asyncio.run(
        _prebuild_templates(
            [*train_rows, *evaluation_rows],
            args=args,
            run_dir=run_dir,
        )
    )

    config = _build_config(args, run_dir=run_dir, row_count=len(train_rows))
    launch = {
        "schema_version": 1,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "train_task_ids": [str(row["task_name"]) for row in train_rows],
        "max_rows": args.max_rows,
        "evaluation_task_ids": list(dataset.evaluation_task_ids),
        "shuffle_seed": args.shuffle_seed,
        "harness": "pi",
        "environment": "e2b",
        "reward": "binary-win",
        **policy_loss_metadata(config),
        "completions_per_prompt": args.completions_per_prompt,
        "evaluation_completions_per_prompt": args.eval_completions_per_prompt,
        "prompt_groups_per_step": args.prompt_groups_per_step,
        "pipeline_chunks_per_step": args.pipeline_chunks_per_step,
        "epochs": config.epochs,
        "max_head_offpolicy_versions": config.max_head_offpolicy_versions,
        "hot_load_transition_type": "SYNC" if args.full_sync else None,
        "training_shape_id": args.training_shape_id,
        "trainer_job_id": args.trainer_job_id,
        "deployment_id": args.deployment_id,
        "max_seq_len": args.max_seq_len,
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": args.temperature,
        "optimizer": {
            "type": "adamw",
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-12,
            "weight_decay": 0.01,
            "learning_rate": args.learning_rate,
            "grad_clip_norm": args.grad_clip_norm,
            "grad_norm_metrics": args.grad_norm_metrics,
            "grad_accumulation_normalization": (
                "num_sequences"
                if args.policy_loss == "gspo"
                else "num_loss_tokens"
                if args.policy_loss == "score_centering"
                else "none"
            ),
        },
    }
    (run_dir / "launch.json").write_text(
        json.dumps(launch, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Starting TextWorld Pi RL over %d training and %d evaluation games",
        len(train_rows),
        len(evaluation_rows),
    )
    main(
        config,
        rollout_fn_factory=make_rollout_fn,
        evaluation_fn=make_fixed_evaluation(
            evaluation_rows,
            completions_per_prompt=args.eval_completions_per_prompt,
            max_concurrency=min(
                args.eval_concurrency,
                args.max_concurrent_trials,
            ),
        ),
        evaluation_interval=args.evaluation_interval,
        rows=train_rows,
        rollout_extras=_rollout_extras(args, run_dir),
    )


if __name__ == "__main__":
    run()
