#!/usr/bin/env python3
"""Generic Harbor/OpenCode training and sampling recipe."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from training.examples.rl.harbor.recipes.dabstep.manifest import (
    AdaptiveTaskSelector,
    DABstepManifest,
    rows_for_tasks,
)
from training.examples.rl.harbor.tito.evaluate import (
    evaluate_rows,
    make_fixed_evaluation,
)
from training.examples.rl.harbor.opencode.rollout import (
    DEFAULT_MAX_CONCURRENT_TRIALS,
    make_rollout_fn,
)
from training.examples.rl.harbor.opencode.constants import DEFAULT_OPENCODE_VERSION
from training.examples.rl.harbor.tito.trial import (
    DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
    DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS,
    load_harbor_rows,
)
from training.recipes.async_rl_loop import Config, RolloutSetup, main
from training.utils import DeployConfig, TrainerConfig, WandBConfig
from training.utils.rl.rollout.lifecycle import close_rollout_fn
from training.utils.tokenizers import load_tokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_HARBOR_DATASET = "terminal-bench@2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fireworks-native Harbor RL with the async loop"
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument(
        "--harbor-dataset",
        default=DEFAULT_HARBOR_DATASET,
        help="Local task/dataset path or Harbor dataset name[@version]",
    )
    parser.add_argument(
        "--harbor-task",
        action="append",
        default=[],
        help="Exact task name to include; repeat to preserve an explicit order",
    )
    parser.add_argument("--harbor-registry-path", default=None)
    parser.add_argument(
        "--harbor-trial-config",
        default=None,
        help="Optional Harbor TrialConfig YAML",
    )
    parser.add_argument(
        "--harbor-environment",
        choices=("docker", "e2b"),
        default="docker",
        help="Harbor sandbox backend; local Docker remains the default",
    )
    parser.add_argument("--harbor-trials-dir", default=None)
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        help="Run rollouts against --deployment-id without starting a trainer",
    )
    parser.add_argument("--tito-debug", action="store_true")
    parser.add_argument(
        "--tito-prompt-mode",
        choices=("full_history", "incremental"),
        default="full_history",
        help=(
            "Prompt construction mode. Incremental is experimental and requires "
            "a model-specific exact-checkpoint suffix/junction implementation."
        ),
    )
    parser.add_argument(
        "--renderer-name",
        required=True,
        help=(
            "Production-certified TITO renderer matching --base-model and "
            "--tokenizer-model; unsupported model/template pairs fail closed"
        ),
    )
    parser.add_argument("--rollout-retries", type=int, default=3)
    parser.add_argument(
        "--retry-include-exception",
        action="append",
        default=None,
        help=(
            "Additional transient provider-create exception type supported by "
            "the installed Harbor/E2B version; repeat to extend the allowlist"
        ),
    )
    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_TRIALS,
        help="Maximum active Harbor agent trials; each trial owns one trajectory",
    )
    parser.add_argument(
        "--terminal-failure-reward",
        type=float,
        default=None,
        help=(
            "Optional reward for terminal agent failures without a verifier "
            "reward; the default retries and discards them"
        ),
    )
    parser.add_argument("--opencode-version", default=DEFAULT_OPENCODE_VERSION)
    parser.add_argument(
        "--harness-tool-timeout-seconds",
        type=int,
        default=DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
        help=(
            "Per-tool timeout inside the agent; it must be positive and below "
            "the Harbor trial's resolved outer agent timeout"
        ),
    )
    parser.add_argument(
        "--dabstep-manifest",
        default=None,
        help="Pinned 67/8 DABstep manifest; enables adaptive training and holdout eval",
    )
    parser.add_argument("--task-seed", type=int, default=20260728)
    parser.add_argument(
        "--evaluation-task",
        action="append",
        default=[],
        help=(
            "Fixed evaluation task from --harbor-dataset; repeat for multiple "
            "tasks. These rows remain in the training population."
        ),
    )
    parser.add_argument("--evaluation-every", type=int, default=3)
    parser.add_argument("--evaluation-concurrency", type=int, default=24)
    parser.add_argument("--output-model-id", default=None)
    parser.add_argument(
        "--warm-start-from-adapter",
        default=None,
        help="Promoted PEFT model resource or adapter path for weights-only recovery",
    )
    parser.add_argument("--max-rows", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--completions-per-prompt", type=int, default=4)
    parser.add_argument("--prompt-groups-per-step", type=int, default=1)
    parser.add_argument("--pipeline-chunks-per-step", type=int, default=1)
    parser.add_argument("--min-group-size", type=int, default=1)
    parser.add_argument("--max-incomplete-group-retries", type=int, default=0)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-beta", type=float, default=0.001)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--max-head-offpolicy-versions", type=int, default=0)
    parser.add_argument(
        "--grad-accumulation-normalization",
        choices=("none", "num_loss_tokens"),
        default="none",
    )
    parser.add_argument(
        "--dcp-save-interval",
        type=int,
        default=0,
        help="Save resumable optimizer state every N steps (0 disables it)",
    )
    parser.add_argument(
        "--weight-sync-timeout",
        type=int,
        default=600,
        help="Seconds to wait for each sampler hotload",
    )
    parser.add_argument("--sample-timeout", type=int, default=600)
    parser.add_argument("--trainer-job-id", default=None)
    parser.add_argument("--training-shape-id", default=None)
    parser.add_argument(
        "--deployment-shape",
        default=None,
        help="Optional versioned RFT deployment shape; defaults to the training profile",
    )
    parser.add_argument(
        "--deployment-id",
        default=None,
        help=(
            "Existing inference deployment to reattach and hot-load in place; "
            "when omitted the SDK creates the recipe deployment"
        ),
    )
    parser.add_argument("--replica-count", type=int, default=1)
    parser.add_argument("--log-path", default="./harbor_opencode_logs")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "harbor-rl-opencode"),
    )
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def _rollout_extras(
    args: argparse.Namespace,
    *,
    selector: AdaptiveTaskSelector | None,
) -> dict[str, object]:
    retry_include_exceptions = None
    if args.retry_include_exception:
        retry_include_exceptions = sorted(
            DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS
            | frozenset(args.retry_include_exception)
        )
    return {
        "renderer_name": args.renderer_name,
        "rollout_retries": args.rollout_retries,
        "retry_include_exceptions": retry_include_exceptions,
        "max_concurrent_trials": args.max_concurrent_trials,
        "terminal_failure_reward": args.terminal_failure_reward,
        "opencode_version": args.opencode_version,
        "harness_tool_timeout_seconds": args.harness_tool_timeout_seconds,
        "task_selector": selector,
        "harbor_trial_config": args.harbor_trial_config,
        "harbor_environment": args.harbor_environment,
        "harbor_trials_dir": args.harbor_trials_dir,
        "tito_sidecar_bundle_root": str(
            Path(args.log_path).expanduser().resolve() / ".tito-sidecar-bundles"
        ),
        "tito_debug_enabled": args.tito_debug,
        "tito_prompt_mode": args.tito_prompt_mode,
    }


def _run_sampling_only(
    args: argparse.Namespace,
    *,
    rows: list[dict],
    selector: AdaptiveTaskSelector | None,
) -> None:
    if not args.deployment_id:
        raise ValueError("--sampling-only requires --deployment-id")
    if args.trainer_job_id:
        raise ValueError("--sampling-only does not accept --trainer-job-id")
    if not args.harbor_trials_dir:
        raise ValueError("--sampling-only requires --harbor-trials-dir")

    tokenizer = load_tokenizer(args.tokenizer_model, args.tokenizer_revision)
    setup = RolloutSetup(
        tokenizer=tokenizer,
        tokenizer_id=args.tokenizer_model,
        sample_kwargs={
            "max_tokens": args.max_completion_tokens,
            "temperature": args.temperature,
            "top_p": 1.0,
            "top_k": 0,
            "max_seq_len": args.max_seq_len,
            "http_timeout": args.sample_timeout,
            "logprobs": True,
            "include_routing_matrix": True,
            "echo": False,
        },
        inference_base_url=os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai"
        ),
        api_key=os.environ["FIREWORKS_API_KEY"],
        model=args.deployment_id,
        completions_per_prompt=args.completions_per_prompt,
        extras=_rollout_extras(args, selector=selector),
    )
    rollout_fn = make_rollout_fn(setup)

    async def evaluate() -> dict[str, float | int]:
        try:
            return await evaluate_rows(
                rollout_fn,
                rows,
                completions_per_prompt=args.completions_per_prompt,
                metric_prefix="sampling",
                step=0,
                max_concurrency=None,
            )
        finally:
            await close_rollout_fn(rollout_fn)

    metrics = asyncio.run(evaluate())
    output_dir = Path(args.log_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": "sampling_only",
        "deployment_id": args.deployment_id,
        "tasks": [row.get("task_name") for row in rows],
        "completions_per_prompt": args.completions_per_prompt,
        "metrics": metrics,
        "trials_dir": str(Path(args.harbor_trials_dir).expanduser().resolve()),
    }
    result_path = output_dir / "sampling-result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**result, "result_path": str(result_path)}, sort_keys=True))


def run() -> None:
    args = parse_args()
    if not args.sampling_only and not args.training_shape_id:
        raise ValueError("training requires --training-shape-id")
    if args.evaluation_every < 1:
        raise ValueError("--evaluation-every must be positive")
    if args.evaluation_concurrency < 1:
        raise ValueError("--evaluation-concurrency must be positive")
    manifest = (
        DABstepManifest.load(args.dabstep_manifest) if args.dabstep_manifest else None
    )
    if manifest is not None and args.evaluation_task:
        raise ValueError(
            "--evaluation-task cannot be combined with --dabstep-manifest; "
            "the manifest already owns its holdout set"
        )
    selector = None
    evaluation_rows: list[dict] = []
    if manifest is not None:
        manifest.verify_task_root(args.harbor_dataset)
        task_names = list(manifest.train_tasks + manifest.holdout_tasks)
        task_rows = load_harbor_rows(
            args.harbor_dataset,
            registry_path=args.harbor_registry_path,
            task_names=task_names,
            n_tasks=None,
        )
        train_rows = rows_for_tasks(task_rows, manifest.train_tasks)
        evaluation_rows = rows_for_tasks(task_rows, manifest.holdout_tasks)
        selector = AdaptiveTaskSelector(
            task_rows=train_rows,
            profile=manifest.profile,
            group_size=args.completions_per_prompt,
            groups_per_batch=args.prompt_groups_per_step,
            seed=args.task_seed,
        )
        rows = [{"id": f"dabstep-group-{index}"} for index in range(args.max_rows)]
    else:
        rows = load_harbor_rows(
            args.harbor_dataset,
            registry_path=args.harbor_registry_path,
            task_names=args.harbor_task or None,
            n_tasks=args.max_rows,
        )
        if args.evaluation_task:
            evaluation_rows = rows_for_tasks(rows, tuple(args.evaluation_task))
    if not rows:
        raise ValueError(f"No Harbor tasks found for {args.harbor_dataset!r}")
    logger.info("Loaded %d Harbor tasks", len(rows))

    if args.sampling_only:
        _run_sampling_only(args, rows=rows, selector=selector)
        return

    config = Config(
        log_path=args.log_path,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        completions_per_prompt=args.completions_per_prompt,
        prompt_groups_per_step=args.prompt_groups_per_step,
        pipeline_chunks_per_step=args.pipeline_chunks_per_step,
        min_group_size=args.min_group_size,
        max_incomplete_group_retries=args.max_incomplete_group_retries,
        max_completion_tokens=args.max_completion_tokens,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=len(rows),
        shuffle=manifest is None,
        lora_rank=args.lora_rank,
        max_head_offpolicy_versions=args.max_head_offpolicy_versions,
        grad_accumulation_normalization=(
            None
            if args.grad_accumulation_normalization == "none"
            else args.grad_accumulation_normalization
        ),
        dcp_save_interval=args.dcp_save_interval,
        weight_sync_timeout=args.weight_sync_timeout,
        warm_start_from_adapter=args.warm_start_from_adapter,
        output_model_id=args.output_model_id,
        trainer=TrainerConfig(
            job_id=args.trainer_job_id,
            training_shape_id=args.training_shape_id,
        ),
        deployment=DeployConfig(
            deployment_id=args.deployment_id,
            tokenizer_model=args.tokenizer_model,
            tokenizer_revision=args.tokenizer_revision,
            deployment_shape=args.deployment_shape,
            replica_count=args.replica_count,
            sample_timeout=args.sample_timeout,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name
            or f"harbor-rl-opencode-{int(time.time()) % 100000}",
        ),
    )

    evaluation_fn = None
    if evaluation_rows:
        evaluation_fn = make_fixed_evaluation(
            evaluation_rows,
            completions_per_prompt=args.completions_per_prompt,
            max_concurrency=args.evaluation_concurrency,
        )

    main(
        config,
        rollout_fn_factory=make_rollout_fn,
        evaluation_fn=evaluation_fn,
        evaluation_interval=args.evaluation_every,
        rows=rows,
        rollout_extras=_rollout_extras(args, selector=selector),
    )


if __name__ == "__main__":
    run()
