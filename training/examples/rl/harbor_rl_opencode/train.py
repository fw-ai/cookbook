#!/usr/bin/env python3
"""Train an OpenCode policy through Harbor's native Trial interface."""

from __future__ import annotations

import argparse
import logging
import os
import time

from training.examples.rl.harbor_rl_opencode.dabstep import (
    AdaptiveTaskSelector,
    DABstepManifest,
    rows_for_tasks,
)
from training.examples.rl.harbor.evaluate import make_fixed_evaluation
from training.examples.rl.harbor.rollout import make_rollout_fn
from training.examples.rl.harbor.trial import (
    DEFAULT_OPENCODE_VERSION,
    load_harbor_rows,
)
from training.recipes.async_rl_loop import Config, main
from training.utils import DeployConfig, TrainerConfig, WandBConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "accounts/fireworks/models/qwen3p5-9b"
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3p5-9b-65k-lora"
DEFAULT_HARBOR_DATASET = "terminal-bench@2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fireworks-native Harbor RL with the async loop"
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument(
        "--harbor-dataset",
        default=DEFAULT_HARBOR_DATASET,
        help="Local task/dataset path or Harbor dataset name[@version]",
    )
    parser.add_argument("--harbor-registry-path", default=None)
    parser.add_argument(
        "--harbor-trial-config",
        default=None,
        help="Optional Harbor TrialConfig YAML; local Docker is the default",
    )
    parser.add_argument("--harbor-trials-dir", default=None)
    parser.add_argument(
        "--renderer-name",
        default="qwen3_5_interleaved",
        help="Cookbook renderer used by the OpenCode recording endpoint",
    )
    parser.add_argument("--rollout-retries", type=int, default=3)
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
        "--dabstep-manifest",
        default=None,
        help="Pinned 67/8 DABstep manifest; enables adaptive training and holdout eval",
    )
    parser.add_argument("--task-seed", type=int, default=20260728)
    parser.add_argument("--holdout-every", type=int, default=3)
    parser.add_argument("--holdout-concurrency", type=int, default=24)
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
    parser.add_argument("--training-shape-id", default=DEFAULT_TRAINING_SHAPE)
    parser.add_argument(
        "--deployment-shape",
        default=None,
        help="Optional versioned RFT deployment shape; defaults to the training profile",
    )
    parser.add_argument("--replica-count", type=int, default=1)
    parser.add_argument("--log-path", default="./harbor_rl_opencode_logs")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "harbor-rl-opencode"),
    )
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    manifest = (
        DABstepManifest.load(args.dabstep_manifest) if args.dabstep_manifest else None
    )
    selector = None
    holdout_rows: list[dict] = []
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
        holdout_rows = rows_for_tasks(task_rows, manifest.holdout_tasks)
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
            n_tasks=args.max_rows,
        )
    if not rows:
        raise ValueError(f"No Harbor tasks found for {args.harbor_dataset!r}")
    logger.info("Loaded %d Harbor tasks", len(rows))

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
        trainer=TrainerConfig(training_shape_id=args.training_shape_id),
        deployment=DeployConfig(
            tokenizer_model=args.tokenizer_model,
            tokenizer_revision=args.tokenizer_revision,
            deployment_shape=args.deployment_shape,
            replica_count=args.replica_count,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name
            or f"harbor-rl-opencode-{int(time.time()) % 100000}",
        ),
    )

    evaluation_fn = None
    if holdout_rows:
        evaluation_fn = make_fixed_evaluation(
            holdout_rows,
            completions_per_prompt=args.completions_per_prompt,
            max_concurrency=args.holdout_concurrency,
        )

    main(
        config,
        rollout_fn_factory=make_rollout_fn,
        evaluation_fn=evaluation_fn,
        evaluation_interval=args.holdout_every,
        rows=rows,
        rollout_extras={
            "renderer_name": args.renderer_name,
            "rollout_retries": args.rollout_retries,
            "terminal_failure_reward": args.terminal_failure_reward,
            "opencode_version": args.opencode_version,
            "task_selector": selector,
            "harbor_trial_config": args.harbor_trial_config,
            "harbor_trials_dir": args.harbor_trials_dir,
        },
    )


if __name__ == "__main__":
    run()
