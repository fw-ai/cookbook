#!/usr/bin/env python3
"""Train Pi on the complete DABstep split with SDK-managed resources."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from training.examples.rl.harbor.pi.rollout import make_rollout_fn
from training.examples.rl.harbor.recipes.dabstep.dataset import (
    ProgressiveDABstepTasks,
    freeze_default_split,
    make_progressive_rollout_factory,
    shuffle_dataset_for_run,
)
from training.examples.rl.harbor.tito.evaluate import make_fixed_evaluation
from training.examples.rl.harbor.tito.trial import (
    DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS,
    DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
)
from training.recipes.async_rl_loop import Config, main
from training.utils import DeployConfig, TrainerConfig, WandBConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

COMPLETIONS_PER_PROMPT = 8
PROMPT_GROUPS_PER_STEP = 8
PIPELINE_CHUNKS_PER_STEP = 2
MAX_COMPLETION_TOKENS = 65_536
MAX_SEQUENCE_TOKENS = 524_288
MAX_CONCURRENT_TRIALS = 256
MAX_HEAD_OFFPOLICY_VERSIONS = 2
EVALUATION_INTERVAL = 5
CHECKPOINT_INTERVAL = 40


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--renderer-name", required=True)
    parser.add_argument("--harbor-dataset", required=True, type=Path)
    parser.add_argument("--harbor-trial-config", default=None)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--replica-count", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument(
        "--max-concurrent-trials", type=int, default=MAX_CONCURRENT_TRIALS
    )
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--template-concurrency", type=int, default=8)
    parser.add_argument("--e2b-request-timeout", type=float, default=900.0)
    parser.add_argument("--sample-timeout", type=int, default=6900)
    parser.add_argument("--weight-sync-timeout", type=int, default=1800)
    parser.add_argument(
        "--harness-tool-timeout-seconds",
        type=int,
        default=DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--retry-include-exception",
        action="append",
        default=[],
        help="Additional qualified transient exception type; repeat as needed",
    )
    parser.add_argument("--start-task-index", type=int, default=0)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "harbor-rl-pi"),
    )
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--tito-debug", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.replica_count < 1:
        raise ValueError("--replica-count must be positive")
    if args.max_concurrent_trials < 1:
        raise ValueError("--max-concurrent-trials must be positive")
    if args.template_concurrency < 1:
        raise ValueError("--template-concurrency must be positive")
    if args.e2b_request_timeout <= 0:
        raise ValueError("--e2b-request-timeout must be positive")
    if args.learning_rate < 0:
        raise ValueError("--learning-rate must be non-negative")
    if args.start_task_index < 0:
        raise ValueError("--start-task-index must be non-negative")
    if bool(args.start_task_index) != bool(args.init_from_checkpoint):
        raise ValueError(
            "--start-task-index and --init-from-checkpoint must be supplied together"
        )
    for name in ("FIREWORKS_API_KEY", "E2B_API_KEY"):
        if not os.environ.get(name):
            raise ValueError(f"{name} must be set")
    if args.wandb_entity and not os.environ.get("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY must be set when --wandb-entity is used")


def _build_config(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    row_count: int,
) -> Config:
    """Build the fixed recipe while leaving resource selection to the SDK."""

    return Config(
        log_path=str(run_dir / "logs"),
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        kl_beta=0.0,
        completions_per_prompt=COMPLETIONS_PER_PROMPT,
        prompt_groups_per_step=PROMPT_GROUPS_PER_STEP,
        pipeline_chunks_per_step=PIPELINE_CHUNKS_PER_STEP,
        min_group_size=1,
        max_incomplete_group_retries=0,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        max_seq_len=MAX_SEQUENCE_TOKENS,
        temperature=1.0,
        epochs=1,
        max_rows=row_count,
        shuffle=False,
        seed=0,
        lora_rank=0,
        max_head_offpolicy_versions=MAX_HEAD_OFFPOLICY_VERSIONS,
        max_concurrency_rollout_sample=None,
        router_replay=True,
        router_replay_completion_only=True,
        grad_clip_norm=0.0,
        eps_clip=0.2,
        anchor_logp="rollout",
        server_side_grpo=True,
        dcp_save_interval=CHECKPOINT_INTERVAL,
        weight_sync_timeout=args.weight_sync_timeout,
        cleanup_on_exit=True,
        init_from_checkpoint=args.init_from_checkpoint,
        save_final_checkpoint=True,
        trainer=TrainerConfig(),
        deployment=DeployConfig(
            tokenizer_model=args.tokenizer_model,
            tokenizer_revision=args.tokenizer_revision,
            replica_count=args.replica_count,
            sample_timeout=args.sample_timeout,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name
            or f"harbor-dabstep-pi-{int(time.time()) % 100000}",
        ),
    )


def _rollout_extras(args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    retry_names = sorted(
        DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS | frozenset(args.retry_include_exception)
    )
    return {
        "renderer_name": args.renderer_name,
        "max_concurrent_trials": args.max_concurrent_trials,
        "terminal_failure_reward": 0.0,
        "harbor_reward_key": "reward",
        "retry_include_exceptions": retry_names,
        "harness_tool_timeout_seconds": args.harness_tool_timeout_seconds,
        "harbor_trial_config": args.harbor_trial_config,
        "harbor_environment": "e2b",
        "harbor_trials_dir": str(run_dir / "trials"),
        "tito_sidecar_bundle_root": str(run_dir / "sidecar-bundles"),
        "tito_debug_enabled": args.tito_debug,
        "tito_debug_run_id": run_dir.name,
        "tito_debug_redact_text": False,
    }


def _write_launch_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    task_names: tuple[str, ...],
    evaluation_tasks: tuple[str, ...],
    dataset_manifest_sha256: str,
) -> None:
    task_order_sha256 = hashlib.sha256(
        ("\n".join(task_names) + "\n").encode()
    ).hexdigest()
    document = {
        "schema_version": 1,
        "dataset": "adyen/DABstep",
        "task_count": len(task_names),
        "task_names": list(task_names),
        "shuffle_seed": args.shuffle_seed,
        "task_order_sha256": task_order_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "harness": "pi",
        "environment": "e2b",
        "resources": {
            "selection": "automatic",
            "replica_count": args.replica_count,
            "cleanup_on_exit": True,
        },
        "training": {
            "algorithm": "server-side-grpo",
            "reward": "harbor-binary",
            "learning_rate": args.learning_rate,
            "completions_per_prompt": COMPLETIONS_PER_PROMPT,
            "prompt_groups_per_step": PROMPT_GROUPS_PER_STEP,
            "pipeline_chunks_per_step": PIPELINE_CHUNKS_PER_STEP,
            "epochs": 1,
            "max_head_offpolicy_versions": MAX_HEAD_OFFPOLICY_VERSIONS,
            "anchor_logp": "rollout",
            "kl_beta": 0.0,
            "lora_rank": 0,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "evaluation_interval": EVALUATION_INTERVAL,
            "evaluation_tasks": list(evaluation_tasks),
            "evaluation_completions_per_prompt": COMPLETIONS_PER_PROMPT,
        },
        "limits": {
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "max_concurrent_trials": args.max_concurrent_trials,
            "e2b_request_timeout_seconds": args.e2b_request_timeout,
            "sample_timeout_seconds": args.sample_timeout,
            "tool_timeout_seconds": args.harness_tool_timeout_seconds,
        },
        "resume": {
            "checkpoint": args.init_from_checkpoint,
            "start_task_index": args.start_task_index,
        },
    }
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run() -> None:
    args = parse_args()
    _validate_args(args)

    # Harbor does not yet expose E2B's control-plane timeout in TrialConfig.
    # Keep the provider request alive while a high-concurrency wave queues.
    import e2b.connection_config as e2b_connection_config

    e2b_connection_config.REQUEST_TIMEOUT = args.e2b_request_timeout
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    dataset = freeze_default_split(
        args.harbor_dataset,
        manifest_path=run_dir / "dataset-manifest.json",
    )
    dataset = shuffle_dataset_for_run(
        dataset,
        seed=args.shuffle_seed,
        order_path=run_dir / "run-order.json",
    )
    all_rows = dataset.rollout_rows()
    if args.start_task_index >= len(all_rows):
        raise ValueError("no DABstep tasks remain at --start-task-index")
    rows = all_rows[args.start_task_index :]
    task_waves = ProgressiveDABstepTasks(
        dataset,
        run_root=run_dir,
        start_task_index=args.start_task_index,
        template_concurrency=args.template_concurrency,
        context_limit=MAX_SEQUENCE_TOKENS,
        output_limit=MAX_COMPLETION_TOKENS,
        trial_config=args.harbor_trial_config,
        tool_timeout_seconds=args.harness_tool_timeout_seconds,
    )
    rollout_factory = make_progressive_rollout_factory(task_waves, make_rollout_fn)
    # Evaluation is a diagnostic view of four training tasks, not a holdout.
    evaluation_fn = make_fixed_evaluation(
        dataset.evaluation_rows(),
        completions_per_prompt=COMPLETIONS_PER_PROMPT,
        max_concurrency=32,
    )
    _write_launch_manifest(
        run_dir / "launch.json",
        args=args,
        task_names=dataset.task_names,
        evaluation_tasks=dataset.evaluation_tasks,
        dataset_manifest_sha256=dataset.manifest_sha256,
    )
    logger.info("Starting DABstep Pi RL over %d tasks", len(rows))
    main(
        _build_config(args, run_dir=run_dir, row_count=len(rows)),
        rollout_fn_factory=rollout_factory,
        evaluation_fn=evaluation_fn,
        evaluation_interval=EVALUATION_INTERVAL,
        rows=rows,
        rollout_extras=_rollout_extras(args, run_dir),
    )


if __name__ == "__main__":
    run()
