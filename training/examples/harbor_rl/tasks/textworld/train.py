#!/usr/bin/env python3
"""Train (or score) Pi on the frozen TextWorld cooking suite in E2B sandboxes.

Assembly: frozen TextWorld games as Harbor tasks (``textworld/dataset.py``),
copied with the pinned Pi CLI baked in + the Pi harness
(``pi.rollout.make_rollout_fn`` with the action-only ``textworld`` tool
profile and a binary win reward) + ``recipes.async_rl_loop.main``
(server-side GRPO). ``--sampling-only`` scores the evaluation games on an
existing deployment to check for headroom before training.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

from training.examples.harbor_rl.harnesses.pi.constants import (
    PI_AGENT_PROVIDER,
    PI_HARBOR_IMPORT_PATH,
    PINNED_PI_VERSION,
)
from training.examples.harbor_rl.harnesses.pi.prepare_tasks import prepare as prepare_pi_tasks
from training.examples.harbor_rl.harnesses.pi.rollout import make_rollout_fn
from training.examples.harbor_rl import common
from training.examples.harbor_rl.records import select_rows, write_json
from training.examples.harbor_rl.tasks.textworld.dataset import load_frozen_dataset
from training.examples.harbor_rl.tito.e2b_templates import prebuild_e2b_templates
from training.examples.harbor_rl.tito.evaluate import make_fixed_evaluation
from training.examples.harbor_rl.tito.trial import load_harbor_rows
from training.recipes.async_rl_loop import Config, RolloutFnFactory, main
from training.utils import WandBConfig

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_model_args(parser)
    common.add_harness_args(
        parser,
        max_concurrent_trials=128,
        tool_timeout_seconds=120,
        terminal_failure_reward=0.0,
        retry_include_exceptions=("AgentSetupTimeoutError",),
    )
    common.add_resource_args(parser, replica_count=2, sample_timeout=1800, weight_sync_timeout=1800)
    common.add_wandb_args(parser, project="harbor-rl-textworld")
    parser.add_argument("--textworld-dataset", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        help="Score the evaluation games on --deployment-id; no trainer",
    )
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None, help="Training-row cap")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--completions-per-prompt", type=int, default=8)
    parser.add_argument("--prompt-groups-per-step", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=8_192)
    parser.add_argument("--max-seq-len", type=int, default=24_576)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--max-head-offpolicy-versions", type=int, default=2)
    parser.add_argument(
        "--full-sync",
        action="store_true",
        help="On-policy batches and SYNC hot-loads (max_head_offpolicy_versions=0)",
    )
    parser.add_argument("--eval-completions-per-prompt", type=int, default=3)
    parser.add_argument("--eval-concurrency", type=int, default=32)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=40)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--template-concurrency", type=int, default=8)
    parser.add_argument("--e2b-request-timeout", type=float, default=900.0)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_completion_tokens >= args.max_seq_len:
        raise ValueError("--max-completion-tokens must be less than --max-seq-len")
    if args.sampling_only:
        common.check_sampling_only_args(args)
    elif args.shuffle_seed is None:
        raise ValueError("--shuffle-seed is required for training")
    common.require_env(
        "FIREWORKS_API_KEY", "E2B_API_KEY", wandb_entity=args.wandb_entity
    )


def rollout_fn_factory(args: argparse.Namespace, run_dir: Path) -> RolloutFnFactory:
    """Pi with the action-only TextWorld tools; debug captures stay text-redacted."""
    return partial(
        make_rollout_fn,
        harbor=common.harbor_settings(
            args,
            run_dir=run_dir,
            environment="e2b",
            trials_dir=run_dir / "trials",
            debug_redact_text=True,
        ),
        tool_profile="textworld",
    )


def build_config(args: argparse.Namespace, *, run_dir: Path, row_count: int) -> Config:
    return Config(
        log_path=str(run_dir / "logs"),
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        completions_per_prompt=args.completions_per_prompt,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_completion_tokens=args.max_completion_tokens,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=row_count,
        shuffle=False,  # the seeded order is recorded in launch.json
        max_head_offpolicy_versions=(0 if args.full_sync else args.max_head_offpolicy_versions),
        grad_clip_norm=args.grad_clip_norm,
        dcp_save_interval=args.checkpoint_interval,
        weight_sync_timeout=args.weight_sync_timeout,
        cleanup_on_exit=not args.skip_cleanup,
        init_from_checkpoint=args.init_from_checkpoint,
        trainer=common.trainer_config(args),
        deployment=common.deploy_config(args, hot_load_transition_type="SYNC" if args.full_sync else None),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"harbor-textworld-pi-{int(time.time()) % 100000}",
        ),
    )


def prebuild_templates(
    rows: list[dict], args: argparse.Namespace, run_dir: Path
) -> None:
    """Build each content-addressed E2B template once before rollout fan-out."""
    asyncio.run(
        prebuild_e2b_templates(
            rows,
            trials_dir=run_dir / "trials",
            agent_import_path=PI_HARBOR_IMPORT_PATH,
            agent_version=PINNED_PI_VERSION,
            agent_provider=PI_AGENT_PROVIDER,
            context_limit=args.max_seq_len,
            output_limit=args.max_completion_tokens,
            trial_config=args.harbor_trial_config,
            max_concurrency=args.template_concurrency,
            tool_timeout_seconds=args.harness_tool_timeout_seconds,
        )
    )


def run() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args()
    validate_args(args)
    common.set_e2b_request_timeout(args.e2b_request_timeout)
    dataset = load_frozen_dataset(args.textworld_dataset)
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    prepare_pi_tasks(dataset.root, run_dir / "prepared", PINNED_PI_VERSION)
    all_rows = load_harbor_rows(run_dir / "prepared")
    train_rows = select_rows(all_rows, dataset.train_task_ids)
    evaluation_rows = select_rows(all_rows, dataset.evaluation_task_ids)
    factory = rollout_fn_factory(args, run_dir)

    if args.sampling_only:
        prebuild_templates(evaluation_rows, args, run_dir)
        common.run_sampling_only(
            args,
            factory,
            evaluation_rows,
            completions_per_prompt=args.eval_completions_per_prompt,
            max_concurrency=args.eval_concurrency,
            result_path=run_dir / "sampling-result.json",
            dataset_manifest_sha256=dataset.manifest_sha256,
        )
        return

    random.Random(args.shuffle_seed).shuffle(train_rows)
    train_rows = train_rows[: args.max_rows]
    prebuild_templates([*train_rows, *evaluation_rows], args, run_dir)
    config = build_config(args, run_dir=run_dir, row_count=len(train_rows))
    write_json(
        run_dir / "launch.json",
        {
            "dataset_manifest_sha256": dataset.manifest_sha256,
            "shuffle_seed": args.shuffle_seed,
            "train_task_ids": [row["task_name"] for row in train_rows],
            "evaluation_task_ids": list(dataset.evaluation_task_ids),
            "config": asdict(config),
        },
    )
    logger.info(
        "Starting TextWorld Pi RL over %d training and %d evaluation games",
        len(train_rows),
        len(evaluation_rows),
    )
    main(
        config,
        rollout_fn_factory=factory,
        evaluation_fn=make_fixed_evaluation(
            evaluation_rows,
            completions_per_prompt=args.eval_completions_per_prompt,
            max_concurrency=min(args.eval_concurrency, args.max_concurrent_trials),
        ),
        evaluation_interval=args.evaluation_interval,
        rows=train_rows,
    )


if __name__ == "__main__":
    run()
