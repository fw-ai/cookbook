#!/usr/bin/env python3
"""Train Pi on the complete DABstep default split in E2B sandboxes.

Assembly: the frozen, seeded 450-task DABstep split, prepared with the pinned
Pi CLI in waves of 64 (``dabstep/dataset.py``) + the Pi harness
(``pi.rollout.make_rollout_fn``, binary Harbor reward) +
``recipes.async_rl_loop.main`` (server-side GRPO) on SDK-selected resources.
Four split tasks provide a diagnostic evaluation and stay in training.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

from training.examples.harbor_rl.harnesses.pi.rollout import make_rollout_fn
from training.examples.harbor_rl import common
from training.examples.harbor_rl.records import write_json
from training.examples.harbor_rl.tasks.dabstep.dataset import (
    ProgressiveDABstepTasks,
    freeze_default_split,
    make_progressive_rollout_factory,
    shuffle_dataset_for_run,
)
from training.examples.harbor_rl.tito.evaluate import make_fixed_evaluation
from training.recipes.async_rl_loop import Config, main
from training.utils import WandBConfig

logger = logging.getLogger(__name__)

COMPLETIONS_PER_PROMPT = 8
PROMPT_GROUPS_PER_STEP = 8
MAX_COMPLETION_TOKENS = 65_536
MAX_SEQUENCE_TOKENS = 524_288
MAX_HEAD_OFFPOLICY_VERSIONS = 2
EVALUATION_INTERVAL = 5
CHECKPOINT_INTERVAL = 40


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_model_args(parser)
    common.add_harness_args(
        parser, max_concurrent_trials=256, terminal_failure_reward=0.0, tito_debug=True
    )
    # Resource shapes are selected by the SDK; only reattachment is exposed.
    common.add_resource_args(
        parser,
        replica_count=4,
        sample_timeout=6900,
        weight_sync_timeout=1800,
        shapes=False,
    )
    common.add_wandb_args(parser, project="harbor-rl-pi")
    parser.add_argument("--harbor-dataset", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--template-concurrency", type=int, default=8)
    parser.add_argument("--e2b-request-timeout", type=float, default=900.0)
    parser.add_argument("--start-task-index", type=int, default=0)
    parser.add_argument("--init-from-checkpoint", default=None)
    args = parser.parse_args(argv)
    if bool(args.start_task_index) != bool(args.init_from_checkpoint):
        parser.error("--start-task-index and --init-from-checkpoint go together")
    return args


def build_config(args: argparse.Namespace, *, run_dir: Path, row_count: int) -> Config:
    return Config(
        log_path=str(run_dir / "logs"),
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        completions_per_prompt=COMPLETIONS_PER_PROMPT,
        prompt_groups_per_step=PROMPT_GROUPS_PER_STEP,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        max_seq_len=MAX_SEQUENCE_TOKENS,
        max_rows=row_count,
        shuffle=False,  # the run order is frozen in run-order.json
        max_head_offpolicy_versions=MAX_HEAD_OFFPOLICY_VERSIONS,
        dcp_save_interval=CHECKPOINT_INTERVAL,
        weight_sync_timeout=args.weight_sync_timeout,
        cleanup_on_exit=not args.skip_cleanup,
        init_from_checkpoint=args.init_from_checkpoint,
        trainer=common.trainer_config(args),
        deployment=common.deploy_config(args),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"harbor-dabstep-pi-{int(time.time()) % 100000}",
        ),
    )


def run() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args()
    common.require_env(
        "FIREWORKS_API_KEY", "E2B_API_KEY", wandb_entity=args.wandb_entity
    )
    common.set_e2b_request_timeout(args.e2b_request_timeout)
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset = freeze_default_split(
        args.harbor_dataset, manifest_path=run_dir / "dataset-manifest.json"
    )
    dataset = shuffle_dataset_for_run(
        dataset, seed=args.shuffle_seed, order_path=run_dir / "run-order.json"
    )
    rows = dataset.rollout_rows()[args.start_task_index :]
    if not rows:
        raise ValueError("no DABstep tasks remain at --start-task-index")
    # Rows are placeholders; each wave is prepared and its E2B templates built
    # just before its first rollout.
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
    config = build_config(args, run_dir=run_dir, row_count=len(rows))
    write_json(
        run_dir / "launch.json",
        {
            "dataset_manifest_sha256": dataset.manifest_sha256,
            "start_task_index": args.start_task_index,
            "evaluation_tasks": list(dataset.evaluation_tasks),
            "config": asdict(config),
        },
    )
    harbor = common.harbor_settings(
        args, run_dir=run_dir, environment="e2b", trials_dir=run_dir / "trials"
    )
    logger.info("Starting DABstep Pi RL over %d tasks", len(rows))
    main(
        config,
        rollout_fn_factory=make_progressive_rollout_factory(
            task_waves, partial(make_rollout_fn, harbor=harbor)
        ),
        evaluation_fn=make_fixed_evaluation(
            dataset.evaluation_rows(),
            completions_per_prompt=COMPLETIONS_PER_PROMPT,
            max_concurrency=32,
        ),
        evaluation_interval=EVALUATION_INTERVAL,
        rows=rows,
    )


if __name__ == "__main__":
    run()
