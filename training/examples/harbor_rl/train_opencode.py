#!/usr/bin/env python3
"""Train OpenCode on any Harbor dataset with the dedicated async RL loop.

Assembly: Harbor task rows (``load_harbor_rows``) + the OpenCode harness
(``opencode.rollout.make_rollout_fn``: one Docker or E2B trial per rollout with
an environment-local TITO sidecar and the Harbor verifier reward) +
``recipes.async_rl_loop.main`` (server-side GRPO). ``--sampling-only`` scores
the same rollout against an existing deployment without a trainer.

    python -m training.examples.harbor_rl.train_opencode \\
      --base-model accounts/<account>/models/<model> \\
      --tokenizer-model <tokenizer> --renderer-name <certified-renderer> \\
      --harbor-dataset ./prepared-tasks --run-dir ./runs/<run-id>
"""

from __future__ import annotations

import argparse
import logging
import time
from functools import partial
from pathlib import Path

from training.examples.harbor_rl.harnesses.opencode.constants import DEFAULT_OPENCODE_VERSION
from training.examples.harbor_rl.harnesses.opencode.rollout import make_rollout_fn
from training.examples.harbor_rl import common
from training.examples.harbor_rl.records import select_rows
from training.examples.harbor_rl.tito.evaluate import make_fixed_evaluation
from training.examples.harbor_rl.tito.trial import load_harbor_rows
from training.recipes.async_rl_loop import Config, RolloutFnFactory, main
from training.utils import WandBConfig

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_model_args(parser)
    common.add_harness_args(parser)
    common.add_resource_args(parser, replica_count=1, sample_timeout=600, weight_sync_timeout=600)
    common.add_wandb_args(parser, project="harbor-rl-opencode")
    parser.add_argument(
        "--harbor-dataset",
        default="terminal-bench@2.0",
        help="Local task/dataset path or Harbor registry name[@version]",
    )
    parser.add_argument(
        "--harbor-task",
        action="append",
        default=[],
        help="Exact task name to include; repeat to freeze an ordered cohort",
    )
    parser.add_argument("--harbor-registry-path", default=None)
    parser.add_argument("--harbor-environment", choices=("docker", "e2b"), default="docker")
    parser.add_argument(
        "--harbor-trials-dir",
        type=Path,
        default=None,
        help="Retain Harbor trials here; default uses temporary directories",
    )
    parser.add_argument("--opencode-version", default=DEFAULT_OPENCODE_VERSION)
    parser.add_argument("--run-dir", type=Path, default=Path("./harbor_opencode_run"))
    parser.add_argument("--sampling-only", action="store_true")
    parser.add_argument(
        "--evaluation-task",
        action="append",
        default=[],
        help="Fixed evaluation task; repeat. It stays in the training population.",
    )
    parser.add_argument("--evaluation-every", type=int, default=3)
    parser.add_argument("--evaluation-concurrency", type=int, default=24)
    parser.add_argument("--max-rows", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--completions-per-prompt", type=int, default=4)
    parser.add_argument("--prompt-groups-per-step", type=int, default=1)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--max-head-offpolicy-versions", type=int, default=0)
    parser.add_argument("--dcp-save-interval", type=int, default=0)
    parser.add_argument("--output-model-id", default=None)
    return parser.parse_args(argv)


def rollout_fn_factory(args: argparse.Namespace, run_dir: Path) -> RolloutFnFactory:
    return partial(
        make_rollout_fn,
        harbor=common.harbor_settings(
            args,
            run_dir=run_dir,
            environment=args.harbor_environment,
            trials_dir=args.harbor_trials_dir,
        ),
        version=args.opencode_version,
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
        lora_rank=args.lora_rank,
        max_head_offpolicy_versions=args.max_head_offpolicy_versions,
        dcp_save_interval=args.dcp_save_interval,
        weight_sync_timeout=args.weight_sync_timeout,
        cleanup_on_exit=not args.skip_cleanup,
        output_model_id=args.output_model_id,
        trainer=common.trainer_config(args),
        deployment=common.deploy_config(args),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"harbor-rl-opencode-{int(time.time()) % 100000}",
        ),
    )


def run() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    rows = load_harbor_rows(
        args.harbor_dataset,
        registry_path=args.harbor_registry_path,
        task_names=args.harbor_task or None,
        n_tasks=args.max_rows,
    )
    if not rows:
        raise ValueError(f"No Harbor tasks found for {args.harbor_dataset!r}")
    logger.info("Loaded %d Harbor tasks", len(rows))
    factory = rollout_fn_factory(args, run_dir)

    if args.sampling_only:
        common.run_sampling_only(
            args,
            factory,
            rows,
            completions_per_prompt=args.completions_per_prompt,
            max_concurrency=args.evaluation_concurrency,
            result_path=run_dir / "sampling-result.json",
        )
        return

    evaluation_fn = None
    if args.evaluation_task:
        evaluation_fn = make_fixed_evaluation(
            select_rows(rows, args.evaluation_task),
            completions_per_prompt=args.completions_per_prompt,
            max_concurrency=args.evaluation_concurrency,
        )
    main(
        build_config(args, run_dir=run_dir, row_count=len(rows)),
        rollout_fn_factory=factory,
        evaluation_fn=evaluation_fn,
        evaluation_interval=args.evaluation_every,
        rows=rows,
    )


if __name__ == "__main__":
    run()
