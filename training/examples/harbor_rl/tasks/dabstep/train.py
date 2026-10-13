#!/usr/bin/env python3
"""Train OpenCode on pinned DABstep tasks with the serverless async RL loop.

Assembly: the manifest's 67 training / 8 holdout tasks
(``dabstep/manifest.py``) + the OpenCode harness behind an adaptive task
selector (``make_adaptive_rollout_factory``) +
``recipes.experiment.async_rl_loop_serverless.main``. ``--sampling-only`` runs
the six reference tasks through ``run_sampling_preflight`` and checks the
reward lands in the calibrated band. A run continues on new resources with
``--resume-from``/``--row-offset``/``--step-offset``/``--selector-state-in``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from functools import partial
from pathlib import Path

from training.examples.harbor_rl.harnesses.opencode.rollout import make_rollout_fn
from training.examples.harbor_rl import common
from training.examples.harbor_rl.records import select_rows, write_json
from training.examples.harbor_rl.tasks.dabstep.manifest import (
    AdaptiveTaskSelector,
    DABstepManifest,
    make_adaptive_rollout_factory,
)
from training.examples.harbor_rl.tito.evaluate import make_fixed_evaluation
from training.examples.harbor_rl.tito.trial import load_harbor_rows
from training.recipes.experiment.async_rl_loop_serverless import (
    Config,
    main,
    run_sampling_preflight,
)
from training.utils import WandBConfig

logger = logging.getLogger(__name__)

COMPLETIONS_PER_PROMPT = 8
PROMPT_GROUPS_PER_STEP = 8
MAX_COMPLETION_TOKENS = 32768
MAX_SEQ_LEN = 524288
TASK_SEED = 20260728
PREFLIGHT_REWARD_RANGE = (0.40, 0.65)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_model_args(parser)
    common.add_harness_args(parser)
    common.add_wandb_args(parser, project="harbor-rl-opencode")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--harbor-dataset", required=True)
    parser.add_argument("--harbor-trials-dir", type=Path, default=None)
    parser.add_argument(
        "--run-dir", type=Path, default=Path(f"./harbor_serverless_{int(time.time())}")
    )
    parser.add_argument("--sampling-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=320)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sample-timeout", type=float, default=2400.0)
    parser.add_argument("--evaluation-interval", type=int, default=3)
    parser.add_argument("--holdout-concurrency", type=int, default=24)
    parser.add_argument("--preflight-concurrency", type=int, default=24)
    parser.add_argument("--dcp-save-interval", type=int, default=0)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--selector-state-in", type=Path, default=None)
    parser.add_argument("--wandb-run-id", default=None)
    args = parser.parse_args(argv)
    _validate_resume(parser, args)
    return args


def _validate_resume(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    resume = (
        args.row_offset > 0,
        args.step_offset > 0,
        bool(args.resume_from),
        bool(args.selector_state_in),
    )
    if args.sampling_only and (any(resume) or args.wandb_run_id):
        parser.error("resume and W&B-run options are training-only")
    if any(resume) and not all(resume):
        parser.error(
            "resume requires --row-offset, --step-offset, --resume-from, and "
            "--selector-state-in together"
        )
    if not 0 <= args.row_offset <= args.max_rows:
        parser.error("--row-offset must be in [0, --max-rows]")
    if args.row_offset % PROMPT_GROUPS_PER_STEP or (
        args.step_offset != args.row_offset // PROMPT_GROUPS_PER_STEP
    ):
        parser.error(
            f"--row-offset must be a multiple of {PROMPT_GROUPS_PER_STEP} and "
            "--step-offset must equal --row-offset / "
            f"{PROMPT_GROUPS_PER_STEP}"
        )
    if args.wandb_run_id and not args.wandb_entity:
        parser.error("--wandb-run-id requires --wandb-entity")
    if all(resume) and args.wandb_entity and not args.wandb_run_id:
        parser.error("W&B-enabled resume requires --wandb-run-id")


def build_config(args: argparse.Namespace, *, run_dir: Path, run_name: str) -> Config:
    return Config(
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        tokenizer_revision=args.tokenizer_revision,
        completions_per_prompt=COMPLETIONS_PER_PROMPT,
        prompt_groups_per_step=PROMPT_GROUPS_PER_STEP,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        max_seq_len=args.max_seq_len,
        max_rows=args.max_rows,
        lora_rank=args.lora_rank,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        sample_timeout=args.sample_timeout,
        snapshot_prefix=run_name,
        metrics_file=str(run_dir / "metrics.jsonl"),
        init_from_checkpoint=args.resume_from,
        step_offset=args.step_offset,
        resolved_rows_offset=args.row_offset,
        dcp_save_interval=args.dcp_save_interval,
        wandb=WandBConfig(entity=args.wandb_entity, project=args.wandb_project, run_name=run_name),
    )


def run() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args()
    manifest = DABstepManifest.load(args.manifest)
    manifest.verify_task_root(args.harbor_dataset)
    task_rows = load_harbor_rows(
        args.harbor_dataset,
        task_names=list(manifest.train_tasks + manifest.holdout_tasks),
    )
    mode = "preflight" if args.sampling_only else "train"
    run_name = args.wandb_run_name or (
        f"harbor-dabstep-serverless-{mode}-{int(time.time()) % 100000}"
    )
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args, run_dir=run_dir, run_name=run_name)
    factory = partial(
        make_rollout_fn,
        harbor=common.harbor_settings(
            args, run_dir=run_dir, environment="docker", trials_dir=args.harbor_trials_dir
        ),
    )

    if args.sampling_only:
        metrics = run_sampling_preflight(
            config,
            rollout_fn_factory=factory,
            evaluation_fn=make_fixed_evaluation(
                select_rows(task_rows, manifest.reference_step_tasks),
                completions_per_prompt=COMPLETIONS_PER_PROMPT,
                metric_prefix="preflight",
                max_concurrency=args.preflight_concurrency,
            ),
        )
        print(json.dumps(metrics, sort_keys=True))
        reward_min, reward_max = PREFLIGHT_REWARD_RANGE
        if not reward_min <= float(metrics["preflight/reward"]) <= reward_max:
            raise RuntimeError(
                f"DABstep preflight reward {metrics['preflight/reward']:.4f} is "
                f"outside [{reward_min:.2f}, {reward_max:.2f}]"
            )
        return

    wandb_run_id = common.pin_wandb_run_id(args.wandb_entity, args.wandb_run_id)
    selector = AdaptiveTaskSelector(
        task_rows=select_rows(task_rows, manifest.train_tasks),
        profile=manifest.profile,
        group_size=COMPLETIONS_PER_PROMPT,
        groups_per_batch=PROMPT_GROUPS_PER_STEP,
        seed=TASK_SEED,
    )
    if args.selector_state_in:
        selector.load_state_dict(
            json.loads(args.selector_state_in.expanduser().read_text(encoding="utf-8"))
        )
    result = main(
        config,
        rollout_fn_factory=make_adaptive_rollout_factory(selector, factory),
        evaluation_fn=make_fixed_evaluation(
            select_rows(task_rows, manifest.holdout_tasks),
            completions_per_prompt=COMPLETIONS_PER_PROMPT,
            max_concurrency=args.holdout_concurrency,
        ),
        evaluation_interval=args.evaluation_interval,
        # Placeholder rows: the selector picks each group's task.
        rows=[{"id": f"dabstep-group-{index}"} for index in range(args.max_rows)],
    )
    result.update(
        selector_state=str(
            write_json(run_dir / "selector-state.json", selector.state_dict())
        ),
        selection_ledger=str(
            write_json(run_dir / "task-selections.json", selector.selected_names())
        ),
        wandb_run_id=wandb_run_id,
    )
    write_json(run_dir / "run-state.json", result)
    logger.info("Serverless DABstep training complete: %s", result)


if __name__ == "__main__":
    run()
