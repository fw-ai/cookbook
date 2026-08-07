#!/usr/bin/env python3
"""Run the DABstep Harbor harness on the serverless async-RL loop."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path

from training.examples.rl.harbor_rl_opencode.dabstep import (
    AdaptiveTaskSelector,
    DABstepManifest,
    rows_for_tasks,
)
from training.examples.rl.harbor_rl_opencode.evaluate import (
    evaluate_rows,
    make_fixed_evaluation,
)
from training.examples.rl.harbor_rl_opencode.harbor import (
    DEFAULT_OPENCODE_VERSION,
    load_harbor_rows,
)
from training.examples.rl.harbor_rl_opencode.rollout import make_rollout_fn
from training.recipes.experiment.async_rl_loop_serverless import (
    Config,
    main,
    run_sampling_preflight,
)
from training.utils import WandBConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "accounts/fireworks/models/kimi-k3"
DEFAULT_TOKENIZER_MODEL = "moonshotai/Kimi-K3"
DEFAULT_TOKENIZER_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
COMPLETIONS_PER_PROMPT = 8
PROMPT_GROUPS_PER_STEP = 8
PIPELINE_CHUNKS_PER_STEP = 2
MAX_COMPLETION_TOKENS = 32768
MAX_SEQ_LEN = 524288
TASK_SEED = 20260728
HOLDOUT_EVERY = 3
PREFLIGHT_REWARD_RANGE = (0.40, 0.65)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--harbor-dataset", required=True)
    parser.add_argument("--harbor-trial-config", default=None)
    parser.add_argument("--harbor-trials-dir", default=None)
    parser.add_argument("--sampling-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=320)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--evaluation-interval", type=int, default=HOLDOUT_EVERY)
    parser.add_argument("--dcp-save-interval", type=int, default=0)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--selector-state-in", default=None)
    parser.add_argument("--holdout-concurrency", type=int, default=24)
    parser.add_argument("--preflight-concurrency", type=int, default=24)
    parser.add_argument("--sample-timeout", type=float, default=2400.0)
    parser.add_argument(
        "--terminal-failure-reward",
        type=float,
        default=None,
        help=(
            "Optional reward for terminal agent failures without a verifier "
            "reward; the default retries and discards them"
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=f"./harbor_serverless_{int(time.time())}",
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "harbor-rl-opencode"),
    )
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if args.max_rows < 1:
        raise ValueError("--max-rows must be >= 1")
    if not 0 <= args.row_offset <= args.max_rows:
        raise ValueError("--row-offset must be in [0, --max-rows]")
    training_only_args = (
        args.row_offset,
        args.step_offset,
        args.resume_from,
        args.selector_state_in,
        args.wandb_run_id,
    )
    if args.sampling_only and any(training_only_args):
        raise ValueError("resume and selector-state options are training-only")
    resume_fields = (
        args.row_offset > 0,
        args.step_offset > 0,
        bool(args.resume_from),
        bool(args.selector_state_in),
    )
    if any(resume_fields) and not all(resume_fields):
        raise ValueError(
            "resume requires --row-offset, --step-offset, --resume-from, and "
            "--selector-state-in together"
        )
    if all(resume_fields):
        if args.row_offset % PROMPT_GROUPS_PER_STEP:
            raise ValueError(
                "--row-offset must land on a "
                f"{PROMPT_GROUPS_PER_STEP}-group optimizer boundary"
            )
        expected_step_offset = args.row_offset // PROMPT_GROUPS_PER_STEP
        if args.step_offset != expected_step_offset:
            raise ValueError(
                "--step-offset must equal --row-offset / "
                f"{PROMPT_GROUPS_PER_STEP} "
                f"(expected {expected_step_offset}, got {args.step_offset})"
            )
    if args.wandb_run_id and not args.wandb_entity:
        raise ValueError("--wandb-run-id requires --wandb-entity")
    if all(resume_fields) and args.wandb_entity and not args.wandb_run_id:
        raise ValueError("W&B-enabled resume requires --wandb-run-id")
    wandb_run_id = None
    if not args.sampling_only and args.wandb_entity:
        wandb_run_id = args.wandb_run_id or uuid.uuid4().hex[:8]
        os.environ["WANDB_RUN_ID"] = wandb_run_id
        os.environ["WANDB_RESUME"] = "must" if args.wandb_run_id else "never"
    manifest = DABstepManifest.load(args.manifest)
    manifest.verify_task_root(args.harbor_dataset)
    task_names = list(manifest.train_tasks + manifest.holdout_tasks)
    task_rows = load_harbor_rows(
        args.harbor_dataset,
        task_names=task_names,
        n_tasks=None,
    )
    train_rows = rows_for_tasks(task_rows, manifest.train_tasks)
    holdout_rows = rows_for_tasks(task_rows, manifest.holdout_tasks)
    reference_rows = rows_for_tasks(task_rows, manifest.reference_step_tasks)

    run_name = args.wandb_run_name or (
        f"harbor-dabstep-serverless-{'preflight' if args.sampling_only else 'train'}-"
        f"{int(time.time()) % 100000}"
    )
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = Config(
        base_model=DEFAULT_BASE_MODEL,
        tokenizer_model=DEFAULT_TOKENIZER_MODEL,
        tokenizer_revision=DEFAULT_TOKENIZER_REVISION,
        completions_per_prompt=COMPLETIONS_PER_PROMPT,
        prompt_groups_per_step=PROMPT_GROUPS_PER_STEP,
        pipeline_chunks_per_step=PIPELINE_CHUNKS_PER_STEP,
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
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=run_name,
        ),
    )
    extras: dict[str, object] = {
        "renderer_name": "kimi_k3",
        "rollout_retries": 3,
        "terminal_failure_reward": args.terminal_failure_reward,
        "opencode_version": DEFAULT_OPENCODE_VERSION,
        "harbor_trial_config": args.harbor_trial_config,
        "harbor_trials_dir": args.harbor_trials_dir,
    }

    if args.sampling_only:

        async def evaluate_preflight(step, rollout_fn):
            return await evaluate_rows(
                rollout_fn,
                reference_rows,
                completions_per_prompt=COMPLETIONS_PER_PROMPT,
                metric_prefix="preflight",
                step=step,
                max_concurrency=args.preflight_concurrency,
            )

        metrics = run_sampling_preflight(
            config,
            rollout_fn_factory=make_rollout_fn,
            evaluation_fn=evaluate_preflight,
            rollout_extras=extras,
        )
        print(json.dumps(metrics, sort_keys=True))
        reward = float(metrics["preflight/reward"])
        reward_min, reward_max = PREFLIGHT_REWARD_RANGE
        if not reward_min <= reward <= reward_max:
            raise RuntimeError(
                f"DABstep preflight reward {reward:.4f} is outside "
                f"[{reward_min:.2f}, {reward_max:.2f}]"
            )
        return

    selector = AdaptiveTaskSelector(
        task_rows=train_rows,
        profile=manifest.profile,
        group_size=COMPLETIONS_PER_PROMPT,
        groups_per_batch=PROMPT_GROUPS_PER_STEP,
        seed=TASK_SEED,
    )
    if args.selector_state_in:
        selector.load_state_dict(
            json.loads(
                Path(args.selector_state_in).expanduser().read_text(encoding="utf-8")
            )
        )
    rows = [{"id": f"dabstep-group-{index}"} for index in range(args.max_rows)]
    extras["task_selector"] = selector

    evaluation_fn = make_fixed_evaluation(
        holdout_rows,
        completions_per_prompt=COMPLETIONS_PER_PROMPT,
        max_concurrency=args.holdout_concurrency,
    )

    result = main(
        config,
        rollout_fn_factory=make_rollout_fn,
        evaluation_fn=evaluation_fn,
        evaluation_interval=args.evaluation_interval,
        rows=rows,
        rollout_extras=extras,
    )
    selector_state_path = run_dir / "selector-state.json"
    selector_state_path.parent.mkdir(parents=True, exist_ok=True)
    selector_state_path.write_text(
        json.dumps(selector.state_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["selector_state"] = str(selector_state_path.resolve())
    selection_ledger_path = run_dir / "task-selections.json"
    selection_ledger_path.write_text(
        json.dumps(selector.selected_names(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["selection_ledger"] = str(selection_ledger_path)
    run_state_path = run_dir / "run-state.json"
    result["run_state"] = str(run_state_path)
    if wandb_run_id:
        result["wandb_run_id"] = wandb_run_id
        result["wandb_url"] = (
            f"https://wandb.ai/{args.wandb_entity}/{args.wandb_project}/runs/"
            f"{wandb_run_id}"
        )
    run_state_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Serverless Harbor training complete: %s", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run()
