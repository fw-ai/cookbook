"""Small, end-to-end SAO actor/critic run on public Qwen3-4B.

Run ``python -m training.examples.rl.SAO.qwen3_4b_sao --help`` from the
``cookbook`` directory. With no dataset, four built-in arithmetic prompts
exercise sampling, a GPU projection-head critic, DIS actor updates, hotload,
and a resumable final checkpoint. Pass ``--dataset`` for your own JSONL rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from training.recipes.experiment.ppo_value_head_loop import (
    QWEN3_4B as MODEL,
    QWEN3_4B_LORA_SHAPE as TRAINING_SHAPE,
    QWEN3_4B_TOKENIZER as TOKENIZER,
    main,
    sao_config,
)
from training.utils import DeployConfig, TrainerConfig, WandBConfig


def smoke_rows() -> list[dict]:
    """Tiny math task with an exact terminal reward from the PPO recipe."""
    problems = [("2 + 2", 4), ("7 + 5", 12), ("9 - 3", 6), ("6 * 3", 18)]
    return [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Solve the arithmetic problem. End your answer with exactly "
                        "one integer inside <answer> and </answer>."
                    ),
                },
                {"role": "user", "content": f"What is {expression}?"},
            ],
            "ground_truth": f"<answer>{answer}</answer>",
        }
        for expression, answer in problems
    ]


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-shape",
        default=os.environ.get("FIREWORKS_TRAINING_SHAPE", TRAINING_SHAPE),
        help=(
            f"Qwen3-4B LoRA training shape (default: {TRAINING_SHAPE}). "
            "FIREWORKS_TRAINING_SHAPE also overrides this default."
        ),
    )
    parser.add_argument("--dataset", help="JSONL rows with messages and ground_truth")
    critic_shape = os.environ.get("FIREWORKS_CRITIC_TRAINING_SHAPE")
    parser.add_argument(
        "--critic-training-shape",
        default=critic_shape,
        help=(
            "Compatible validated Qwen3-4B LoRA shape; "
            "critic_projection_head_dim configures the critic head"
        ),
    )
    parser.add_argument(
        "--task",
        choices=("arithmetic", "deepmath"),
        default="arithmetic",
        help="Select the answer format and reward function for the dataset",
    )
    parser.add_argument(
        "--deployment-shape",
        help="Optional validated inference shape override for actor sampling",
    )
    parser.add_argument("--actor-job-id", help="Reattach an existing actor trainer")
    parser.add_argument("--critic-job-id", help="Reattach an existing critic trainer")
    parser.add_argument("--deployment-id", help="Reattach an existing actor sampler deployment")
    parser.add_argument("--output-dir", default="./qwen3_4b_sao_run")
    parser.add_argument(
        "--preserve-jobs",
        action="store_true",
        help="Keep trainers and sampler after the run for checkpoint inspection or resume",
    )
    parser.add_argument("--max-rows", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--prompt-groups-per-batch", type=int, default=2)
    parser.add_argument("--wandb-project", help="Optional W&B project")
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="Optional W&B entity",
    )
    args = parser.parse_args()
    if not args.critic_training_shape and not args.critic_job_id:
        parser.error("provide --critic-training-shape or --critic-job-id")
    return args


def run(args: argparse.Namespace) -> dict:
    load_dotenv()
    if not os.environ.get("FIREWORKS_API_KEY"):
        raise ValueError("Set FIREWORKS_API_KEY to a training-scoped key.")
    if os.environ.get("FIREWORKS_BASE_URL", "").rstrip("/").endswith("/training"):
        raise ValueError(
            "FIREWORKS_BASE_URL must be the API root (https://api.fireworks.ai), "
            "not the legacy /training endpoint."
        )
    if args.max_rows < 1 or args.max_completion_tokens < 1 or args.max_seq_len < 1:
        raise ValueError(
            "max-rows, max-completion-tokens, and max-seq-len must be positive"
        )
    if args.wandb_project and not args.wandb_entity:
        raise ValueError("--wandb-project also requires --wandb-entity")
    if args.task == "deepmath" and not args.dataset:
        raise ValueError("--task deepmath requires --dataset from deepmath/prepare_data.py")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["COOKBOOK_METRICS_FILE"] = str(output_dir / "metrics.jsonl")
    trainer = TrainerConfig(
        job_id=args.actor_job_id,
        training_shape_id=args.training_shape,
        inactivity_timeout="1800s",
    )
    config = sao_config(
        log_path=str(output_dir / "checkpoints"),
        actor_base_model=MODEL,
        critic_base_model=MODEL,
        dataset=args.dataset,
        actor_trainer=trainer,
        critic_trainer=TrainerConfig(
            job_id=args.critic_job_id,
            training_shape_id=args.critic_training_shape,
            inactivity_timeout="1800s",
        ),
        deployment=DeployConfig(
            deployment_id=args.deployment_id,
            tokenizer_model=TOKENIZER,
            deployment_shape=args.deployment_shape,
            replica_count=1,
            wait_for_trainer_before_deployment=True,
        ),
        critic_projection_head_dim=1,
        max_rows=args.max_rows,
        max_completion_tokens=args.max_completion_tokens,
        max_seq_len=args.max_seq_len,
        prompt_groups_per_batch=args.prompt_groups_per_batch,
        critic_warmup_batches=0,
        # Keep the smoke run short; full runs can supply ValuePretrainData.
        value_pretrain_steps=0,
        cleanup_on_exit=not args.preserve_jobs,
        wandb=(
            WandBConfig(entity=args.wandb_entity, project=args.wandb_project)
            if args.wandb_project
            else WandBConfig()
        ),
    )
    if args.task == "deepmath":
        from training.examples.rl.deepmath.train_deepmath import deepmath_reward

        grading_reward = deepmath_reward
    else:
        grading_reward = None
    result = main(
        config,
        rows=None if args.dataset else smoke_rows(),
        reward=grading_reward,
    )
    if result["actor_steps"] < 1 or result["critic_steps"] < 1:
        raise RuntimeError(f"No actor/critic update completed: {result}")
    if not result["deployment_id"]:
        raise RuntimeError(f"No rollout deployment was provisioned: {result}")
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True), flush=True)
