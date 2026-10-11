#!/usr/bin/env python3
"""GRPO on FrozenLake multi-turn tool calls, with eval-protocol rollouts.

The policy plays FrozenLake through a ``lake_move`` tool: each turn it sees the
grid (text, or a rendered image with ``--observation-mode image``) and calls
the tool once. An episode ends in a hole, at the goal or after ``--max-steps``
moves; its reward is 1 when the agent reaches the goal. Training is
``recipes/async_rl_loop.py`` driven by this file's task code:

- ``make_frozen_lake_evaluator``: the eval-protocol ``@evaluation_test`` that
  owns the seed rows, the ``FrozenLakeToolRolloutProcessor`` and the reward.
- ``episode_to_sample``: one played episode -> one token-native sample. The
  model's tool calls are trained; tool replies and turn framing are context.
- ``make_eval_protocol_rollout_fn_factory`` turns both into the recipe's
  ``rollout_fn``. Groups whose episodes all score the same are skipped.

Usage:
    Follow the setup instructions in ../../../README.md.
    export FIREWORKS_API_KEY=...
    python -m training.examples.rl.frozen_lake.train_frozen_lake --training-shape <shape_id>

Reuse a trainer and deployment with ``--policy-job-id`` and ``--deployment-id``.
``verify_rollout.py`` plays the same episodes against a serving model without
a trainer.
"""

# No ``from __future__ import annotations``: eval-protocol checks the reward
# function's annotations by identity.
import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from eval_protocol.dataset_logger import default_logger
from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata
from eval_protocol.pytest import evaluation_test

from training.examples.rl.frozen_lake.frozen_lake_rollout import (
    FrozenLakeToolRolloutProcessor,
)
from training.recipes import async_rl_loop
from training.recipes.async_rl_loop import Config, RolloutFnFactory, RolloutSetup
from training.utils import DeployConfig, TrainerConfig, WandBConfig
from training.utils.rl.losses import has_reward_variance
from training.utils.rl.rollout import RolloutSample
from training.utils.rl.rollout.eval_protocol import (
    load_eval_protocol_input_rows,
    make_eval_protocol_rollout_fn_factory,
)
from training.utils.rl.rollout.turn_traces import turn_traces_to_sample

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = os.path.join(os.path.dirname(__file__), "seeds.jsonl")


@dataclass
class FrozenLakeTask:
    seed_jsonl_path: str = DEFAULT_SEEDS
    """One JSON object per line; each is an episode's ``environment_context``
    (``{"seed": 7}`` draws a random 4x4 map with that seed)."""
    max_seeds: int = 20
    max_steps: int = 12
    """Moves per episode."""
    observation_mode: str = "text"
    """``text`` grid, or a rendered ``image`` for vision models."""
    allow_plaintext_action_fallback: bool = False
    """Accept a bare LEFT/DOWN/RIGHT/UP when the model emits no tool call."""
    log_rollouts: bool = False
    """Log every episode to eval-protocol's local store (browse with ``ep logs``)."""


# -- Task: rows, environment and reward (eval-protocol) -----------------------


def frozen_lake_reward(row: EvaluationRow) -> EvaluationRow:
    """Score 1.0 when the episode's last move reached the goal."""
    step_rewards = (row.execution_metadata.extra or {}).get("step_rewards") or []
    reached_goal = bool(step_rewards) and float(step_rewards[-1]) > 0
    row.evaluation_result = EvaluateResult(
        score=1.0 if reached_goal else 0.0,
        reason="reached the goal" if reached_goal else "did not reach the goal",
    )
    return row


def make_frozen_lake_evaluator(task: FrozenLakeTask):
    """The eval-protocol evaluation test the trainer rolls out."""
    with open(task.seed_jsonl_path) as f:
        contexts = [json.loads(line) for line in f if line.strip()][: task.max_seeds]
    rows = [
        EvaluationRow(
            input_metadata=InputMetadata(
                row_id=f"seed_{context['seed']}",
                dataset_info={"environment_context": context},
            )
        )
        for context in contexts
    ]
    processor = FrozenLakeToolRolloutProcessor(
        observation_mode=task.observation_mode,
        allow_plaintext_action_fallback=task.allow_plaintext_action_fallback,
    )
    return evaluation_test(
        input_rows=[rows],
        rollout_processor=processor,
        steps=task.max_steps,
        mode="pointwise",
        logger=default_logger if task.log_rollouts else None,
    )(frozen_lake_reward)


def evaluator_rows(evaluator) -> list[dict[str, Any]]:
    """The evaluator's rows as recipe dataset rows."""
    return [
        {"id": row.input_metadata.row_id, "evaluation_row": row}
        for row in load_eval_protocol_input_rows(evaluator)
    ]


# -- Rollout: one episode -> one training sample -----------------------------


def episode_to_sample(row: EvaluationRow) -> RolloutSample | None:
    """Pack a played episode; drop it when a tool call could not be parsed."""
    extra = row.execution_metadata.extra or {}
    if extra.get("rollout_error"):
        logger.warning(
            "Dropping FrozenLake episode %s: %s",
            row.input_metadata.row_id,
            extra["rollout_error"],
        )
        return None
    return turn_traces_to_sample(
        extra["token_turn_traces"],
        reward=row.evaluation_result.score,
        finish_reason=row.execution_metadata.finish_reason or "stop",
    )


def completion_params(setup: RolloutSetup, _params: Any) -> dict[str, Any]:
    """Sample from the recipe's deployment with the recipe's sampling settings."""
    gateway = setup.inference_base_url.rstrip("/").removesuffix("/inference")
    sample_kwargs = setup.sample_kwargs
    return {
        "model": setup.model,
        "base_url": f"{gateway}/inference",
        "api_key": setup.api_key,
        "tokenizer_name_or_path": setup.tokenizer_id,
        "temperature": sample_kwargs["temperature"],
        "max_tokens": sample_kwargs["max_tokens"],
        # Full-vocabulary sampling: the trainer normalizes over the full vocabulary.
        "top_p": sample_kwargs.get("top_p", 1.0),
        "top_k": sample_kwargs.get("top_k", 0),
    }


def make_rollout_fn_factory(evaluator) -> RolloutFnFactory:
    return make_eval_protocol_rollout_fn_factory(
        evaluator,
        sample_converter=episode_to_sample,
        completion_params_factory=completion_params,
    )


# -- Assembly -----------------------------------------------------------------


def main(cfg: Config, task: FrozenLakeTask | None = None) -> dict[str, Any]:
    evaluator = make_frozen_lake_evaluator(task or FrozenLakeTask())
    return async_rl_loop.main(
        cfg,
        rollout_fn_factory=make_rollout_fn_factory(evaluator),
        rows=evaluator_rows(evaluator),
        dynamic_filter_fn=has_reward_variance,
    )


def parse_args(argv: list[str] | None = None) -> tuple[Config, FrozenLakeTask]:
    parser = argparse.ArgumentParser(description="GRPO on FrozenLake tool calls")
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3-8b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--training-shape", default=os.environ.get("TRAINING_SHAPE"))
    parser.add_argument("--policy-job-id", help="Reuse this trainer job.")
    parser.add_argument("--deployment-id", help="Reuse this deployment.")
    parser.add_argument("--deployment-shape")
    parser.add_argument("--deployment-replica-count", type=int)
    parser.add_argument("--seed-jsonl-path", default=DEFAULT_SEEDS)
    parser.add_argument("--max-seeds", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=12, help="Moves per episode.")
    parser.add_argument("--observation-mode", choices=("text", "image"), default="text")
    parser.add_argument("--allow-plaintext-action-fallback", action="store_true")
    parser.add_argument("--log-rollouts", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--completions-per-prompt", type=int, default=4)
    parser.add_argument("--prompt-groups-per-step", type=int, default=4)
    parser.add_argument("--max-concurrent", type=int, default=16, help="Episodes in flight.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--log-path", default="./frozen_lake_logs")
    parser.add_argument("--output-model-id")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "frozen-lake-grpo")
    )
    args = parser.parse_args(argv)

    cfg = Config(
        log_path=args.log_path,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        epochs=args.epochs,
        lora_rank=args.lora_rank,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_concurrency_rollout_sample=args.max_concurrent,
        min_group_size=2,
        weight_sync_timeout=900,
        trainer=TrainerConfig(
            job_id=args.policy_job_id, training_shape_id=args.training_shape
        ),
        deployment=DeployConfig(
            deployment_id=args.deployment_id,
            deployment_shape=args.deployment_shape,
            replica_count=args.deployment_replica_count,
            tokenizer_model=args.tokenizer_model,
        ),
        wandb=WandBConfig(entity=args.wandb_entity, project=args.wandb_project),
        output_model_id=args.output_model_id,
    )
    task = FrozenLakeTask(
        seed_jsonl_path=args.seed_jsonl_path,
        max_seeds=args.max_seeds,
        max_steps=args.max_steps,
        observation_mode=args.observation_mode,
        allow_plaintext_action_fallback=args.allow_plaintext_action_fallback,
        log_rollouts=args.log_rollouts,
    )
    return cfg, task


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main(*parse_args())
