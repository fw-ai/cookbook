#!/usr/bin/env python3
"""IGPO on multi-hop QA: search a paragraph pool, then submit an answer.

Each dataset row (``prepare_data.py``) holds a question, its answer and a pool
of paragraphs. Each rollout plays one episode with
``MultiHopQARolloutProcessor``: ``search(query)`` returns the pool's best
TF-IDF matches and ``submit_answer(answer)`` ends the episode with the
answer's token F1 as its reward. IGPO (``training.utils.rl.algorithm.igpo``)
rewards every turn but the last with its information gain toward the answer
and the last turn with the F1. This file is the task code:

- ``make_rollout_fn_factory``: one search episode per ``rollout_fn(row)``
  call, packed by ``episode_to_run`` into a run whose ``TURN_SPANS`` mark the
  model's tool calls and whose ``ANSWER_TOKENS`` hold the ground truth.
- ``main``: ``rl_loop.main(cfg, rollout_fn_factory=..., prepare_batch_fn=igpo.prepare_batch(...))``.

Usage:
    Follow the setup instructions in ../../README.md.
    export FIREWORKS_API_KEY=...
    python prepare_data.py                     # HotpotQA -> dataset.jsonl
    python train_multihop_qa_igpo.py --training-shape <shape_id> --output-model-id <id>

Reuse a trainer and deployment with ``--policy-job-id`` and ``--deployment-id``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from eval_protocol.models import EvaluationRow, InputMetadata, Message
from eval_protocol.pytest.types import RolloutProcessorConfig

from training.examples.rl.multihop_qa.multihop_qa_rollout import (
    MultiHopQARolloutProcessor,
)
from training.recipes import rl_loop
from training.recipes.rl_loop import Config
from training.utils import DeployConfig, TrainerConfig, WandBConfig, build_renderer
from training.utils.rl.algorithm import igpo
from training.utils.rl.algorithm.igpo import ANSWER_TOKENS, TURN_SPANS
from training.utils.rl.rollout import RolloutRun
from training.utils.rl.rollout.setup import RolloutFn, RolloutFnFactory, RolloutSetup
from training.utils.rl.rollout.turn_traces import turn_spans, turn_traces_to_sample

logger = logging.getLogger(__name__)

DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "dataset.jsonl")


@dataclass
class MultiHopQATask:
    max_steps: int = 8
    """Tool calls per episode."""
    search_top_k: int = 2
    """Paragraphs returned per search."""
    gamma: float = 0.95
    """IGPO discount for turn-level return accumulation."""
    renderer_name: str = ""
    """Cookbook renderer; empty infers it from the tokenizer."""


def episode_row(row: dict[str, Any]) -> EvaluationRow:
    """The eval-protocol row one episode of a dataset row starts from."""
    return EvaluationRow(
        input_metadata=InputMetadata(
            dataset_info={
                "context": row.get("context") or {},
                "ground_truth": str(row.get("ground_truth", "")),
            },
        ),
        messages=[
            Message(role=message["role"], content=message["content"])
            for message in row.get("messages") or []
        ],
    )


def episode_to_run(episode: EvaluationRow, answer_tokens: list[int]) -> RolloutRun | None:
    """Pack a finished episode into one run with the turn spans and answer IGPO scores."""
    extra = episode.execution_metadata.extra or {}
    if extra.get("rollout_error"):
        logger.warning("Dropping multi-hop QA episode: %s", extra["rollout_error"])
        return None
    traces = extra["token_turn_traces"]
    sample = turn_traces_to_sample(
        traces,
        reward=extra["step_rewards"][-1],
        finish_reason=episode.execution_metadata.finish_reason or "stop",
    )
    return RolloutRun(
        segments=[sample],
        metadata={TURN_SPANS: turn_spans(traces), ANSWER_TOKENS: answer_tokens},
    )


def make_rollout_fn_factory(task: MultiHopQATask) -> RolloutFnFactory:
    def make_rollout_fn(setup: RolloutSetup) -> RolloutFn:
        processor = MultiHopQARolloutProcessor(
            tokenizer=setup.tokenizer,
            renderer=build_renderer(
                setup.tokenizer,
                setup.tokenizer_id,
                task.renderer_name,
                load_image_processor=False,
            ),
            search_top_k=task.search_top_k,
        )
        gateway = setup.inference_base_url.rstrip("/").removesuffix("/inference")
        completion_params = {
            "model": setup.model,
            "base_url": f"{gateway}/inference",
            "api_key": setup.api_key,
            "temperature": setup.sample_kwargs["temperature"],
            "max_tokens": setup.sample_kwargs["max_tokens"],
            "top_p": setup.sample_kwargs["top_p"],
            "top_k": setup.sample_kwargs["top_k"],
        }

        async def rollout_fn(row: dict[str, Any]) -> RolloutRun | None:
            answer_tokens = setup.tokenizer.encode(
                str(row.get("ground_truth", "")), add_special_tokens=False
            )
            if not answer_tokens:
                return None  # IGPO has no answer to score
            config = RolloutProcessorConfig(
                completion_params=completion_params,
                mcp_config_path="",
                steps=task.max_steps,
                semaphore=asyncio.Semaphore(1),
            )
            [episode] = processor([episode_row(row)], config)
            return episode_to_run(await episode, answer_tokens)

        return rollout_fn

    return make_rollout_fn


def main(cfg: Config, task: MultiHopQATask | None = None) -> dict[str, Any]:
    task = task or MultiHopQATask()
    return rl_loop.main(
        cfg,
        rollout_fn_factory=make_rollout_fn_factory(task),
        prepare_batch_fn=igpo.prepare_batch(gamma=task.gamma),
        # IGPO's turn-level information gain varies even when every outcome
        # reward in a group is equal, so keep those groups.
        dynamic_filter_fn=None,
    )


def parse_args(argv: list[str] | None = None) -> tuple[Config, MultiHopQATask]:
    parser = argparse.ArgumentParser(description="IGPO on multi-hop QA with search tools")
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3-8b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--renderer-name", default="", help="Empty infers it from the tokenizer.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--training-shape", default=os.environ.get("TRAINING_SHAPE"))
    parser.add_argument("--policy-job-id", help="Reuse this trainer job.")
    parser.add_argument("--deployment-id", help="Reuse this deployment.")
    parser.add_argument("--deployment-shape")
    parser.add_argument("--deployment-replica-count", type=int)
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=8, help="Tool calls per episode.")
    parser.add_argument("--search-top-k", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--completions-per-prompt", type=int, default=4)
    parser.add_argument("--prompt-groups-per-step", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--log-path", default="./multihop_qa_igpo_logs")
    parser.add_argument("--output-model-id")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "multihop-qa-igpo")
    )
    args = parser.parse_args(argv)

    cfg = Config(
        log_path=args.log_path,
        base_model=args.base_model,
        dataset=args.dataset_path,
        max_rows=args.max_rows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        completions_per_prompt=args.completions_per_prompt,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        lora_rank=args.lora_rank,
        eps_clip=args.eps_clip,
        weight_sync_timeout=1800,
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
    task = MultiHopQATask(
        max_steps=args.max_steps,
        search_top_k=args.search_top_k,
        gamma=args.gamma,
        renderer_name=args.renderer_name,
    )
    return cfg, task


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main(*parse_args())
