#!/usr/bin/env python3
"""Run a tiny Qwen3-4B SAO/PPO actor-critic convergence check.

This is deliberately a batch-size-one overfit test: repeating one
deterministic trajectory keeps the run small and makes movement in the critic
loss easy to distinguish from dataset variance.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import tinker

from training.recipes.experiment.ppo_value_head_loop import (
    QWEN3_4B,
    QWEN3_4B_LORA_SHAPE,
    QWEN3_4B_TOKENIZER,
    Config,
    main,
)
from training.utils import DeployConfig, TrainerConfig, WandBConfig
from training.utils.rl.losses import PromptGroup


def _metric_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run() -> dict[str, object]:
    artifact_dir = Path(os.environ.get("SAO_CONVERGENCE_ARTIFACT_DIR", "/tmp/sao-convergence"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "metrics.jsonl"
    os.environ["COOKBOOK_METRICS_FILE"] = str(metrics_path)

    base_model = os.environ.get("FIREWORKS_E2E_MODEL", QWEN3_4B)
    training_shape = os.environ.get(
        "FIREWORKS_E2E_LORA_TRAINING_SHAPE", QWEN3_4B_LORA_SHAPE
    )
    tokenizer_model = os.environ.get(
        "FIREWORKS_E2E_TOKENIZER_MODEL", QWEN3_4B_TOKENIZER
    )
    trainer = dict(
        training_shape_id=training_shape,
        use_reservation=False,
        inactivity_timeout="1800s",
        pending_timeout_s=60 * 60,
    )

    rows = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Solve the arithmetic problem. End with exactly one integer "
                        "inside <answer> and </answer>."
                    ),
                },
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            "ground_truth": "<answer>4</answer>",
        }
    ]

    async def deterministic_trajectory(
        _row: dict[str, object],
        *,
        cursor_index: int,
    ) -> PromptGroup:
        del cursor_index
        tokens = [1, 2, 3, 4]
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=tokens,
                    dtype="int64",
                    shape=[len(tokens)],
                ),
                "weights": tinker.TensorData(
                    data=[0.0, 0.0, 1.0, 1.0],
                    dtype="float32",
                    shape=[len(tokens)],
                ),
            },
        )
        return PromptGroup(
            data=[datum],
            advantages=[1.0],
            ref_logprobs=None,
            prompt_len=2,
            rewards=[1.0],
            completion_lens=[2],
            truncated=[False],
            completions=["<answer>4</answer>"],
        )

    result = main(
        Config(
            log_path=str(artifact_dir / "checkpoints"),
            actor_base_model=base_model,
            critic_base_model=base_model,
            actor_learning_rate=1e-6,
            critic_learning_rate=1e-4,
            critic_warmup_batches=0,
            critic_projection_head_dim=1,
            completions_per_prompt=1,
            prompt_groups_per_batch=1,
            max_completion_tokens=64,
            temperature=0.2,
            epochs=6,
            shuffle=False,
            max_rows=1,
            max_seq_len=512,
            lora_rank=64,
            lora_alpha=128,
            critic_lora_rank=64,
            critic_lora_alpha=128,
            actor_trainer=TrainerConfig(**trainer),
            critic_trainer=TrainerConfig(**trainer),
            deployment=DeployConfig(
                tokenizer_model=tokenizer_model,
                replica_count=1,
                sample_timeout=600,
                deployment_timeout_s=90 * 60,
            ),
            dcp_save_interval=0,
            save_final_checkpoint=True,
            cleanup_on_exit=True,
            wandb=WandBConfig(),
        ),
        rows=rows,
        sample_prompt_fn=deterministic_trajectory,
    )

    records = _metric_records(metrics_path)
    losses = [
        float(record["train/critic-value_loss"])
        for record in records
        if "train/critic-value_loss" in record
    ]
    rewards = [
        float(record["rollout/filtered_reward"])
        for record in records
        if "rollout/filtered_reward" in record
    ]
    if result["actor_steps"] != 6 or result["critic_steps"] != 6:
        raise AssertionError(f"expected six actor and critic steps, got {result}")
    if len(losses) != 6 or not all(math.isfinite(value) for value in losses):
        raise AssertionError(f"expected six finite critic losses, got {losses}")
    if min(losses[1:]) >= losses[0]:
        raise AssertionError(f"critic loss never improved from its first step: {losses}")

    summary: dict[str, object] = {
        **result,
        "status": "success",
        "base_model": base_model,
        "training_shape": training_shape,
        "critic_losses": losses,
        "critic_loss_first": losses[0],
        "critic_loss_best_after_first": min(losses[1:]),
        "critic_loss_improvement": losses[0] - min(losses[1:]),
        "rewards": rewards,
    }
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("SAO_PROJECTION_CONVERGENCE_PASS", flush=True)
    return summary


if __name__ == "__main__":
    run()
