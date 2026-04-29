from __future__ import annotations

import asyncio
from dataclasses import dataclass
import sys
import types
from pathlib import Path
import importlib.util

errors_module = types.ModuleType("fireworks.training.sdk.errors")
errors_module.request_with_retries = lambda fn, *args, **kwargs: fn(*args, **kwargs)
sys.modules.setdefault("fireworks", types.ModuleType("fireworks"))
sys.modules.setdefault("fireworks.training", types.ModuleType("fireworks.training"))
sys.modules.setdefault("fireworks.training.sdk", types.ModuleType("fireworks.training.sdk"))
sys.modules.setdefault("fireworks.training.sdk.errors", errors_module)

spec = importlib.util.spec_from_file_location(
    "training_utils_data",
    Path(__file__).parents[1] / "utils" / "data.py",
)
assert spec and spec.loader
data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_module)
RLPromptDataset = data_module.RLPromptDataset


@dataclass
class _PromptGroup:
    rewards: list[float]


losses_module = types.ModuleType("training.utils.rl.losses")
losses_module.PromptGroup = _PromptGroup
sys.modules.setdefault("training.utils.rl.losses", losses_module)

train_spec = importlib.util.spec_from_file_location(
    "training_utils_rl_train",
    Path(__file__).parents[1] / "utils" / "rl" / "train.py",
)
assert train_spec and train_spec.loader
train_module = importlib.util.module_from_spec(train_spec)
sys.modules[train_spec.name] = train_module
train_spec.loader.exec_module(train_module)
TrainStepFns = train_module.TrainStepFns
run_rl_loop = train_module.run_rl_loop


def test_rl_prompt_dataset_resumes_from_persisted_cursor():
    dataset = RLPromptDataset([{"id": i} for i in range(5)], prompts_per_step=2)

    dataset.resume(3, fallback_step=4, minibatches_per_step=2)

    assert dataset.data_consumed == 3
    assert dataset.rows_from_cursor() == [{"id": 3}, {"id": 4}]


def test_rl_prompt_dataset_falls_back_to_step_cursor_for_legacy_resume():
    dataset = RLPromptDataset([{"id": i} for i in range(6)], prompts_per_step=2)

    dataset.resume(0, fallback_step=4, minibatches_per_step=2)

    assert dataset.data_consumed == 4
    assert dataset.rows_from_cursor() == [{"id": 4}, {"id": 5}]


def test_rl_prompt_dataset_records_dynamic_filter_consumption():
    dataset = RLPromptDataset([{"id": i} for i in range(8)], prompts_per_step=2)
    dataset.resume(2)

    dataset.record_batch(
        {"total_sampled": 5, "filter_drops": 2, "sample_fails": 1},
        accepted_rows=2,
    )

    assert dataset.data_consumed == 7
    assert dataset.rows_from_cursor() == [{"id": 7}]


def test_rl_prompt_dataset_records_accepted_rows_without_stats():
    dataset = RLPromptDataset([{"id": i} for i in range(3)], prompts_per_step=2)

    dataset.record_batch(None, accepted_rows=2)

    assert dataset.data_consumed == 2


def test_rl_prompt_dataset_tracks_actual_pipeline_consumption_with_dynamic_filter_and_ppo_minibatches():
    dataset = RLPromptDataset(
        [
            {"id": 0, "reward": 1.0},
            {"id": 1, "fail": True},
            {"id": 2, "reward": 0.0},
            {"id": 3, "reward": 1.0},
            {"id": 4, "reward": 1.0},
            {"id": 5, "reward": 0.0},
            {"id": 6, "reward": 1.0},
        ],
        prompts_per_step=2,
    )
    dataset.resume(0)
    ppo_minibatches = 3
    train_log = []
    sync_log = []

    async def sample_one(row):
        await asyncio.sleep(0)
        if row.get("fail"):
            return None
        return _PromptGroup(rewards=[row["reward"]])

    def dynamic_filter(group):
        return group.rewards[0] > 0

    def train_step(step, groups, stats):
        dataset.record_batch(stats, accepted_rows=len(groups))
        train_log.append(
            {
                "step": step,
                "n_groups": len(groups),
                "total_sampled": stats["total_sampled"],
                "filter_drops": stats["filter_drops"],
                "sample_fails": stats["sample_fails"],
                "cursor": dataset.data_consumed,
            }
        )
        return step + ppo_minibatches, {"loss": 0.1}

    global_step = asyncio.run(
        run_rl_loop(
            sample_fns=(sample_one(row) for row in dataset.rows_from_cursor()),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=dataset.prompts_per_step,
            dynamic_filter_fn=dynamic_filter,
            global_step=0,
            weight_sync_fn=sync_log.append,
            weight_sync_interval=2,
        )
    )

    assert global_step == 6
    assert dataset.data_consumed == 7
    assert dataset.rows_from_cursor() == []
    assert sync_log == [3, 6]
    assert train_log == [
        {
            "step": 0,
            "n_groups": 2,
            "total_sampled": 4,
            "filter_drops": 1,
            "sample_fails": 1,
            "cursor": 4,
        },
        {
            "step": 3,
            "n_groups": 2,
            "total_sampled": 3,
            "filter_drops": 1,
            "sample_fails": 0,
            "cursor": 7,
        },
    ]
