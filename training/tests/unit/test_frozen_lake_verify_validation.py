from __future__ import annotations

import asyncio
import json

from training.examples.rl.frozen_lake.frozen_lake_rollout import (
    FrozenLakeToolRolloutProcessor,
)
from training.examples.rl.frozen_lake.train_frozen_lake import FrozenLakeTask
from training.examples.rl.frozen_lake.verify_rollout import verify
from training.recipes.async_rl_loop import RolloutSetup

GOOD_TRACES = [
    {"prompt_ids": [1, 2], "completion_ids": [10], "completion_logprobs": [-0.1]},
    {"prompt_ids": [1, 2, 10, 90, 20], "completion_ids": [11], "completion_logprobs": [-0.2]},
]
# The second prompt re-tokenized the first completion (12 instead of 10).
RE_RENDERED_TRACES = [GOOD_TRACES[0], {**GOOD_TRACES[1], "prompt_ids": [1, 2, 12, 90, 20]}]


def test_verify_reports_trainable_and_malformed_episodes(monkeypatch, tmp_path):
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text(json.dumps({"seed": 1}) + "\n" + json.dumps({"seed": 2}) + "\n")
    traces_by_row = {"seed_1": GOOD_TRACES, "seed_2": RE_RENDERED_TRACES}

    def fake_play(self, rows, config):
        async def play(row):
            row.execution_metadata.extra = {
                "token_turn_traces": traces_by_row[row.input_metadata.row_id],
                "step_rewards": [0.0, 1.0],
            }
            return row

        return [asyncio.ensure_future(play(row)) for row in rows]

    monkeypatch.setattr(FrozenLakeToolRolloutProcessor, "__call__", fake_play)
    setup = RolloutSetup(
        sampler=None,
        tokenizer=None,
        tokenizer_id="Qwen/Qwen3-8B",
        sample_kwargs={"temperature": 0.0, "max_tokens": 32},
        inference_base_url="https://api.fireworks.ai",
        api_key="key",
        model="accounts/fireworks/models/qwen3-8b",
        completions_per_prompt=1,
    )

    lines = asyncio.run(verify(setup, FrozenLakeTask(seed_jsonl_path=str(seeds))))

    assert lines == [
        "seed_1: reward=1 tokens=6 trained=2 finish=stop",
        "seed_2: not trainable (reason logged above)",
    ]
