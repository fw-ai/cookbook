from __future__ import annotations

import asyncio

from training.utils.rl.losses import PromptGroup
from training.utils.rl.train import TrainStepFns, run_rl_loop


def _prompt_group(*, rewards: list[float]) -> PromptGroup:
    return PromptGroup(
        data=[],
        advantages=[0.0 for _ in rewards],
        ref_logprobs=[],
        prompt_len=0,
        rewards=rewards,
    )


async def _sample_after(delay_s: float, pg: PromptGroup | None) -> PromptGroup | None:
    await asyncio.sleep(delay_s)
    return pg


def test_run_rl_loop_trains_with_partial_valid_groups():
    events: dict[str, object] = {"steps": []}

    def train_step(step: int, prompt_groups: list[PromptGroup], loop_stats: dict | None):
        events["steps"].append(
            {
                "step": step + 1,
                "n_groups": len(prompt_groups),
                "total_sampled": loop_stats["total_sampled"],
                "filter_drops": loop_stats["filter_drops"],
            }
        )
        return step + 1, {}

    async def _run():
        return await run_rl_loop(
            sample_fns=[
                _sample_after(0.0, _prompt_group(rewards=[0.0, 0.0])),
                _sample_after(0.0, _prompt_group(rewards=[0.0, 1.0])),
            ],
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            prompt_groups_per_policy=2,
            dynamic_filter_fn=lambda pg: len(set(pg.rewards)) > 1,
        )

    global_step = asyncio.run(_run())

    assert events["steps"] == [
        {
            "step": 1,
            "n_groups": 1,
            "total_sampled": 2,
            "filter_drops": 1,
        }
    ]
    assert global_step == 1


def test_run_rl_loop_overlaps_within_policy_window():
    events: list[str] = []

    async def _sample(name: str) -> PromptGroup:
        events.append(f"start:{name}")
        await asyncio.sleep(0.01 if name == "a" else 0.0)
        events.append(f"done:{name}")
        return _prompt_group(rewards=[0.0, 1.0])

    def train_step(step: int, prompt_groups: list[PromptGroup], loop_stats: dict | None):
        events.append(f"train:{step + 1}")
        return step + 1, {}

    def policy_boundary(step: int):
        events.append(f"sync:{step}")

    async def _run():
        return await run_rl_loop(
            sample_fns=[_sample("a"), _sample("b"), _sample("c"), _sample("d")],
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            prompt_groups_per_policy=4,
            policy_boundary_fn=policy_boundary,
        )

    global_step = asyncio.run(_run())

    assert global_step == 2
    assert "sync:2" not in events
    assert events.index("start:c") < events.index("train:1")
