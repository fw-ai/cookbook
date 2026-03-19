from __future__ import annotations

import asyncio

from training.utils.rl.losses import PromptGroup
from training.utils.rl.train import AsyncRLState, TrainStepFns, async_rl_loop


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


def test_async_rl_loop_refills_until_full_accepted_batch():
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
        return await async_rl_loop(
            sample_fns=[
                _sample_after(0.0, _prompt_group(rewards=[0.0, 0.0])),
                _sample_after(0.0, _prompt_group(rewards=[0.0, 1.0])),
                _sample_after(0.0, _prompt_group(rewards=[1.0, 0.0])),
            ],
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            prompt_groups_per_policy=2,
            dynamic_filter_fn=lambda pg: len(set(pg.rewards)) > 1,
        )

    state = asyncio.run(_run())

    assert events["steps"] == [
        {
            "step": 1,
            "n_groups": 2,
            "total_sampled": 3,
            "filter_drops": 1,
        }
    ]
    assert isinstance(state, AsyncRLState)
    assert state.global_step == 1
    assert state.rows_submitted == 3
    assert state.accepted_total == 2


def test_async_rl_loop_triggers_policy_boundary_by_policy_quota():
    boundaries: list[tuple[int, int]] = []

    def train_step(step: int, prompt_groups: list[PromptGroup], loop_stats: dict | None):
        return step + 1, {}

    def policy_boundary(state: AsyncRLState):
        boundaries.append((state.global_step, state.current_launch_version))

    async def _accepted() -> PromptGroup:
        await asyncio.sleep(0.0)
        return _prompt_group(rewards=[0.0, 1.0])

    async def _run():
        return await async_rl_loop(
            sample_fns=[
                _accepted(),
                _accepted(),
                _accepted(),
                _accepted(),
            ],
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            prompt_groups_per_policy=2,
            max_head_offpolicy_versions=0,
            policy_boundary_fn=policy_boundary,
        )

    state = asyncio.run(_run())

    assert state.global_step == 2
    assert boundaries == [(1, 0)]
    assert state.current_launch_version == 1
