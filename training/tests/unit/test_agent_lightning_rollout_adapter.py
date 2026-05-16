import asyncio
from types import SimpleNamespace

import pytest

import training.examples.rl.agent_lightning_protocol.rollout as agl_example_rollout
from training.utils.rl.rollout import (
    PrefixMismatch,
)


def _run(coro):
    return asyncio.run(coro)


def test_agent_lightning_triplets_pack_multi_turn_dicts():
    triplets = [
        {
            "prompt": {"token_ids": [10, 11]},
            "response": {
                "token_ids": [20],
                "logprobs": [{"logprob": -0.2}],
                "finish_reason": "stop",
            },
            "reward": None,
        },
        {
            "prompt": {"token_ids": [10, 11, 20, 30]},
            "response": {
                "token_ids": [40, 41],
                "logprobs": [{"logprob": -0.4}, {"logprob": -0.5}],
                "finish_reason": "length",
            },
            "reward": 1.0,
        },
    ]

    sample = _run(agl_example_rollout.agent_lightning_triplets_to_sample(
        triplets,
        tokenizer_id="tok",
    ))

    assert sample.tokens == [10, 11, 20, 30, 40, 41]
    assert sample.logprobs == [0.0, 0.0, -0.2, 0.0, -0.4, -0.5]
    assert sample.loss_mask == [0, 0, 1, 0, 1, 1]
    assert sample.reward == 1.0
    assert sample.finish_reason == "length"


def test_agent_lightning_triplets_accept_objects_and_total_reward_override():
    triplet = SimpleNamespace(
        prompt=SimpleNamespace(token_ids=[1, 2]),
        response=SimpleNamespace(
            token_ids=[3],
            logprobs=[SimpleNamespace(logprob=-0.3)],
            finish_reason="stop",
        ),
        reward=0.0,
    )

    sample = _run(agl_example_rollout.agent_lightning_triplets_to_sample(
        [triplet],
        total_reward=0.75,
    ))

    assert sample.tokens == [1, 2, 3]
    assert sample.logprobs == [0.0, 0.0, -0.3]
    assert sample.loss_mask == [0, 0, 1]
    assert sample.reward == 0.75


def test_agent_lightning_payload_uses_trajectory_assembler_prefix_guard():
    triplets = [
        {
            "prompt": {"token_ids": [1, 2]},
            "response": {"token_ids": [3], "logprobs": [-0.3]},
            "reward": None,
        },
        {
            "prompt": {"token_ids": [1, 99, 3, 4]},
            "response": {"token_ids": [5], "logprobs": [-0.5]},
            "reward": 1.0,
        },
    ]

    with pytest.raises(PrefixMismatch):
        agl_example_rollout.agent_lightning_triplets_to_payload(triplets)


def test_make_agent_lightning_rollout_fn_drops_bad_sample_by_default():
    async def provider(_row):
        return [
            {
                "prompt": {"token_ids": [1, 2]},
                "response": {"token_ids": [3]},
                "reward": 1.0,
            }
        ]

    rollout_fn = agl_example_rollout.make_agent_lightning_rollout_fn(provider)

    assert _run(rollout_fn({"id": "row"})) is None


def test_make_agent_lightning_rollout_fn_can_raise_for_debugging():
    def provider(_row):
        return [
            {
                "prompt": {"token_ids": [1, 2]},
                "response": {"token_ids": [3]},
                "reward": 1.0,
            }
        ]

    rollout_fn = agl_example_rollout.make_agent_lightning_rollout_fn(
        provider,
        swallow_exceptions=False,
    )

    with pytest.raises(ValueError, match="logprobs"):
        _run(rollout_fn({"id": "row"}))


def test_agent_lightning_protocol_example_rollout(monkeypatch):
    class FakeSampler:
        async def sample_with_prompt_tokens(self, prompt_token_ids, *, n, **_kwargs):
            assert prompt_token_ids == [10, 11]
            assert n == 1
            return [
                SimpleNamespace(
                    prompt_len=2,
                    full_tokens=[10, 11, 12, 13],
                    inference_logprobs=[-0.12, -0.13],
                    logprobs_echoed=False,
                    finish_reason="stop",
                )
            ]

    monkeypatch.setattr(
        agl_example_rollout,
        "build_deployment_sampler",
        lambda _setup: FakeSampler(),
    )
    setup = SimpleNamespace(
        sample_kwargs={"temperature": 0.7},
        tokenizer_id="tok",
    )

    rollout_fn = agl_example_rollout.make_rollout_fn(setup)
    sample = _run(rollout_fn({"prompt_token_ids": [10, 11], "reward": 0.5}))

    assert sample.tokens == [10, 11, 12, 13]
    assert sample.logprobs == [0.0, 0.0, -0.12, -0.13]
    assert sample.loss_mask == [0, 0, 1, 1]
    assert sample.reward == 0.5
