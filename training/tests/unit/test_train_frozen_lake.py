from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import eval_protocol.integrations.fireworks_v1_completions_client as ep_completions_client
import pytest
from eval_protocol.models import EvaluationRow, InputMetadata

import training.examples.rl.frozen_lake.train_frozen_lake as train_module
from training.examples.rl.frozen_lake.frozen_lake_rollout import (
    FrozenLakeToolRolloutProcessor,
)
from training.examples.rl.frozen_lake.train_frozen_lake import (
    FrozenLakeTask,
    episode_to_sample,
    evaluator_rows,
    frozen_lake_reward,
    has_reward_variance,
    make_frozen_lake_evaluator,
    make_rollout_fn_factory,
    parse_args,
)
from training.recipes.async_rl_loop import Config, RolloutSetup
from training.utils import DeployConfig
from training.utils.rl.losses import build_policy_datums, combine_prompt_groups
from training.utils.rl.rollout import Rollout, RolloutRun, rollout_to_prompt_group
from training.utils.runner import UserConfigError
from training.utils.tokenizers import load_tokenizer

# Two turns: the second prompt is the first prompt + completion + close (90)
# + tool reply (20, 21).
TRACES = [
    {"prompt_ids": [1, 2, 3], "completion_ids": [10, 11], "completion_logprobs": [-0.1, -0.2]},
    {
        "prompt_ids": [1, 2, 3, 10, 11, 90, 20, 21],
        "completion_ids": [30, 31],
        "completion_logprobs": [-0.3, -0.4],
    },
]


def _episode(*, step_rewards, rollout_error=None) -> EvaluationRow:
    row = EvaluationRow(input_metadata=InputMetadata(row_id="seed_1"))
    row.execution_metadata.extra = {"token_turn_traces": TRACES, "step_rewards": step_rewards}
    if rollout_error:
        row.execution_metadata.extra["rollout_error"] = rollout_error
    row.execution_metadata.finish_reason = "stop"
    return frozen_lake_reward(row)


def _seeds(tmp_path, *seeds: int) -> str:
    path = tmp_path / "seeds.jsonl"
    path.write_text("\n".join(json.dumps({"seed": seed}) for seed in seeds) + "\n")
    return str(path)


def test_parse_args_maps_cli_to_recipe_config():
    cfg, task = parse_args(
        [
            "--base-model", "accounts/test/models/qwen3-4b",
            "--tokenizer-model", "Qwen/Qwen3-4B",
            "--training-shape", "ts-qwen3-4b-smoke-v1",
            "--policy-job-id", "job-1",
            "--deployment-id", "dep-1",
            "--deployment-replica-count", "3",
            "--max-seeds", "3",
            "--max-steps", "9",
            "--observation-mode", "image",
            "--allow-plaintext-action-fallback",
            "--max-concurrent", "8",
            "--output-model-id", "out-model",
        ]
    )

    assert cfg.base_model == "accounts/test/models/qwen3-4b"
    assert cfg.deployment.tokenizer_model == "Qwen/Qwen3-4B"
    assert cfg.deployment.deployment_id == "dep-1"
    assert cfg.deployment.replica_count == 3
    assert cfg.trainer.job_id == "job-1"
    assert cfg.trainer.training_shape_id == "ts-qwen3-4b-smoke-v1"
    assert cfg.max_concurrency_rollout_sample == 8
    assert cfg.min_group_size == 2
    assert cfg.output_model_id == "out-model"
    assert task == FrozenLakeTask(
        max_seeds=3,
        max_steps=9,
        observation_mode="image",
        allow_plaintext_action_fallback=True,
    )


def test_main_rejects_invalid_output_model_id(monkeypatch, tmp_path):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    cfg = Config(
        log_path=str(tmp_path),
        output_model_id="bad_name",
        deployment=DeployConfig(tokenizer_model="Qwen/Qwen3-8B"),
    )

    with pytest.raises(UserConfigError, match="output_model_id.*invalid|invalid.*output_model_id"):
        train_module.main(cfg, FrozenLakeTask())


def test_main_runs_async_rl_loop_with_seed_rows(monkeypatch, tmp_path):
    captured = {}

    def fake_recipe_main(cfg, **kwargs):
        captured.update(kwargs, cfg=cfg)
        return {"steps": 0}

    monkeypatch.setattr(train_module.async_rl_loop, "main", fake_recipe_main)
    cfg = Config(log_path=str(tmp_path))

    result = train_module.main(cfg, FrozenLakeTask(seed_jsonl_path=_seeds(tmp_path, 7, 8, 9), max_seeds=2))

    assert result == {"steps": 0}
    assert captured["cfg"] is cfg
    assert [row["id"] for row in captured["rows"]] == ["seed_7", "seed_8"]
    assert captured["rows"][0]["evaluation_row"].input_metadata.dataset_info == {
        "environment_context": {"seed": 7}
    }
    assert captured["dynamic_filter_fn"] is has_reward_variance
    assert callable(captured["rollout_fn_factory"])


def test_frozen_lake_reward_scores_reaching_the_goal():
    assert _episode(step_rewards=[0.0, 1.0]).evaluation_result.score == 1.0
    assert _episode(step_rewards=[0.0, 0.0]).evaluation_result.score == 0.0
    assert _episode(step_rewards=[]).evaluation_result.score == 0.0


def test_episode_to_sample_trains_tool_calls_and_masks_replies():
    sample = episode_to_sample(_episode(step_rewards=[0.0, 1.0]))

    assert sample is not None
    assert sample.tokens == [1, 2, 3, 10, 11, 90, 20, 21, 30, 31]
    assert sample.loss_mask == [0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
    assert sample.logprobs == [0.0, 0.0, 0.0, -0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4]
    assert sample.reward == 1.0


def test_episode_to_sample_drops_unparsed_tool_calls():
    assert episode_to_sample(_episode(step_rewards=[0.0], rollout_error="bad tool call")) is None


def test_episode_groups_build_server_grpo_datums():
    """The training step clips against the episode's rollout logprobs."""
    won = episode_to_sample(_episode(step_rewards=[0.0, 1.0]))
    lost = episode_to_sample(_episode(step_rewards=[0.0, 0.0]))
    group = rollout_to_prompt_group(
        Rollout(runs=[RolloutRun(segments=[won]), RolloutRun(segments=[lost])])
    )
    assert group is not None and has_reward_variance(group)

    datums = build_policy_datums(combine_prompt_groups([group]))

    inputs = datums[0].loss_fn_inputs
    generated = inputs["response_mask"].data
    assert generated == [0, 0, 1, 1, 0, 0, 0, 1, 1]
    assert [lp for lp, g in zip(inputs["logprobs"].data, generated) if g] == pytest.approx(
        [-0.1, -0.2, -0.3, -0.4]
    )
    assert all(a > 0.0 for a, g in zip(inputs["advantages"].data, generated) if g)


def test_rollout_fn_plays_one_episode_on_the_recipe_deployment(monkeypatch, tmp_path):
    seen_configs = []

    def fake_play(self, rows, config):
        seen_configs.append(config)

        async def play(row):
            row.execution_metadata.extra = {"token_turn_traces": TRACES, "step_rewards": [0.0, 1.0]}
            row.execution_metadata.finish_reason = "stop"
            return row

        return [asyncio.ensure_future(play(row)) for row in rows]

    monkeypatch.setattr(FrozenLakeToolRolloutProcessor, "__call__", fake_play)
    evaluator = make_frozen_lake_evaluator(
        FrozenLakeTask(seed_jsonl_path=_seeds(tmp_path, 7), max_steps=5)
    )
    setup = RolloutSetup(
        sampler=None,
        tokenizer=None,
        tokenizer_id="Qwen/Qwen3-8B",
        sample_kwargs={
            "temperature": 1.0,
            "max_tokens": 64,
            "top_p": 1.0,
            "top_k": 0,
            "include_routing_matrix": True,
        },
        inference_base_url="https://api.fireworks.ai",
        api_key="key",
        model="accounts/a/models/m#accounts/a/deployments/d",
        completions_per_prompt=2,
    )
    rollout_fn = make_rollout_fn_factory(evaluator)(setup)

    run = asyncio.run(rollout_fn(evaluator_rows(evaluator)[0]))

    assert run.segments[0].reward == 1.0
    assert run.segments[0].loss_mask == [0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
    [config] = seen_configs
    assert config.steps == 5
    assert config.completion_params == {
        "model": "accounts/a/models/m#accounts/a/deployments/d",
        "base_url": "https://api.fireworks.ai/inference",
        "api_key": "key",
        "tokenizer_name_or_path": "Qwen/Qwen3-8B",
        "temperature": 1.0,
        "max_tokens": 64,
        "top_p": 1.0,
        "top_k": 0,
    }


def test_has_reward_variance_skips_uniform_groups():
    class Group:
        rewards = [1.0, 1.0]

    assert not has_reward_variance(Group())
    Group.rewards = [0.0, 1.0]
    assert has_reward_variance(Group())


def test_text_episode_on_the_qwen3_6_template_end_to_end(monkeypatch, tmp_path):
    """Real tokenizer, template and parser; only the completions endpoint is faked."""
    tokenizer = load_tokenizer("Qwen/Qwen3.6-35B-A3B")
    tool_call = (
        "<tool_call>\n<function=lake_move>\n<parameter=action>\n{}\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>"
    )
    actions = iter(["RIGHT", "DOWN"])  # S F / F G: two moves reach the goal
    requests = []

    class Completions:
        async def create(self, **payload):
            requests.append(payload)
            ids = tokenizer.encode(tool_call.format(next(actions)), add_special_tokens=False)
            content = [{"token_id": token, "sampling_logprob": -0.1} for token in ids]
            choice = {"finish_reason": "stop", "logprobs": {"content": content}}
            return SimpleNamespace(model_dump=lambda: {"choices": [choice], "usage": {}})

    class AsyncFireworks:
        def __init__(self, **kwargs):
            self.completions = Completions()

        async def close(self):
            return None

    monkeypatch.setattr(ep_completions_client, "AsyncFireworks", AsyncFireworks)
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text(json.dumps({"seed": 1, "desc": ["SF", "FG"]}) + "\n")
    evaluator = make_frozen_lake_evaluator(FrozenLakeTask(seed_jsonl_path=str(seeds)))
    setup = RolloutSetup(
        sampler=None,
        tokenizer=tokenizer,
        tokenizer_id="Qwen/Qwen3.6-35B-A3B",
        sample_kwargs={"temperature": 1.0, "max_tokens": 64, "top_p": 1.0, "top_k": 0},
        inference_base_url="https://api.fireworks.ai",
        api_key="key",
        model="m",
        completions_per_prompt=2,
    )

    run = asyncio.run(make_rollout_fn_factory(evaluator)(setup)(evaluator_rows(evaluator)[0]))

    sample = run.segments[0]
    assert sample.reward == 1.0
    first, second = (request["prompt"] for request in requests)
    assert second[: len(first)] == first
    assert tokenizer.decode(second[len(first):]).endswith(
        "</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n"
        '{"action":"RIGHT","reward":0.0,"terminated":false,"truncated":false,'
        '"position":1,"row":0,"col":1,"tile":"F","step_index":1}\n</tool_response><|im_end|>\n'
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    trained = [token for token, mask in zip(sample.tokens, sample.loss_mask) if mask]
    assert tokenizer.decode(trained) == tool_call.format("RIGHT") + tool_call.format("DOWN")
