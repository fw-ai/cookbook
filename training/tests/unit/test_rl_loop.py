from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import training.recipes.rl_loop as module
from training.utils.rl.losses import PromptGroup


def test_extract_answer_reads_digits_from_answer_block():
    assert module.extract_answer("<answer> 42 apples </answer>") == "42"
    assert module.extract_answer("no answer block") is None


def test_reward_fn_requires_matching_numeric_answer():
    assert module.reward_fn("<answer>7</answer>", {"ground_truth": "<answer>7</answer>"}) == 1.0
    assert module.reward_fn("<answer>8</answer>", {"ground_truth": "<answer>7</answer>"}) == 0.0
    assert module.reward_fn("missing", {"ground_truth": "<answer>7</answer>"}) == 0.0


def test_should_accept_requires_reward_variance():
    same_rewards = PromptGroup(data=[], advantages=[], ref_logprobs=[], prompt_len=0, rewards=[0.0, 0.0])
    varied_rewards = PromptGroup(data=[], advantages=[], ref_logprobs=[], prompt_len=0, rewards=[0.0, 1.0])

    assert module.should_accept(same_rewards) is False
    assert module.should_accept(varied_rewards) is True


def test_compute_group_advantages_supports_countdown_mean_centering():
    assert module._compute_group_advantages([1.0, 0.0], "mean_centered") == pytest.approx([0.5, -0.5])

    zscore = module._compute_group_advantages([1.0, 0.0], "zscore")
    assert zscore == pytest.approx([0.70710677, -0.70710677])

    with pytest.raises(ValueError, match="Unsupported advantage_normalization"):
        module._compute_group_advantages([1.0], "unknown")  # type: ignore[arg-type]


def test_completion_text_uses_generated_tokens_not_echoed_text():
    class FakeRenderer:
        def __init__(self):
            self.seen_tokens = []

        def parse_response(self, tokens):
            self.seen_tokens.append(tokens)
            return {"role": "assistant", "content": "<answer>good</answer>"}, True

    sample = SimpleNamespace(
        full_tokens=[101, 102, 201, 202],
        prompt_len=2,
        text="PROMPT <answer>bad</answer>",
    )
    renderer = FakeRenderer()

    assert (
        module._completion_text_from_sample(sample, tokenizer=None, renderer=renderer)
        == "<answer>good</answer>"
    )
    assert renderer.seen_tokens == [[201, 202]]


def test_dump_trajectory_writes_one_record_per_completion(tmp_path):
    prompt_groups = [
        PromptGroup(
            data=[],
            advantages=[0.5, -0.5],
            ref_logprobs=[],
            prompt_len=4,
            rewards=[1.0, 0.0],
            completion_lens=[3, 4],
            truncated=[False, True],
            prompt=[{"role": "user", "content": "Solve"}],
            completions=["<answer>1</answer>", "<answer>2</answer>"],
            row_meta={"ground_truth": "<answer>1</answer>"},
        )
    ]

    module._dump_trajectory(str(tmp_path), step=3, prompt_groups=prompt_groups)

    path = tmp_path / "step_0003.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert records == [
        {
            "step": 3,
            "prompt_group": 0,
            "completion_index": 0,
            "prompt": [{"role": "user", "content": "Solve"}],
            "completion": "<answer>1</answer>",
            "reward": 1.0,
            "advantage": 0.5,
            "completion_len": 3,
            "truncated": False,
            "ground_truth": "<answer>1</answer>",
        },
        {
            "step": 3,
            "prompt_group": 0,
            "completion_index": 1,
            "prompt": [{"role": "user", "content": "Solve"}],
            "completion": "<answer>2</answer>",
            "reward": 0.0,
            "advantage": -0.5,
            "completion_len": 4,
            "truncated": True,
            "ground_truth": "<answer>1</answer>",
        },
    ]


def test_main_requires_deployment_tokenizer_model(monkeypatch):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        deployment=module.DeployConfig(tokenizer_model=""),
    )

    with pytest.raises(ValueError, match="deployment.tokenizer_model"):
        module.main(cfg)
