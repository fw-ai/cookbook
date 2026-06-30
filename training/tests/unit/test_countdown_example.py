from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import tinker

from training.examples.rl.countdown import rollout
from training.examples.rl.countdown.reward import (
    check_numbers_used,
    composite_reward,
    safe_eval_equation,
)
from training.utils.rl.rollout import VisionCompletionsResult


def test_countdown_reward_accepts_valid_equation() -> None:
    row = {"numbers": [1, 2, 3, 4], "target": 24}
    completion = "<think>try multiply</think><answer>1 * 2 * 3 * 4</answer>"

    assert composite_reward(completion, row) == pytest.approx(1.0)


def test_countdown_reward_rejects_missing_or_extra_numbers() -> None:
    row = {"numbers": [1, 2, 3, 4], "target": 24}

    assert check_numbers_used("1 * 2 * 12", [1, 2, 3, 4]) is False
    assert composite_reward("<answer>6 * 4</answer>", row) < 1.0


def test_safe_eval_equation_rejects_unsafe_expressions() -> None:
    assert safe_eval_equation("1 + 2 * 3") == pytest.approx(7.0)
    assert safe_eval_equation("__import__('os').system('true')") is None
    assert safe_eval_equation("2 ** 10") is None


def test_build_countdown_vision_messages_contains_base64_image() -> None:
    row = {"numbers": [1, 2, 3, 4], "target": 24}

    messages = rollout.build_countdown_messages(row, variant="vision")

    content = messages[1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"
    assert content[1]["image"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_vision_rollout_uses_renderer_backed_vision_sampling(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Renderer:
        def build_generation_prompt(self, messages):
            captured["messages"] = messages
            return tinker.ModelInput(
                chunks=[
                    tinker.types.EncodedTextChunk(tokens=[1, 2]),
                    tinker.types.ImageAssetPointerChunk(
                        location=messages[1]["content"][1]["image"],
                        format="png",
                        expected_tokens=4,
                    ),
                ]
            )

        def parse_response(self, tokens):
            captured["parse_tokens"] = list(tokens)
            return {"content": "<answer>1 * 2 * 3 * 4</answer>"}, True

        def get_stop_sequences(self):
            return []

    class _Sampler:
        async def sample_with_prompt_tokens(self, prompt_token_ids, **kwargs):
            raise AssertionError("vision run should use sample_with_vision")

    async def fake_sample_vision_completion(**kwargs):
        captured["vision_kwargs"] = kwargs
        return VisionCompletionsResult(
            prompt_token_ids=[10, 11],
            completion_token_ids=[20, 21],
            completion_logprobs=[-0.1, -0.2],
            finish_reason="stop",
            text="<answer>1 * 2 * 3 * 4</answer>",
        )

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<|im_start|>user\n"
    monkeypatch.setattr(rollout, "build_renderer", lambda *_args, **_kwargs: _Renderer())
    monkeypatch.setattr(rollout, "build_deployment_sampler", lambda _setup: _Sampler())
    monkeypatch.setattr(rollout, "sample_vision_completion", fake_sample_vision_completion)

    setup = SimpleNamespace(
        tokenizer=tokenizer,
        tokenizer_id="Qwen/Qwen3-VL-8B-Instruct",
        sample_kwargs={"max_tokens": 8, "temperature": 1.0},
        inference_base_url="https://api.fireworks.ai/inference",
        api_key="test-key",
        model="accounts/test/deployments/countdown",
        completions_per_prompt=2,
        extras={"variant": "vision"},
    )

    rollout_fn = rollout.make_rollout_fn(setup)
    result = await rollout_fn({"numbers": [1, 2, 3, 4], "target": 24})

    assert result is not None
    assert result.segments[0].reward == pytest.approx(1.0)
    assert result.segments[0].prompt_model_input is not None
    assert captured["parse_tokens"] == [20, 21]
    assert captured["vision_kwargs"]["deployment_model"] == "accounts/test/deployments/countdown"
