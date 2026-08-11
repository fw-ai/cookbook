"""Focused tests for the VisualToolBench rollout runtime."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
from types import SimpleNamespace

import pytest
import tinker

from training.examples.rl.visual_toolbench import reward as vtb_reward
from training.examples.rl.visual_toolbench import rollout as vtb_rollout
from training.examples.rl.visual_toolbench.image_tools import (
    decode_data_url,
    encode_data_url,
    execute_tool_call,
)
from training.examples.rl.visual_toolbench.reward import (
    JudgeResult,
    compute_rubric_score,
    parse_judge_verdicts,
)
from training.examples.rl.visual_toolbench.rollout import make_rollout_fn

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


_RUBRICS = [
    {"id": "a", "description": "Mentions 6 rebars", "weight": 5, "critical": True},
    {"id": "b", "description": "Uses correct units", "weight": 1, "critical": False},
]


def test_rubric_judge_uses_fireworks_inference_endpoint(monkeypatch):
    client = SimpleNamespace()
    captured = {}

    def make_client(**kwargs):
        captured.update(kwargs)
        return client

    monkeypatch.setattr(vtb_reward, "AsyncFireworks", make_client)

    vtb_reward.RubricJudge(api_key="test-key")

    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://api.fireworks.ai/inference"


def _data_url(width: int = 64, height: int = 48) -> str:
    return encode_data_url(Image.new("RGB", (width, height), (200, 30, 30)))


def _image_chunk() -> tinker.types.ImageChunk:
    return tinker.types.ImageChunk(
        data=base64.b64decode(_data_url().partition(",")[2]),
        format="jpeg",
        expected_tokens=4,
    )


def test_image_tools_cap_and_transform_images():
    original = encode_data_url(
        Image.new("RGB", (3000, 1500), (200, 30, 30)), max_dim=1024
    )
    assert max(decode_data_url(original).size) == 1024

    cropped, message = execute_tool_call(
        "crop_image",
        {
            "image_index": 0,
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": 0.5,
            "y_max": 0.5,
        },
        [original],
    )
    assert cropped is not None
    assert "image 1" in message
    assert decode_data_url(cropped).size == (512, 256)


def test_zoom_enlarges_center_without_exceeding_image_cap():
    image = Image.new("RGB", (1024, 768), (0, 0, 0))
    image.paste((255, 255, 255), (256, 192, 768, 576))
    original = encode_data_url(image)

    zoomed, message = execute_tool_call(
        "zoom_image",
        {"image_index": 0, "factor": 2.0},
        [original],
    )

    assert zoomed is not None
    decoded = decode_data_url(zoomed)
    assert decoded.size == (1024, 768)
    assert decoded.getpixel((10, 10))[0] > 240
    assert "center" in message


def test_crop_rejects_region_that_rounds_to_zero_pixels():
    original = _data_url(width=10, height=10)

    cropped, message = execute_tool_call(
        "crop_image",
        {
            "image_index": 0,
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": 0.01,
            "y_max": 0.01,
        },
        [original],
    )

    assert cropped is None
    assert "empty after conversion to image pixels" in message


def test_reward_is_dense_and_critical_aware():
    result = compute_rubric_score(_RUBRICS, [True, False])
    assert result.score == pytest.approx(5 / 6)
    assert result.critical_fraction == 1.0
    assert result.reward == pytest.approx(0.8 * (5 / 6) + 0.2)
    assert result.passed is True

    failed = compute_rubric_score(_RUBRICS, [False, True])
    assert 0.0 < failed.reward < result.reward
    assert failed.passed is False


def test_judge_parser_requires_explicit_boolean_verdicts():
    assert parse_judge_verdicts(
        '{"verdicts":[{"index":1,"pass":true},{"index":2,"pass":false}]}',
        2,
    ) == [True, False]
    assert (
        parse_judge_verdicts(
            '{"verdicts":[{"index":1,"pass":"false"}]}',
            1,
        )
        is None
    )


class _ToolCall:
    def __init__(self):
        self.id = "tool-1"
        self.function = SimpleNamespace(
            name="crop_image",
            arguments=json.dumps(
                {
                    "image_index": 0,
                    "x_min": 0.0,
                    "y_min": 0.0,
                    "x_max": 0.5,
                    "y_max": 0.5,
                }
            ),
        )


class _Renderer:
    image_placeholder_token_id = 99

    def __init__(self):
        self.parse_calls = 0
        self.generation_messages = []

    def create_conversation_prefix_with_tools(self, tools, system_prompt):
        assert tools
        return [{"role": "system", "content": system_prompt}]

    def build_generation_prompt(self, messages):
        self.generation_messages.append(copy.deepcopy(messages))
        chunks = [tinker.types.EncodedTextChunk(tokens=[1, 2])]
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    chunks.append(_image_chunk())
        chunks.append(tinker.types.EncodedTextChunk(tokens=[3]))
        return tinker.ModelInput(chunks=chunks)

    def parse_response(self, _tokens):
        self.parse_calls += 1
        if self.parse_calls == 1:
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": "Let me crop."}],
                "tool_calls": [_ToolCall()],
            }, SimpleNamespace(is_clean=True)
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": "The answer is 6."}],
        }, SimpleNamespace(is_clean=True)

    def get_stop_sequences(self):
        return ["<|im_end|>"]


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [99] if text == "<|image_pad|>" else [7]

    def decode(self, _token_ids, **_kwargs):
        return "<|im_end|>"


class _Sampler:
    def __init__(self):
        self.calls = []

    async def sample_with_prompt_tokens(self, prompt_token_ids, **kwargs):
        self.calls.append({"prompt_token_ids": list(prompt_token_ids), **kwargs})
        expanded_prompt = []
        for token in prompt_token_ids:
            expanded_prompt.extend([0] * 4 if token == 99 else [token])
        completion_tokens = [101, 102]
        return [
            SimpleNamespace(
                prompt_len=len(expanded_prompt),
                full_tokens=expanded_prompt + completion_tokens,
                inference_logprobs=[-0.1, -0.2],
                sampling_logprobs=[-0.11, -0.22],
                logprobs_echoed=False,
                routing_matrices=None,
                finish_reason="stop",
                text="",
            )
        ]


class _Judge:
    def __init__(self, **_kwargs):
        pass

    async def grade(self, _row, answer):
        assert "answer is 6" in answer
        return JudgeResult(
            score=0.8,
            passed=True,
            verdicts=[True],
            critical_fraction=1.0,
            reward=0.84,
        )

    async def close(self):
        pass


def test_rollout_uses_session_sampler_and_trains_each_assistant_turn(monkeypatch):
    sampler = _Sampler()
    renderer = _Renderer()
    monkeypatch.setattr(
        vtb_rollout,
        "build_deployment_sampler",
        lambda _setup: pytest.fail("session sampler should be used"),
    )
    monkeypatch.setattr(vtb_rollout, "build_renderer", lambda *_args: renderer)
    monkeypatch.setattr(vtb_rollout, "RubricJudge", _Judge)

    setup = SimpleNamespace(
        tokenizer=_Tokenizer(),
        tokenizer_id="Qwen/Qwen3.6-27B",
        sample_kwargs={"max_tokens": 128, "temperature": 1.0},
        inference_base_url="https://api.fireworks.ai/training/v1/serverless",
        api_key="test-key",
        model="saved/checkpoint",
        completions_per_prompt=8,
        sampler=sampler,
        extras={"max_turns": 4},
    )
    rollout_fn = make_rollout_fn(setup)
    run = asyncio.run(
        rollout_fn(
            {
                "id": "tool-episode",
                "prompt": "How many rebars?",
                "golden_answer": "Six.",
                "rubrics": _RUBRICS,
                "images": [_data_url(width=200, height=100)],
            }
        )
    )

    assert run is not None
    assert len(run.segments) == 2
    assert len(sampler.calls) == 2
    assert len(sampler.calls[0]["images"]) == 1
    assert len(sampler.calls[1]["images"]) == 2
    for segment in run.segments:
        assert segment.reward == pytest.approx(0.84)
        assert segment.loss_mask[-2:] == [1, 1]
        assert all(value == 0 for value in segment.loss_mask[:-2])
        assert segment.logprobs[-2:] == [-0.11, -0.22]
        assert segment.raw_logprobs is not None
        assert segment.raw_logprobs[-2:] == [-0.1, -0.2]
        assert segment.prompt_model_input is not None
    assert run.metadata["num_tool_calls"] == 1
    assert run.metadata["metrics"]["rollout/reward"] == pytest.approx(0.84)

    close = getattr(rollout_fn, "close")
    asyncio.run(close())
