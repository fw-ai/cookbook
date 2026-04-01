from __future__ import annotations

import asyncio

import pytest

from eval_protocol.models import EvaluationRow, InputMetadata
from eval_protocol.pytest.types import RolloutProcessorConfig

from training.examples.rl.frozen_lake.frozen_lake_rollout import (
    FireworksV1ImageCompletionsClient,
    FrozenLakeToolRolloutProcessor,
    _build_visual_user_prompt,
    build_frozen_lake_tool_call_parser,
)
from training.examples.rl.frozen_lake.frozen_lake_schema import FROZEN_LAKE_TOOLS


class _FakeImageClient:
    requested_prompt_ids: list[list[int]] = []
    requested_image_counts: list[int] = []
    requested_prompt_texts: list[str] = []

    def __init__(self, **_: object) -> None:
        self._call_count = 0

    def build_prompt_token_ids(self, *, messages, tools):
        assert tools, "visual rollout should still pass tool specs into prompt construction"
        assert messages[-1]["role"] == "user"
        assert isinstance(messages[-1]["content"], list)
        return [10, 11, 12]

    def decode_token_ids(self, *, token_ids):
        normalized = list(token_ids)
        if normalized == [10, 11, 12]:
            return "prompt:initial"
        if normalized == [91, 92]:
            return "suffix:tool"
        return f"prompt:{normalized}"

    def build_prompt_text(self, *, messages, tools):
        assert tools, "visual rollout should still pass tool specs into prompt construction"
        assert messages[-1]["role"] == "user"
        return "prompt:initial"

    def encode_assistant_turn_suffix(self):
        return [90]

    def assistant_turn_suffix_text(self):
        return "<|im_end|>"

    def build_tool_response_suffix_token_ids(self, *, tool_message):
        assert tool_message["role"] == "tool"
        assert isinstance(tool_message["content"], list)
        return [91, 92]

    def build_tool_response_suffix_text(self, *, tool_message):
        assert tool_message["role"] == "tool"
        return "suffix:tool"

    async def create_completion_from_prompt_ids(self, *, prompt_token_ids, prompt_text=None, images, tools):
        del tools
        self._call_count += 1
        type(self).requested_prompt_ids.append(list(prompt_token_ids))
        type(self).requested_image_counts.append(len(images))
        type(self).requested_prompt_texts.append(str(prompt_text or ""))

        action = "RIGHT" if self._call_count == 1 else "DOWN"
        completion_id = 100 + self._call_count
        tool_call_id = f"tc_{self._call_count}"
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": "lake_move",
                                    "arguments": f'{{"action":"{action}"}}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                    "raw_output": {},
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": 1,
                "total_tokens": len(prompt_token_ids) + 1,
            },
            "prompt_ids": list(prompt_token_ids),
            "prompt_text": f"prompt:{prompt_token_ids}",
            "completion_ids": [completion_id],
            "completion_text": f"action:{action}",
            "completion_logprobs": [-0.1],
            "finish_reason": "stop",
            "raw_output": {},
        }

    async def close(self) -> None:
        return None


class _CapturingImageClient(_FakeImageClient):
    init_kwargs: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).init_kwargs.append(dict(kwargs))
        super().__init__(**kwargs)


def test_visual_user_prompt_includes_text_observation_without_placeholder():
    rendered = _build_visual_user_prompt(
        prompt_template="Inspect the image and choose one action.",
        observation="Current state:\nSFFF",
        default_prompt="unused",
    )

    assert "Inspect the image and choose one action." in rendered
    assert "Current textual observation:" in rendered
    assert "Current state:\nSFFF" in rendered


def test_frozen_lake_tool_call_parser_accepts_json_schema_output_with_suffix():
    parser = build_frozen_lake_tool_call_parser(model_id="accounts/fireworks/models/kimi-k2p5")

    parsed = parser(
        '{\n"tool_calls":[{"name":"lake_move","arguments":{"action":"RIGHT"}}]\n}<|im_end|>',
        [1, 2, 3],
        None,
    )

    assert parsed["parser"] == "json_schema"
    assert parsed["assistant_content"] == ""
    assert parsed["parsed_tool_call"].name == "lake_move"
    assert parsed["parsed_tool_call"].arguments == {"action": "RIGHT"}


def test_frozen_lake_tool_call_parser_accepts_kimi_native_output():
    parser = build_frozen_lake_tool_call_parser(model_id="accounts/fireworks/models/kimi-k2p5")

    parsed = parser(
        '<|tool_call_begin|>functions.lake_move:0<|tool_call_argument_begin|>{"action":"DOWN"}<|tool_call_end|><|tool_calls_section_end|>',
        [1, 2, 3],
        None,
    )

    assert parsed["parsed_tool_call"].name == "lake_move"
    assert parsed["parsed_tool_call"].arguments == {"action": "DOWN"}


def test_kimi_visual_prompt_includes_empty_think_block_when_thinking_disabled():
    client = FireworksV1ImageCompletionsClient(
        model_id="accounts/fireworks/models/kimi-k2p5",
        tokenizer_name_or_path="moonshotai/Kimi-K2.5",
        enable_thinking=False,
    )
    try:
        messages = [
            {"role": "system", "content": "You are an RL policy for FrozenLake."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "obs"},
                ],
            },
        ]
        prompt_text = client.build_prompt_text(messages=messages, tools=list(FROZEN_LAKE_TOOLS))
        prompt_ids = client.build_prompt_token_ids(messages=messages, tools=list(FROZEN_LAKE_TOOLS))
        decoded_prompt = client.decode_token_ids(token_ids=prompt_ids)

        assert prompt_text.endswith("<think></think>")
        assert "<think>" in decoded_prompt
        assert "</think>" in decoded_prompt
        assert decoded_prompt.rstrip().endswith("</think>")
    finally:
        asyncio.run(client.close())


def test_kimi_visual_tool_suffix_includes_empty_think_block_when_thinking_disabled():
    client = FireworksV1ImageCompletionsClient(
        model_id="accounts/fireworks/models/kimi-k2p5",
        tokenizer_name_or_path="moonshotai/Kimi-K2.5",
        enable_thinking=False,
    )
    try:
        tool_message = {
            "role": "tool",
            "name": "lake_move",
            "tool_call_id": "functions.lake_move:0",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
        }
        suffix_text = client.build_tool_response_suffix_text(tool_message=tool_message)
        suffix_ids = client.build_tool_response_suffix_token_ids(tool_message=tool_message)
        decoded_suffix = client.decode_token_ids(token_ids=suffix_ids)

        assert suffix_text.endswith("<think></think>")
        assert "<think>" in decoded_suffix
        assert "</think>" in decoded_suffix
        assert decoded_suffix.rstrip().endswith("</think>")
    finally:
        asyncio.run(client.close())


@pytest.mark.asyncio
async def test_visual_rollout_preserves_prior_completion_tokens(monkeypatch):
    _FakeImageClient.requested_prompt_ids = []
    _FakeImageClient.requested_image_counts = []
    _FakeImageClient.requested_prompt_texts = []
    monkeypatch.setattr(
        "training.examples.rl.frozen_lake.frozen_lake_rollout.FireworksV1ImageCompletionsClient",
        _FakeImageClient,
    )

    processor = FrozenLakeToolRolloutProcessor(
        model_id="accounts/fireworks/models/kimi-k2p5",
        tokenizer_name_or_path="moonshotai/Kimi-K2.5",
        observation_mode="image",
        logprobs=True,
    )
    row = EvaluationRow(
        input_metadata=InputMetadata(
            row_id="visual_seed",
            dataset_info={"environment_context": {"desc": ["SF", "FG"]}},
        )
    )
    config = RolloutProcessorConfig(
        completion_params={"model": "accounts/fireworks/models/kimi-k2p5"},
        mcp_config_path="",
        steps=4,
        semaphore=asyncio.Semaphore(1),
        logger=None,
    )

    [task] = processor([row], config)
    result = await task

    extra = result.execution_metadata.extra or {}
    token_turn_traces = extra["token_turn_traces"]

    assert len(token_turn_traces) == 2
    assert token_turn_traces[0]["prompt_ids"] == [10, 11, 12]
    assert token_turn_traces[0]["completion_ids"] == [163595, 101, 90]
    assert token_turn_traces[1]["prompt_ids"] == [10, 11, 12, 163595, 101, 90, 91, 92]
    assert token_turn_traces[1]["completion_ids"] == [163595, 102, 90]

    assert _FakeImageClient.requested_prompt_ids == [
        [10, 11, 12, 163595],
        [10, 11, 12, 163595, 101, 90, 91, 92, 163595],
    ]
    assert _FakeImageClient.requested_prompt_texts == [
        "prompt:initial<|tool_calls_section_begin|>",
        "prompt:initial<|tool_calls_section_begin|>action:RIGHT<|im_end|>suffix:tool<|tool_calls_section_begin|>",
    ]
    assert _FakeImageClient.requested_image_counts == [1, 2]
    assert extra["model_request_traces"][0]["assistant_turn_len"] == 3
    assert extra["model_request_traces"][1]["assistant_turn_len"] == 3
    assert extra["observation_mode"] == "image"
    assert extra["tool_call_generation_mode"] == "prompt_only"
    assert [message.role for message in result.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert str(result.messages[2].content).startswith("action:RIGHT")
    assert isinstance(result.messages[1].content, list)
    assert result.messages[1].content[0]["type"] == "image_url"
    assert isinstance(result.messages[3].content, list)
    assert result.messages[3].content[0]["type"] == "image_url"


@pytest.mark.asyncio
async def test_kimi_visual_rollout_keeps_raw_prompt_only_generation(monkeypatch):
    _CapturingImageClient.init_kwargs = []
    monkeypatch.setattr(
        "training.examples.rl.frozen_lake.frozen_lake_rollout.FireworksV1ImageCompletionsClient",
        _CapturingImageClient,
    )

    processor = FrozenLakeToolRolloutProcessor(
        model_id="accounts/fireworks/models/kimi-k2p5",
        tokenizer_name_or_path="moonshotai/Kimi-K2.5",
        observation_mode="image",
        logprobs=True,
    )
    row = EvaluationRow(
        input_metadata=InputMetadata(
            row_id="visual_seed_json_schema",
            dataset_info={"environment_context": {"desc": ["SF", "FG"]}},
        )
    )
    config = RolloutProcessorConfig(
        completion_params={"model": "accounts/fireworks/models/kimi-k2p5"},
        mcp_config_path="",
        steps=2,
        semaphore=asyncio.Semaphore(1),
        logger=None,
    )

    [task] = processor([row], config)
    result = await task

    init_kwargs = _CapturingImageClient.init_kwargs[0]
    request_params = init_kwargs["request_params"]
    assert isinstance(request_params, dict)
    assert request_params["thinking"] == {"type": "disabled"}
    assert "response_format" not in request_params
    assert "logit_bias" not in request_params
    assert init_kwargs["max_tokens"] == 256
    assert (result.execution_metadata.extra or {})["tool_call_generation_mode"] == "prompt_only"
