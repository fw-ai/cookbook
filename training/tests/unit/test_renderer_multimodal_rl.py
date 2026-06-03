"""Unit tests for multimodal renderer-backed RL rollouts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import tinker

from training.utils.rl.rollout.renderer import (
    MultimodalRenderingNotSupported,
    VisionCompletionsResult,
    _collect_base64_images,
    _model_input_to_completions_prompt_text,
    _parse_vision_completions_payload,
    _build_multimodal_rollout_sample,
    build_multimodal_completions_prompt_token_ids,
    build_multimodal_completions_request,
    model_input_to_token_ids,
    single_turn_renderer_rollout,
)
from fireworks.training.sdk.sampling import SampledCompletion
from training.utils.rl.rollout.types import rollout_to_prompt_group, Rollout, RolloutRun
from training.utils.supervised import build_multimodal_policy_datum, has_non_text_chunks

_BASE64_PNG = "data:image/png;base64,iVBORw0KGgo="


async def _test_messages(_row: object) -> list[dict[str, str]]:
    return [{"role": "user", "content": "hi"}]


async def _test_reward(_row: object, _msg: object, _ok: bool) -> float:
    return 1.0


def _multimodal_prompt(*, image_location: str = _BASE64_PNG) -> tinker.ModelInput:
    return tinker.ModelInput(
        chunks=[
            tinker.types.EncodedTextChunk(tokens=[10, 11]),
            tinker.types.ImageAssetPointerChunk(
                location=image_location,
                format="png",
                expected_tokens=4,
            ),
            tinker.types.EncodedTextChunk(tokens=[12]),
        ]
    )


def test_model_input_to_token_ids_rejects_image_chunks():
    with pytest.raises(MultimodalRenderingNotSupported):
        model_input_to_token_ids(_multimodal_prompt())


def test_collect_base64_images_from_model_input_and_messages():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image/png;base64,abc"},
                {"type": "text", "text": "hi"},
            ],
        }
    ]
    images = _collect_base64_images(messages, _multimodal_prompt())
    assert images == [_BASE64_PNG, "data:image/png;base64,abc"]


def test_collect_base64_images_rejects_http_url():
    with pytest.raises(MultimodalRenderingNotSupported, match="base64-encoded"):
        _collect_base64_images(
            [],
            _multimodal_prompt(image_location="https://example.com/a.png"),
        )


def test_collect_base64_images_error_includes_payload_snippet():
    with pytest.raises(MultimodalRenderingNotSupported, match="got 'not-a-data-url'"):
        _collect_base64_images(
            [],
            _multimodal_prompt(image_location="not-a-data-url"),
        )


def test_model_input_to_completions_prompt_text_inserts_vision_placeholder():
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "USER:"
    text = _model_input_to_completions_prompt_text(_multimodal_prompt(), tokenizer)
    assert "USER:" in text
    assert "<|image_pad|>" in text
    assert tokenizer.decode.call_count == 2


def test_build_multimodal_completions_prompt_token_ids_uses_chat_template():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [1, 248056, 2]
    tokenizer.special_ids = MagicMock(image=248056)
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    prompt_ids, images = build_multimodal_completions_prompt_token_ids(
        messages,
        _multimodal_prompt(),
        tokenizer,
    )
    assert prompt_ids == [1, 248056, 2]
    assert images == [_BASE64_PNG]
    tokenizer.apply_chat_template.assert_called_once()


def test_build_multimodal_completions_prompt_token_ids_raises_when_chat_template_fails():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = ValueError("no multimodal template")
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    with pytest.raises(MultimodalRenderingNotSupported, match="apply_chat_template"):
        build_multimodal_completions_prompt_token_ids(
            messages,
            _multimodal_prompt(),
            tokenizer,
        )


def test_build_multimodal_completions_request_uses_chat_template():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<|im_start|>user\nimg\n"
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    prompt_text, images = build_multimodal_completions_request(
        messages,
        _multimodal_prompt(),
        tokenizer,
    )
    assert prompt_text == "<|im_start|>user\nimg\n"
    assert images == [_BASE64_PNG]
    tokenizer.apply_chat_template.assert_called_once()


def test_build_multimodal_completions_request_falls_back_to_model_input_stitch():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = ValueError("no template")
    tokenizer.decode.return_value = "USER:"
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    prompt_text, images = build_multimodal_completions_request(
        messages,
        _multimodal_prompt(),
        tokenizer,
    )
    assert "USER:" in prompt_text
    assert "<|image_pad|>" in prompt_text
    assert images == [_BASE64_PNG]


def test_parse_vision_completions_payload():
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "text": "ok",
                "prompt_token_ids": [1, 2, 3],
                "token_ids": [4, 5],
                "logprobs": {"token_logprobs": [-0.1, -0.2]},
            }
        ]
    }
    result = _parse_vision_completions_payload(payload)
    assert result == VisionCompletionsResult(
        prompt_token_ids=[1, 2, 3],
        completion_token_ids=[4, 5],
        completion_logprobs=[-0.1, -0.2],
        finish_reason="stop",
        text="ok",
    )


def test_build_multimodal_policy_datum_preserves_image_chunk():
    prompt = _multimodal_prompt()
    datum = build_multimodal_policy_datum(prompt, [20, 21, 22])
    targets = list(datum.loss_fn_inputs["target_tokens"].data)
    assert 0 not in targets
    chunk_types = [c.type for c in datum.model_input.chunks]
    assert "image_asset_pointer" in chunk_types or any(
        "image" in str(t) for t in chunk_types
    )


def test_multimodal_rollout_sample_packs_prompt_group():
    prompt = _multimodal_prompt()
    run = _build_multimodal_rollout_sample(
        prompt_model_input=prompt,
        completion_tokens=[30, 31],
        completion_logprobs=[-0.1, -0.2],
        reward=1.0,
        finish_reason="stop",
        text="hi",
    )
    segment = run.segments[0]
    assert segment.prompt_model_input is not None
    pg = rollout_to_prompt_group(
        Rollout(runs=[run, run]),
        advantage_fn=lambda rewards: list(rewards),
    )
    assert pg is not None
    assert len(pg.data) == 2
    assert has_non_text_chunks(pg.data[0].model_input)
    assert has_non_text_chunks(segment.prompt_model_input)


def test_multimodal_reference_datum_mask_shape_matches_weights():
    prompt = _multimodal_prompt()
    run = _build_multimodal_rollout_sample(
        prompt_model_input=prompt,
        completion_tokens=[30, 31],
        completion_logprobs=[-0.1, -0.2],
        reward=1.0,
        finish_reason="stop",
        text="hi",
    )
    pg = rollout_to_prompt_group(
        Rollout(runs=[run, run]),
        with_reference=True,
        advantage_fn=lambda rewards: list(rewards),
    )
    assert pg is not None
    policy_weights = pg.data[0].loss_fn_inputs["weights"]
    ref_mask = pg.ref_data[0].loss_fn_inputs["loss_mask"]
    assert len(ref_mask.data) == ref_mask.shape[0]
    assert ref_mask.shape[0] == policy_weights.shape[0]
    assert ref_mask.shape[0] > len(pg.data[0].loss_fn_inputs["target_tokens"].data)


def test_multimodal_inf_logprobs_match_datum_weight_index_space():
    """GRPO slices inf_logprobs with prompt_lens in shifted-weights space."""
    prompt = _multimodal_prompt()
    run = _build_multimodal_rollout_sample(
        prompt_model_input=prompt,
        completion_tokens=[30, 31],
        completion_logprobs=[-0.1, -0.2],
        reward=1.0,
        finish_reason="stop",
        text="hi",
    )
    pg = rollout_to_prompt_group(
        Rollout(runs=[run, run]),
        advantage_fn=lambda rewards: list(rewards),
    )
    assert pg is not None
    weights = [float(w) for w in pg.data[0].loss_fn_inputs["weights"].data]
    inf_lp = pg.inf_logprobs[0]
    assert len(inf_lp) == len(weights)
    active = [i for i, w in enumerate(weights) if w > 0]
    assert [inf_lp[i] for i in active] == [-0.1, -0.2]
    assert all(inf_lp[i] == 0.0 for i in range(len(weights)) if i not in active)
    assert pg.prompt_lens[0] == active[0] + 1
    response_start = pg.prompt_lens[0] - 1
    assert response_start == active[0]
    assert len(inf_lp) >= response_start + (len(weights) - response_start)


@pytest.mark.asyncio
async def test_single_turn_renderer_rollout_multimodal_uses_token_in_completions():
    captured: dict[str, object] = {}

    class _Renderer:
        def build_generation_prompt(self, messages, **kwargs):
            return _multimodal_prompt()

        def parse_response(self, tokens):
            return ({"content": "x"}, True)

        def get_stop_sequences(self):
            return []

    async def sample_with_prompt_tokens(prompt_token_ids, **kwargs):
        captured["prompt_token_ids"] = list(prompt_token_ids)
        captured["kwargs"] = kwargs
        expanded_prompt = [10, 11, 248056, 12, 13, 14, 15]
        return [
            SampledCompletion(
                text="gen",
                full_tokens=expanded_prompt + [99, 100],
                prompt_len=len(expanded_prompt),
                finish_reason="stop",
                inference_logprobs=[-0.3, -0.4],
            )
        ]

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [10, 11, 248056, 12]
    tokenizer.special_ids = MagicMock(image=248056)

    run = await single_turn_renderer_rollout(
        {"id": 1},
        renderer=_Renderer(),
        sample_with_prompt_tokens=sample_with_prompt_tokens,
        message_builder=_test_messages,
        reward_fn=_test_reward,
        tokenizer=tokenizer,
    )
    assert run is not None
    assert captured["prompt_token_ids"] == [10, 11, 248056, 12]
    assert captured["kwargs"]["images"] == [_BASE64_PNG]
    assert captured["kwargs"]["logprobs"] is True
    assert captured["kwargs"]["return_token_ids"] is True
    segment = run.segments[0]
    assert segment.prompt_model_input is not None
    assert segment.tokens[-2:] == [99, 100]


@pytest.mark.asyncio
async def test_single_turn_renderer_rollout_vision_accepts_sync_reward_fn():
    class _Renderer:
        def build_generation_prompt(self, messages, **kwargs):
            return _multimodal_prompt()

        def parse_response(self, tokens):
            return ({"content": "x"}, True)

        def get_stop_sequences(self):
            return []

    async def sample_with_prompt_tokens(prompt_token_ids, **kwargs):
        raise AssertionError("token path should not run")

    async def sample_with_vision(*, prompt_text, images, **kwargs):
        return VisionCompletionsResult(
            prompt_token_ids=[1, 2, 3],
            completion_token_ids=[99, 100],
            completion_logprobs=[-0.3, -0.4],
            finish_reason="stop",
            text="gen",
        )

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<|im_start|>user\n"

    def sync_reward(_row, _msg, _ok):
        return 0.5

    run = await single_turn_renderer_rollout(
        {"id": 1},
        renderer=_Renderer(),
        sample_with_prompt_tokens=sample_with_prompt_tokens,
        sample_with_vision=sample_with_vision,
        message_builder=lambda row: [{"role": "user", "content": "hi"}],
        reward_fn=sync_reward,
        tokenizer=tokenizer,
    )
    assert run is not None
    assert run.segments[0].reward == 0.5


@pytest.mark.asyncio
async def test_single_turn_renderer_rollout_multimodal_uses_sample_with_vision():
    captured: dict[str, object] = {}

    class _Renderer:
        def build_generation_prompt(self, messages, **kwargs):
            return _multimodal_prompt()

        def parse_response(self, tokens):
            return ({"content": "x"}, True)

        def get_stop_sequences(self):
            return []

    async def sample_with_vision(*, prompt_text, images, **kwargs):
        captured["prompt_text"] = prompt_text
        captured["images"] = images
        captured["kwargs"] = kwargs
        return VisionCompletionsResult(
            prompt_token_ids=[1, 2, 3],
            completion_token_ids=[99, 100],
            completion_logprobs=[-0.3, -0.4],
            finish_reason="stop",
            text="gen",
        )

    async def sample_with_prompt_tokens(prompt_token_ids, **kwargs):
        raise AssertionError("token path should not run when sample_with_vision is set")

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<|im_start|>user\n"

    run = await single_turn_renderer_rollout(
        {"id": 1},
        renderer=_Renderer(),
        sample_with_prompt_tokens=sample_with_prompt_tokens,
        sample_with_vision=sample_with_vision,
        message_builder=_test_messages,
        reward_fn=_test_reward,
        tokenizer=tokenizer,
    )
    assert run is not None
    assert captured["prompt_text"] == "<|im_start|>user\n"
    assert captured["images"] == [_BASE64_PNG]
    assert run.segments[0].tokens[-2:] == [99, 100]
