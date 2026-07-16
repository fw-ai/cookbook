"""Unit tests for multimodal renderer-backed RL rollouts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import tinker
import torch

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
from training.utils.rl.losses import (
    LossConfig,
    build_builtin_loss_datums,
    build_loss_fn,
)
from training.utils.rl.rollout.types import rollout_to_prompt_group, Rollout
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


def test_collect_base64_images_prefers_model_input_over_messages():
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
    assert images == [_BASE64_PNG]


def test_collect_base64_images_uses_materialized_image_chunk_bytes():
    model_input = tinker.ModelInput(
        chunks=[
            tinker.types.EncodedTextChunk(tokens=[10]),
            tinker.types.ImageChunk(
                data=b"renderer-jpeg",
                format="jpeg",
                expected_tokens=4,
            ),
            tinker.types.ImageChunk(
                data=b"renderer-jpeg",
                format="jpeg",
                expected_tokens=4,
            ),
            tinker.types.EncodedTextChunk(tokens=[11]),
        ]
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": _BASE64_PNG},
            ],
        }
    ]

    images = _collect_base64_images(messages, model_input)

    assert images == [
        "data:image/jpeg;base64,cmVuZGVyZXItanBlZw==",
        "data:image/jpeg;base64,cmVuZGVyZXItanBlZw==",
    ]


def test_collect_base64_images_preserves_duplicate_message_images():
    model_input = tinker.ModelInput(
        chunks=[tinker.types.EncodedTextChunk(tokens=[10])]
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": _BASE64_PNG},
                {"type": "image", "image": _BASE64_PNG},
            ],
        }
    ]

    images = _collect_base64_images(messages, model_input)

    assert images == [_BASE64_PNG, _BASE64_PNG]


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


def test_build_multimodal_completions_prompt_token_ids_uses_renderer_model_input():
    tokenizer = MagicMock()
    # The tokenizer default can add thinking tokens that the selected renderer
    # intentionally omitted. Sampling must not re-render the prompt here.
    tokenizer.apply_chat_template.return_value = [10, 11, 248056, 12, 248068, 198]
    tokenizer.special_ids = MagicMock(image=248056)
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    prompt_ids, images = build_multimodal_completions_prompt_token_ids(
        messages,
        _multimodal_prompt(),
        tokenizer,
    )
    assert prompt_ids == [10, 11, 248056, 12]
    assert images == [_BASE64_PNG]
    tokenizer.apply_chat_template.assert_not_called()


def test_build_multimodal_completions_prompt_token_ids_requires_image_token_id():
    tokenizer = MagicMock()
    tokenizer.special_ids = MagicMock(image=None)
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    with pytest.raises(MultimodalRenderingNotSupported, match="image placeholder token ID"):
        build_multimodal_completions_prompt_token_ids(
            messages,
            _multimodal_prompt(),
            tokenizer,
        )


def test_build_multimodal_completions_prompt_token_ids_accepts_hf_image_token_id():
    tokenizer = MagicMock()
    tokenizer.special_ids = None
    tokenizer.image_token_id = 248056
    prompt_ids, _ = build_multimodal_completions_prompt_token_ids(
        [],
        _multimodal_prompt(),
        tokenizer,
    )
    assert prompt_ids == [10, 11, 248056, 12]


def test_build_multimodal_completions_request_uses_renderer_model_input():
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = ["<|im_start|>user\n", "img\n"]
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    prompt_text, images = build_multimodal_completions_request(
        messages,
        _multimodal_prompt(),
        tokenizer,
    )
    assert prompt_text == (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>img\n"
    )
    assert images == [_BASE64_PNG]
    tokenizer.apply_chat_template.assert_not_called()


def test_build_multimodal_completions_request_accepts_materialized_image_chunk():
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = ["USER:", "ASSISTANT:"]
    model_input = tinker.ModelInput(
        chunks=[
            tinker.types.EncodedTextChunk(tokens=[10]),
            tinker.types.ImageChunk(
                data=b"renderer-jpeg",
                format="jpeg",
                expected_tokens=4,
            ),
            tinker.types.EncodedTextChunk(tokens=[11]),
        ]
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    prompt_text, images = build_multimodal_completions_request(
        messages,
        model_input,
        tokenizer,
    )
    assert prompt_text == ("USER:<|vision_start|><|image_pad|><|vision_end|>ASSISTANT:")
    assert images == ["data:image/jpeg;base64,cmVuZGVyZXItanBlZw=="]
    tokenizer.apply_chat_template.assert_not_called()


def test_parse_vision_completions_payload():
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "text": "ok",
                "prompt_token_ids": [1, 2, 3],
                "token_ids": [4, 5],
                "logprobs": {
                    "content": [
                        {"logprob": -0.11, "sampling_logprob": -0.1},
                        {"logprob": -0.22, "sampling_logprob": -0.2},
                    ],
                },
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


def test_parse_vision_completions_requires_sampling_logprob():
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "text": "ok",
                "prompt_token_ids": [1, 2, 3],
                "token_ids": [4, 5],
                "logprobs": {
                    "content": [
                        {"logprob": -0.11},
                        {"logprob": -0.22},
                    ],
                    "token_logprobs": [-0.11, -0.22],
                },
            }
        ]
    }
    with pytest.raises(RuntimeError, match="sampling_logprob"):
        _parse_vision_completions_payload(payload)


def test_build_multimodal_policy_datum_preserves_image_chunk():
    prompt = _multimodal_prompt()
    datum = build_multimodal_policy_datum(prompt, [20, 21, 22])
    targets = list(datum.loss_fn_inputs["target_tokens"].data)
    weights = list(datum.loss_fn_inputs["weights"].data)
    assert targets == [11, 0, 0, 0, 0, 12, 20, 21, 22]
    assert len(targets) == len(weights) == datum.model_input.length
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
        raw_completion_logprobs=[-1.1, -1.2],
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
        raw_completion_logprobs=[-1.1, -1.2],
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
    assert ref_mask.shape[0] == len(pg.data[0].loss_fn_inputs["target_tokens"].data)


def test_multimodal_inf_logprobs_match_datum_weight_index_space():
    """GRPO slices inf_logprobs with prompt_lens in shifted-weights space."""
    prompt = _multimodal_prompt()
    run = _build_multimodal_rollout_sample(
        prompt_model_input=prompt,
        completion_tokens=[30, 31],
        completion_logprobs=[-0.1, -0.2],
        raw_completion_logprobs=[-1.1, -1.2],
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
    raw_inf_lp = pg.raw_inf_logprobs[0]
    assert len(inf_lp) == len(weights)
    assert len(raw_inf_lp) == len(weights)
    active = [i for i, w in enumerate(weights) if w > 0]
    assert [inf_lp[i] for i in active] == [-0.1, -0.2]
    assert [raw_inf_lp[i] for i in active] == [-1.1, -1.2]
    assert all(inf_lp[i] == 0.0 for i in range(len(weights)) if i not in active)
    assert all(raw_inf_lp[i] == 0.0 for i in range(len(weights)) if i not in active)
    assert pg.prompt_lens[0] == active[0] + 1
    response_start = pg.prompt_lens[0] - 1
    assert response_start == active[0]
    assert len(inf_lp) >= response_start + (len(weights) - response_start)


def test_multimodal_client_grpo_preserves_expanded_coordinates():
    """Client-side GRPO receives one logprob/gradient per expanded target."""
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
    assert pg.prompt_lens is not None

    target_lengths = [
        len(datum.loss_fn_inputs["target_tokens"].data) for datum in pg.data
    ]
    assert target_lengths == [len(values) for values in pg.inf_logprobs]

    build_client_loss = build_loss_fn(
        LossConfig(policy_loss="grpo", loss_path="client", kl_beta=0.0)
    )
    loss_fn = build_client_loss(
        pg.advantages,
        [[0.0] * n for n in target_lengths],
        pg.prompt_lens,
        pg.inf_logprobs,
        pg.inf_logprobs,
    )
    forward_logprobs = [
        torch.tensor(values, dtype=torch.float32, requires_grad=True)
        for values in pg.inf_logprobs
    ]

    loss, metrics = loss_fn(pg.data, forward_logprobs)
    loss.backward()

    assert metrics["active_tokens"] == 4
    for datum, logprobs in zip(pg.data, forward_logprobs):
        weights = [float(x) for x in datum.loss_fn_inputs["weights"].data]
        assert logprobs.grad is not None
        assert all(
            logprobs.grad[i].item() == pytest.approx(0.0)
            for i, weight in enumerate(weights)
            if weight == 0.0
        )
        assert any(
            abs(logprobs.grad[i].item()) > 0
            for i, weight in enumerate(weights)
            if weight > 0.0
        )


def test_multimodal_builtin_loss_datums_use_expanded_coordinates():
    """Built-in loss fields align with image slots instead of reshaping."""
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
    assert pg.prompt_lens is not None

    builtin_datums = build_builtin_loss_datums(
        data=pg.data,
        advantages=pg.advantages,
        old_policy_logprobs=pg.inf_logprobs,
        inf_logprobs=pg.inf_logprobs,
        prompt_lens=pg.prompt_lens,
        policy_loss="grpo",
    )

    for policy_datum, builtin_datum in zip(pg.data, builtin_datums):
        expected_len = policy_datum.model_input.length
        assert set(builtin_datum.loss_fn_inputs) == {
            "target_tokens",
            "logprobs",
            "advantages",
        }
        assert all(
            len(tensor.data) == tensor.shape[0] == expected_len
            for tensor in builtin_datum.loss_fn_inputs.values()
        )
        target_tokens = builtin_datum.loss_fn_inputs["target_tokens"].data
        assert target_tokens[1:5] == [0, 0, 0, 0]
        packed_advantages = builtin_datum.loss_fn_inputs["advantages"].data
        weights = policy_datum.loss_fn_inputs["weights"].data
        assert all(
            packed_advantages[i] == pytest.approx(0.0)
            for i, weight in enumerate(weights)
            if float(weight) == 0.0
        )


def test_multimodal_router_replay_uses_expanded_datum_boundary():
    prompt = _multimodal_prompt()
    run = _build_multimodal_rollout_sample(
        prompt_model_input=prompt,
        completion_tokens=[30, 31],
        completion_logprobs=[-0.1, -0.2],
        routing_matrices=["route-30", "route-31"],
        reward=1.0,
        finish_reason="stop",
        text="hi",
    )

    pg = rollout_to_prompt_group(
        Rollout(runs=[run, run]),
        advantage_fn=lambda rewards: list(rewards),
        with_reference=True,
        router_replay_completion_only=True,
    )

    assert pg is not None
    weights = [float(w) for w in pg.data[0].loss_fn_inputs["weights"].data]
    active = [i for i, w in enumerate(weights) if w > 0]
    routes = pg.data[0].model_input.routing_matrices
    assert routes is not None
    assert len(routes) == len(weights)
    assert [routes[i] for i in active] == ["route-30", "route-31"]
    assert all(routes[i] == "" for i in range(active[0]))
    assert pg.ref_data[0].model_input.routing_matrices is None


@pytest.mark.asyncio
async def test_single_turn_renderer_rollout_parses_completion_tokens_only():
    """Prompt-prefilled tokens are sliced off before renderer.parse_response."""
    captured: dict[str, object] = {}
    prompt_tokens = [10, 20, 30]
    completion_tokens = [40, 50]

    class _Renderer:
        def build_generation_prompt(
            self, messages: object, **kwargs: object
        ) -> tinker.ModelInput:
            del messages, kwargs
            return tinker.ModelInput(
                chunks=[tinker.types.EncodedTextChunk(tokens=prompt_tokens)]
            )

        def parse_response(self, tokens: list[int]) -> tuple[dict[str, str], bool]:
            captured["parse_tokens"] = list(tokens)
            return ({"content": "x"}, True)

        def get_stop_sequences(self) -> list[int]:
            return []

    async def sample_with_prompt_tokens(
        prompt_token_ids: list[int], **kwargs: object
    ) -> list[SampledCompletion]:
        captured["prompt_token_ids"] = list(prompt_token_ids)
        captured["kwargs"] = kwargs
        return [
            SampledCompletion(
                text="gen",
                full_tokens=prompt_tokens + completion_tokens,
                prompt_len=len(prompt_tokens),
                finish_reason="stop",
                inference_logprobs=[-0.3, -0.4],
                sampling_logprobs=[-0.31, -0.41],
            )
        ]

    run = await single_turn_renderer_rollout(
        {"id": 1},
        renderer=_Renderer(),
        sample_with_prompt_tokens=sample_with_prompt_tokens,
        message_builder=_test_messages,
        reward_fn=_test_reward,
    )

    assert run is not None
    assert captured["prompt_token_ids"] == prompt_tokens
    assert captured["parse_tokens"] == completion_tokens
    assert captured["kwargs"]["n"] == 1
    segment = run.segments[0]
    assert segment.tokens == prompt_tokens + completion_tokens
    assert segment.logprobs[-2:] == [-0.31, -0.41]
    assert segment.raw_logprobs is not None
    assert segment.raw_logprobs[-2:] == [-0.3, -0.4]


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
                sampling_logprobs=[-0.31, -0.41],
                routing_matrices=["route-99", "route-100"],
            )
        ]

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [10, 11, 248056, 12, 248068, 198]
    tokenizer.special_ids = MagicMock(image=248056)

    run = await single_turn_renderer_rollout(
        {"id": 1},
        renderer=_Renderer(),
        sample_with_prompt_tokens=sample_with_prompt_tokens,
        message_builder=_test_messages,
        reward_fn=_test_reward,
        tokenizer=tokenizer,
        sample_kwargs={"include_routing_matrix": True},
    )
    assert run is not None
    assert captured["prompt_token_ids"] == [10, 11, 248056, 12]
    tokenizer.apply_chat_template.assert_not_called()
    assert captured["kwargs"]["images"] == [_BASE64_PNG]
    assert captured["kwargs"]["logprobs"] is True
    assert captured["kwargs"]["return_token_ids"] is True
    assert captured["kwargs"]["include_routing_matrix"] is True
    segment = run.segments[0]
    assert segment.prompt_model_input is not None
    assert segment.tokens[-2:] == [99, 100]
    assert segment.logprobs[-2:] == [-0.31, -0.41]
    assert segment.routing_matrices == ["route-99", "route-100"]


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
    tokenizer.decode.side_effect = ["PROMPT:", "ASSISTANT:"]

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
    tokenizer.apply_chat_template.assert_not_called()


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
    tokenizer.decode.side_effect = ["PROMPT:", "ASSISTANT:"]

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
    assert captured["prompt_text"] == (
        "PROMPT:<|vision_start|><|image_pad|><|vision_end|>ASSISTANT:"
    )
    assert captured["images"] == [_BASE64_PNG]
    tokenizer.apply_chat_template.assert_not_called()
    assert run.segments[0].tokens[-2:] == [99, 100]
