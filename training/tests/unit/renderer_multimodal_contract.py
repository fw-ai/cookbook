"""Reusable real-renderer contracts for image requests and next-token datums."""

from __future__ import annotations

import base64

import pytest
import tinker
from PIL import Image

from training.renderer import TrainOnWhat
from training.tests.unit.renderer_contract import _split_by_weights
from training.utils.rl.rollout.renderer import (
    build_multimodal_completions_prompt_token_ids,
)
from training.utils.supervised import (
    build_multimodal_policy_datum,
    normalize_messages,
    render_messages_to_datum,
)


def _expanded_tokens(model_input):
    return [
        token
        for chunk in model_input.chunks
        for token in (
            chunk.tokens
            if isinstance(chunk, tinker.types.EncodedTextChunk)
            else [0] * chunk.length
        )
    ]


class MultimodalRendererContractTests:
    """Adapters provide a real renderer and independently pinned image metadata."""

    cases = []
    # Each entry supplies (image dimensions, expected visual token count).
    image_fixtures = []
    # True when the checkpoint's SFT generation mask includes the same
    # assistant suffix that build_generation_prompt appends before sampling.
    sft_trains_generation_suffix = False

    def load_image_renderer(self, case):
        """Return (tokenizer, renderer, native request placeholder from config)."""
        raise NotImplementedError

    def pytest_generate_tests(self, metafunc):
        if "case" in metafunc.fixturenames:
            metafunc.parametrize("case", self.cases, ids=lambda case: case.renderer)

    def test_image_matrix_is_nonempty(self):
        assert self.cases
        assert self.image_fixtures

    @pytest.mark.parametrize("role", ["user", "tool"])
    @pytest.mark.parametrize("multiple_images", [False, True])
    def test_image_request_and_training_contract(self, case, role, multiple_images):
        tokenizer, renderer, native_image_id = self.load_image_renderer(case)
        fixtures = self.image_fixtures if multiple_images else self.image_fixtures[:1]
        content = [{"type": "text", "text": "Inspect these images: "}]
        for index, (size, _) in enumerate(fixtures):
            content.append(
                {
                    "type": "image",
                    "image": Image.new("RGB", size, color=(index * 50, 30, 90)),
                }
            )
            content.append({"type": "text", "text": f" image {index}. "})
        messages = []
        if role == "tool":
            messages = [
                {"role": "user", "content": "Inspect the result."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "inspect-1",
                            "type": "function",
                            "function": {"name": "inspect", "arguments": {}},
                        }
                    ],
                },
            ]
        messages.append(
            {
                "role": role,
                "content": content,
                **(
                    {"name": "inspect", "tool_call_id": "inspect-1"}
                    if role == "tool"
                    else {}
                ),
            }
        )
        normalized = normalize_messages(messages)
        prompt = renderer.build_generation_prompt(normalized)
        image_chunks = [
            chunk
            for chunk in prompt.chunks
            if isinstance(chunk, tinker.types.ImageChunk)
        ]
        assert [chunk.expected_tokens for chunk in image_chunks] == [
            count for _, count in fixtures
        ]

        request_tokens, images = build_multimodal_completions_prompt_token_ids(
            normalized,
            prompt,
            tokenizer,
            renderer=renderer,
        )
        # The expected placeholder comes from checkpoint config, never from
        # the renderer whose request contract this test is verifying.
        assert renderer.image_placeholder_token_id == native_image_id
        assert request_tokens.count(native_image_id) == len(fixtures) == len(images)
        assert images == [
            f"data:image/{chunk.format};base64,{base64.b64encode(chunk.data).decode()}"
            for chunk in image_chunks
        ]
        expected_request = [
            token
            for chunk in prompt.chunks
            for token in (
                chunk.tokens
                if isinstance(chunk, tinker.types.EncodedTextChunk)
                else [native_image_id]
            )
        ]
        assert request_tokens == expected_request

        full_messages = normalized + [
            {"role": "assistant", "content": "These are two colored panels."}
        ]
        full_input, full_weights = renderer.build_supervised_example(full_messages)
        expanded = _expanded_tokens(full_input)
        weights = [float(weight) for weight in full_weights.tolist()]
        observation, completion = _split_by_weights(expanded, weights)
        prompt_tokens = _expanded_tokens(prompt)
        assert expanded[: prompt.length] == prompt_tokens
        if self.sft_trains_generation_suffix:
            # SFT trains that header, while rollout sampling starts after the
            # identical suffix already present in the generation prompt.
            trainable_suffix_length = prompt.length - len(observation)
            assert trainable_suffix_length > 0
            assert (
                completion[:trainable_suffix_length]
                == prompt_tokens[-trainable_suffix_length:]
            )
            rollout_completion = completion[trainable_suffix_length:]
        else:
            assert observation == prompt_tokens
            rollout_completion = completion
        assert completion

        # Both hosted SFT and token-in RL must preserve images and use the
        # same expanded N-1 target/mask coordinates after the next-token shift.
        sft = render_messages_to_datum(
            messages
            + [{"role": "assistant", "content": "These are two colored panels."}],
            renderer=renderer,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        ).datum
        policy = build_multimodal_policy_datum(prompt, rollout_completion)
        for datum in (sft, policy):
            assert list(datum.loss_fn_inputs["target_tokens"].data) == expanded[1:]
            assert datum.model_input.length == len(expanded) - 1
            assert [
                chunk
                for chunk in datum.model_input.chunks
                if isinstance(chunk, tinker.types.ImageChunk)
            ] == image_chunks
        assert list(sft.loss_fn_inputs["weights"].data) == weights[1:]
        policy_weights = [0.0] * prompt.length + [1.0] * len(rollout_completion)
        assert list(policy.loss_fn_inputs["weights"].data) == policy_weights[1:]
        assert weights[: len(observation)] == [0.0] * len(observation)

        # Hosted SFT defaults to training every assistant turn, including the
        # historical tool call. Image positions must remain masked in that mode.
        all_input, all_weights = renderer.build_supervised_example(
            full_messages,
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
        all_sft = render_messages_to_datum(
            messages
            + [{"role": "assistant", "content": "These are two colored panels."}],
            renderer=renderer,
        ).datum
        assert _expanded_tokens(all_input) == expanded
        all_weights = [float(weight) for weight in all_weights.tolist()]
        assert list(all_sft.loss_fn_inputs["target_tokens"].data) == expanded[1:]
        assert list(all_sft.loss_fn_inputs["weights"].data) == all_weights[1:]
        assert all(
            all_weight >= last_weight
            for all_weight, last_weight in zip(all_weights, weights)
        )
        if role == "tool":
            assert sum(all_weights) > sum(weights)
        cursor = 0
        for chunk in all_input.chunks:
            if isinstance(chunk, tinker.types.ImageChunk):
                assert (
                    all_weights[cursor : cursor + chunk.length] == [0.0] * chunk.length
                )
            cursor += chunk.length
