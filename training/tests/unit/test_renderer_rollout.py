"""Unit tests for ``training.utils.rl.renderer_rollout``.

Covers the ``model_input_to_token_ids`` adapter (text-only acceptance,
multimodal rejection) and ``single_turn_renderer_rollout`` against a curated
canary matrix of two renderers: ``gemma4`` (cookbook) and ``kimi_k25``
(Tinker upstream).  The sampler is mocked so the tests run hermetically and
deterministically; the goal is to verify the helper preserves the renderer's
prompt verbatim, packs assistant tokens / logprobs / loss-mask correctly,
and never re-renders chat templates.
"""

from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock

import pytest
import tinker

# Make the Tinker upstream cookbook (kimi_k25 etc.) importable when this
# repo is checked out next to it.
_TINKER_COOKBOOK = Path(__file__).resolve().parents[4] / "tinker-cookbook"
if _TINKER_COOKBOOK.exists() and str(_TINKER_COOKBOOK) not in sys.path:
    sys.path.insert(0, str(_TINKER_COOKBOOK))

from training.utils.rl.renderer_rollout import (
    MultimodalRenderingNotSupported,
    RolloutHelperInfo,
    helper_info,  # backward-compat alias of renderer_helper_info
    make_remote_rollout_fn,
    model_input_to_token_ids,
    renderer_helper_info,
    single_turn_renderer_rollout,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal stub renderers and sampler
# ---------------------------------------------------------------------------


@dataclass
class _FakeCtx:
    completions_per_prompt: int = 1
    sample_kwargs: dict = field(default_factory=dict)
    tokenizer_id: str = "test-tokenizer"


class _StubRenderer:
    """Minimal renderer-shaped stub used when we don't want to load a real
    upstream renderer.  Tests against gemma4 and kimi_k25 swap in real ones.
    """

    def __init__(
        self,
        prompt_tokens: List[int],
        stop_sequences: List[Any],
        parsed: tuple[Any, bool] = (SimpleNamespace(role="assistant"), True),
        name: str = "stub",
    ) -> None:
        self._prompt_tokens = list(prompt_tokens)
        self._stop = list(stop_sequences)
        self._parsed = parsed
        self.name = name

    def build_generation_prompt(self, messages: List[Any]) -> tinker.ModelInput:
        return tinker.ModelInput.from_ints(self._prompt_tokens)

    def parse_response(self, tokens: List[int]) -> tuple[Any, bool]:
        return self._parsed

    def get_stop_sequences(self) -> List[Any]:
        return list(self._stop)


def _completion(prompt_token_ids: List[int], out_tokens: List[int],
                logprobs: List[float] | None = None,
                finish_reason: str = "stop", text: str = "ok"):
    """Build a SampledCompletion-shaped object for the fake sampler."""
    return SimpleNamespace(
        text=text,
        full_tokens=list(prompt_token_ids) + list(out_tokens),
        prompt_len=len(prompt_token_ids),
        finish_reason=finish_reason,
        completion_len=len(out_tokens),
        inference_logprobs=logprobs,
        logprobs_echoed=False,
        routing_matrices=None,
    )


def _make_sampler(returns):
    """Return an async callable mirroring ``sample_with_prompt_tokens``.

    ``returns`` may be a single completion or a list of completions; the
    callable records its kwargs into ``last_call`` for assertions.
    """
    seq = list(returns) if isinstance(returns, list) else [returns]
    captured: dict[str, Any] = {}

    async def _sampler(prompt_token_ids, **kwargs):
        captured["prompt_token_ids"] = list(prompt_token_ids)
        captured["kwargs"] = dict(kwargs)
        return list(seq)

    return _sampler, captured


async def _identity_messages(row, ctx):
    return [{"role": "user", "content": row["q"]}]


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# model_input_to_token_ids — text only
# ---------------------------------------------------------------------------


class TestModelInputAdapter:
    def test_text_only_chunks_flatten(self):
        mi = tinker.ModelInput.from_ints([10, 20, 30])
        assert model_input_to_token_ids(mi) == [10, 20, 30]

    def test_concatenates_multiple_text_chunks(self):
        chunks = [
            tinker.EncodedTextChunk(tokens=[1, 2]),
            tinker.EncodedTextChunk(tokens=[3]),
            tinker.EncodedTextChunk(tokens=[4, 5, 6]),
        ]
        mi = tinker.ModelInput(chunks=chunks)
        assert model_input_to_token_ids(mi) == [1, 2, 3, 4, 5, 6]

    def test_empty_model_input(self):
        assert model_input_to_token_ids(tinker.ModelInput.empty()) == []

    def test_multimodal_raises_typed_error(self):
        class _ImageChunk:  # not an EncodedTextChunk
            type = "image"

        # Bypass StrictBase typing by using a plain wrapper attribute.
        mi = tinker.ModelInput.empty()
        object.__setattr__(mi, "chunks", [_ImageChunk()])

        with pytest.raises(MultimodalRenderingNotSupported, match="_ImageChunk"):
            model_input_to_token_ids(mi)


# ---------------------------------------------------------------------------
# single_turn_renderer_rollout — happy path against stub renderer
# ---------------------------------------------------------------------------


class TestSingleTurnHelperBasics:
    def test_produces_rollout_with_correct_token_layout(self):
        prompt = [101, 102, 103]
        out = [501, 502, 503]
        logprobs = [-0.1, -0.2, -0.3]

        renderer = _StubRenderer(prompt, ["</s>"])
        sampler, captured = _make_sampler(_completion(prompt, out, logprobs=logprobs))

        async def _reward(row, parsed, ok):
            assert ok is True
            return 0.7

        rollout = _run(single_turn_renderer_rollout(
            row={"q": "hi"},
            ctx=_FakeCtx(),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
        ))

        assert rollout is not None
        assert len(rollout.samples) == 1
        s = rollout.samples[0]
        assert s.tokens[: len(prompt)] == prompt
        assert s.tokens[len(prompt) :] == out
        assert s.logprobs[: len(prompt)] == [0.0] * len(prompt)
        assert s.logprobs[len(prompt) :] == logprobs
        assert s.loss_mask == [0] * len(prompt) + [1] * len(out)
        assert s.reward == pytest.approx(0.7)
        assert s.finish_reason == "stop"

    def test_default_stop_comes_from_renderer(self):
        renderer = _StubRenderer([1, 2], ["<eot>", "</s>"])
        sampler, captured = _make_sampler(_completion([1, 2], [9]))

        async def _reward(row, parsed, ok):
            return 0.0

        _run(single_turn_renderer_rollout(
            row={"q": "x"}, ctx=_FakeCtx(),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
        ))

        assert captured["kwargs"]["stop"] == ["<eot>", "</s>"]

    def test_explicit_stop_overrides_renderer_default(self):
        renderer = _StubRenderer([1, 2], ["<eot>"])
        sampler, captured = _make_sampler(_completion([1, 2], [9]))

        async def _reward(row, parsed, ok):
            return 0.0

        _run(single_turn_renderer_rollout(
            row={"q": "x"}, ctx=_FakeCtx(),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
            stop=[99, 100],  # int stop ids, must round-trip unchanged
        ))

        assert captured["kwargs"]["stop"] == [99, 100]
        assert all(isinstance(s, int) for s in captured["kwargs"]["stop"])

    def test_stop_str_shape_preserved(self):
        renderer = _StubRenderer([1, 2], ["<a>", "<b>"])
        sampler, captured = _make_sampler(_completion([1, 2], [9]))

        async def _reward(row, parsed, ok):
            return 0.0

        _run(single_turn_renderer_rollout(
            row={"q": "x"}, ctx=_FakeCtx(),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
        ))

        assert captured["kwargs"]["stop"] == ["<a>", "<b>"]
        assert all(isinstance(s, str) for s in captured["kwargs"]["stop"])

    def test_n_completions_packed_into_one_rollout(self):
        prompt = [1, 2]
        renderer = _StubRenderer(prompt, ["</s>"])
        completions = [
            _completion(prompt, [10, 11], logprobs=[-0.1, -0.2]),
            _completion(prompt, [20, 21, 22], logprobs=[-0.3, -0.4, -0.5]),
            _completion(prompt, [30], logprobs=[-0.6]),
        ]
        sampler, _ = _make_sampler(completions)

        async def _reward(row, parsed, ok):
            return 1.0

        rollout = _run(single_turn_renderer_rollout(
            row={"q": "x"},
            ctx=_FakeCtx(completions_per_prompt=3),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
        ))

        assert rollout is not None
        assert len(rollout.samples) == 3
        # n forwarded to the SDK primitive
        # (captured via the third call by the sampler factory; check separately)

    def test_drop_completion_when_reward_fn_returns_none(self):
        prompt = [1, 2]
        renderer = _StubRenderer(prompt, ["</s>"])
        completions = [
            _completion(prompt, [10]),
            _completion(prompt, [20]),
            _completion(prompt, [30]),
        ]
        sampler, _ = _make_sampler(completions)

        async def _reward(row, parsed, ok):
            return None  # caller chose DROP for every completion

        rollout = _run(single_turn_renderer_rollout(
            row={"q": "x"},
            ctx=_FakeCtx(completions_per_prompt=3),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
        ))

        assert rollout is None

    def test_zero_reward_completion_is_emitted(self):
        prompt = [1, 2]
        renderer = _StubRenderer(prompt, ["</s>"], parsed=(None, False))
        sampler, _ = _make_sampler(_completion(prompt, [10], logprobs=[-0.1]))

        async def _reward(row, parsed, ok):
            assert ok is False  # parse failed; user chose zero-reward
            return 0.0

        rollout = _run(single_turn_renderer_rollout(
            row={"q": "x"}, ctx=_FakeCtx(),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
        ))
        assert rollout is not None
        assert rollout.samples[0].reward == 0.0

    def test_length_finish_reason_flows_through_unchanged(self):
        prompt = [1, 2]
        renderer = _StubRenderer(prompt, ["</s>"])
        sampler, _ = _make_sampler(_completion(
            prompt, [10, 11], logprobs=[-0.1, -0.2], finish_reason="length",
        ))

        async def _reward(row, parsed, ok):
            return 0.5

        rollout = _run(single_turn_renderer_rollout(
            row={"q": "x"}, ctx=_FakeCtx(),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
        ))
        assert rollout is not None
        assert rollout.samples[0].finish_reason == "length"
        assert rollout.samples[0].reward == 0.5

    def test_drop_completion_when_inference_logprobs_missing(self):
        """The sampler MUST return per-token ``inference_logprobs``;
        fabricating zeros silently corrupts PPO/GRPO ratio/KL.  The
        helper drops the completion at the rollout boundary — same as
        ``extract_completion`` / ``pack_payload_to_sample``."""
        prompt = [1, 2]
        renderer = _StubRenderer(prompt, ["</s>"])
        # ``logprobs=None`` simulates an integration that forgot to
        # request logprobs.
        sampler, _ = _make_sampler(_completion(prompt, [10, 11], logprobs=None))

        async def _reward(row, parsed, ok):
            return 1.0

        rollout = _run(single_turn_renderer_rollout(
            row={"q": "x"}, ctx=_FakeCtx(),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
        ))
        # All completions dropped — no rollout emitted.
        assert rollout is None

    def test_drop_completion_when_inference_logprobs_misaligned(self):
        prompt = [1, 2]
        renderer = _StubRenderer(prompt, ["</s>"])
        # 2 assistant tokens but only 1 logprob — misalignment.
        sampler, _ = _make_sampler(_completion(prompt, [10, 11], logprobs=[-0.1]))

        async def _reward(row, parsed, ok):
            return 1.0

        rollout = _run(single_turn_renderer_rollout(
            row={"q": "x"}, ctx=_FakeCtx(),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
        ))
        assert rollout is None

    def test_echoed_logprobs_sliced_to_assistant_tokens(self):
        """When the sampler returns ``logprobs_echoed=True`` (caller
        passed ``echo=True`` in ``ctx.sample_kwargs``), inference
        logprobs cover the full ``prompt + completion`` span.  The
        helper must slice off the prompt prefix instead of dropping
        the completion as misaligned — mirrors the rl_loop fast path.
        """
        prompt = [1, 2]
        out = [10, 11]
        # Echoed sampler: logprobs has length prompt_len + len(out) = 4.
        c = _completion(prompt, out, logprobs=[-9.0, -9.0, -0.1, -0.2])
        c.logprobs_echoed = True
        sampler, _ = _make_sampler(c)

        async def _reward(row, parsed, ok):
            return 0.7

        rollout = _run(single_turn_renderer_rollout(
            row={"q": "x"}, ctx=_FakeCtx(),
            renderer=_StubRenderer(prompt, ["</s>"]),
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
        ))
        assert rollout is not None
        s = rollout.samples[0]
        # Tokens: prompt + assistant.
        assert s.tokens == [1, 2, 10, 11]
        # Per-token logprobs: 0.0 on prompt, real values on assistant.
        # (The prompt-prefix [-9.0, -9.0] from the echoed span is sliced off.)
        assert s.logprobs == [0.0, 0.0, -0.1, -0.2]
        assert s.loss_mask == [0, 0, 1, 1]

    def test_skip_empty_completions(self):
        prompt = [1, 2]
        renderer = _StubRenderer(prompt, ["</s>"])
        sampler, _ = _make_sampler(_completion(prompt, []))

        async def _reward(row, parsed, ok):
            return 1.0  # would produce a reward — but completion is empty

        rollout = _run(single_turn_renderer_rollout(
            row={"q": "x"}, ctx=_FakeCtx(),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
        ))
        assert rollout is None

    def test_max_tokens_forwarded_to_sampler(self):
        renderer = _StubRenderer([1], ["</s>"])
        sampler, captured = _make_sampler(_completion([1], [2]))

        async def _reward(row, parsed, ok):
            return 0.0

        _run(single_turn_renderer_rollout(
            row={"q": "x"}, ctx=_FakeCtx(),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
            max_tokens=128,
        ))

        assert captured["kwargs"]["max_tokens"] == 128

    def test_n_passed_through(self):
        renderer = _StubRenderer([1], ["</s>"])
        sampler, captured = _make_sampler([_completion([1], [2])])

        async def _reward(row, parsed, ok):
            return 0.0

        _run(single_turn_renderer_rollout(
            row={"q": "x"},
            ctx=_FakeCtx(completions_per_prompt=4),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_identity_messages,
            reward_fn=_reward,
        ))

        assert captured["kwargs"]["n"] == 4


# ---------------------------------------------------------------------------
# Canary renderers — gemma4 (cookbook) and kimi_k25 (Tinker upstream)
# ---------------------------------------------------------------------------


def _try_load_gemma4():
    try:
        from training.renderer.gemma4 import Gemma4Renderer
    except ImportError:
        return None, "gemma4 renderer module not importable"
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None, "transformers not installed"
    import os
    model_path = os.environ.get("GEMMA4_MODEL_PATH")
    if not model_path:
        return None, "GEMMA4_MODEL_PATH not set"
    try:
        tok = AutoTokenizer.from_pretrained(model_path)
        return Gemma4Renderer(tok), None
    except Exception as e:  # noqa: BLE001
        return None, f"gemma4 tokenizer load failed: {e}"


def _try_load_kimi_k25():
    try:
        from tinker_cookbook.renderers import get_renderer  # type: ignore
    except ImportError:
        return None, "tinker_cookbook.renderers not importable"
    import os
    model_name = os.environ.get("KIMI_K25_MODEL_NAME", "moonshotai/Kimi-K2-Instruct")
    try:
        return get_renderer("kimi_k25", model_name), None
    except Exception as e:  # noqa: BLE001
        return None, f"kimi_k25 renderer load failed: {e}"


@pytest.mark.parametrize("renderer_loader", [_try_load_gemma4, _try_load_kimi_k25])
class TestCanaryRenderers:
    """Run against real renderers when their model files are available locally.

    Skips automatically when the tokenizer / renderer cannot be constructed
    so the suite stays green in environments without the model weights.
    """

    def test_helper_round_trip(self, renderer_loader):
        renderer, skip_reason = renderer_loader()
        if renderer is None:
            pytest.skip(skip_reason)

        messages = [
            {"role": "user", "content": "Solve 2+2."},
        ]
        prompt_tokens = model_input_to_token_ids(
            renderer.build_generation_prompt(messages)
        )
        assert prompt_tokens, "renderer produced empty prompt"

        out_tokens = [prompt_tokens[-1] + 1, prompt_tokens[-1] + 2]
        sampler, captured = _make_sampler(_completion(prompt_tokens, out_tokens,
                                                       logprobs=[-0.1, -0.2]))

        async def _build(row, ctx):
            return list(messages)

        async def _reward(row, parsed, ok):
            return 1.0

        rollout = _run(single_turn_renderer_rollout(
            row={}, ctx=_FakeCtx(),
            renderer=renderer,
            sample_with_prompt_tokens=sampler,
            message_builder=_build,
            reward_fn=_reward,
        ))

        assert rollout is not None
        s = rollout.samples[0]
        assert s.tokens[: len(prompt_tokens)] == prompt_tokens
        assert s.tokens[len(prompt_tokens) :] == out_tokens
        assert s.loss_mask == [0] * len(prompt_tokens) + [1] * len(out_tokens)
        # Stop sequences round-trip preserves their shape (list[str] | list[int])
        stops = renderer.get_stop_sequences()
        assert isinstance(stops, list)
        captured_stops = captured["kwargs"]["stop"]
        if stops and isinstance(stops[0], int):
            assert all(isinstance(s, int) for s in captured_stops)
        elif stops:
            assert all(isinstance(s, str) for s in captured_stops)


# ---------------------------------------------------------------------------
# Structural / boundary checks on the helper module itself
# ---------------------------------------------------------------------------


_HELPER_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "utils" / "rl" / "renderer_rollout.py"
)


class TestHelperModuleStructure:
    def test_helper_does_not_reference_apply_chat_template(self):
        text = _HELPER_MODULE_PATH.read_text()
        # Reject usage; mentioning the name in a docstring as a *negation* is fine
        # but the canonical structural test for AC-8 / AC-9 is "no call".
        assert "apply_chat_template(" not in text

    def test_helper_does_not_import_eval_protocol(self):
        text = _HELPER_MODULE_PATH.read_text()
        assert not re.search(r"^\s*(from|import)\s+eval_protocol\b",
                             text, re.MULTILINE)

    def test_helper_does_not_reference_sample_with_tokens_messages_kwarg(self):
        text = _HELPER_MODULE_PATH.read_text()
        assert "sample_with_tokens(messages=" not in text

    def test_helper_info_exposes_triage_fields(self):
        renderer = _StubRenderer([1, 2], ["</s>"], name="stub-name")
        info = renderer_helper_info(renderer, tokenizer_id="tok", max_tokens=512)
        assert isinstance(info, RolloutHelperInfo)
        assert info.tokenizer_id == "tok"
        assert info.renderer_name == "stub-name"
        assert info.stop_condition == ["</s>"]
        assert info.max_tokens == 512

    def test_helper_info_alias_kept_for_back_compat(self):
        # `helper_info` was the Round-0 name; `renderer_helper_info` is the
        # public AC-8 surface.  Both should work and return identical data.
        renderer = _StubRenderer([1, 2], ["</s>"], name="stub-name")
        a = helper_info(renderer, tokenizer_id="t", max_tokens=64)
        b = renderer_helper_info(renderer, tokenizer_id="t", max_tokens=64)
        assert a == b

    def test_single_turn_helper_exposes_helper_info_accessor(self):
        # AC-8 triage metadata: the helper function itself exposes a
        # `helper_info` callable so runtime triage code can grab the four
        # triage fields without re-deriving them.
        info_fn = getattr(single_turn_renderer_rollout, "helper_info", None)
        assert info_fn is not None
        renderer = _StubRenderer([1], ["</s>"], name="r")
        info = info_fn(renderer, tokenizer_id="tk", max_tokens=42)
        assert isinstance(info, RolloutHelperInfo)
        assert info.tokenizer_id == "tk"
        assert info.max_tokens == 42

    def test_helper_info_is_in_public_all(self):
        import training.utils.rl.renderer_rollout as mod
        assert "RolloutHelperInfo" in mod.__all__
        assert "renderer_helper_info" in mod.__all__

    def test_remote_rollout_helper_is_reexported(self):
        # Renderer-backed remote-rollout surface — same callable as
        # text_rollout.make_text_rollout_fn (renderer is applied service-side
        # so the helper is renderer-name-agnostic by design).
        from training.utils.rl import text_rollout as tr
        assert make_remote_rollout_fn is tr.make_text_rollout_fn
