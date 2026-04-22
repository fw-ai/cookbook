"""Unit tests for training.utils.rl.tokenize helpers."""

from __future__ import annotations

import pytest

from training.utils.rl import tokenize as tokenize_mod
from training.utils.rl.tokenize import (
    _normalize_inference_base_url,
    get_prefill_logprobs,
    tokenize_chat_turn,
)


class _FakeTokenizer:
    """Minimal chat-template tokenizer.

    Each message contributes a deterministic list of token IDs:
    role=user -> [1, <content_len>], role=assistant -> [2, <content_len>,
    3 (eot)], system -> [0, <content_len>]. ``add_generation_prompt`` adds
    a trailing ``[99]`` marker.  Crucially, rendering the same message list
    always yields the same prefix when more messages are appended, so the
    prefix-diff in tokenize_chat_turn works.
    """

    _ROLE_TOKEN = {"system": 0, "user": 1, "assistant": 2}
    _EOT = 3
    _GEN_PROMPT = 99

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False, **_):
        out: list[int] = []
        for m in messages:
            role = m.get("role", "user")
            content_len = len(str(m.get("content", "")))
            out.append(self._ROLE_TOKEN.get(role, 1))
            out.append(content_len)
            if role == "assistant":
                out.append(self._EOT)
        if add_generation_prompt:
            out.append(self._GEN_PROMPT)
        if not tokenize:
            raise NotImplementedError("fake only supports tokenize=True")
        return out


class TestTokenizeChatTurn:
    def test_raises_when_prefix_invariant_violated(self):
        """If the assistant turn isn't a strict extension of the rendered
        prompt, the helper must fail loudly rather than silently derive
        garbage completion_ids.  The default fake tokenizer violates the
        invariant (the generation-prompt marker is outside the assistant
        turn), so it's the natural negative-test fixture."""
        tokenizer = _FakeTokenizer()
        messages = [{"role": "user", "content": "hi"}]
        assistant = {"role": "assistant", "content": "yo"}

        with pytest.raises(RuntimeError, match="prefix extension"):
            tokenize_chat_turn(messages, assistant, tokenizer)

    def test_prefix_preserving_tokenizer_produces_expected_split(self):
        """A chat template where the generation-prompt marker is *inside*
        the assistant turn satisfies the strict-prefix invariant; the
        helper then returns the assistant-added suffix as completion_ids."""

        class _PrefixTokenizer:
            _ROLE_TOKEN = {"system": 0, "user": 1, "assistant": 2}
            _EOT = 3
            _GEN_PROMPT = 99

            def apply_chat_template(
                self, messages, *, tokenize=True, add_generation_prompt=False, **_
            ):
                out: list[int] = []
                for m in messages:
                    role = m.get("role", "user")
                    content_len = len(str(m.get("content", "")))
                    if role == "assistant":
                        out.append(self._GEN_PROMPT)  # part of assistant turn
                        out.append(self._ROLE_TOKEN[role])
                        out.append(content_len)
                        out.append(self._EOT)
                    else:
                        out.append(self._ROLE_TOKEN.get(role, 1))
                        out.append(content_len)
                if add_generation_prompt:
                    out.append(self._GEN_PROMPT)
                return out

        tokenizer = _PrefixTokenizer()
        messages = [{"role": "user", "content": "hi"}]  # 2 chars
        assistant = {"role": "assistant", "content": "yo"}  # 2 chars

        prompt_ids, completion_ids = tokenize_chat_turn(messages, assistant, tokenizer)

        # user: [1, 2], gen prompt: [99] -> prompt_ids = [1, 2, 99]
        assert prompt_ids == [1, 2, 99]
        # Full: user [1, 2] + assistant [99, 2, 2, 3] -> [1, 2, 99, 2, 2, 3].
        # completion_ids is the suffix past the rendered prompt.
        assert completion_ids == [2, 2, 3]

    def test_multi_turn_prompt_includes_earlier_messages(self):
        class _PrefixTokenizer:
            _ROLE_TOKEN = {"system": 0, "user": 1, "assistant": 2}
            _EOT = 3
            _GEN_PROMPT = 99

            def apply_chat_template(
                self, messages, *, tokenize=True, add_generation_prompt=False, **_
            ):
                out: list[int] = []
                for m in messages:
                    role = m.get("role", "user")
                    content_len = len(str(m.get("content", "")))
                    if role == "assistant":
                        out.extend([self._GEN_PROMPT, self._ROLE_TOKEN[role], content_len, self._EOT])
                    else:
                        out.extend([self._ROLE_TOKEN.get(role, 1), content_len])
                if add_generation_prompt:
                    out.append(self._GEN_PROMPT)
                return out

        tokenizer = _PrefixTokenizer()
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        assistant = {"role": "assistant", "content": "a2"}

        prompt_ids, completion_ids = tokenize_chat_turn(messages, assistant, tokenizer)

        assert prompt_ids[-1] == 99, "generation prompt marker must end the prompt"
        assert len(completion_ids) > 0
        # The first assistant turn must appear before the generation prompt.
        assert 3 in prompt_ids, "earlier assistant EOT should be preserved in prompt"


class TestNormalizeInferenceBaseUrl:
    def test_strips_inference_suffix(self):
        assert _normalize_inference_base_url("https://host.example/inference") == "https://host.example"

    def test_strips_inference_v1_suffix(self):
        assert (
            _normalize_inference_base_url("https://host.example/inference/v1")
            == "https://host.example"
        )

    def test_strips_trailing_slash(self):
        assert _normalize_inference_base_url("https://host.example/") == "https://host.example"

    def test_leaves_other_urls_alone(self):
        assert _normalize_inference_base_url("https://host.example/v1") == "https://host.example/v1"


class TestGetPrefillLogprobs:
    def test_returns_empty_for_no_tokens(self):
        assert get_prefill_logprobs(url="x", tokens=[], api_key="k", model="m") == []

    def test_returns_zero_padding_for_single_token(self):
        assert get_prefill_logprobs(url="x", tokens=[5], api_key="k", model="m") == []

    def test_aligns_and_drops_leading_none(self, monkeypatch):
        captured: dict = {}

        class _Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {"logprobs": {"token_logprobs": [None, -0.1, -0.2, -0.3]}}
                    ]
                }

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json
            captured["headers"] = headers
            return _Resp()

        monkeypatch.setattr(tokenize_mod.requests, "post", fake_post)

        out = get_prefill_logprobs(
            url="https://host/inference/v1",
            tokens=[10, 20, 30, 40],
            api_key="secret",
            model="model-x",
        )

        assert out == [-0.1, -0.2, -0.3]
        assert captured["url"].endswith("/inference/v1/completions")
        assert captured["url"].startswith("https://host/")
        assert captured["payload"]["prompt"] == [10, 20, 30, 40]
        assert captured["payload"]["echo"] is True
        assert captured["payload"]["max_tokens"] == 1
        assert captured["payload"]["model"] == "model-x"
        assert captured["headers"]["Authorization"] == "Bearer secret"

    def test_pads_when_server_returns_fewer_logprobs(self, monkeypatch):
        class _Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"logprobs": {"token_logprobs": [None, -0.5]}}]}

        monkeypatch.setattr(tokenize_mod.requests, "post", lambda *a, **kw: _Resp())

        out = get_prefill_logprobs(
            url="https://host", tokens=[1, 2, 3, 4], api_key="k", model="m"
        )
        # Expected length = len(tokens) - 1 = 3; server returned 1 aligned value.
        assert out == [-0.5, 0.0, 0.0]

    def test_replaces_none_logprobs_with_zero(self, monkeypatch):
        class _Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"logprobs": {"token_logprobs": [None, -0.1, None, -0.3]}}]}

        monkeypatch.setattr(tokenize_mod.requests, "post", lambda *a, **kw: _Resp())

        out = get_prefill_logprobs(url="https://host", tokens=[1, 2, 3, 4], api_key="k", model="m")
        assert out == [-0.1, 0.0, -0.3]

    def test_handles_empty_choices(self, monkeypatch):
        class _Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": []}

        monkeypatch.setattr(tokenize_mod.requests, "post", lambda *a, **kw: _Resp())

        out = get_prefill_logprobs(url="https://host", tokens=[1, 2, 3], api_key="k", model="m")
        assert out == [0.0, 0.0]
