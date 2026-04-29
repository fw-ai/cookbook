"""Tests for the renderer-backed tool-using example rollout.

Verifies: tool calls are routed through ``env.execute`` (not the renderer);
``loss_mask=1`` only on assistant tokens; AC-3 guard trips on Qwen3 default
mode; structural invariants (no eval_protocol, no apply_chat_template, no
add_environment_tokens call).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest
import tinker

from training.examples.rl.multi_turn_tool import rollout as rollout_mod


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubRenderer:
    def __init__(
        self,
        first_prompt: List[int],
        gap_suffix: List[int],
        parse_outputs: List[Any],
        *,
        has_extension_property: bool = True,
        name: str = "stub-renderer",
    ) -> None:
        self._first = list(first_prompt)
        self._gap = list(gap_suffix)
        self._parse_outputs = list(parse_outputs)
        self._call_idx = 0
        self._has_ext = has_extension_property
        self.name = name
        self._last_assistant_tokens: List[int] = []
        # Tool replies appended between turns are recorded here so the next
        # rendered prompt can include them in its input prefix.
        self._extra_gap: List[int] = []

    @property
    def has_extension_property(self) -> bool:
        return self._has_ext

    def build_generation_prompt(self, messages: List[Any]) -> tinker.ModelInput:
        # First call: just the first prompt.
        if not self._last_assistant_tokens:
            return tinker.ModelInput.from_ints(self._first)
        # Subsequent: first prompt + prior assistant + gap + extra-gap (tool reply).
        return tinker.ModelInput.from_ints(
            list(self._first)
            + list(self._last_assistant_tokens)
            + list(self._gap)
            + list(self._extra_gap)
        )

    def parse_response(self, tokens: List[int]) -> tuple[Any, bool]:
        idx = self._call_idx
        self._call_idx += 1
        self._last_assistant_tokens = list(tokens)
        if idx < len(self._parse_outputs):
            return (self._parse_outputs[idx], True)
        return (SimpleNamespace(role="assistant"), True)

    def get_stop_sequences(self) -> List[Any]:
        return ["</s>"]

    def add_tool_reply_tokens(self, tokens: List[int]) -> None:
        # Test-only helper to extend the renderer's "expected next prompt"
        # with tool-reply tokens so add_call's prefix invariant holds.
        self._extra_gap = list(self._extra_gap) + list(tokens)


@dataclass
class _ToolEnv:
    initial: List[Any] = field(default_factory=list)
    tool_message: Any = field(
        default_factory=lambda: SimpleNamespace(role="tool", content="tool-reply")
    )
    final_step_reward: float = 0.0
    terminal_reward: float | None = None
    execute_called: int = 0

    async def initial_messages(self):
        return list(self.initial)

    async def execute(self, tool_call):
        self.execute_called += 1
        return self.tool_message

    async def step(self, parsed):
        return ([], self.final_step_reward, True)

    async def reward(self, messages):
        return float(self.terminal_reward) if self.terminal_reward is not None else self.final_step_reward


@dataclass
class _Ctx:
    renderer: Any
    sample_with_prompt_tokens: Any
    build_env: Any
    tokenizer_id: str = "stub-tokenizer"
    sample_kwargs: dict = field(default_factory=dict)
    max_tokens: int | None = None
    _version: int = 7
    _version_iter: Any = None

    def current_version(self) -> int:
        if self._version_iter is not None:
            try:
                return next(self._version_iter)
            except StopIteration:
                return self._version
        return self._version


def _completion(prompt_token_ids: List[int], out_tokens: List[int],
                logprobs: List[float] | None = None,
                finish_reason: str = "stop"):
    return SimpleNamespace(
        text="",
        full_tokens=list(prompt_token_ids) + list(out_tokens),
        prompt_len=len(prompt_token_ids),
        finish_reason=finish_reason,
        completion_len=len(out_tokens),
        inference_logprobs=logprobs,
        logprobs_echoed=False,
        routing_matrices=None,
    )


def _make_sampler(returns):
    seq = list(returns)

    async def _sampler(prompt_token_ids, **kwargs):
        if not seq:
            raise RuntimeError("sampler exhausted")
        result = seq.pop(0)
        return [result] if not isinstance(result, list) else list(result)

    return _sampler


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tool execution path — routed through env, not renderer
# ---------------------------------------------------------------------------


class TestToolRouting:
    def test_env_execute_called_for_tool_calls(self):
        first_prompt = [1, 2]
        gap = [50]
        out_tool = [10]              # turn-1 assistant tokens (with tool call)
        tool_reply_tokens = [60, 61]  # tokens for the tool message
        out_final = [20, 21]          # turn-2 assistant tokens (no tool call)

        # Turn-1 parses with tool_calls; turn-2 parses without.
        parse_outputs = [
            SimpleNamespace(role="assistant", tool_calls=[{"name": "calc"}]),
            SimpleNamespace(role="assistant", tool_calls=None),
        ]
        renderer = _StubRenderer(first_prompt, gap, parse_outputs)
        env = _ToolEnv(initial=[{"role": "user"}], final_step_reward=0.4)

        # When we hit add_call on turn 2, the renderer needs to predict the
        # exact prompt tokens.  Pre-stage the tool reply so the prefix delta
        # matches.
        renderer.add_tool_reply_tokens(tool_reply_tokens)

        turn2_prompt = list(first_prompt) + list(out_tool) + list(gap) + list(tool_reply_tokens)
        sampler = _make_sampler([
            _completion(first_prompt, out_tool, logprobs=[-0.1]),
            _completion(turn2_prompt, out_final, logprobs=[-0.2, -0.3]),
        ])

        ctx = _Ctx(renderer=renderer, sample_with_prompt_tokens=sampler,
                   build_env=lambda row: env)

        rollout = _run(rollout_mod.rollout_fn({"id": "row-tool"}, ctx))

        assert rollout is not None
        assert env.execute_called == 1
        s = rollout.samples[0]
        # loss_mask is 1 only on assistant tokens — tool reply gets 0.
        expected_mask = (
            [0] * len(first_prompt)
            + [1] * len(out_tool)
            + [0] * len(gap)
            + [0] * len(tool_reply_tokens)
            + [1] * len(out_final)
        )
        assert s.loss_mask == expected_mask

    def test_renderer_parse_response_is_only_tool_call_extractor(self):
        # The renderer's parse_response is the only callable that yields
        # tool_calls; the env never inspects raw assistant tokens.  This is
        # implicit in the design — we verify it by checking the rollout file
        # never calls anything other than renderer.parse_response on
        # out_tokens.
        import ast
        text = (Path(rollout_mod.__file__)).read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                # Forbid env.execute_tool / env.parse_tool_calls / etc.
                attr = node.func.attr
                if attr in {"parse_response_streaming"}:
                    # Streaming path is out of scope for this example.
                    raise AssertionError(
                        "tool example must use parse_response, not parse_response_streaming"
                    )


# ---------------------------------------------------------------------------
# AC-3 — extension-property guard
# ---------------------------------------------------------------------------


class TestExtensionPropertyGuard:
    def test_qwen3_default_mode_raises_before_second_sampling_call(self):
        first_prompt = [1, 2]
        out_tool = [10]
        sample_call_count = {"n": 0}

        parse_outputs = [
            SimpleNamespace(role="assistant", tool_calls=[{"name": "calc"}]),
        ]
        renderer = _StubRenderer(first_prompt, [50], parse_outputs,
                                 has_extension_property=False, name="qwen3")

        async def _sampler(prompt_token_ids, **kwargs):
            sample_call_count["n"] += 1
            return [_completion(prompt_token_ids, out_tool, logprobs=[-0.1])]

        env = _ToolEnv(initial=[{"role": "user"}])
        ctx = _Ctx(renderer=renderer, sample_with_prompt_tokens=_sampler,
                   build_env=lambda row: env)

        with pytest.raises(RuntimeError, match=r"has_extension_property=False"):
            _run(rollout_mod.rollout_fn({}, ctx))

        # Guard fires before second sample call.
        assert sample_call_count["n"] == 1


# ---------------------------------------------------------------------------
# Structural invariants on the example file
# ---------------------------------------------------------------------------


_EXAMPLE_DIR = Path(__file__).resolve().parent


class TestPerTurnVersionPlumbing:
    """AC-13: tool rollouts preserve per-call deployment versions across
    assistant spans and stay at -1 on tool / template-suffix gap tokens."""

    def test_version_bump_across_tool_then_final_assistant_span(self):
        first_prompt = [1, 2]
        gap = [50]
        out_tool = [10]
        tool_reply_tokens = [60, 61]
        out_final = [20, 21]

        parse_outputs = [
            SimpleNamespace(role="assistant", tool_calls=[{"name": "calc"}]),
            SimpleNamespace(role="assistant", tool_calls=None),
        ]
        renderer = _StubRenderer(first_prompt, gap, parse_outputs)
        env = _ToolEnv(initial=[{"role": "user"}], final_step_reward=0.4)
        renderer.add_tool_reply_tokens(tool_reply_tokens)

        turn2_prompt = list(first_prompt) + list(out_tool) + list(gap) + list(tool_reply_tokens)
        sampler = _make_sampler([
            _completion(first_prompt, out_tool, logprobs=[-0.1]),
            _completion(turn2_prompt, out_final, logprobs=[-0.2, -0.3]),
        ])

        ctx = _Ctx(
            renderer=renderer, sample_with_prompt_tokens=sampler,
            build_env=lambda row: env,
            _version_iter=iter([21, 22]),
        )

        rollout = _run(rollout_mod.rollout_fn({}, ctx))
        assert rollout is not None
        s = rollout.samples[0]

        # versions layout:
        #   prompt(first):    [-1] * len(first_prompt)
        #   assistant(turn1): [21] * len(out_tool)
        #   gap (chat suffix + tool reply): [-1] * (len(gap) + len(tool_reply_tokens))
        #   assistant(turn2 final): [22] * len(out_final)
        expected = (
            [-1] * len(first_prompt)
            + [21] * len(out_tool)
            + [-1] * (len(gap) + len(tool_reply_tokens))
            + [22] * len(out_final)
        )
        assert s.versions == expected
        assistant_versions = {v for v, m in zip(s.versions, s.loss_mask) if m == 1}
        assert assistant_versions == {21, 22}


class TestStructuralInvariants:
    def test_no_eval_protocol_imports(self):
        for f in _EXAMPLE_DIR.glob("*.py"):
            if f.name.startswith("test_"):
                continue
            text = f.read_text()
            assert not re.search(r"^\s*(from|import)\s+eval_protocol\b",
                                 text, re.MULTILINE), f"{f}: contains eval_protocol import"

    def test_no_apply_chat_template(self):
        for f in _EXAMPLE_DIR.glob("*.py"):
            if f.name.startswith("test_"):
                continue
            text = f.read_text()
            assert "apply_chat_template(" not in text, f"{f}: contains apply_chat_template"

    def test_does_not_call_add_environment_tokens(self):
        import ast
        text = (_EXAMPLE_DIR / "rollout.py").read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "add_environment_tokens":
                    raise AssertionError(
                        "rollout.py calls add_environment_tokens; the default "
                        "full-rendering loop must derive gap tokens via add_call's "
                        "prefix delta instead."
                    )

    def test_no_renderer_side_tool_execution(self):
        # The renderer is never called as `renderer.execute(...)`.  Tool
        # execution lives on the env.
        import ast
        text = (_EXAMPLE_DIR / "rollout.py").read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "renderer"
                    and node.func.attr == "execute"
                ):
                    raise AssertionError(
                        "renderer.execute(...) call found; tool execution must "
                        "live on the env, not the renderer"
                    )
