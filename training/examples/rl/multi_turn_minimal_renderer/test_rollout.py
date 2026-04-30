"""Tests for the renderer-backed multi-turn example rollout.

The example file under test is a *concrete rollout function*, not a
framework helper.  These tests exercise the canonical happy path, the
AC-3 extension-property guard (Qwen3 default mode), the PrefixMismatch
drop behavior, the `_assembled=True` payload path through
`pack_payload_to_sample`, and the structural invariant that this example
imports zero `eval_protocol`.
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

from training.examples.rl.multi_turn_minimal_renderer import rollout as rollout_mod


# ---------------------------------------------------------------------------
# Stub renderer + env
# ---------------------------------------------------------------------------


class _StubRenderer:
    """Renderer-shaped stub: returns *deterministic* token sequences per call.

    The first turn's prompt is `[1, 2, 3]`; the second turn's prompt extends
    that with the previous-turn assistant tokens plus a fixed gap suffix
    (mimicking a chat-template suffix).  This satisfies the prefix-equality
    invariant unless the test deliberately breaks it.
    """

    def __init__(
        self,
        first_prompt: List[int],
        gap_suffix: List[int],
        *,
        has_extension_property: bool = True,
        parse_outputs: List[tuple[Any, bool]] | None = None,
        name: str = "stub-renderer",
    ) -> None:
        self._first = list(first_prompt)
        self._gap = list(gap_suffix)
        self._has_ext = has_extension_property
        self._parse_outputs = parse_outputs or [
            (SimpleNamespace(role="assistant", content="t1"), True),
            (SimpleNamespace(role="assistant", content="t2"), True),
        ]
        self._call_idx = 0
        self.name = name
        self._last_assistant_tokens: List[int] = []

    @property
    def has_extension_property(self) -> bool:
        return self._has_ext

    def build_generation_prompt(self, messages: List[Any]) -> tinker.ModelInput:
        # Turn 1: just the first prompt.
        # Turn 2+: first prompt + prior assistant tokens + gap suffix.
        if not self._last_assistant_tokens:
            tokens = list(self._first)
        else:
            tokens = list(self._first) + list(self._last_assistant_tokens) + list(self._gap)
        return tinker.ModelInput.from_ints(tokens)

    def parse_response(self, tokens: List[int]) -> tuple[Any, bool]:
        idx = self._call_idx
        self._call_idx += 1
        self._last_assistant_tokens = list(tokens)
        if idx < len(self._parse_outputs):
            return self._parse_outputs[idx]
        return (SimpleNamespace(role="assistant"), True)

    def get_stop_sequences(self) -> List[Any]:
        return ["</s>"]


class _DriftingRenderer(_StubRenderer):
    """Renderer that re-renders earlier turns, breaking prefix equality."""

    def build_generation_prompt(self, messages: List[Any]) -> tinker.ModelInput:
        # Always emit a different first-prompt sequence on the second call,
        # which breaks the assembler's strict-prefix invariant.
        if not self._last_assistant_tokens:
            return tinker.ModelInput.from_ints(self._first)
        # Drifted prompt: completely different first segment.
        return tinker.ModelInput.from_ints([999, 998] + list(self._last_assistant_tokens))


@dataclass
class _Env:
    initial: List[Any] = field(default_factory=list)
    n_turns: int = 2
    rewards: List[float] = field(default_factory=list)
    _step_count: int = 0

    async def initial_messages(self):
        return list(self.initial)

    async def step(self, parsed):
        self._step_count += 1
        done = self._step_count >= self.n_turns
        reward = self.rewards[self._step_count - 1] if self._step_count - 1 < len(self.rewards) else 0.0
        return ([], reward, done)


@dataclass
class _Ctx:
    renderer: Any
    sample_with_prompt_tokens: Any
    build_env: Any
    tokenizer_id: str = "stub-tokenizer"
    sample_kwargs: dict = field(default_factory=dict)
    max_tokens: int | None = None
    _version: int = 7
    _version_iter: Any = None  # optional iterator yielding per-call versions

    def current_version(self) -> int:
        if self._version_iter is not None:
            try:
                return next(self._version_iter)
            except StopIteration:
                return self._version
        return self._version


def _completion(prompt_token_ids: List[int], out_tokens: List[int],
                logprobs: List[float] | None = None,
                finish_reason: str = "stop", text: str = "ok"):
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


def _make_sampler(returns_per_call):
    """returns_per_call: list of completions to return on successive calls."""
    seq = list(returns_per_call)

    async def _sampler(prompt_token_ids, **kwargs):
        if not seq:
            raise RuntimeError("sampler exhausted")
        result = seq.pop(0)
        return [result] if not isinstance(result, list) else list(result)

    return _sampler


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Happy path — two-turn episode produces a single RolloutSample
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_two_turn_episode_produces_one_sample(self):
        first_prompt = [1, 2, 3]
        gap = [50, 51]
        out1 = [10, 11]  # turn-1 assistant tokens
        out2 = [20, 21, 22]  # turn-2 assistant tokens
        # Turn 2's prompt = first_prompt + out1 + gap
        turn2_prompt_len = len(first_prompt) + len(out1) + len(gap)

        renderer = _StubRenderer(first_prompt, gap)
        sampler = _make_sampler([
            _completion(first_prompt, out1, logprobs=[-0.1, -0.2]),
            _completion([0] * turn2_prompt_len, out2, logprobs=[-0.3, -0.4, -0.5]),
        ])
        env = _Env(initial=[{"role": "user", "content": "hi"}], n_turns=2,
                   rewards=[0.0, 0.9])
        ctx = _Ctx(renderer=renderer, sample_with_prompt_tokens=sampler,
                   build_env=lambda row: env)

        rollout = _run(rollout_mod.rollout_fn({"id": "row-1"}, ctx))

        assert rollout is not None
        assert len(rollout.samples) == 1
        s = rollout.samples[0]
        # Concatenation: first_prompt + out1 + gap + out2
        expected_tokens = list(first_prompt) + list(out1) + list(gap) + list(out2)
        assert s.tokens == expected_tokens
        # loss_mask: 1 only on assistant tokens.
        expected_mask = (
            [0] * len(first_prompt)
            + [1] * len(out1)
            + [0] * len(gap)
            + [1] * len(out2)
        )
        assert s.loss_mask == expected_mask
        # Reward from the terminal env step.
        assert s.reward == pytest.approx(0.9)
        assert rollout.row_meta == {"row_id": "row-1"}


# ---------------------------------------------------------------------------
# AC-3 — extension-property guard trips on Qwen3 default mode
# ---------------------------------------------------------------------------


class TestExtensionPropertyGuard:
    def test_qwen3_default_mode_raises_before_second_sampling_call(self):
        first_prompt = [1, 2, 3]
        gap = [50, 51]
        out1 = [10, 11]
        sample_call_count = {"n": 0}

        renderer = _StubRenderer(first_prompt, gap, has_extension_property=False,
                                 name="qwen3")

        async def _sampler(prompt_token_ids, **kwargs):
            sample_call_count["n"] += 1
            return [_completion(prompt_token_ids, out1, logprobs=[-0.1, -0.2])]

        env = _Env(initial=[{"role": "user", "content": "hi"}], n_turns=3, rewards=[0.0, 0.0, 0.0])
        ctx = _Ctx(renderer=renderer, sample_with_prompt_tokens=_sampler,
                   build_env=lambda row: env)

        with pytest.raises(RuntimeError, match=r"has_extension_property=False"):
            _run(rollout_mod.rollout_fn({"id": "row-2"}, ctx))

        # The guard fires BEFORE the second sampling call, so exactly one
        # sample request is made.
        assert sample_call_count["n"] == 1

    def test_extension_property_true_renderer_completes(self):
        renderer = _StubRenderer([1, 2], [], has_extension_property=True)
        out1 = [10, 11]
        sampler = _make_sampler([_completion([1, 2], out1, logprobs=[-0.1, -0.2])])
        env = _Env(initial=[{"role": "user"}], n_turns=1, rewards=[1.0])
        ctx = _Ctx(renderer=renderer, sample_with_prompt_tokens=sampler,
                   build_env=lambda row: env)

        rollout = _run(rollout_mod.rollout_fn({}, ctx))
        assert rollout is not None


# ---------------------------------------------------------------------------
# PrefixMismatch — propagate, do NOT silently drop
# ---------------------------------------------------------------------------


class TestEchoedLogprobs:
    """``echo=True`` in ``sample_kwargs`` makes the sampler return
    logprobs for the full ``prompt + completion`` span; the multi-turn
    helper must slice off the prompt prefix instead of treating the
    different length as a misalignment and dropping the turn.  Without
    this, every assistant turn under echoed sampling was discarded and
    the rollout yielded no trainable samples.
    """

    def test_echoed_completion_keeps_turn(self):
        first_prompt = [1, 2, 3]
        out = [10, 11]
        # Echoed sampler: logprobs covers prompt + completion (length 5).
        c = _completion(first_prompt, out, logprobs=[-9.0, -9.0, -9.0, -0.1, -0.2])
        c.logprobs_echoed = True

        async def reward_fn(*_):
            return 1.0

        async def sampler(_prompt, **_kwargs):
            return [c]

        env = _Env(initial=[{"role": "user"}], n_turns=1, rewards=[1.0])
        ctx = _Ctx(
            renderer=_StubRenderer(first_prompt, []),
            sample_with_prompt_tokens=sampler,
            build_env=lambda row: env,
        )
        rollout = _run(rollout_mod.rollout_fn({}, ctx))
        assert rollout is not None, (
            "echoed-logprob rollout must be retained, not dropped as "
            "misaligned"
        )
        s = rollout.samples[0]
        # The assistant logprobs (slice past the prompt prefix) should
        # appear at the end of the per-token logprob sequence.
        assistant_lps = [lp for lp, m in zip(s.logprobs, s.loss_mask) if m == 1]
        assert assistant_lps == [-0.1, -0.2]


class TestPrefixMismatchPropagates:
    def test_drifting_renderer_raises_prefix_mismatch(self):
        """A drifting renderer (or env that re-renders history
        differently between turns) violates the
        ``TrajectoryAssembler`` token-native invariant.  Earlier the
        rollout step swallowed ``PrefixMismatch`` and reported a
        benign sample miss, so training churned through every prompt
        with no actionable signal.  The exception now propagates so
        the integration bug fails loud."""
        from training.utils.rl.rollout import PrefixMismatch

        first_prompt = [1, 2, 3]
        out1 = [10, 11]

        renderer = _DriftingRenderer(first_prompt, [])
        sampler = _make_sampler([
            _completion(first_prompt, out1, logprobs=[-0.1, -0.2]),
            _completion([999, 998] + out1, [20], logprobs=[-0.3]),
        ])
        env = _Env(initial=[{"role": "user"}], n_turns=2, rewards=[0.0, 1.0])
        ctx = _Ctx(renderer=renderer, sample_with_prompt_tokens=sampler,
                   build_env=lambda row: env)

        with pytest.raises(PrefixMismatch):
            _run(rollout_mod.rollout_fn({}, ctx))


# ---------------------------------------------------------------------------
# _assembled=True payload path
# ---------------------------------------------------------------------------


class TestPerTurnVersionPlumbing:
    """AC-13: per-turn ``output_versions`` are preserved across assistant
    spans and stay at -1 on non-assistant gap tokens."""

    def test_version_bump_between_turns_preserved_per_span(self):
        first_prompt = [1, 2, 3]
        gap = [50, 51]
        out1 = [10, 11]
        out2 = [20, 21, 22]
        turn2_prompt_len = len(first_prompt) + len(out1) + len(gap)

        renderer = _StubRenderer(first_prompt, gap)
        sampler = _make_sampler([
            _completion(first_prompt, out1, logprobs=[-0.1, -0.2]),
            _completion([0] * turn2_prompt_len, out2, logprobs=[-0.3, -0.4, -0.5]),
        ])
        env = _Env(initial=[{"role": "user"}], n_turns=2, rewards=[0.0, 0.9])

        # Simulate a version bump between turns: turn 1 samples on v=11,
        # turn 2 samples on v=12.
        ctx = _Ctx(
            renderer=renderer, sample_with_prompt_tokens=sampler,
            build_env=lambda row: env,
            _version_iter=iter([11, 12]),
        )

        rollout = _run(rollout_mod.rollout_fn({}, ctx))
        assert rollout is not None
        s = rollout.samples[0]

        # versions layout:
        #   prompt(first):    [-1] * len(first_prompt)
        #   assistant(turn1): [11] * len(out1)
        #   gap (chat suffix between turns): [-1] * len(gap)
        #   assistant(turn2): [12] * len(out2)
        expected = (
            [-1] * len(first_prompt)
            + [11] * len(out1)
            + [-1] * len(gap)
            + [12] * len(out2)
        )
        assert s.versions == expected
        # Sanity: not collapsed to a single terminal scalar.
        assistant_versions = {v for v, m in zip(s.versions, s.loss_mask) if m == 1}
        assert assistant_versions == {11, 12}


class TestAssembledPackingPath:
    """The example uses ``pack_assembled_to_sample`` (assembler.to_flat-based)
    so per-call ``output_versions`` are preserved on assistant tokens.  The
    older ``pack_payload_to_sample`` path collapsed the whole trajectory
    onto a single terminal version and is no longer used here.
    """

    def test_sample_has_per_turn_versions_not_terminal_scalar(self):
        first_prompt = [1, 2]
        out1 = [10, 11]

        renderer = _StubRenderer(first_prompt, [])
        sampler = _make_sampler([_completion(first_prompt, out1, logprobs=[-0.1, -0.2])])
        env = _Env(initial=[{"role": "user"}], n_turns=1, rewards=[0.5])
        # ``current_version`` returns 7 — assistant tokens should carry 7;
        # prompt (gap) tokens stay at the assembler's default -1.
        ctx = _Ctx(renderer=renderer, sample_with_prompt_tokens=sampler,
                   build_env=lambda row: env, _version=7)

        rollout = _run(rollout_mod.rollout_fn({}, ctx))
        assert rollout is not None
        s = rollout.samples[0]
        # Prompt span: -1; assistant span: 7.
        assert s.versions == [-1] * len(first_prompt) + [7] * len(out1)


# ---------------------------------------------------------------------------
# Structural invariants on the example file
# ---------------------------------------------------------------------------


_EXAMPLE_DIR = Path(__file__).resolve().parent


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
        # The default full-rendering loop must use add_call, NOT
        # add_environment_tokens (reserved for incremental engine adapters).
        # We check for the call pattern (with paren) rather than mere mention,
        # so docstrings can still explain the design rationale.
        text = (_EXAMPLE_DIR / "rollout.py").read_text()
        # Strip module docstring before grepping for calls.
        import ast
        tree = ast.parse(text)
        ast.get_docstring(tree)  # ensure parse succeeds
        # Find any Call whose attribute is add_environment_tokens.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "add_environment_tokens":
                    raise AssertionError(
                        "rollout.py calls add_environment_tokens; the default "
                        "full-rendering loop must derive gap tokens via add_call's "
                        "prefix delta instead."
                    )
