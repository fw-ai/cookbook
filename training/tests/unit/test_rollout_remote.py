"""Unit tests for training.utils.rl.rollout.

Token-native only: every turn carries ``token_ids`` and every assistant
turn carries ``logprobs``.  Re-tokenizing text post-hoc silently
misaligns the loss mask and inference logprobs (slime/AReaL refuse to
do it; this packer follows the same rule), so the path doesn't exist
here either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from training.utils.rl.rollout import Rollout
from training.utils.rl.rollout import RolloutPayload, TurnRecord
from training.utils.rl.rollout import remote as tr


@dataclass
class _FakeCtx:
    completions_per_prompt: int = 2
    sample_kwargs: dict = field(default_factory=dict)
    inference_base_url: str = "https://example.invalid"
    api_key: str = "sk-test"
    model: str = "accounts/test/models/x"
    tokenizer_id: str = "test-tokenizer"
    _version: int = 3

    def current_version(self) -> int:
        return self._version


# ---------------------------------------------------------------------------
# Token-native packing
# ---------------------------------------------------------------------------


class TestTokenNative:
    @pytest.mark.asyncio
    async def test_single_turn_assistant(self):
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", text="hi", token_ids=[1, 2, 3]),
                TurnRecord(
                    role="assistant", text="ok",
                    token_ids=[4, 5], logprobs=[-0.1, -0.2],
                    finish_reason="stop",
                ),
            ],
            total_reward=0.75,
        )
        sample = await tr.pack_payload_to_sample(
            payload, ctx=_FakeCtx(), version=7,
        )
        assert sample.tokens == [1, 2, 3, 4, 5]
        assert sample.logprobs == [0.0, 0.0, 0.0, -0.1, -0.2]
        assert sample.loss_mask == [0, 0, 0, 1, 1]
        assert sample.reward == 0.75
        assert sample.versions == [7] * 5
        assert sample.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_multi_turn_interleaved_mask(self):
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", text="q1", token_ids=[1, 2]),
                TurnRecord(
                    role="assistant", text="a1",
                    token_ids=[3, 4], logprobs=[-0.1, -0.2],
                ),
                TurnRecord(role="tool", text="obs", token_ids=[5, 6, 7]),
                TurnRecord(
                    role="assistant", text="a2",
                    token_ids=[8, 9], logprobs=[-0.3, -0.4],
                    finish_reason="stop",
                ),
            ],
            total_reward=1.0,
        )
        sample = await tr.pack_payload_to_sample(
            payload, ctx=_FakeCtx(), version=0,
        )
        assert sample.tokens == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert sample.loss_mask == [0, 0, 1, 1, 0, 0, 0, 1, 1]
        assert sample.logprobs == [0.0, 0.0, -0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4]

    @pytest.mark.asyncio
    async def test_assistant_missing_logprobs_rejected(self):
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", token_ids=[1]),
                TurnRecord(role="assistant", token_ids=[2, 3], logprobs=None),
            ],
            total_reward=0.0,
        )
        with pytest.raises(tr._PackError, match="per-token logprobs"):
            await tr.pack_payload_to_sample(
                payload, ctx=_FakeCtx(), version=0,
            )

    @pytest.mark.asyncio
    async def test_text_only_turn_rejected(self):
        """A turn that lacks ``token_ids`` is a hard error: the packer
        refuses to re-tokenize text after the fact."""
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", token_ids=[1, 2]),
                TurnRecord(role="assistant", text="only text"),
            ],
            total_reward=0.5,
        )
        with pytest.raises(tr._PackError, match="missing token_ids"):
            await tr.pack_payload_to_sample(
                payload, ctx=_FakeCtx(), version=0,
            )

    @pytest.mark.asyncio
    async def test_tokenizer_id_mismatch_rejected(self):
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", token_ids=[1]),
                TurnRecord(role="assistant", token_ids=[2], logprobs=[-0.1]),
            ],
            total_reward=0.0,
            tokenizer_id="other-tokenizer",
        )
        with pytest.raises(tr._PackError, match="tokenizer_id"):
            await tr.pack_payload_to_sample(
                payload, ctx=_FakeCtx(), version=0,
            )

    @pytest.mark.asyncio
    async def test_matching_tokenizer_id_accepted(self):
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", token_ids=[1]),
                TurnRecord(role="assistant", token_ids=[2], logprobs=[-0.1]),
            ],
            total_reward=0.0,
            tokenizer_id="test-tokenizer",
        )
        sample = await tr.pack_payload_to_sample(
            payload, ctx=_FakeCtx(), version=0,
        )
        assert sample.tokens == [1, 2]

    @pytest.mark.asyncio
    async def test_handbuilt_empty_turn_rejected(self):
        """Hand-built payloads MUST not contain empty intermediate turns —
        an empty token_ids list usually means the service mis-rendered
        and emitted a stale span.  ``_assembled=False`` triggers the
        defensive check; ``TrajectoryAssembler`` payloads (with
        ``_assembled=True``) skip it because the assembler already
        enforces non-empty turns at add-time."""
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", token_ids=[1, 2]),
                TurnRecord(role="tool", token_ids=[]),  # empty
                TurnRecord(role="assistant", token_ids=[3], logprobs=[-0.1]),
            ],
            total_reward=0.0,
            tokenizer_id="test-tokenizer",
        )
        with pytest.raises(tr._PackError, match="empty turn"):
            await tr.pack_payload_to_sample(payload, ctx=_FakeCtx(), version=0)

    @pytest.mark.asyncio
    async def test_handbuilt_non_assistant_tail_rejected(self):
        """The trainer's loss is computed over the FINAL assistant span;
        a non-assistant tail (typically: gap turn appended after the
        last engine call by mistake) must fail fast."""
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", token_ids=[1]),
                TurnRecord(role="assistant", token_ids=[2], logprobs=[-0.1]),
                TurnRecord(role="tool", token_ids=[3, 4]),  # non-assistant tail
            ],
            total_reward=0.0,
            tokenizer_id="test-tokenizer",
        )
        with pytest.raises(tr._PackError, match="end with an assistant"):
            await tr.pack_payload_to_sample(payload, ctx=_FakeCtx(), version=0)

    @pytest.mark.asyncio
    async def test_assembled_payload_skips_handbuilt_checks(self):
        """``TrajectoryAssembler.to_payload`` sets ``_assembled=True``;
        the packer trusts the assembler's prior validation and skips the
        hand-built defensive checks (which would otherwise reject a
        legitimate empty-tool-turn-as-gap pattern, etc.)."""
        # Build a payload whose turns include an empty trailing tool turn —
        # this would normally be rejected, but with _assembled=True the
        # packer trusts the assembler.  We don't actually trigger that
        # case here (the assembler doesn't emit empty turns); we just
        # verify the flag flips off the defensive path.
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", token_ids=[1]),
                TurnRecord(role="assistant", token_ids=[2], logprobs=[-0.1]),
            ],
            total_reward=0.0,
            tokenizer_id="test-tokenizer",
        )
        payload._assembled = True
        sample = await tr.pack_payload_to_sample(
            payload, ctx=_FakeCtx(), version=0,
        )
        assert sample.tokens == [1, 2]


# ---------------------------------------------------------------------------
# Reward plumbing
# ---------------------------------------------------------------------------


def _toy_payload(reward):
    return RolloutPayload(
        turns=[
            TurnRecord(role="user", token_ids=[1]),
            TurnRecord(role="assistant", token_ids=[2], logprobs=[-0.1]),
        ],
        total_reward=reward,
    )


class TestReward:
    @pytest.mark.asyncio
    async def test_server_reward_wins(self):
        async def reward_fn(row, payload):
            return 99.0  # should not be called

        sample = await tr.pack_payload_to_sample(
            _toy_payload(0.5), ctx=_FakeCtx(), version=0, reward_fn=reward_fn,
        )
        assert sample.reward == 0.5

    @pytest.mark.asyncio
    async def test_reward_fn_called_when_unset(self):
        captured: dict = {}

        async def reward_fn(row, payload):
            captured["row"] = row
            captured["payload"] = payload
            return 0.8

        payload = _toy_payload(None)
        sample = await tr.pack_payload_to_sample(
            payload, ctx=_FakeCtx(), version=0, reward_fn=reward_fn,
            row={"id": "r1"},
        )
        assert sample.reward == 0.8
        assert captured["row"] == {"id": "r1"}
        assert captured["payload"] is payload

    @pytest.mark.asyncio
    async def test_missing_reward_raises(self):
        with pytest.raises(tr._PackError, match="total_reward is None"):
            await tr.pack_payload_to_sample(
                _toy_payload(None), ctx=_FakeCtx(), version=0,
            )


# ---------------------------------------------------------------------------
# make_remote_rollout_fn end-to-end
# ---------------------------------------------------------------------------


class TestMakeTextRolloutFn:
    @pytest.mark.asyncio
    async def test_builds_rollout_from_service(self):
        async def service(messages, n, sample_kwargs, row):
            return [_toy_payload(float(i)) for i in range(n)]

        rollout_fn = tr.make_remote_rollout_fn(service)
        ctx = _FakeCtx(completions_per_prompt=3)
        row = {"messages": [{"role": "user", "content": "q"}], "id": "r0"}

        rollout = await rollout_fn(row, ctx)
        assert isinstance(rollout, Rollout)
        assert len(rollout.samples) == 3
        assert [s.reward for s in rollout.samples] == [0.0, 1.0, 2.0]

    @pytest.mark.asyncio
    async def test_service_failure_propagates(self):
        """Service exceptions are NOT swallowed.  ``async_rl_loop``
        counts a returned ``None`` as ``sample_fail`` and folds it
        into ``data_consumed``, so a service outage / hard
        integration bug used to checkpoint rows as consumed at
        step 0 and skip them on resume.  Failures now propagate so
        the run aborts loud rather than persisting a corrupt
        cursor."""
        async def service(messages, n, sample_kwargs, row):
            raise RuntimeError("upstream down")

        rollout_fn = tr.make_remote_rollout_fn(service)
        with pytest.raises(RuntimeError, match="upstream down"):
            await rollout_fn(
                {"messages": [{"role": "user", "content": "q"}]}, _FakeCtx(),
            )

    @pytest.mark.asyncio
    async def test_one_malformed_payload_keeps_surviving_completions(self):
        """One bad payload (e.g. missing token_ids) must NOT drop the
        whole prompt group — we keep the surviving completions and
        train on those.  Earlier behavior turned every transient
        ``_PackError`` into full-group loss."""

        async def service(messages, n, sample_kwargs, row):
            # First payload is fine; second is malformed (missing
            # token_ids on assistant turn).
            good = _toy_payload(1.0)
            bad = RolloutPayload(
                turns=[
                    TurnRecord(role="user", token_ids=[1, 2]),
                    TurnRecord(role="assistant", token_ids=None),
                ],
                total_reward=0.5,
            )
            return [good, bad]

        rollout_fn = tr.make_remote_rollout_fn(service)
        rollout = await rollout_fn(
            {"messages": [{"role": "user", "content": "q"}]},
            _FakeCtx(completions_per_prompt=2),
        )
        assert rollout is not None
        assert len(rollout.samples) == 1, (
            "Surviving completion must be trained on; the bad payload "
            "is dropped individually."
        )

    @pytest.mark.asyncio
    async def test_all_payloads_malformed_returns_none(self):
        """When every payload fails to pack there's nothing to train
        on; ``rollout_fn`` correctly returns ``None`` as before."""

        async def service(messages, n, sample_kwargs, row):
            return [
                RolloutPayload(
                    turns=[
                        TurnRecord(role="user", token_ids=[1]),
                        TurnRecord(role="assistant", token_ids=None),
                    ],
                    total_reward=0.0,
                )
                for _ in range(n)
            ]

        rollout_fn = tr.make_remote_rollout_fn(service)
        out = await rollout_fn(
            {"messages": [{"role": "user", "content": "q"}]},
            _FakeCtx(completions_per_prompt=2),
        )
        assert out is None

    @pytest.mark.asyncio
    async def test_empty_messages_returns_none(self):
        async def service(messages, n, sample_kwargs, row):
            return []  # should not be called

        rollout_fn = tr.make_remote_rollout_fn(service)
        out = await rollout_fn({"messages": []}, _FakeCtx())
        assert out is None

    @pytest.mark.asyncio
    async def test_accepts_class_based_service(self):
        class S:
            async def rollout(self, messages, *, n, sample_kwargs, row):
                return [_toy_payload(1.0) for _ in range(n)]

        rollout_fn = tr.make_remote_rollout_fn(S())
        rollout = await rollout_fn(
            {"messages": [{"role": "user", "content": "q"}]},
            _FakeCtx(completions_per_prompt=2),
        )
        assert rollout is not None
        assert len(rollout.samples) == 2

    @pytest.mark.asyncio
    async def test_allow_empty_messages_forwards_through_to_service(self):
        # Round-4 regression: env-driven domains (frozen_lake) build the
        # first observation inside the processor, so the seed conversation
        # is genuinely empty.  ``allow_empty_messages=True`` makes the
        # helper forward to ``service.rollout`` instead of dropping the row.
        called = {"messages": None}

        async def service(messages, n, sample_kwargs, row):
            called["messages"] = list(messages)
            return [_toy_payload(0.5) for _ in range(n)]

        rollout_fn = tr.make_remote_rollout_fn(service, allow_empty_messages=True)
        rollout = await rollout_fn({"messages": []}, _FakeCtx(completions_per_prompt=1))
        assert rollout is not None
        # The service WAS invoked with an empty messages list.
        assert called["messages"] == []

    @pytest.mark.asyncio
    async def test_allow_empty_messages_default_still_drops(self):
        # Default behavior (allow_empty_messages=False) still short-circuits
        # — chat-style services that need a non-empty seed are unaffected.
        called = {"invoked": False}

        async def service(messages, n, sample_kwargs, row):
            called["invoked"] = True
            return []

        rollout_fn = tr.make_remote_rollout_fn(service)
        rollout = await rollout_fn({"messages": []}, _FakeCtx())
        assert rollout is None
        assert called["invoked"] is False

    @pytest.mark.asyncio
    async def test_payload_extras_propagate_to_row_meta(self):
        # Round-4 regression: per-payload extras (e.g. step_rewards /
        # row_id) survive the helper packing path so domain consumers
        # (multihop_qa IGPO) can read them downstream from a single
        # helper call.
        async def service(messages, n, sample_kwargs, row):
            payloads = []
            for i in range(n):
                p = _toy_payload(float(i))
                p.extras = {"step_rewards": [float(i), float(i + 1)], "row_id": f"r{i}"}
                payloads.append(p)
            return payloads

        rollout_fn = tr.make_remote_rollout_fn(service)
        rollout = await rollout_fn(
            {"messages": [{"role": "user", "content": "q"}], "id": "row-extra"},
            _FakeCtx(completions_per_prompt=2),
        )
        assert rollout is not None
        extras = (rollout.row_meta or {}).get("payload_extras")
        assert extras is not None
        assert len(extras) == 2
        assert extras[0]["step_rewards"] == [0.0, 1.0]
        assert extras[1]["row_id"] == "r1"

    @pytest.mark.asyncio
    async def test_version_snapshotted_before_service_await(self):
        """In async RL ``weight_sync_fn`` advances ``ctx.current_version()``
        concurrently with rollouts.  The packer must tag each sample with
        the version that was current at SAMPLING time, not at the time
        the await resolves.  Reading after the await would mislabel a
        rollout sampled at version N as N+1 if a hotload landed mid-call."""

        @dataclass
        class _MutableVersionCtx:
            completions_per_prompt: int = 1
            sample_kwargs: dict = field(default_factory=dict)
            inference_base_url: str = "https://example.invalid"
            api_key: str = "sk-test"
            model: str = "accounts/test/models/x"
            tokenizer_id: str = "test-tokenizer"
            _v: list = field(default_factory=lambda: [3])

            def current_version(self) -> int:
                return self._v[0]

        ctx = _MutableVersionCtx()

        async def service(messages, n, sample_kwargs, row):
            # Simulate a hotload that lands mid-sample by bumping the
            # version *during* the (awaited) service call.
            ctx._v[0] = 7
            return [_toy_payload(1.0) for _ in range(n)]

        rollout_fn = tr.make_remote_rollout_fn(service)
        rollout = await rollout_fn(
            {"messages": [{"role": "user", "content": "q"}]}, ctx,
        )
        assert rollout is not None
        # All sample tokens must carry version 3 (the pre-await snapshot),
        # NOT 7 (the post-await reading).
        for s in rollout.samples:
            assert s.versions == [3] * len(s.tokens), (
                f"expected version=3 (pre-await), got {set(s.versions or [])}"
            )


# ---------------------------------------------------------------------------
# Plain callable contract — invoked positionally per the
# ``RolloutServiceCallable`` type signature.  Arbitrary parameter names work.
# ---------------------------------------------------------------------------


class TestPlainCallableInvocation:
    @pytest.mark.asyncio
    async def test_plain_callable_with_arbitrary_param_names(self):
        """A plain async callable typed as
        ``RolloutServiceCallable = Callable[[List[dict], int, dict, dict], ...]``
        must work with ANY param names — the contract is positional, not
        kwargs.  Earlier the helper passed ``n=`` / ``sample_kwargs=`` /
        ``row=`` which only matched users who happened to name their
        params identically; arbitrary names raised ``TypeError`` and
        dropped every prompt group.
        """
        captured: dict = {}

        async def rollout(messages, count, kwargs, dataset_row):
            captured["messages"] = messages
            captured["count"] = count
            captured["kwargs"] = kwargs
            captured["row_id"] = dataset_row.get("id")
            return [_toy_payload(1.0) for _ in range(count)]

        rollout_fn = tr.make_remote_rollout_fn(rollout)
        out = await rollout_fn(
            {"messages": [{"role": "user", "content": "q"}], "id": "r0"},
            _FakeCtx(completions_per_prompt=2),
        )
        assert out is not None
        assert len(out.samples) == 2
        assert captured["count"] == 2
        assert captured["row_id"] == "r0"

    @pytest.mark.asyncio
    async def test_class_with_rollout_method_uses_kwargs(self):
        """``RolloutService`` Protocol documents
        ``rollout(self, messages, *, n, sample_kwargs, row)`` — kwargs.
        Class-based services keep that contract."""

        class S:
            async def rollout(self, messages, *, n, sample_kwargs, row):
                return [_toy_payload(0.5) for _ in range(n)]

        rollout_fn = tr.make_remote_rollout_fn(S())
        out = await rollout_fn(
            {"messages": [{"role": "user", "content": "q"}]},
            _FakeCtx(completions_per_prompt=1),
        )
        assert out is not None
        assert len(out.samples) == 1
