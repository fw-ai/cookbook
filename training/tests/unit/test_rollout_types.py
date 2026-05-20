"""Unit tests for training.utils.rl.rollout."""

from __future__ import annotations

import pytest

from training.utils.rl.rollout import Rollout, RolloutSample, rollout_to_prompt_group


def _sample(
    *,
    tokens=(1, 2, 3, 4, 5),
    loss_mask=(0, 0, 1, 1, 1),
    logprobs=None,
    reward=0.0,
    finish_reason="stop",
    text="x",
):
    if logprobs is None:
        logprobs = [0.0 if m == 0 else -0.1 for m in loss_mask]
    return RolloutSample(
        tokens=list(tokens),
        logprobs=list(logprobs),
        loss_mask=list(loss_mask),
        reward=float(reward),
        finish_reason=finish_reason,
        text=text,
    )


class TestValidation:
    def test_rejects_empty_samples(self):
        with pytest.raises(ValueError, match="empty"):
            rollout_to_prompt_group(Rollout(samples=[]))

    def test_rejects_length_mismatch(self):
        s = RolloutSample(
            tokens=[1, 2, 3],
            logprobs=[0.0, -0.1],  # wrong length
            loss_mask=[0, 1, 1],
            reward=1.0,
        )
        with pytest.raises(ValueError, match="length mismatch"):
            rollout_to_prompt_group(Rollout(samples=[s]))

    def test_rejects_tokens_too_short(self):
        s = RolloutSample(
            tokens=[1],
            logprobs=[0.0],
            loss_mask=[1],
            reward=0.0,
        )
        with pytest.raises(ValueError, match="length >= 2"):
            rollout_to_prompt_group(Rollout(samples=[s]))

    def test_rejects_all_zero_loss_mask(self):
        s = _sample(tokens=[1, 2, 3], loss_mask=[0, 0, 0], logprobs=[0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="all zeros"):
            rollout_to_prompt_group(Rollout(samples=[s]))


class TestSingleTurnPacking:
    def test_two_samples_build_two_datums(self):
        """Basic packing structure check.  Uses 2 samples since
        ``rollout_to_prompt_group`` now drops singleton groups (the
        default ``compute_advantages`` would produce NaN advantages
        on length-1 inputs and poison the training step)."""
        r = Rollout(samples=[
            _sample(reward=1.0),
            _sample(reward=0.0),
        ])
        pg = rollout_to_prompt_group(r)
        assert pg is not None
        assert len(pg.data) == 2
        assert pg.rewards == [1.0, 0.0]
        # target lengths are tokens[1:] so 4 entries for tokens of length 5
        td = pg.data[0].loss_fn_inputs["target_tokens"]
        assert td.shape == [4]
        mask_td = pg.data[0].loss_fn_inputs["weights"]
        assert mask_td.shape == [4]

    def test_singleton_rollout_with_default_advantages_dropped(self):
        """The default GRPO-style ``compute_advantages`` z-score-normalizes
        by ``torch.std(rewards)``, which is NaN on a length-1 tensor.
        Without a guard the NaN advantage would flow through to the
        loss kernel and poison the entire training step.  The helper
        validates ``advantage_fn`` output and drops the group if any
        advantage is non-finite — preserves the NaN-poison protection
        without pre-rejecting all N=1 groups (REINFORCE-style runs
        below).
        """
        r = Rollout(samples=[_sample(reward=1.0)])
        pg = rollout_to_prompt_group(r)
        assert pg is None

    def test_singleton_rollout_with_custom_advantage_fn_emitted(self):
        """REINFORCE-style async RL is a legitimate single-sample
        objective (``completions_per_prompt=1``); a previous coarse
        ``len(samples) < 2`` precheck silently dropped every such
        rollout and made the recipe make no training progress despite
        advertising REINFORCE support.  When the caller supplies a
        custom ``advantage_fn`` (e.g. ``lambda r: r``) that returns
        finite values on N=1, the rollout MUST be packed into a
        PromptGroup like any other.
        """
        r = Rollout(samples=[_sample(reward=0.7)])
        pg = rollout_to_prompt_group(r, advantage_fn=lambda rewards: list(rewards))
        assert pg is not None
        assert pg.rewards == [0.7]
        # The custom advantage_fn passes the reward through unchanged.
        assert pg.advantages == [0.7]
        # Single-sample group still emits a single Datum at the
        # same packing shape as multi-sample groups.
        assert len(pg.data) == 1

    def test_non_finite_advantage_drops_even_with_custom_fn(self):
        """The post-compute non-finite check must catch NaN/inf
        advantages from ANY ``advantage_fn``, not just the default —
        the protection is on the output, not the sample count.  A
        custom advantage_fn that happens to return NaN on a
        degenerate input still produces a non-finite advantage that
        would poison the loss; drop the group."""
        r = Rollout(samples=[_sample(reward=1.0), _sample(reward=2.0)])
        pg = rollout_to_prompt_group(
            r, advantage_fn=lambda rewards: [float("nan")] * len(rewards),
        )
        assert pg is None

    def test_n_samples_center_advantages(self):
        r = Rollout(samples=[
            _sample(reward=1.0),
            _sample(reward=0.0),
        ])
        pg = rollout_to_prompt_group(r)
        assert pg is not None
        assert pg.rewards == [1.0, 0.0]
        assert sum(pg.advantages) == pytest.approx(0.0, abs=1e-5)

    def test_prompt_len_derived_from_first_mask_one(self):
        """prompt_len field reflects where assistant tokens start in sample 0."""
        r = Rollout(samples=[
            _sample(tokens=[1, 2, 3, 4, 5], loss_mask=[0, 0, 0, 1, 1]),
            _sample(tokens=[1, 2, 3, 4, 5], loss_mask=[0, 0, 0, 1, 1]),
        ])
        pg = rollout_to_prompt_group(r)
        assert pg is not None
        assert pg.prompt_len == 3
        # Per-sample list mirrors prompt_len when all samples agree.
        assert pg.prompt_lens == [3, 3]

    def test_per_sample_prompt_lens_for_heterogeneous_rollout(self):
        """Multi-turn / tool branches produce samples with different
        prompt prefix lengths in the same rollout group.  ``prompt_lens``
        must record each sample's first-assistant-token index so the
        downstream loss slices each sample at its own boundary —
        replicating a single ``prompt_len`` would drop tokens for
        short-prefix samples and leak prompt tokens for long-prefix ones.
        """
        r = Rollout(samples=[
            # Sample A: 2-token prefix, then 3 assistant tokens.
            _sample(tokens=[1, 2, 3, 4, 5], loss_mask=[0, 0, 1, 1, 1]),
            # Sample B (different branch): 4-token prefix, 1 assistant token.
            _sample(tokens=[1, 2, 9, 9, 7], loss_mask=[0, 0, 0, 0, 1]),
        ])
        pg = rollout_to_prompt_group(r)
        assert pg is not None
        assert pg.prompt_lens == [2, 4]
        # Back-compat scalar reflects sample 0 only (callers that already
        # consume ``prompt_lens`` see the per-sample truth).
        assert pg.prompt_len == 2

    def test_combine_prompt_groups_uses_per_sample_prompt_lens(self):
        """``combine_prompt_groups`` must prefer per-sample ``prompt_lens``
        over the scalar ``prompt_len`` so heterogeneous rollouts flow
        through to the loss with correct boundaries.
        """
        from training.utils.rl.losses import combine_prompt_groups

        r = Rollout(samples=[
            _sample(tokens=[1, 2, 3, 4, 5], loss_mask=[0, 0, 1, 1, 1]),
            _sample(tokens=[1, 2, 9, 9, 7], loss_mask=[0, 0, 0, 0, 1]),
        ])
        pg = rollout_to_prompt_group(r)
        _, _, _, prompt_lens, _ = combine_prompt_groups([pg])
        assert prompt_lens == [2, 4], (
            "combine_prompt_groups must extend with per-sample prompt_lens "
            f"when set; got {prompt_lens}"
        )

    def test_with_reference_mirrors_policy_datums(self):
        r = Rollout(samples=[_sample(), _sample()])
        pg = rollout_to_prompt_group(r, with_reference=True)
        assert pg is not None
        assert len(pg.ref_data) == 2

    def test_completion_lens_counts_mask_ones(self):
        r = Rollout(samples=[
            _sample(tokens=[1, 2, 3, 4, 5], loss_mask=[0, 0, 1, 1, 1]),  # 3 mask ones
            _sample(tokens=[1, 2, 3, 4, 5], loss_mask=[0, 0, 1, 1, 1]),
        ])
        pg = rollout_to_prompt_group(r)
        assert pg is not None
        assert pg.completion_lens == [3, 3]

    def test_truncated_from_finish_reason(self):
        r = Rollout(samples=[
            _sample(finish_reason="length"),
            _sample(finish_reason="stop"),
        ])
        pg = rollout_to_prompt_group(r)
        assert pg is not None
        assert pg.truncated == [True, False]


class TestMultiTurnPacking:
    def test_interleaved_mask_preserved_in_datum(self):
        """Tool-use shape: assistant turn, then tool result (mask 0), then
        assistant turn again.  The loss_mask must be preserved verbatim
        (shifted by one for the target-alignment)."""
        # tokens: [prompt, prompt, asst1, asst1, tool, tool, asst2, asst2]
        # mask:   [0,      0,      1,     1,     0,    0,    1,     1    ]
        def _interleaved_sample(reward: float):
            return _sample(
                tokens=[10, 11, 20, 21, 30, 31, 40, 41],
                loss_mask=[0, 0, 1, 1, 0, 0, 1, 1],
                logprobs=[0.0, 0.0, -0.1, -0.2, 0.0, 0.0, -0.3, -0.4],
                reward=reward,
            )
        # Two samples since singleton groups now drop (NaN-advantage guard).
        pg = rollout_to_prompt_group(Rollout(samples=[
            _interleaved_sample(reward=1.0),
            _interleaved_sample(reward=0.0),
        ]))
        assert pg is not None
        mask_td = pg.data[0].loss_fn_inputs["weights"]
        # Target is tokens[1:] = 7 entries; mask is loss_mask[1:] = 7 entries.
        assert list(mask_td.data) == [0, 1, 1, 0, 0, 1, 1]
        # Completion-len counts the original mask ones, not the shifted one.
        assert pg.completion_lens == [4, 4]


class TestInferenceLogprobs:
    def test_inf_logprobs_shifted_by_one(self):
        """inf_logprobs on PromptGroup is logprobs[1:] (target-aligned)."""
        def mk(reward: float):
            return _sample(
                tokens=[1, 2, 3, 4, 5],
                loss_mask=[0, 0, 1, 1, 1],
                logprobs=[0.0, 0.0, -0.1, -0.2, -0.3],
                reward=reward,
            )
        # Two samples — singleton groups now drop (NaN-advantage guard).
        pg = rollout_to_prompt_group(Rollout(samples=[mk(1.0), mk(0.0)]))
        assert pg is not None
        assert pg.inf_logprobs == [
            [0.0, -0.1, -0.2, -0.3],
            [0.0, -0.1, -0.2, -0.3],
        ]


class TestRowMeta:
    def test_row_meta_copied_when_present(self):
        r = Rollout(
            samples=[_sample(), _sample()],
            row_meta={"ground_truth": "42"},
        )
        pg = rollout_to_prompt_group(r)
        assert pg is not None
        assert pg.row_meta == {"ground_truth": "42"}
        # Defensive copy.
        assert pg.row_meta is not r.row_meta

    def test_row_meta_none_stays_none(self):
        pg = rollout_to_prompt_group(
            Rollout(samples=[_sample(), _sample()])
        )
        assert pg is not None
        assert pg.row_meta is None
