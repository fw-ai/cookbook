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
    versions=None,
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
        versions=list(versions) if versions is not None else None,
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

    def test_rejects_versions_length_mismatch(self):
        s = _sample(versions=[1, 2])  # tokens is length 5
        with pytest.raises(ValueError, match="versions length"):
            rollout_to_prompt_group(Rollout(samples=[s]))


class TestSingleTurnPacking:
    def test_one_sample_builds_one_datum(self):
        r = Rollout(samples=[_sample(reward=1.0)])
        pg = rollout_to_prompt_group(r)
        assert pg is not None
        assert len(pg.data) == 1
        assert pg.rewards == [1.0]
        # target lengths are tokens[1:] so 4 entries for tokens of length 5
        td = pg.data[0].loss_fn_inputs["target_tokens"]
        assert td.shape == [4]
        mask_td = pg.data[0].loss_fn_inputs["loss_mask"]
        assert mask_td.shape == [4]

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

    def test_with_reference_mirrors_policy_datums(self):
        r = Rollout(samples=[_sample()])
        pg = rollout_to_prompt_group(r, with_reference=True)
        assert pg is not None
        assert len(pg.ref_data) == 1

    def test_completion_lens_counts_mask_ones(self):
        r = Rollout(samples=[
            _sample(tokens=[1, 2, 3, 4, 5], loss_mask=[0, 0, 1, 1, 1]),  # 3 mask ones
        ])
        pg = rollout_to_prompt_group(r)
        assert pg is not None
        assert pg.completion_lens == [3]

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
        s = _sample(
            tokens=[10, 11, 20, 21, 30, 31, 40, 41],
            loss_mask=[0, 0, 1, 1, 0, 0, 1, 1],
            logprobs=[0.0, 0.0, -0.1, -0.2, 0.0, 0.0, -0.3, -0.4],
            reward=1.0,
        )
        pg = rollout_to_prompt_group(Rollout(samples=[s]))
        assert pg is not None
        mask_td = pg.data[0].loss_fn_inputs["loss_mask"]
        # Target is tokens[1:] = 7 entries; mask is loss_mask[1:] = 7 entries.
        assert list(mask_td.data) == [0, 1, 1, 0, 0, 1, 1]
        # Completion-len counts the original mask ones, not the shifted one.
        assert pg.completion_lens == [4]


class TestInferenceLogprobs:
    def test_inf_logprobs_shifted_by_one(self):
        """inf_logprobs on PromptGroup is logprobs[1:] (target-aligned)."""
        s = _sample(
            tokens=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[0.0, 0.0, -0.1, -0.2, -0.3],
        )
        pg = rollout_to_prompt_group(Rollout(samples=[s]))
        assert pg is not None
        assert pg.inf_logprobs == [[0.0, -0.1, -0.2, -0.3]]


class TestRowMeta:
    def test_row_meta_copied_when_present(self):
        r = Rollout(
            samples=[_sample()],
            row_meta={"ground_truth": "42"},
        )
        pg = rollout_to_prompt_group(r)
        assert pg is not None
        assert pg.row_meta == {"ground_truth": "42"}
        # Defensive copy.
        assert pg.row_meta is not r.row_meta

    def test_row_meta_none_stays_none(self):
        pg = rollout_to_prompt_group(Rollout(samples=[_sample()]))
        assert pg is not None
        assert pg.row_meta is None
