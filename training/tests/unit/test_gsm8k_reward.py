"""Unit tests for the GSM8K multi-turn example's reward function."""

from __future__ import annotations

from training.examples.rl.multi_turn_message_in.reward import gsm8k_reward


GT_FROM_GSM8K = (
    "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
    "She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n"
    "#### 18"
)


class TestCorrectAnswers:
    def test_boxed_integer_matches_gt(self):
        assert gsm8k_reward("The answer is \\boxed{18}.", GT_FROM_GSM8K) == 1.0

    def test_boxed_with_spaces(self):
        assert gsm8k_reward("Therefore \\boxed{ 18 }", GT_FROM_GSM8K) == 1.0

    def test_uses_last_boxed_when_multiple(self):
        # First guess wrong, second guess right — last boxed wins.
        completion = "First I thought \\boxed{16}, but actually \\boxed{18}."
        assert gsm8k_reward(completion, GT_FROM_GSM8K) == 1.0

    def test_comma_separated_thousands(self):
        gt = "#### 1,200"
        assert gsm8k_reward("\\boxed{1200}", gt) == 1.0
        assert gsm8k_reward("\\boxed{1,200}", gt) == 1.0


class TestWrongAnswers:
    def test_wrong_number(self):
        assert gsm8k_reward("\\boxed{17}", GT_FROM_GSM8K) == 0.0

    def test_no_boxed(self):
        assert gsm8k_reward("The answer is 18.", GT_FROM_GSM8K) == 0.0

    def test_unmatched_braces(self):
        assert gsm8k_reward("\\boxed{18", GT_FROM_GSM8K) == 0.0

    def test_empty_completion(self):
        assert gsm8k_reward("", GT_FROM_GSM8K) == 0.0


class TestEdgeCases:
    def test_gt_with_negative_number(self):
        assert gsm8k_reward("\\boxed{-5}", "Some reasoning.\n#### -5") == 1.0

    def test_nested_braces_preserved(self):
        # Should still extract the (incorrect-here) content, then fail to match.
        assert gsm8k_reward("\\boxed{\\frac{1}{2}}", GT_FROM_GSM8K) == 0.0

    def test_missing_gt_marker(self):
        # No '#### N' anchor → ground-truth extraction yields None;
        # math_verify fallback will try the raw answer string.
        assert gsm8k_reward("\\boxed{18}", "no answer marker here") == 0.0

    def test_close_but_not_equal_floats_reject(self):
        gt = "#### 18"
        assert gsm8k_reward("\\boxed{18.0001}", gt) == 0.0
