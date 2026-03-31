#!/usr/bin/env python3
"""Unit tests for the DeepMath reward function."""

import os
import json

from train_deepmath import (
    extract_boxed,
    deepmath_reward,
    extract_answer_from_completion,
)


def test_extract_boxed():
    assert extract_boxed(r"The answer is \boxed{42}") == "42"
    assert extract_boxed(r"\boxed{\dfrac{1}{2}}") == r"\dfrac{1}{2}"
    assert extract_boxed(r"\boxed{\sqrt{2}}") == r"\sqrt{2}"
    assert extract_boxed(r"So \boxed{x^2 + 1}") == "x^2 + 1"
    assert extract_boxed(r"\boxed{2^n}") == "2^n"
    assert extract_boxed(r"\boxed{-\frac{2}{3}}") == r"-\frac{2}{3}"
    assert extract_boxed(r"\boxed{\frac{a}{b+c}}") == r"\frac{a}{b+c}"
    assert extract_boxed(r"\boxed{\left(\frac{1}{2}\right)^{3}}") == r"\left(\frac{1}{2}\right)^{3}"
    assert extract_boxed(r"First \boxed{wrong} then \boxed{42}") == "42"
    assert extract_boxed("No boxed here") is None
    assert extract_boxed(r"\boxed{ 0 }") == "0"
    print("  [PASS] test_extract_boxed")


def test_extract_answer_fallbacks():
    assert extract_answer_from_completion("<answer>42</answer>") == "42"
    assert extract_answer_from_completion("**Answer:** 42") == "42"
    assert extract_answer_from_completion("**ANSWER:** 100\n") == "100"
    assert extract_answer_from_completion(r"\boxed{42} and <answer>99</answer>") == "42"
    print("  [PASS] test_extract_answer_fallbacks")


def test_reward_exact_match():
    row = {"ground_truth": "42"}
    assert deepmath_reward(r"The answer is \boxed{42}.", row) == 1.0
    assert deepmath_reward(r"The answer is \boxed{43}.", row) == 0.0
    assert deepmath_reward("No answer here.", row) == 0.0
    print("  [PASS] test_reward_exact_match")


def test_reward_numeric():
    assert deepmath_reward(r"\boxed{0}", {"ground_truth": "0"}) == 1.0
    assert deepmath_reward(r"\boxed{0.0}", {"ground_truth": "0"}) == 1.0
    assert deepmath_reward(r"\boxed{34}", {"ground_truth": "34"}) == 1.0
    print("  [PASS] test_reward_numeric")


def test_reward_latex_fractions():
    assert deepmath_reward(r"\boxed{-\frac{2}{3}}", {"ground_truth": r"-\dfrac{2}{3}"}) == 1.0
    assert deepmath_reward(r"\boxed{-\dfrac{2}{3}}", {"ground_truth": r"-\dfrac{2}{3}"}) == 1.0
    assert deepmath_reward(r"\boxed{\frac{1}{2}}", {"ground_truth": r"\dfrac{1}{2}"}) == 1.0
    assert deepmath_reward(r"\boxed{0.5}", {"ground_truth": r"\dfrac{1}{2}"}) == 1.0
    print("  [PASS] test_reward_latex_fractions")


def test_reward_symbolic():
    assert deepmath_reward(r"\boxed{\pi}", {"ground_truth": r"\pi"}) == 1.0
    assert deepmath_reward(r"\boxed{\sqrt{2}}", {"ground_truth": r"\sqrt{2}"}) == 1.0
    assert deepmath_reward(r"\boxed{\infty}", {"ground_truth": r"\infty"}) == 1.0
    assert deepmath_reward(r"\boxed{2^n}", {"ground_truth": "2^n"}) == 1.0
    print("  [PASS] test_reward_symbolic")


def test_reward_with_real_dataset():
    dataset_path = os.path.join(os.path.dirname(__file__), "deepmath_103k.jsonl")
    if not os.path.exists(dataset_path):
        print("  [SKIP] test_reward_with_real_dataset (run prepare_data.py first)")
        return

    with open(dataset_path) as f:
        rows = [json.loads(line) for _, line in zip(range(20), f)]

    passed = 0
    for i, row in enumerate(rows):
        gt = row["ground_truth"]
        fake = f"After careful analysis, the answer is \\boxed{{{gt}}}"
        reward = deepmath_reward(fake, row)
        if reward == 1.0:
            passed += 1
        else:
            print(f"    [WARN] Row {i}: gt={gt!r} got reward={reward}")

    print(f"  [PASS] test_reward_with_real_dataset: {passed}/{len(rows)} matched")
    assert passed >= 15, f"Too many failures: only {passed}/{len(rows)} matched"


def main():
    print("Running reward function unit tests...")
    test_extract_boxed()
    test_extract_answer_fallbacks()
    test_reward_exact_match()
    test_reward_numeric()
    test_reward_latex_fractions()
    test_reward_symbolic()
    test_reward_with_real_dataset()
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
