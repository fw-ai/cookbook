#!/usr/bin/env python3
"""Unit tests for the DeepMath reward function."""

import json
import os
import tempfile
import textwrap
import threading
import time

from train_deepmath import (
    extract_boxed,
    deepmath_reward,
    extract_answer_from_completion,
    math_verify,
    _MathVerifyService,
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


def test_math_verify_correctness():
    """The out-of-process math_verify returns correct symbolic results."""
    assert math_verify(r"\boxed{42}", r"\boxed{42}") is True
    assert math_verify(r"\boxed{42}", r"\boxed{43}") is False
    assert math_verify(r"\boxed{\frac{1}{2}}", r"\boxed{0.5}") is True
    print("  [PASS] test_math_verify_correctness")


def test_math_verify_thread_safe():
    """math_verify must work when called from a non-main thread.

    math_verify's native timeout uses signal.SIGALRM, which only works on the
    main thread; our out-of-process wrapper must not depend on that.
    """
    results = []

    def _worker():
        results.append(math_verify(r"\boxed{7}", r"\boxed{7}"))

    t = threading.Thread(target=_worker)
    t.start()
    t.join(15.0)
    assert not t.is_alive(), "math_verify hung when called from a worker thread"
    assert results == [True], f"expected [True], got {results}"
    print("  [PASS] test_math_verify_thread_safe")


def test_math_verify_killable_timeout():
    """A wedged worker is killed on timeout — fast False, no leaked thread/proc.

    This is the property the in-process (thread-based) approach could not
    provide: a runaway sympy expansion can only be bounded by killing the
    process it runs in.
    """
    # A stand-in worker that hangs forever instead of computing.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent("""
            import sys, time
            for _line in sys.stdin:
                time.sleep(3600)
        """))
        slow_worker = f.name

    try:
        svc = _MathVerifyService(worker_path=slow_worker)
        t0 = time.time()
        result = svc.verify(r"\\boxed{1}", r"\\boxed{2}", timeout=0.5)
        elapsed = time.time() - t0
        assert result is False, f"wedged worker should yield False, got {result}"
        assert elapsed < 5.0, f"timeout not enforced — took {elapsed:.1f}s"
        assert svc._proc is None, "wedged worker process was not killed"
        # Reader threads must not leak: after the kill, none should survive.
        leaked = [
            th for th in threading.enumerate()
            if th.name.startswith("Thread-") and th.is_alive() and th.daemon
        ]
        # Give any just-killed reader a moment to observe EOF and exit.
        time.sleep(0.5)
        still = [th for th in leaked if th.is_alive()]
        assert not still, f"reader thread leaked after worker kill: {still}"
    finally:
        os.unlink(slow_worker)
    print("  [PASS] test_math_verify_killable_timeout")


def main():
    print("Running reward function unit tests...")
    test_extract_boxed()
    test_extract_answer_fallbacks()
    test_reward_exact_match()
    test_reward_numeric()
    test_reward_latex_fractions()
    test_reward_symbolic()
    test_math_verify_correctness()
    test_math_verify_thread_safe()
    test_math_verify_killable_timeout()
    test_reward_with_real_dataset()
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
