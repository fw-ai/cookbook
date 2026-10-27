from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

from training.examples.rl.code_repair.rewards import code_repair_reward, extract_code
from training.examples.rl.code_repair.sandbox import SANDBOX_EXEC, grade, rewrite_tests

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not os.path.exists(SANDBOX_EXEC), reason="sandbox-exec is macOS only"
)

TEST = """
def assertion(out, exp, atol):
    assert out == exp, f"out: {out}, exp: {exp}"

inputs = [[1], [2], [3], [4]]
results = [2, 4, 6, 8]
for i, (inp, exp) in enumerate(zip(inputs, results)):
    assertion(double(*inp), exp, 0)
"""
HE_TEST = """
def check(candidate):
    inputs = [[1], [2]]
    results = [2, 4]
    for i, (inp, exp) in enumerate(zip(inputs, results)):
        assert candidate(*inp) == exp
"""
FIXED = "def double(x):\n    return 2 * x\n"
BUGGY = "def double(x):\n    return 2 * x if x < 3 else x\n"
TRUTH = {"entry_point": "double", "test": TEST, "num_cases": 4, "buggy_pass_rate": 0.5}


def run(coro):
    return asyncio.run(coro)


def test_counts_passing_cases():
    assert run(grade(FIXED, TEST, "double")).passed == 4
    buggy = run(grade(BUGGY, TEST, "double"))
    assert (buggy.status, buggy.passed, buggy.seen) == ("ok", 2, 4)
    assert run(grade(FIXED, HE_TEST, "double")).passed == 2


def test_rewrite_requires_single_loop():
    with pytest.raises(ValueError):
        rewrite_tests("assert double(1) == 2\n", "double", "t")


@pytest.mark.parametrize(
    "code,status",
    [
        ("def double(x):\n    while True: pass\n", "timeout"),
        ("import os\nos._exit(0)\n", "crashed"),
        ("def double(x) return x\n", "syntax_error"),
        ("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=2)\n" + FIXED, "candidate_error"),
        (f"open({os.path.expanduser('~/code_repair_probe.txt')!r}, 'w').write('x')\n" + FIXED, "candidate_error"),
        (f"open({os.path.expanduser('~/.zshrc')!r}).read()\n" + FIXED, "candidate_error"),
        ("import subprocess\nsubprocess.run(['/bin/echo'])\n" + FIXED, "candidate_error"),
    ],
)
def test_sandbox_blocks_escapes(code, status):
    result = run(grade(code, TEST, "double", timeout=6))
    assert (result.status, result.passed) == (status, 0)
    assert not os.path.exists(os.path.expanduser("~/code_repair_probe.txt"))


def test_cheats_pass_no_cases():
    fake_eq = "class A:\n    def __eq__(self, other): return True\ndef double(x):\n    return A()\n"
    exits = "import sys\ndef double(x):\n    sys.exit(0)\n"
    for code in (fake_eq, exits):
        assert run(grade(code, TEST, "double")).passed == 0


def test_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "secret")
    code = "import os\nassert 'FIREWORKS_API_KEY' not in os.environ\n" + FIXED
    assert run(grade(code, TEST, "double")).passed == 4


def test_extract_code_prefers_block_defining_entry_point():
    reply = "<think>try ```python\ndef helper(): pass\n```</think>Fix:\n```python\n" + FIXED + "```\n```python\nprint(double(2))\n```"
    assert extract_code(reply, "double") == FIXED
    assert extract_code("no code here", "double") is None


def test_reward_is_share_of_cases():
    assert run(code_repair_reward(f"```python\n{FIXED}```", TRUTH)) == 1.0
    assert run(code_repair_reward(f"```python\n{BUGGY}```", TRUTH)) == 0.5
    assert run(code_repair_reward(FIXED, TRUTH)) == 0.0


def test_metrics_and_keep_rule(tmp_path):
    import json

    from training.examples.rl.code_repair.build_dataset import keep
    from training.examples.rl.code_repair.train_code_repair_serverless import repair_metrics

    def sample(reward):
        return SimpleNamespace(segments=[SimpleNamespace(reward=reward, finish_reason="stop", loss_mask=[1] * 5)])

    rows = [{"id": "a", "ground_truth": TRUTH}, {"id": "b", "ground_truth": TRUTH}]
    runs = [[sample(1.0), sample(0.25), sample(0.0), None], [sample(1.0)] * 4]
    metrics = repair_metrics(rows, runs, rows_out=tmp_path / "rows.jsonl")
    assert metrics["eval/fully_fixed"] == 5 / 8
    assert metrics["eval/regressed"] == 2 / 8
    assert metrics["eval/zero_reward"] == 1 / 8
    sampled = [json.loads(line) for line in (tmp_path / "rows.jsonl").open()]
    assert [keep(s) for s in sampled] == [True, False]
