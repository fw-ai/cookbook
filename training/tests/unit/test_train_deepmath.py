from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import tinker
from fireworks.training.sdk.sampling import SampledCompletion

from training.examples.rl.deepmath import math_verify_service
from training.examples.rl.deepmath import train_deepmath as module


def _argv(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["train_deepmath.py", *args])


def test_import_has_no_environment_side_effects(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    import importlib

    importlib.reload(module)
    assert callable(module.deepmath_reward)


def test_parse_args_parses_core_fields(monkeypatch):
    _argv(
        monkeypatch,
        "--base-model",
        "accounts/test/models/qwen3-4b",
        "--tokenizer-model",
        "Qwen/Qwen3-4B",
        "--dataset-path",
        "/tmp/deepmath.jsonl",
        "--training-shape",
        "ts-qwen3-4b-smoke-v1",
        "--policy-job-id",
        "job-1",
        "--deployment-id",
        "dep-1",
        "--skip-cleanup",
        "--output-model-id",
        "out-model",
    )

    args = module.parse_args()

    assert args.base_model == "accounts/test/models/qwen3-4b"
    assert args.tokenizer_model == "Qwen/Qwen3-4B"
    assert args.dataset_path == "/tmp/deepmath.jsonl"
    assert args.training_shape == "ts-qwen3-4b-smoke-v1"
    assert args.policy_job_id == "job-1"
    assert args.deployment_id == "dep-1"
    assert args.skip_cleanup is True
    assert args.max_head_offpolicy_versions == 0


def test_parse_args_rejects_removed_deployment_extra_values(monkeypatch):
    _argv(monkeypatch, "--deployment-extra-values", "testValue=test-value")

    with pytest.raises(SystemExit):
        module.parse_args()


def test_extract_boxed_handles_nested_braces():
    assert module.extract_boxed(r"work \boxed{\frac{1}{x+1}} tail") == r"\frac{1}{x+1}"
    assert module.extract_boxed(r"\boxed{\left(\frac{1}{2}\right)^{3}}") == (
        r"\left(\frac{1}{2}\right)^{3}"
    )
    assert module.extract_boxed(r"First \boxed{wrong} then \boxed{42}") == "42"
    assert module.extract_boxed("no boxed answer") is None


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        (r"Solution \boxed{42}", "42"),
        ("<answer> 17 </answer>", "17"),
        ("**Answer:** x^2 + 1", "x^2 + 1"),
        (r"\boxed{42} and <answer>99</answer>", "42"),
    ],
)
def test_extract_answer_from_completion_supports_multiple_formats(
    completion, expected
):
    assert module.extract_answer_from_completion(completion) == expected


def test_deepmath_reward_supports_exact_and_numeric_fallback(monkeypatch):
    # Disable the symbolic (out-of-process) path to exercise only the
    # string-exact and numeric fallbacks.
    monkeypatch.setattr(module, "math_verify", lambda _pred, _gt: False)

    assert module.deepmath_reward(r"Result \boxed{2}", {"ground_truth": "2"}) == 1.0
    assert module.deepmath_reward(r"Result \boxed{2.0}", {"ground_truth": "2"}) == 1.0
    assert (
        module.deepmath_reward(r"\boxed{\frac{1}{2}}", {"ground_truth": r"\dfrac{1}{2}"})
        == 1.0
    )
    assert module.deepmath_reward(r"\boxed{43}", {"ground_truth": "42"}) == 0.0
    assert module.deepmath_reward("No answer here", {"ground_truth": "2"}) == 0.0


def test_deepmath_reward_uses_symbolic_verifier():
    pytest.importorskip("math_verify")

    assert module.deepmath_reward(r"\boxed{0.5}", {"ground_truth": r"\dfrac{1}{2}"}) == 1.0
    assert module.deepmath_reward(r"\boxed{\sqrt{2}}", {"ground_truth": r"\sqrt{2}"}) == 1.0
    assert module.math_verify(r"\boxed{42}", r"\boxed{43}") is False


def test_math_verify_service_works_from_worker_threads():
    pytest.importorskip("math_verify")
    results: list[bool] = []
    thread = threading.Thread(
        target=lambda: results.append(
            math_verify_service.math_verify(r"\boxed{7}", r"\boxed{7}")
        )
    )
    thread.start()
    thread.join(30.0)
    assert not thread.is_alive()
    assert results == [True]


def test_missing_math_verify_fails_instead_of_scoring_zero(monkeypatch):
    real_find_spec = math_verify_service.importlib.util.find_spec
    monkeypatch.setattr(
        math_verify_service.importlib.util,
        "find_spec",
        lambda name, *a: None if name == "math_verify" else real_find_spec(name, *a),
    )
    math_verify_service._require_math_verify.cache_clear()
    try:
        with pytest.raises(ImportError, match="math-verify"):
            math_verify_service.MathVerifyService().verify(r"\boxed{7}", r"\boxed{7}")
    finally:
        math_verify_service._require_math_verify.cache_clear()


def test_math_verify_service_kills_a_wedged_worker(tmp_path):
    slow_worker = tmp_path / "slow_worker.py"
    slow_worker.write_text(
        "import sys, time\nfor _line in sys.stdin:\n    time.sleep(3600)\n"
    )
    service = math_verify_service.MathVerifyService(worker_path=str(slow_worker))

    started = time.monotonic()
    assert service.verify(r"\boxed{1}", r"\boxed{2}", timeout=0.5) is False
    assert time.monotonic() - started < 5.0
    assert service._proc is None


def test_rollout_grades_parsed_completion_with_deepmath_reward(monkeypatch):
    prompt_tokens, completion_tokens = [10, 20, 30], [40, 50]
    captured: dict[str, object] = {}

    class Renderer:
        def build_generation_prompt(self, messages, **_kwargs):
            captured["messages"] = messages
            return tinker.ModelInput(
                chunks=[tinker.types.EncodedTextChunk(tokens=prompt_tokens)]
            )

        def parse_response(self, tokens):
            return {"role": "assistant", "content": r"so \boxed{4}"}, True

        def get_stop_sequences(self):
            return []

    class Sampler:
        async def sample_with_prompt_tokens(self, prompt_token_ids, **kwargs):
            captured["kwargs"] = kwargs
            return [
                SampledCompletion(
                    text="gen",
                    full_tokens=prompt_token_ids + completion_tokens,
                    prompt_len=len(prompt_token_ids),
                    finish_reason="stop",
                    sampling_logprobs=[-0.1, -0.2],
                )
            ]

    renderer_names = []
    monkeypatch.setattr(
        module, "build_renderer", lambda *args, **_k: renderer_names.append(args[2]) or Renderer()
    )
    monkeypatch.setattr(module, "math_verify", lambda _pred, _gt: False)
    setup = SimpleNamespace(
        tokenizer=object(),
        tokenizer_id="tokenizer",
        sample_kwargs={"max_tokens": 8, "temperature": 1.0, "logprobs": True},
        sampler=Sampler(),
    )
    rollout_fn = module.make_rollout_fn(setup, renderer_name="qwen3")
    row = {
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "stale tail"},
        ],
        "ground_truth": "4",
    }

    run = asyncio.run(rollout_fn(row))

    assert run is not None
    assert renderer_names == ["qwen3"]
    assert captured["messages"] == [{"role": "user", "content": "2+2?"}]
    assert captured["kwargs"]["max_tokens"] == 8
    segment = run.segments[0]
    assert segment.reward == 1.0
    assert segment.tokens == prompt_tokens + completion_tokens
    assert segment.loss_mask == [0, 0, 0, 1, 1]


def test_reward_variance_filter_drops_uniform_groups():
    assert module.has_reward_variance(SimpleNamespace(rewards=[0.0, 1.0]))
    assert not module.has_reward_variance(SimpleNamespace(rewards=[1.0, 1.0]))
