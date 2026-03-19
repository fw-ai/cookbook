from __future__ import annotations

import importlib
import sys

import pytest


def _load_module(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    import training.examples.deepmath_rl.train_deepmath as module

    return importlib.reload(module)


def test_parse_args_parses_extra_values(monkeypatch):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_deepmath.py",
            "--base-model",
            "accounts/test/models/qwen3-4b",
            "--tokenizer-model",
            "Qwen/Qwen3-4B",
            "--dataset-path",
            "/tmp/deepmath.jsonl",
            "--training-shape",
            "ts-qwen3-4b-smoke-v1",
            "--deployment-replica-count",
            "3",
            "--deployment-extra-values",
            "priorityClass=deployment",
            "featureFlag=on",
            "--output-model-id",
            "out-model",
        ],
    )

    args = module.parse_args()

    assert args.base_model == "accounts/test/models/qwen3-4b"
    assert args.tokenizer_model == "Qwen/Qwen3-4B"
    assert args.dataset_path == "/tmp/deepmath.jsonl"
    assert args.training_shape == "ts-qwen3-4b-smoke-v1"
    assert args.deployment_region is None
    assert args.deployment_replica_count == 3
    assert args.deployment_extra_values == {
        "priorityClass": "deployment",
        "featureFlag": "on",
    }


def test_parse_args_rejects_invalid_extra_values(monkeypatch):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_deepmath.py",
            "--deployment-extra-values",
            "priorityClass",
        ],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_extract_boxed_handles_nested_braces(monkeypatch):
    module = _load_module(monkeypatch)

    assert module.extract_boxed(r"work \boxed{\frac{1}{x+1}} tail") == r"\frac{1}{x+1}"
    assert module.extract_boxed("no boxed answer") is None


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        (r"Solution \boxed{42}", "42"),
        ("<answer> 17 </answer>", "17"),
        ("**Answer:** x^2 + 1", "x^2 + 1"),
    ],
)
def test_extract_answer_from_completion_supports_multiple_formats(monkeypatch, completion, expected):
    module = _load_module(monkeypatch)

    assert module.extract_answer_from_completion(completion) == expected


def test_deepmath_reward_supports_exact_and_numeric_fallback(monkeypatch):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "math_parse", lambda _text: None)
    monkeypatch.setattr(module, "math_verify", lambda _lhs, _rhs: False)

    assert module.deepmath_reward(r"Result \boxed{2}", {"ground_truth": "2"}) == 1.0
    assert module.deepmath_reward(r"Result \boxed{2.0}", {"ground_truth": "2"}) == 1.0
    assert module.deepmath_reward("No answer here", {"ground_truth": "2"}) == 0.0


def test_main_passes_replica_count_to_deploy_config(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    dataset_path = tmp_path / "deepmath.jsonl"
    dataset_path.write_text("{}\n")
    captured = {}

    args = module.TrainArgs(
        base_model="accounts/test/models/qwen3-4b",
        tokenizer_model="Qwen/Qwen3-4B",
        dataset_path=str(dataset_path),
        training_shape="ts-qwen3-4b-smoke-v1",
        deployment_replica_count=4,
        output_model_id="out-model",
    )

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "TrainerJobManager", lambda **kwargs: captured.setdefault("trainer_mgr", kwargs))
    monkeypatch.setattr(module, "DeploymentManager", lambda **kwargs: captured.setdefault("deploy_mgr", kwargs))

    def fake_rl_main(config, **kwargs):
        captured["config"] = config
        captured["rl_main_kwargs"] = kwargs
        return {"steps": 0}

    monkeypatch.setattr(module.rl_loop, "main", fake_rl_main)

    module.main()

    assert captured["config"].deployment.replica_count == 4
    assert captured["rl_main_kwargs"]["cleanup_on_exit"] is True


