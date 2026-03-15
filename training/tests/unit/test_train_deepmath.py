from __future__ import annotations

import importlib
import sys

import pytest


def _load_module(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
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


def test_main_builds_rl_config_and_calls_recipe(monkeypatch):
    module = _load_module(monkeypatch)
    dataset_path = "/tmp/deepmath.jsonl"
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
            dataset_path,
            "--training-shape",
            "ts-qwen3-4b-smoke-v1",
            "--deployment-id",
            "dep-123",
            "--region",
            "US_OHIO_1",
            "--deployment-region",
            "US_VIRGINIA_1",
            "--max-rows",
            "32",
            "--epochs",
            "2",
            "--completions-per-prompt",
            "3",
            "--learning-rate",
            "2e-5",
            "--kl-beta",
            "0.05",
            "--temperature",
            "0.7",
            "--max-completion-tokens",
            "128",
            "--prompt-groups-per-step",
            "4",
            "--trajectory-dir",
            "/tmp/traj",
            "--router-replay",
            "--deployment-extra-values",
            "priorityClass=deployment",
            "--wandb-entity",
            "fw",
            "--wandb-project",
            "deepmath-tests",
            "--output-model-id",
            "out-model",
        ],
    )
    monkeypatch.setattr(module.os.path, "exists", lambda path: path == dataset_path)
    monkeypatch.setattr(module.time, "time", lambda: 123456.0)

    created: dict[str, object] = {}

    class FakeTrainerJobManager:
        def __init__(self, *, api_key, account_id, base_url):
            created["trainer_mgr"] = {
                "api_key": api_key,
                "account_id": account_id,
                "base_url": base_url,
            }

    class FakeDeploymentManager:
        def __init__(self, *, api_key, account_id, base_url, hotload_api_url):
            created["deploy_mgr"] = {
                "api_key": api_key,
                "account_id": account_id,
                "base_url": base_url,
                "hotload_api_url": hotload_api_url,
            }

    def fake_rl_main(config, *, rlor_mgr=None, deploy_mgr=None, cleanup_on_exit=False):
        created["config"] = config
        created["rlor_mgr"] = rlor_mgr
        created["deploy_mgr_obj"] = deploy_mgr
        created["cleanup_on_exit"] = cleanup_on_exit
        return {"reward": 1.0}

    monkeypatch.setattr(module, "TrainerJobManager", FakeTrainerJobManager)
    monkeypatch.setattr(module, "DeploymentManager", FakeDeploymentManager)
    monkeypatch.setattr(module.rl_loop, "main", fake_rl_main)

    module.main()

    cfg = created["config"]
    assert cfg.base_model == "accounts/test/models/qwen3-4b"
    assert cfg.dataset == dataset_path
    assert cfg.learning_rate == 2e-5
    assert cfg.kl_beta == 0.05
    assert cfg.completions_per_prompt == 3
    assert cfg.max_completion_tokens == 128
    assert cfg.temperature == 0.7
    assert cfg.epochs == 2
    assert cfg.max_rows == 32
    assert cfg.prompt_groups_per_step == 4
    assert cfg.trajectory_dir == "/tmp/traj"
    assert cfg.router_replay is True
    assert cfg.router_replay_completion_only is True
    assert cfg.is_correction.tis_cap == 2.0
    assert cfg.infra.training_shape_id == "ts-qwen3-4b-smoke-v1"
    assert cfg.infra.region == "US_OHIO_1"
    assert cfg.deployment.deployment_id == "dep-123"
    assert cfg.deployment.deployment_region == "US_VIRGINIA_1"
    assert cfg.deployment.tokenizer_model == "Qwen/Qwen3-4B"
    assert cfg.deployment.extra_values == {"priorityClass": "deployment"}
    assert cfg.weight_sync.weight_sync_interval == 1
    assert cfg.weight_sync.dcp_save_interval == 20
    assert cfg.wandb.entity == "fw"
    assert cfg.wandb.project == "deepmath-tests"
    assert cfg.wandb.run_name == "dep-123"
    assert created["cleanup_on_exit"] is True
    assert module.rl_loop.reward_fn is module.deepmath_reward
