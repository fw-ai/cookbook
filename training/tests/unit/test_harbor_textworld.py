from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.examples.harbor_rl.tasks.textworld import dataset as textworld


def _fake_tw_make(command, *, check):
    assert check is True
    assert command[1] == "tw-cooking"
    output = Path(command[command.index("--output") + 1])
    seed = command[command.index("--seed") + 1]
    output.write_bytes(f"compiled-textworld-game:{seed}".encode())
    output.with_suffix(".json").write_text(
        json.dumps({"metadata": {"walkthrough": ["look"]}}),
        encoding="utf-8",
    )


def _generate(tmp_path, monkeypatch, name="dataset", **kwargs):
    monkeypatch.setattr(textworld.shutil, "which", lambda _: "/usr/bin/tw-make")
    monkeypatch.setattr(textworld.subprocess, "run", _fake_tw_make)
    return textworld.generate_dataset(
        tmp_path / name,
        seed=17,
        train_tasks=3,
        evaluation_tasks=2,
        **kwargs,
    )


def test_generate_textworld_dataset_is_frozen_and_reproducible(tmp_path, monkeypatch):
    first = _generate(tmp_path, monkeypatch, "first")
    second = _generate(tmp_path, monkeypatch, "second")

    assert len(first.train_task_ids) == 3
    assert len(first.evaluation_task_ids) == 2
    assert not set(first.train_task_ids) & set(first.evaluation_task_ids)
    first_manifest = json.loads((first.root / textworld.MANIFEST_NAME).read_text())
    second_manifest = json.loads((second.root / textworld.MANIFEST_NAME).read_text())
    assert first_manifest == second_manifest

    task = first.root / first.directory_by_task_id[first.train_task_ids[0]]
    assert (task / "environment" / "game.ulx").is_file()
    assert not (task / "environment" / "game.json").exists()
    assert (task / "tests" / "game.json").is_file()
    assert "textworld==1.6.2" in (task / "environment" / "Dockerfile").read_text()
    assert "`textworld_action`" in (task / "instruction.md").read_text()
    assert (task / "tests" / "test.sh").stat().st_mode & 0o111
    assert "state.won" in (task / "tests" / "grader.py").read_text()


def test_frozen_textworld_dataset_rejects_content_drift(tmp_path, monkeypatch):
    dataset = _generate(tmp_path, monkeypatch)
    task = dataset.root / dataset.directory_by_task_id[dataset.train_task_ids[0]]
    (task / "instruction.md").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="content differs"):
        textworld.load_frozen_dataset(dataset.root)


def test_textworld_generator_requires_fresh_destination(tmp_path, monkeypatch):
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        textworld.generate_dataset(
            destination,
            seed=1,
            train_tasks=1,
            evaluation_tasks=1,
        )


def test_existing_generator_environment_must_match_pins(tmp_path, monkeypatch):
    environment = tmp_path / "generator"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin" / "python").write_text("", encoding="utf-8")
    (environment / "bin" / "tw-make").write_text("", encoding="utf-8")
    calls = []

    class Result:
        returncode = 0

    monkeypatch.setattr(
        textworld.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Result(),
    )

    assert textworld.ensure_generator_environment(environment) == (
        environment / "bin" / "tw-make"
    )
    assert textworld.TEXTWORLD_VERSION in calls[0][0][0][2]
    assert textworld.GENERATOR_NUMPY_VERSION in calls[0][0][0][2]


def test_incomplete_generator_environment_fails_closed(tmp_path):
    environment = tmp_path / "generator"
    environment.mkdir()

    with pytest.raises(RuntimeError, match="incomplete"):
        textworld.ensure_generator_environment(environment)


def test_generator_cli_defaults_to_calibrated_cooking_suite(tmp_path):
    args = textworld.parse_args(
        ["--output", str(tmp_path / "dataset"), "--seed", "42"]
    )

    assert args.train_tasks == 256
    assert args.evaluation_tasks == 32
    assert (args.recipe, args.take, args.go) == (5, 5, 12)
    assert (args.open_, args.cook, args.cut, args.drop) == (True, True, True, True)
    assert args.split == "test"
    assert args.tw_make is None


def test_cooking_tasks_use_only_the_game_native_goal(tmp_path, monkeypatch):
    dataset = _generate(
        tmp_path,
        monkeypatch,
        settings={"recipe": 3, "take": 3, "go": 6, "cook": True, "cut": True},
    )

    task = dataset.root / dataset.directory_by_task_id[dataset.train_task_ids[0]]
    instruction = (task / "instruction.md").read_text()
    assert "game prints its own goal" in instruction
    assert "cookbook" not in instruction
    assert "ingredient" not in instruction
    config = (task / "task.toml").read_text()
    assert 'challenge = "cooking"' in config
    manifest = json.loads((dataset.root / textworld.MANIFEST_NAME).read_text())
    assert manifest["challenge"] == "cooking"
    assert manifest["settings"]["recipe"] == 3


def test_cooking_argv_varies_the_recipe_per_game() -> None:
    settings = {"recipe": 2, "take": 1, "go": 9, "cook": True, "cut": False}

    first = textworld._cooking_argv(settings, 111)
    second = textworld._cooking_argv(settings, 222)

    assert first[0] == "tw-cooking"
    assert "--cook" in first and "--cut" not in first
    assert first[first.index("--recipe-seed") + 1] == "111"
    assert second[second.index("--recipe-seed") + 1] == "222"


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"recipe": 0, "take": 0, "go": 6}, "between 1 and 5"),
        ({"recipe": 6, "take": 6, "go": 6}, "between 1 and 5"),
        ({"recipe": 2, "take": 3, "go": 6}, "take must be between"),
        ({"recipe": 2, "take": 1, "go": 5}, "go must be one of"),
    ],
)
def test_cooking_settings_fail_closed(tmp_path, settings, message):
    with pytest.raises(ValueError, match=message):
        textworld.generate_dataset(
            tmp_path / "cooking",
            seed=1,
            train_tasks=1,
            evaluation_tasks=1,
            settings=settings,
        )


def test_textworld_pi_recipe_uses_e2b_and_managed_server_grpo(tmp_path):
    from training.examples.harbor_rl.tasks.textworld import train as train_textworld

    args = train_textworld.parse_args(
        [
            "--base-model",
            "accounts/example/models/policy",
            "--tokenizer-model",
            "example/tokenizer",
            "--renderer-name",
            "example-renderer",
            "--textworld-dataset",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--shuffle-seed",
            "19",
        ]
    )

    config = train_textworld.build_config(
        args,
        run_dir=tmp_path / "run",
        row_count=256,
    )
    factory = train_textworld.rollout_fn_factory(args, tmp_path / "run")
    harbor = factory.keywords["harbor"]

    assert config.completions_per_prompt == 8
    assert config.prompt_groups_per_step == 8
    assert config.epochs == 1
    assert config.max_head_offpolicy_versions == 2
    assert config.eps_clip == 0.2
    assert config.grad_clip_norm == 0.0
    assert config.cleanup_on_exit is True
    assert config.max_rows == 256
    assert harbor.environment == "e2b"
    assert harbor.terminal_failure_reward == 0.0
    assert "AgentSetupTimeoutError" in harbor.retry_include_exceptions
    assert factory.keywords["tool_profile"] == "textworld"
    assert harbor.tool_timeout_seconds == 120
    assert harbor.tito.debug_enabled is False
    assert harbor.tito.debug_redact_text is True


def test_textworld_full_sync_shape_and_batch_contract(tmp_path):
    from training.examples.harbor_rl.tasks.textworld import train as train_textworld

    shape = "accounts/example/trainingShapes/textworld"
    args = train_textworld.parse_args(
        [
            "--base-model",
            "accounts/example/models/policy",
            "--tokenizer-model",
            "Qwen/Qwen3.8-27B",
            "--renderer-name",
            "qwen3_8",
            "--textworld-dataset",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--shuffle-seed",
            "19",
            "--training-shape-id",
            shape,
            "--completions-per-prompt",
            "8",
            "--prompt-groups-per-step",
            "16",
            "--max-rows",
            "96",
            "--full-sync",
            "--grad-clip-norm",
            "1.5",
            "--learning-rate",
            "2e-6",
        ]
    )

    config = train_textworld.build_config(
        args,
        run_dir=tmp_path / "run",
        row_count=256,
    )

    assert config.learning_rate == 2e-6
    assert args.max_rows == 96
    assert config.completions_per_prompt == 8
    assert config.prompt_groups_per_step == 16
    assert config.max_head_offpolicy_versions == 0
    assert config.grad_clip_norm == 1.5
    assert config.trainer.training_shape_id == shape
    assert config.deployment.hot_load_transition_type == "SYNC"


def test_textworld_async_offpolicy_and_epoch_controls(tmp_path) -> None:
    from training.examples.harbor_rl.tasks.textworld import train as train_textworld

    args = train_textworld.parse_args(
        [
            "--base-model",
            "accounts/example/models/policy",
            "--tokenizer-model",
            "Qwen/Qwen3.8-27B",
            "--renderer-name",
            "qwen3_8",
            "--textworld-dataset",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--shuffle-seed",
            "19",
            "--max-head-offpolicy-versions",
            "8",
            "--epochs",
            "2",
        ]
    )

    config = train_textworld.build_config(
        args,
        run_dir=tmp_path / "run",
        row_count=256,
    )

    assert config.max_head_offpolicy_versions == 8
    assert config.epochs == 2
    assert config.deployment.hot_load_transition_type is None


def test_textworld_rows_require_unique_task_ids():
    from training.examples.harbor_rl.records import select_rows

    with pytest.raises(ValueError, match="unique"):
        select_rows([{"task_name": "same"}, {"task_name": "same"}], ["same"])


def _sampling_args(tmp_path, *extra):
    from training.examples.harbor_rl.tasks.textworld import train as train_textworld

    return train_textworld.parse_args(
        [
            "--base-model",
            "accounts/example/models/policy",
            "--tokenizer-model",
            "example/tokenizer",
            "--renderer-name",
            "example-renderer",
            "--textworld-dataset",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            *extra,
        ]
    )


def test_textworld_sampling_only_requires_a_deployment(tmp_path, monkeypatch):
    from training.examples.harbor_rl.tasks.textworld import train as train_textworld

    monkeypatch.setenv("FIREWORKS_API_KEY", "key")
    monkeypatch.setenv("E2B_API_KEY", "key")
    args = _sampling_args(tmp_path, "--sampling-only")

    with pytest.raises(ValueError, match="--sampling-only requires --deployment-id"):
        train_textworld.validate_args(args)


def test_textworld_training_accepts_managed_resource_reattach(tmp_path, monkeypatch):
    from training.examples.harbor_rl.tasks.textworld import train as train_textworld

    monkeypatch.setenv("FIREWORKS_API_KEY", "key")
    monkeypatch.setenv("E2B_API_KEY", "key")
    args = _sampling_args(
        tmp_path,
        "--deployment-id",
        "accounts/example/deployments/abc",
        "--trainer-job-id",
        "accounts/example/rlorTrainerJobs/job",
        "--shuffle-seed",
        "19",
    )

    train_textworld.validate_args(args)
    config = train_textworld.build_config(
        args,
        run_dir=tmp_path / "run",
        row_count=256,
    )

    assert config.trainer.job_id == "accounts/example/rlorTrainerJobs/job"
    assert config.deployment.deployment_id == "accounts/example/deployments/abc"


def test_textworld_sampling_only_skips_trainer_arguments(tmp_path, monkeypatch):
    from training.examples.harbor_rl.tasks.textworld import train as train_textworld

    monkeypatch.setenv("FIREWORKS_API_KEY", "key")
    monkeypatch.setenv("E2B_API_KEY", "key")
    args = _sampling_args(
        tmp_path,
        "--sampling-only",
        "--deployment-id",
        "accounts/example/deployments/abc",
    )

    train_textworld.validate_args(args)

    assert args.shuffle_seed is None
    assert args.eval_completions_per_prompt == 3
    assert args.temperature == 1.0
    assert args.max_seq_len == 24_576
    assert args.max_completion_tokens == 8_192


def test_textworld_sampling_only_rejects_trainer_reattach(tmp_path, monkeypatch):
    from training.examples.harbor_rl.tasks.textworld import train as train_textworld

    monkeypatch.setenv("FIREWORKS_API_KEY", "key")
    monkeypatch.setenv("E2B_API_KEY", "key")
    args = _sampling_args(
        tmp_path,
        "--sampling-only",
        "--deployment-id",
        "accounts/example/deployments/abc",
        "--trainer-job-id",
        "accounts/example/rlorTrainerJobs/job",
    )

    with pytest.raises(ValueError, match="does not accept --trainer-job-id"):
        train_textworld.validate_args(args)


def test_textworld_rejects_completion_budget_at_context_limit(
    tmp_path, monkeypatch
):
    from training.examples.harbor_rl.tasks.textworld import train as train_textworld

    monkeypatch.setenv("FIREWORKS_API_KEY", "key")
    monkeypatch.setenv("E2B_API_KEY", "key")
    args = _sampling_args(
        tmp_path,
        "--sampling-only",
        "--deployment-id",
        "accounts/example/deployments/abc",
        "--max-seq-len",
        "8192",
        "--max-completion-tokens",
        "8192",
    )

    with pytest.raises(ValueError, match="must be less than"):
        train_textworld.validate_args(args)
