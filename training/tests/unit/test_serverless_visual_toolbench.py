"""Unit tests for the pooled VisualToolBench validation path."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.examples.serverless_rl import visual_toolbench_rl as serverless_vtb
from training.utils import GradAccNormalization


def test_serverless_defaults_use_qwen3p6():
    args = serverless_vtb.parse_args([])
    cfg = serverless_vtb.config_from_args(args)

    assert cfg.base_model == "accounts/fireworks/models/qwen3p6-27b"
    assert cfg.tokenizer_model == "Qwen/Qwen3.6-27B"
    assert cfg.renderer_name == "qwen3_6_disable_thinking_interleaved"
    assert cfg.steps == 15
    assert cfg.prompt_groups_per_step == 8
    assert cfg.group_size == 8
    assert cfg.rollout_concurrency == 8
    assert cfg.max_completion_tokens == 32768
    assert cfg.max_seq_len == 131072
    assert cfg.learning_rate == 3e-5
    assert cfg.eval_temperature == 1.0
    assert cfg.eval_top_p == 0.95
    assert cfg.eval_top_k == 20
    assert cfg.eval_max_completion_tokens == 26666
    assert cfg.require_complete_eval is False
    assert cfg.adam_beta2 == 0.95
    assert cfg.adam_eps == 1e-12
    assert cfg.adam_weight_decay == 0.0
    assert cfg.judge_max_tokens == 65536
    assert cfg.judge_max_concurrency == 4
    assert cfg.judge_timeout_s == 900.0
    assert cfg.dcp_save_interval == 2
    assert cfg.grad_accumulation_normalization is (GradAccNormalization.NUM_LOSS_TOKENS)
    assert cfg.router_replay is False
    assert cfg.router_replay_completion_only is False
    assert cfg.require_tool_aligned_data is True


def test_serverless_eval_args_map_to_config():
    args = serverless_vtb.parse_args(
        [
            "--tokenizer-model",
            "/tmp/qwen3p6-tokenizer",
            "--eval-dataset",
            "/tmp/eval.jsonl",
            "--eval-interval",
            "5",
            "--eval-upfront",
            "--eval-at-end",
            "--eval-group-size",
            "1",
            "--eval-temperature",
            "1",
            "--eval-top-p",
            "0.95",
            "--eval-top-k",
            "20",
            "--eval-max-completion-tokens",
            "26666",
            "--require-complete-eval",
            "--adam-beta2",
            "0.95",
            "--adam-eps",
            "1e-12",
            "--adam-weight-decay",
            "0",
        ]
    )
    cfg = serverless_vtb.config_from_args(args)

    assert cfg.eval_dataset == "/tmp/eval.jsonl"
    assert cfg.eval_interval == 5
    assert cfg.eval_upfront is True
    assert cfg.eval_at_end is True
    assert cfg.eval_group_size == 1
    assert cfg.eval_temperature == 1.0
    assert cfg.eval_top_p == 0.95
    assert cfg.eval_top_k == 20
    assert cfg.eval_max_completion_tokens == 26666
    assert cfg.require_complete_eval is True
    assert cfg.adam_beta2 == 0.95
    assert cfg.adam_eps == 1e-12
    assert cfg.adam_weight_decay == 0.0


def test_serverless_dcp_interval_maps_to_config():
    args = serverless_vtb.parse_args(
        [
            "--tokenizer-model",
            "/tmp/qwen3p6-tokenizer",
            "--dcp-save-interval",
            "4",
        ]
    )

    assert serverless_vtb.config_from_args(args).dcp_save_interval == 4


def test_wandb_eval_completeness_metrics_are_defined_and_logged(
    tmp_path: Path,
    monkeypatch,
):
    defined_metrics = []
    logged = []
    fake_run = SimpleNamespace(url="https://wandb.example/run")
    fake_wandb = SimpleNamespace(
        run=fake_run,
        init=lambda **_kwargs: fake_run,
        define_metric=lambda *args, **kwargs: defined_metrics.append((args, kwargs)),
        log=lambda payload, *, step: logged.append((payload, step)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "test-only")

    cfg = serverless_vtb.Config(
        tokenizer_model="/tmp/qwen3p6-tokenizer",
        wandb_entity="test-entity",
        wandb_project="test-project",
    )
    assert serverless_vtb._maybe_init_wandb(cfg, tmp_path) is fake_run
    serverless_vtb._log_wandb_eval(
        5,
        {
            "eval/return_ratio": 0.98,
            "eval/truncated_ratio": 0.02,
        },
    )

    assert (
        ("eval/return_ratio",),
        {"step_metric": "train/step", "summary": "min"},
    ) in defined_metrics
    assert (
        ("eval/truncated_ratio",),
        {"step_metric": "train/step", "summary": "max"},
    ) in defined_metrics
    assert logged == [
        (
            {
                "train/step": 5,
                "eval/return_ratio": 0.98,
                "eval/truncated_ratio": 0.02,
            },
            5,
        )
    ]


def test_wandb_step_logs_custom_loss_and_optimizer_diagnostics(monkeypatch):
    logged = []
    fake_wandb = SimpleNamespace(
        run=object(),
        log=lambda payload, *, step: logged.append((payload, step)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    serverless_vtb._log_wandb_step(
        3,
        {
            "train/step": 3,
            "train/grad_norm": 1.25,
            "train/ppo_clip_frac": 0.1,
            "train/tis/weight_mean": 0.99,
            "train/policy_gradient/estimator_variance_proxy": 0.02,
            "rollout/raw_reward": 0.4,
            "snapshot": "not-a-metric",
        },
    )

    assert logged == [
        (
            {
                "train/step": 3,
                "train/grad_norm": 1.25,
                "train/ppo_clip_frac": 0.1,
                "train/tis/weight_mean": 0.99,
                "train/policy_gradient/estimator_variance_proxy": 0.02,
                "rollout/raw_reward": 0.4,
            },
            3,
        )
    ]


def test_custom_loss_diagnostics_keep_stable_numeric_metrics():
    result = SimpleNamespace(
        metrics={
            "ppo_clip_frac": 0.2,
            "tis/weight_mean": 0.98,
            "active_tokens": 1234,
            "policy_gradient/coefficient_variance_proxy": 0.5,
            "policy_gradient/estimator_variance_proxy": 0.01,
            "policy_gradient/estimator_std_error_proxy": 0.1,
            "policy_gradient/sample_count": 50,
            "inference_kld:last": 0.001,
            "ignored": 7.0,
        }
    )

    assert serverless_vtb._extract_train_diagnostics(result) == {
        "train/ppo_clip_frac": 0.2,
        "train/tis/weight_mean": 0.98,
        "train/active_tokens": 1234.0,
        "train/policy_gradient/coefficient_variance_proxy": 0.5,
        "train/policy_gradient/estimator_variance_proxy": 0.01,
        "train/policy_gradient/estimator_std_error_proxy": 0.1,
        "train/policy_gradient/sample_count": 50.0,
        "train/inference_kld": 0.001,
    }


def test_complete_eval_gate_rejects_missing_or_truncated_episodes():
    with pytest.raises(RuntimeError, match="returned 49/50.*sampling calls=1"):
        serverless_vtb._validate_eval_completeness(
            label="baseline",
            required=True,
            returned_episodes=49,
            expected_episodes=50,
            truncated_samples=1,
            prompt_budget_exhaustions=0,
        )


def test_complete_eval_gate_allows_complete_untruncated_eval():
    serverless_vtb._validate_eval_completeness(
        label="baseline",
        required=True,
        returned_episodes=50,
        expected_episodes=50,
        truncated_samples=0,
        prompt_budget_exhaustions=0,
    )


def test_training_batches_cover_two_complete_epochs_without_drops():
    rows = [{"id": str(index)} for index in range(214)]
    batches = list(
        serverless_vtb._iter_training_batches(
            rows,
            epochs=2,
            shuffle=True,
            seed=0,
            batch_size=4,
        )
    )

    assert len(batches) == 107
    assert all(len(batch) == 4 for batch in batches)
    assert Counter(row["id"] for batch in batches for row in batch) == Counter(
        {str(index): 2 for index in range(214)}
    )


def test_training_batches_keep_partial_final_batch_for_eight_group_steps():
    rows = [{"id": str(index)} for index in range(214)]
    batches = list(
        serverless_vtb._iter_training_batches(
            rows,
            epochs=2,
            shuffle=True,
            seed=0,
            batch_size=8,
        )
    )

    assert len(batches) == 54
    assert all(len(batch) == 8 for batch in batches[:-1])
    assert len(batches[-1]) == 4
    assert Counter(row["id"] for batch in batches for row in batch) == Counter(
        {str(index): 2 for index in range(214)}
    )


def test_rollout_setup_uses_fireworks_production_endpoint():
    runner = object.__new__(serverless_vtb.ServerlessVisualToolbenchRL)
    runner.cfg = serverless_vtb.Config(
        tokenizer_model="/tmp/qwen3p6-tokenizer",
        api_key="test-key",
        max_completion_tokens=32768,
        eval_max_completion_tokens=26666,
    )
    runner.tokenizer = object()
    runner.router_replay_enabled = False
    sampler = object()

    setup = runner._rollout_setup(
        "snapshot", sampler=sampler, event_sink=lambda _event: None
    )
    eval_setup = runner._rollout_setup(
        "snapshot",
        sampler=sampler,
        event_sink=lambda _event: None,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        max_completion_tokens=runner.cfg.eval_max_completion_tokens,
    )

    assert setup.sample_kwargs["max_tokens"] == 32768
    assert setup.sample_kwargs["temperature"] == 1.0
    assert setup.sample_kwargs["top_p"] == 1.0
    assert setup.sample_kwargs["top_k"] == 0
    assert eval_setup.sample_kwargs["max_tokens"] == 26666
    assert eval_setup.sample_kwargs["temperature"] == 1.0
    assert eval_setup.sample_kwargs["top_p"] == 0.95
    assert eval_setup.sample_kwargs["top_k"] == 20
    assert setup.inference_base_url == (
        "https://api.fireworks.ai/training/v1/serverless"
    )
    assert setup.api_key == "test-key"
    assert "judge_base_url" not in setup.extras
    assert "judge_api_key" not in setup.extras
    assert setup.extras["judge_max_tokens"] == 65536
    assert setup.extras["judge_max_concurrency"] == 4
    assert setup.extras["judge_timeout_s"] == 900.0
    assert setup.sampler is sampler


def test_serverless_loader_rejects_unaligned_rows(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"id": "stale"}) + "\n")

    with pytest.raises(ValueError, match="not four-image-tool aligned"):
        serverless_vtb._load_rows(path)

    assert serverless_vtb._load_rows(path, require_tool_aligned=False) == [
        {"id": "stale"}
    ]


def test_eval_path_uses_fixed_denominator_without_optimizer(
    tmp_path: Path,
    monkeypatch,
):
    runner = object.__new__(serverless_vtb.ServerlessVisualToolbenchRL)
    runner.cfg = serverless_vtb.Config(
        tokenizer_model="/tmp/qwen3p6-tokenizer",
        checkpoint_name="eval-test",
        eval_group_size=2,
        eval_temperature=1.0,
        eval_top_p=0.95,
        eval_top_k=20,
        eval_max_completion_tokens=26666,
        require_complete_eval=False,
    )
    runner.tokenizer = object()
    runner.router_replay_enabled = True
    runner.eval_rows = [
        {
            "id": "eval-1",
            "prompt": "prompt",
            "category": "biology",
            "eval_focus": "region_switch_qa",
        }
    ]
    runner.eval_metrics_path = tmp_path / "eval_metrics.jsonl"
    runner.eval_completions_dir = tmp_path / "eval_completions"
    runner.eval_completions_dir.mkdir()

    save_calls = []

    class TrainingClient:
        def save_weights_for_sampler(self, name):
            save_calls.append(name)
            return SimpleNamespace(
                result=lambda: SimpleNamespace(path="saved/eval-checkpoint")
            )

    class SamplingClient:
        deployment_sampler = object()

        def close(self):
            pass

    runner.training_client = TrainingClient()
    runner.service = SimpleNamespace(
        create_sampling_client=lambda **_kwargs: SamplingClient(),
    )
    run = SimpleNamespace(
        segments=[SimpleNamespace(reward=0.6, text="answer")],
        metadata={
            "mean_official_score": 0.5,
            "mean_critical_fraction": 1.0,
            "judge_passed": True,
            "num_turns": 1,
            "num_tool_calls": 2,
        },
    )

    async def collect(setup, rows, *, completions_per_prompt):
        assert setup.sample_kwargs["temperature"] == 1.0
        assert setup.sample_kwargs["top_p"] == 0.95
        assert setup.sample_kwargs["top_k"] == 20
        assert setup.sample_kwargs["max_tokens"] == 26666
        assert "include_routing_matrix" not in setup.sample_kwargs
        assert rows == runner.eval_rows
        assert completions_per_prompt == 2
        return [[run]]

    monkeypatch.setattr(runner, "_collect_runs", collect)

    rec = runner._evaluate(completed_steps=0, label="baseline")

    assert save_calls == ["eval-test-b0000"]
    assert rec["eval/reward"] == 0.3
    assert rec["eval/reward_returned"] == 0.6
    assert rec["eval/official_score"] == 0.25
    assert rec["eval/official_score_returned"] == 0.5
    assert rec["eval/returned_episodes"] == 1
    assert rec["eval/return_ratio"] == 0.5
    assert json.loads(runner.eval_metrics_path.read_text())["label"] == "baseline"


def test_run_saves_dcp_every_two_successful_optimizer_updates_and_final(
    tmp_path: Path,
    monkeypatch,
):
    runner = object.__new__(serverless_vtb.ServerlessVisualToolbenchRL)
    runner.cfg = serverless_vtb.Config(
        tokenizer_model="/tmp/qwen3p6-tokenizer",
        steps=5,
        epochs=1,
        prompt_groups_per_step=1,
        dcp_save_interval=2,
        plot_reward_curve=False,
    )
    runner.rows = [{"id": str(index)} for index in range(5)]
    runner.metrics_path = tmp_path / "metrics.jsonl"
    runner.dcp_metrics_path = tmp_path / "dcp_metrics.jsonl"

    saved_dcp_names = []

    class TrainingClient:
        def save_state(self, name):
            saved_dcp_names.append(name)
            return SimpleNamespace(
                result=lambda: SimpleNamespace(path=name)
            )

        def save_weights_for_sampler(self, _name):
            return SimpleNamespace(
                result=lambda: SimpleNamespace(path="saved/final-sampler")
            )

    runner.training_client = TrainingClient()
    trained_by_step = [True, False, True, True, False]

    def step(step, _batch):
        trained = trained_by_step[step]
        return {
            "step": step,
            "rollout/raw_reward": 0.1,
            "rollout/judge_pass": 0.0,
            "train/trained": trained,
            "train/inference_kld": 0.001 if trained else None,
        }

    monkeypatch.setattr(runner, "_step", step)

    runner.run()

    assert saved_dcp_names == [
        "vtb-serverless-d0002s0003",
        "vtb-serverless-d0003s0005",
    ]
    rows = [
        json.loads(line) for line in runner.dcp_metrics_path.read_text().splitlines()
    ]
    assert [(row["trained_steps"], row["completed_steps"]) for row in rows] == [
        (2, 3),
        (3, 5),
    ]
    assert rows[-1]["final"] is True


@pytest.mark.parametrize(
    "name",
    ["UPPERCASE", "contains_underscore", "-leading", "trailing-", "x" * 50],
)
def test_checkpoint_prefix_rejects_names_that_sdk_cannot_save(name: str):
    with pytest.raises(ValueError, match="lowercase DNS-label"):
        serverless_vtb._validate_checkpoint_prefix(name, "checkpoint_name")


def test_checkpoint_prefix_accepts_formal_launcher_shape():
    serverless_vtb._validate_checkpoint_prefix(
        "vtb-four-tools-20260729t064821z-2572002",
        "checkpoint_name",
    )


def test_inference_kld_uses_expanded_response_slice():
    response_start = 1134
    inference = [0.0] * response_start + [-0.35, -1.20, -0.05, -2.10, -0.60]
    trainer = [0.0] * response_start + [-0.34, -1.22, -0.05, -2.08, -0.61]

    values = serverless_vtb._inference_kld(
        trainer,
        inference,
        response_start=response_start,
    )

    assert len(values) == 5
    assert sum(values) / len(values) < 1e-3


def test_qwen_launcher_has_public_demo_train_and_eval_shape():
    script = (
        Path(serverless_vtb.__file__).resolve().parent
        / "runs"
        / "run_qwen3p6_serverless_2epoch.sh"
    ).read_text()

    for expected in (
        "--base-model accounts/fireworks/models/qwen3p6-27b",
        '--tokenizer-model "$tokenizer_model"',
        "--renderer-name qwen3_6_disable_thinking_interleaved",
        "--steps 54",
        "--epochs 2",
        "--learning-rate 3e-5",
        "--max-completion-tokens 32768",
        "--eval-max-completion-tokens 26666",
        "--eval-interval 5",
        "--eval-upfront",
        "--eval-at-end",
        "--eval-group-size 1",
        "--eval-temperature 1.0",
        "--eval-top-p 0.95",
        "--eval-top-k 20",
        "--rollout-concurrency 8",
        "--no-router-replay",
        "--grad-accumulation-normalization num_loss_tokens",
        "--judge-max-tokens 65536",
        "--judge-max-concurrency 4",
        "--judge-timeout-s 900",
        "--dcp-save-interval 2",
    ):
        assert expected in script
    assert "--training-shape" not in script
    assert "--region" not in script
    assert "validate_visual_toolbench_run" not in script


def test_kimi_launcher_matches_public_demo_train_and_eval_shape():
    script = (
        Path(serverless_vtb.__file__).resolve().parent
        / "runs"
        / "run_kimi_k3_serverless_2epoch.sh"
    ).read_text()

    for expected in (
        "--base-model accounts/fireworks/models/kimi-k3",
        'tokenizer_model="${KIMI_K3_TOKENIZER:-moonshotai/Kimi-K3}"',
        "--renderer-name kimi_k3_disable_thinking",
        "--steps 27",
        "--prompt-groups-per-step 16",
        "--group-size 8",
        "--rollout-concurrency 8",
        "--epochs 2",
        "--lora-rank 64",
        "--learning-rate 1e-4",
        "--adam-beta2 0.95",
        "--adam-eps 1e-12",
        "--adam-weight-decay 0",
        "--max-completion-tokens 32768",
        "--eval-max-completion-tokens 26666",
        "--eval-interval 5",
        "--eval-upfront",
        "--eval-at-end",
        "--eval-temperature 1.0",
        "--eval-top-p 0.95",
        "--eval-top-k 20",
        "--no-router-replay",
        "--grad-accumulation-normalization num_loss_tokens",
        "--judge-max-tokens 65536",
        "--dcp-save-interval 2",
    ):
        assert expected in script
    assert "--training-shape" not in script
    assert "--region" not in script
    assert "validate_visual_toolbench_run" not in script
