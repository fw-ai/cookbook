from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from eval_protocol.models import EvaluationRow, InputMetadata
import httpx

import training.examples.frozen_lake.train_frozen_lake as train_module
from training.examples.frozen_lake.masking import (
    build_training_loss_mask,
    build_ui_token_mask,
    compute_model_output_spans,
)
from training.examples.frozen_lake.train_frozen_lake import (
    FrozenLakeConfig,
    evaluation_row_to_training_data,
    load_seed_contexts,
    parse_args,
)


def test_parse_args_applies_cli_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_frozen_lake.py",
            "--base-model",
            "accounts/test/models/qwen3-4b",
            "--tokenizer-model",
            "Qwen/Qwen3-4B",
            "--max-seeds",
            "3",
            "--max-steps",
            "9",
            "--observation-mode",
            "image",
            "--allow-plaintext-action-fallback",
            "--training-shape",
            "ts-qwen3-4b-smoke-v1",
        ],
    )

    cfg = parse_args()

    assert isinstance(cfg, FrozenLakeConfig)
    assert cfg.base_model == "accounts/test/models/qwen3-4b"
    assert cfg.tokenizer_model == "Qwen/Qwen3-4B"
    assert cfg.max_seeds == 3
    assert cfg.max_steps == 9
    assert cfg.observation_mode == "image"
    assert cfg.allow_plaintext_action_fallback is True
    assert cfg.training_shape == "ts-qwen3-4b-smoke-v1"


def test_load_seed_contexts_respects_limit_and_defaults(tmp_path):
    seed_path = tmp_path / "seeds.jsonl"
    seed_path.write_text(
        "\n".join(
            [
                json.dumps({"seed": 101, "map_name": "4x4"}),
                json.dumps({"seed": 202}),
                json.dumps({"seed": 303, "map_name": "8x8"}),
            ]
        )
    )

    contexts = load_seed_contexts(str(seed_path), max_seeds=2)

    assert contexts == [
        {"map_name": "4x4", "use_random_map": True, "seed": 101},
        {"map_name": "4x4", "use_random_map": True, "seed": 202},
    ]


def test_evaluation_row_to_training_data_returns_empty_without_traces():
    row = EvaluationRow(input_metadata=InputMetadata(row_id="empty"))
    row.execution_metadata.extra = {}

    datums, prompt_len, inf_logprobs, rewards = evaluation_row_to_training_data(row)

    assert datums == []
    assert prompt_len == 0
    assert inf_logprobs == []
    assert rewards == []


def test_evaluation_row_to_training_data_builds_weighted_episode():
    token_turn_traces = [
        {
            "prompt_ids": [1, 2, 3],
            "completion_ids": [10, 11],
            "completion_logprobs": [-0.1, -0.2],
        },
        {
            "prompt_ids": [1, 2, 3, 10, 11, 20, 21, 22],
            "completion_ids": [30, 31, 32],
            "completion_logprobs": [-0.3, -0.4, -0.5],
        },
    ]
    model_request_traces = [{"assistant_turn_len": 2}, {}]
    step_rewards = [0.0, 1.0]

    row = EvaluationRow(input_metadata=InputMetadata(row_id="episode"))
    row.execution_metadata.extra = {
        "token_turn_traces": token_turn_traces,
        "model_request_traces": model_request_traces,
        "step_rewards": step_rewards,
    }

    datums, prompt_len, inf_logprobs, rewards = evaluation_row_to_training_data(row)

    full_tokens = token_turn_traces[-1]["prompt_ids"] + token_turn_traces[-1]["completion_ids"]
    spans = compute_model_output_spans(token_turn_traces, model_request_traces)
    ui_mask = build_ui_token_mask(spans, len(full_tokens))
    loss_mask = build_training_loss_mask(spans, len(full_tokens) - 1)

    assert prompt_len == 3
    assert rewards == [1.0]
    assert inf_logprobs == [[0.0, 0.0, -0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4, -0.5]]
    assert len(datums) == 1
    assert datums[0].loss_fn_inputs["target_tokens"].data == full_tokens[1:]
    assert datums[0].loss_fn_inputs["weights"].data == loss_mask
    assert datums[0].loss_fn_inputs["loss_mask"].data == loss_mask
    assert ui_mask == [0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 2]


def test_main_bootstraps_without_reference_and_cleans_up(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "trainer_jobs": [],
        "deleted_jobs": [],
        "deleted_deployments": [],
        "wandb_logs": [],
        "wandb_finished": 0,
    }

    cfg = train_module.FrozenLakeConfig(
        base_model="accounts/test/models/qwen3-4b",
        tokenizer_model="Qwen/Qwen3-4B",
        completions_per_prompt=2,
        prompt_groups_per_step=4,
        min_samples_per_fwd_bwd=2,
        max_concurrent=8,
        max_steps=5,
        max_seeds=1,
        kl_beta=0.0,
        training_shape="ts-qwen3-4b-smoke-v1",
        deployment_id="dep-123",
        observation_mode="image",
        allow_plaintext_action_fallback=True,
    )

    class FakeTrainerJobManager:
        def __init__(self, *, api_key, account_id, base_url):
            events["trainer_mgr_init"] = {
                "api_key": api_key,
                "account_id": account_id,
                "base_url": base_url,
            }

        def resolve_training_profile(self, shape_id):
            events["resolved_shape"] = shape_id
            return SimpleNamespace(
                deployment_shape_version="shape/versions/7",
                pipeline_parallelism=1,
                max_supported_context_length=256,
            )

        def delete(self, job_id):
            events["deleted_jobs"].append(job_id)

        def get(self, _job_id):
            return None

    class FakeDeploymentManager:
        inference_url = "https://deployments.unit.test"

        def __init__(self, *, api_key, account_id, base_url, hotload_api_url):
            events["deploy_mgr_init"] = {
                "api_key": api_key,
                "account_id": account_id,
                "base_url": base_url,
                "hotload_api_url": hotload_api_url,
            }

        def delete(self, deployment_id):
            events["deleted_deployments"].append(deployment_id)

        def get(self, _deployment_id):
            return None

    class FakeClient:
        def __init__(self, _mgr, job_id, *_args, **_kwargs):
            self.job_id = job_id
            self.inner = object()

        def save_state(self, name, timeout):
            events["final_save"] = (name, timeout)

    class FakeWeightSyncer:
        def __init__(self, **kwargs):
            events["weight_syncer_init"] = kwargs

    class FakeRolloutProcessor:
        def __init__(self, **kwargs):
            events["rollout_processor_init"] = kwargs

        def __call__(self, rows, rollout_config):
            events["rollout_processor_call"] = (rows, rollout_config)
            return []

    async def fake_run_rl_loop(**kwargs):
        events["run_rl_loop_kwargs"] = kwargs
        return 0

    def fake_create_trainer_job(*args, **kwargs):
        events["trainer_jobs"].append(kwargs)
        return SimpleNamespace(job_id="policy-job")

    monkeypatch.setattr(train_module, "parse_args", lambda: cfg)
    monkeypatch.setattr(train_module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "wandb_finish", lambda: events.__setitem__("wandb_finished", 1))
    monkeypatch.setattr(train_module, "wandb_log", lambda payload, step=0: events["wandb_logs"].append((step, payload)))
    monkeypatch.setattr(
        train_module,
        "load_seed_contexts",
        lambda *args, **kwargs: [{"seed": 101, "map_name": "4x4", "use_random_map": True}],
    )
    monkeypatch.setattr(train_module, "TrainerJobManager", FakeTrainerJobManager)
    monkeypatch.setattr(train_module, "DeploymentManager", FakeDeploymentManager)
    monkeypatch.setattr(
        train_module,
        "setup_deployment",
        lambda *args, **kwargs: SimpleNamespace(inference_model="accounts/test/models/deployed"),
    )
    monkeypatch.setattr(train_module, "create_trainer_job", fake_create_trainer_job)
    monkeypatch.setattr(train_module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(train_module, "WeightSyncer", FakeWeightSyncer)
    monkeypatch.setattr(train_module, "FrozenLakeToolRolloutProcessor", FakeRolloutProcessor)
    monkeypatch.setattr(train_module, "build_loss_fn", lambda **kwargs: ("loss-builder", kwargs))
    monkeypatch.setattr(train_module, "run_rl_loop", fake_run_rl_loop)
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: SimpleNamespace(status_code=200))

    train_module.main()

    assert cfg.max_seq_len == 256
    assert events["resolved_shape"] == "ts-qwen3-4b-smoke-v1"
    assert len(events["trainer_jobs"]) == 1
    assert events["trainer_jobs"][0]["display_name"] == "frozen-lake-policy"
    assert events["trainer_jobs"][0]["grad_accum"] == 4
    assert events["trainer_jobs"][0]["hot_load_deployment_id"] == "dep-123"
    assert events["weight_syncer_init"]["deployment_id"] == "dep-123"
    assert events["rollout_processor_init"]["observation_mode"] == "image"
    assert events["rollout_processor_init"]["allow_plaintext_action_fallback"] is True
    assert events["run_rl_loop_kwargs"]["prompt_groups_per_step"] == 4
    assert events["run_rl_loop_kwargs"]["min_prompt_groups_per_fwd_bwd"] == 1
    assert events["run_rl_loop_kwargs"]["completions_per_prompt"] == 2
    assert events["deleted_jobs"] == ["policy-job"]
    assert events["deleted_deployments"] == ["dep-123"]
    assert events["wandb_finished"] == 1
