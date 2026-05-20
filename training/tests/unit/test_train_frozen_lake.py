from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from eval_protocol.models import EvaluationRow, InputMetadata, Message
import httpx
import pytest

import training.examples.rl.frozen_lake.train_frozen_lake as train_module
from training.utils.rl.rollout import Rollout, rollout_to_prompt_group
from training.examples.rl.frozen_lake.masking import (
    build_training_loss_mask,
    build_ui_token_mask,
    compute_model_output_spans,
)
from training.examples.rl.frozen_lake.train_frozen_lake import (
    FrozenLakeConfig,
    evaluation_row_to_rollout_sample,
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
            "--deployment-replica-count",
            "3",
            "--output-model-id",
            "out-model",
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
    assert cfg.deployment_region is None
    assert cfg.deployment_replica_count == 3


def test_main_rejects_invalid_output_model_id(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    cfg = FrozenLakeConfig(output_model_id="bad_name")

    with pytest.raises(RuntimeError, match="output_model_id.*invalid|invalid.*output_model_id"):
        train_module.main(cfg)


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


def test_evaluation_row_to_rollout_sample_builds_token_native_sample():
    row = EvaluationRow(input_metadata=InputMetadata(row_id="rollout"))
    row.execution_metadata.extra = {
        "token_turn_traces": [
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
        ],
        "model_request_traces": [{"assistant_turn_len": 2}, {}],
        "step_rewards": [0.0, 1.0],
    }

    sample = evaluation_row_to_rollout_sample(row)

    assert sample is not None
    assert sample.tokens == [1, 2, 3, 10, 11, 20, 21, 22, 30, 31, 32]
    assert sample.loss_mask == [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
    assert sample.logprobs == [0.0, 0.0, 0.0, -0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4, -0.5]
    assert sample.reward == 1.0


def test_rollout_sample_adapter_matches_legacy_training_data():
    row = EvaluationRow(input_metadata=InputMetadata(row_id="parity"))
    row.execution_metadata.extra = {
        "token_turn_traces": [
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
        ],
        "model_request_traces": [{"assistant_turn_len": 2}, {}],
        "step_rewards": [0.0, 1.0],
    }

    legacy_datums, _prompt_len, legacy_inf_logprobs, legacy_rewards = (
        evaluation_row_to_training_data(row)
    )
    sample = evaluation_row_to_rollout_sample(row)
    assert sample is not None
    contrasting_sample = sample.__class__(
        tokens=list(sample.tokens),
        logprobs=list(sample.logprobs),
        loss_mask=list(sample.loss_mask),
        reward=0.0,
    )

    prompt_group = rollout_to_prompt_group(Rollout(samples=[sample, contrasting_sample]))

    assert prompt_group is not None
    assert (
        prompt_group.data[0].loss_fn_inputs["target_tokens"].data
        == legacy_datums[0].loss_fn_inputs["target_tokens"].data
    )
    assert (
        prompt_group.data[0].loss_fn_inputs["weights"].data
        == legacy_datums[0].loss_fn_inputs["weights"].data
    )
    assert prompt_group.inf_logprobs[0] == legacy_inf_logprobs[0]
    assert prompt_group.rewards[0] == legacy_rewards[0]


def test_evaluation_row_to_training_data_handles_multimodal_row_messages():
    row = EvaluationRow(input_metadata=InputMetadata(row_id="visual-episode"))
    row.messages = [
        Message(
            role="user",
            content=[
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "FrozenLake grid observation"},
            ],
        ),
        Message(
            role="tool",
            content=[
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,def"}},
            ],
        ),
    ]
    row.execution_metadata.extra = {
        "token_turn_traces": [
            {
                "prompt_ids": [1, 2, 3],
                "completion_ids": [10, 11],
                "completion_logprobs": [-0.1, -0.2],
            },
            {
                "prompt_ids": [1, 2, 3, 10, 11, 20, 21],
                "completion_ids": [30],
                "completion_logprobs": [-0.3],
            },
        ],
        "model_request_traces": [{"assistant_turn_len": 2}, {}],
        "step_rewards": [0.0, 1.0],
        "observation_mode": "image",
    }

    datums, prompt_len, inf_logprobs, rewards = evaluation_row_to_training_data(row)

    assert prompt_len == 3
    assert rewards == [1.0]
    assert inf_logprobs == [[0.0, 0.0, -0.1, -0.2, 0.0, 0.0, -0.3]]
    assert len(datums) == 1
    assert datums[0].loss_fn_inputs["target_tokens"].data == [2, 3, 10, 11, 20, 21, 30]
    assert datums[0].loss_fn_inputs["weights"].data == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]


def test_main_bootstraps_without_reference_and_cleans_up(monkeypatch, tmp_path):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "trainer_jobs": [],
        "deleted_jobs": [],
        "deleted_deployments": [],
        "wandb_logs": [],
        "wandb_finished": 0,
    }

    cfg = train_module.FrozenLakeConfig(
        log_path=str(tmp_path / "frozen_lake_logs"),
        base_model="accounts/test/models/qwen3-4b",
        tokenizer_model="Qwen/Qwen3-4B",
        completions_per_prompt=2,
        prompt_groups_per_step=4,
        max_concurrent=8,
        max_steps=5,
        max_seeds=1,
        kl_beta=0.0,
        training_shape="ts-qwen3-4b-smoke-v1",
        deployment_id="dep-123",
        deployment_replica_count=2,
        observation_mode="image",
        allow_plaintext_action_fallback=True,
    )

    class FakeTrainerJobManager:
        def __init__(self, *, api_key, base_url):
            events["trainer_mgr_init"] = {
                "api_key": api_key,
                "base_url": base_url,
            }

        def resolve_training_profile(self, shape_id):
            events["resolved_shape"] = shape_id
            return SimpleNamespace(
                deployment_shape="shape/versions/7",
                deployment_shape_version="shape/versions/7",
                pipeline_parallelism=1,
                max_supported_context_length=256,
            )

        def cancel(self, job_id):
            events["deleted_jobs"].append(job_id)

        def delete(self, job_id):
            self.cancel(job_id)

        def get(self, _job_id):
            return None

    class FakeDeploymentManager:
        inference_url = "https://deployments.unit.test"

        def __init__(self, *, api_key, base_url, hotload_api_url, inference_url=None):
            events["deploy_mgr_init"] = {
                "api_key": api_key,
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

        def save_state(self, name, timeout=None):
            events["final_save"] = (name, timeout)

    class FakeWeightSyncer:
        def __init__(self, **kwargs):
            events["weight_syncer_init"] = kwargs

        def save_and_hotload(self, name, checkpoint_type="base"):
            pass

    class FakeRolloutProcessor:
        def __init__(self, **kwargs):
            events["rollout_processor_init"] = kwargs

        def __call__(self, rows, rollout_config):
            events["rollout_processor_call"] = (rows, rollout_config)
            return []

    async def fake_run_async_rl_loop(**kwargs):
        events["run_async_rl_loop_kwargs"] = kwargs
        return 0, {"resolved_rows": 0}

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

    def fake_setup_deployment(_deploy_mgr, deploy_cfg, *_args, **_kwargs):
        events["deploy_config"] = deploy_cfg
        return SimpleNamespace(inference_model="accounts/test/models/deployed")

    monkeypatch.setattr(train_module, "setup_deployment", fake_setup_deployment)
    monkeypatch.setattr(train_module, "create_trainer_job", fake_create_trainer_job)
    monkeypatch.setattr(train_module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(train_module, "WeightSyncer", FakeWeightSyncer)
    monkeypatch.setattr(train_module, "FrozenLakeToolRolloutProcessor", FakeRolloutProcessor)
    monkeypatch.setattr(train_module, "build_loss_fn", lambda args: ("loss-builder", args))
    monkeypatch.setattr(train_module, "run_async_rl_loop", fake_run_async_rl_loop)
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: SimpleNamespace(status_code=200))

    train_module.main()

    assert cfg.max_seq_len == 256
    assert events["resolved_shape"] == "ts-qwen3-4b-smoke-v1"
    assert len(events["trainer_jobs"]) == 1
    assert events["trainer_jobs"][0]["display_name"] == "frozen-lake-policy"
    assert events["trainer_jobs"][0]["hot_load_deployment_id"] == "dep-123"
    assert events["deploy_config"].replica_count == 2
    assert events["weight_syncer_init"]["deployment_id"] == "dep-123"
    assert events["rollout_processor_init"]["observation_mode"] == "image"
    assert events["rollout_processor_init"]["allow_plaintext_action_fallback"] is True
    assert events["run_async_rl_loop_kwargs"]["prompt_groups_per_step"] == 4
    assert events["run_async_rl_loop_kwargs"]["completions_per_prompt"] == 2
    assert events["run_async_rl_loop_kwargs"]["max_concurrent"] == 8
    assert "train_fns" in events["run_async_rl_loop_kwargs"]
    assert events["run_async_rl_loop_kwargs"]["weight_sync_interval"] == 1
    assert events["run_async_rl_loop_kwargs"]["weight_sync_fn"] is not None
    assert events["deleted_jobs"] == ["policy-job"]
    assert events["deleted_deployments"] == []
    assert events["wandb_finished"] == 1


def test_main_skips_consumed_eval3_rows_on_resume(monkeypatch, tmp_path):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "trainer_jobs": [],
        "deleted_jobs": [],
        "deleted_deployments": [],
        "weight_sync_saves": [],
        "wandb_finished": 0,
    }

    cfg = train_module.FrozenLakeConfig(
        log_path=str(tmp_path / "frozen_lake_logs"),
        base_model="accounts/test/models/qwen3-4b",
        tokenizer_model="Qwen/Qwen3-4B",
        completions_per_prompt=2,
        prompt_groups_per_step=4,
        max_concurrent=8,
        max_steps=5,
        max_seeds=3,
        epochs=1,
        kl_beta=0.0,
        training_shape="ts-qwen3-4b-smoke-v1",
        deployment_id="dep-123",
    )

    class FakeTrainerJobManager:
        def __init__(self, *, api_key, base_url):
            pass

        def resolve_training_profile(self, shape_id):
            return SimpleNamespace(
                deployment_shape="shape/versions/7",
                deployment_shape_version="shape/versions/7",
                pipeline_parallelism=1,
                max_supported_context_length=256,
            )

        def cancel(self, job_id):
            events["deleted_jobs"].append(job_id)

        def delete(self, job_id):
            self.cancel(job_id)

        def get(self, _job_id):
            return None

    class FakeDeploymentManager:
        inference_url = "https://deployments.unit.test"

        def __init__(self, *, api_key, base_url, hotload_api_url, inference_url=None):
            pass

        def delete(self, deployment_id):
            events["deleted_deployments"].append(deployment_id)

        def get(self, _deployment_id):
            return None

    class FakeClient:
        def __init__(self, _mgr, job_id, *_args, **_kwargs):
            self.job_id = job_id
            self.inner = object()

    class FakeWeightSyncer:
        def __init__(self, **kwargs):
            pass

        def save_and_hotload(self, name, checkpoint_type="base"):
            events["weight_sync_saves"].append((name, checkpoint_type))

    class FakeTrainingCheckpoints:
        def __init__(self, *args, **kwargs):
            pass

        def resume(self):
            return SimpleNamespace(step=7, data_consumed=1)

        def save(self, *args, **kwargs):
            events.setdefault("checkpoint_saves", []).append((args, kwargs))

    class FakeRolloutProcessor:
        def __init__(self, **kwargs):
            pass

        def __call__(self, rows, rollout_config):
            return []

    async def fake_run_async_rl_loop(**kwargs):
        row_requests = list(kwargs["rows"])
        events["row_request_ids"] = [request.row_id for request in row_requests]
        events["row_request_seeds"] = [
            request.row_meta["seed"] for request in row_requests
        ]
        events["run_async_rl_loop_kwargs"] = {
            key: value for key, value in kwargs.items() if key != "rows"
        }
        return kwargs["global_step"], {"resolved_rows": kwargs["resolved_rows_offset"]}

    def fake_create_trainer_job(*args, **kwargs):
        events["trainer_jobs"].append(kwargs)
        return SimpleNamespace(job_id="policy-job")

    monkeypatch.setattr(train_module, "parse_args", lambda: cfg)
    monkeypatch.setattr(train_module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_module,
        "wandb_finish",
        lambda: events.__setitem__("wandb_finished", 1),
    )
    monkeypatch.setattr(train_module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_module,
        "load_seed_contexts",
        lambda *args, **kwargs: [
            {"seed": 101, "map_name": "4x4", "use_random_map": True},
            {"seed": 202, "map_name": "4x4", "use_random_map": True},
            {"seed": 303, "map_name": "4x4", "use_random_map": True},
        ],
    )
    monkeypatch.setattr(train_module, "TrainerJobManager", FakeTrainerJobManager)
    monkeypatch.setattr(train_module, "DeploymentManager", FakeDeploymentManager)
    monkeypatch.setattr(
        train_module,
        "setup_deployment",
        lambda *args, **kwargs: SimpleNamespace(
            inference_model="accounts/test/models/deployed",
        ),
    )
    monkeypatch.setattr(train_module, "create_trainer_job", fake_create_trainer_job)
    monkeypatch.setattr(train_module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(train_module, "WeightSyncer", FakeWeightSyncer)
    monkeypatch.setattr(
        train_module,
        "FrozenLakeToolRolloutProcessor",
        FakeRolloutProcessor,
    )
    monkeypatch.setattr(
        train_module,
        "build_loss_fn",
        lambda args: ("loss-builder", args),
    )
    monkeypatch.setattr(train_module, "run_async_rl_loop", fake_run_async_rl_loop)
    monkeypatch.setattr(
        "training.utils.checkpoints.TrainingCheckpoints",
        FakeTrainingCheckpoints,
    )
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: SimpleNamespace(status_code=200),
    )

    train_module.main()

    assert events["row_request_ids"] == [1, 2]
    assert events["row_request_seeds"] == [202, 303]
    assert events["run_async_rl_loop_kwargs"]["global_step"] == 7
    assert events["run_async_rl_loop_kwargs"]["resolved_rows_offset"] == 1
    assert events["weight_sync_saves"] == [("resume-7-base", "base")]
    assert events.get("checkpoint_saves", []) == []
    assert events["deleted_jobs"] == ["policy-job"]
    assert events["deleted_deployments"] == []
    assert events["wandb_finished"] == 1


def test_main_runs_sampling_and_training_with_reference(monkeypatch, tmp_path):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "trainer_jobs": [],
        "deleted_jobs": [],
        "deleted_deployments": [],
        "wandb_logs": [],
        "weight_sync_saves": [],
        "weight_sync_dcp": [],
        "build_loss_fn_calls": [],
        "sleeps": [],
        "promotions": [],
    }

    cfg = train_module.FrozenLakeConfig(
        log_path=str(tmp_path / "frozen_lake_logs"),
        base_model="accounts/test/models/qwen3-4b",
        tokenizer_model="Qwen/Qwen3-4B",
        completions_per_prompt=2,
        prompt_groups_per_step=1,
        max_concurrent=8,
        max_steps=5,
        max_seeds=1,
        epochs=1,
        kl_beta=0.1,
        ratio_log_cap=9.0,
        training_shape="ts-qwen3-4b-smoke-v1",
        deployment_id="dep-123",
        observation_mode="image",
        allow_plaintext_action_fallback=True,
        output_model_id="promoted-rl-model",
    )

    class FakeTrainerJobManager:
        def __init__(self, *, api_key, base_url):
            events["trainer_mgr_init"] = {
                "api_key": api_key,
                "base_url": base_url,
            }

        def resolve_training_profile(self, shape_id):
            events.setdefault("resolved_shapes", []).append(shape_id)
            return SimpleNamespace(
                deployment_shape="shape/versions/7",
                deployment_shape_version="shape/versions/7",
                pipeline_parallelism=1,
                max_supported_context_length=256,
            )

        def cancel(self, job_id):
            events["deleted_jobs"].append(job_id)

        def delete(self, job_id):
            self.cancel(job_id)

        def get(self, _job_id):
            return None

        def promote_checkpoint(self, *args, name=None, output_model_id=None, base_model=None, **kwargs):
            assert not args, (
                "Cookbook should call promote_checkpoint with the 4-segment "
                f"name= form (4-segment resource path); got positional args {args}"
            )
            assert name, "Expected name= 4-segment resource path"
            short = name.rstrip("/").rsplit("/", 1)[-1]
            job_part = name.split("/rlorTrainerJobs/", 1)[1].split("/", 1)[0]
            events["promotions"].append((job_part, short, output_model_id))

        def list_checkpoints(self, job_id, **kwargs):
            # Reflect what save_state / save_weights have produced so far,
            # so resume() at startup sees no rows (fresh start) but the
            # post-save polling and promote_latest see the just-saved rows.
            rows = []
            for state_name in events.get("save_state_calls") or []:
                rows.append({
                    "name": (
                        f"accounts/test/rlorTrainerJobs/{job_id}/"
                        f"checkpoints/{state_name}"
                    ),
                    "checkpointType": "CHECKPOINT_TYPE_TRAINING_LORA",
                    "promotable": False,
                    "createTime": "2099-04-27T00:00:00Z",
                })
            saved_weights = events.get("save_weights") or []
            for w in saved_weights:
                w_name = w[0] if isinstance(w, tuple) else w
                rows.append({
                    "name": (
                        f"accounts/test/rlorTrainerJobs/{job_id}/"
                        f"checkpoints/{w_name}-session"
                    ),
                    "checkpointType": "CHECKPOINT_TYPE_INFERENCE_LORA",
                    "promotable": True,
                    "createTime": "2099-04-27T00:00:01Z",
                })
            return rows

    class FakeDeploymentManager:
        inference_url = "https://deployments.unit.test"

        def __init__(self, *, api_key, base_url, hotload_api_url, inference_url=None):
            events["deploy_mgr_init"] = {
                "api_key": api_key,
                "base_url": base_url,
                "hotload_api_url": hotload_api_url,
            }

        def delete(self, deployment_id):
            events["deleted_deployments"].append(deployment_id)

        def get(self, _deployment_id):
            return None

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeThreadPoolExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return FakeFuture(fn(*args, **kwargs))

    class FakeClient:
        def __init__(self, _mgr, job_id, *_args, **_kwargs):
            self.job_id = job_id
            self.inner = object()

        def forward(self, data, loss_fn):
            return SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": SimpleNamespace(data=[-0.3] * len(d.loss_fn_inputs["target_tokens"].data))}
                    for d in data
                ]
            )

        def forward_backward(self, data, loss_fn="cross_entropy", loss_fn_config=None):
            events["fwd_bwd_call"] = {"data_len": len(data), "loss_fn": loss_fn, "loss_fn_config": loss_fn_config}
            return SimpleNamespace(metrics={"loss": 1.0})

        def forward_backward_custom(self, data, loss_fn):
            events["fwd_bwd_call"] = {"data_len": len(data), "loss_fn": loss_fn}
            return SimpleNamespace(metrics={"loss": 1.0})

        def optim_step(self, _params, **kwargs):
            events["optim_step_called"] = True
            return SimpleNamespace(metrics={"optimizer/lr": 1e-4})

        def save_state(self, name, timeout=None):
            events["final_save"] = (name, timeout)
            events.setdefault("save_state_calls", []).append(name)

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            events.setdefault("save_weights", []).append((name, checkpoint_type))
            return SimpleNamespace(
                path=f"tinker://unit/sampler/{name}-session",
                snapshot_name=f"{name}-session",
            )

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return f"tinker://unit/state/{name}"

    class FakeWeightSyncer:
        def __init__(self, **kwargs):
            events["weight_syncer_init"] = kwargs

        def save_and_hotload(self, name, checkpoint_type="base"):
            events["weight_sync_saves"].append((name, checkpoint_type))

        def save_dcp(self, name):
            events["weight_sync_dcp"].append(name)

    class FakeRolloutProcessor:
        def __init__(self, **kwargs):
            events["rollout_processor_init"] = kwargs

        def __call__(self, rows, rollout_config):
            row_id = rows[0].input_metadata.row_id
            events.setdefault("rollout_processor_calls", []).append(
                {"row_ids": [row.input_metadata.row_id for row in rows], "steps": rollout_config.steps}
            )

            async def _finish(row, reward):
                row.execution_metadata.extra = {
                    "token_turn_traces": [
                        {
                            "prompt_ids": [1, 2, 3],
                            "completion_ids": [10],
                            "completion_logprobs": [-0.1],
                        }
                    ],
                    "model_request_traces": [{}],
                    "step_rewards": [reward],
                }
                row.messages = []
                return row

            reward = 1.0 if row_id.endswith("_0") else 0.0
            return [_finish(rows[0], reward)]

    def fake_create_trainer_job(*args, **kwargs):
        events["trainer_jobs"].append(kwargs)
        display_name = kwargs["display_name"]
        job_id = "policy-job" if display_name == "frozen-lake-policy" else "reference-job"
        return SimpleNamespace(job_id=job_id)

    def fake_setup_deployment(_deploy_mgr, deploy_cfg, *_args, **_kwargs):
        events["deployment_shape"] = deploy_cfg.deployment_shape
        return SimpleNamespace(inference_model="accounts/test/models/deployed")

    def fake_build_loss_fn(args):
        events["build_loss_fn_args"] = args
        def _builder(adv, ref_lp, prompt_lens, inf_lp, prox_lp):
            events["build_loss_fn_calls"].append(
                {
                    "advantages": adv,
                    "ref_logprobs": ref_lp,
                    "prompt_lens": prompt_lens,
                    "inf_logprobs": inf_lp,
                    "prox_logprobs": prox_lp,
                }
            )
            return "loss-fn"

        return _builder

    readiness_status = iter([503, 200])

    monkeypatch.setattr(train_module, "parse_args", lambda: cfg)
    monkeypatch.setattr(train_module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "wandb_finish", lambda: events.__setitem__("wandb_finished", 1))
    monkeypatch.setattr(train_module, "wandb_log", lambda payload, step=0: events["wandb_logs"].append((step, payload)))
    monkeypatch.setattr(train_module, "log_metrics_json", lambda step, **kwargs: events.setdefault("metrics_logs", []).append((step, kwargs)))
    monkeypatch.setattr(
        train_module,
        "load_seed_contexts",
        lambda *args, **kwargs: [{"seed": 101, "map_name": "4x4", "use_random_map": True}],
    )
    monkeypatch.setattr(train_module, "TrainerJobManager", FakeTrainerJobManager)
    monkeypatch.setattr(train_module, "DeploymentManager", FakeDeploymentManager)
    monkeypatch.setattr(train_module, "setup_deployment", fake_setup_deployment)
    monkeypatch.setattr(train_module, "ThreadPoolExecutor", FakeThreadPoolExecutor)
    monkeypatch.setattr(train_module, "create_trainer_job", fake_create_trainer_job)
    monkeypatch.setattr(train_module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(train_module, "WeightSyncer", FakeWeightSyncer)
    monkeypatch.setattr(train_module, "FrozenLakeToolRolloutProcessor", FakeRolloutProcessor)
    monkeypatch.setattr(train_module, "build_loss_fn", fake_build_loss_fn)
    monkeypatch.setattr(train_module, "compute_step_metrics", lambda **kwargs: {
        "rollout/reward": 0.5,
        "train/mean_kl": 0.02,
        "train/mean_loss": 0.9,
        "train/mean_adv_loss": 0.4,
        "train/mean_kl_penalty": 0.1,
        "train/mask_ratio": 0.5,
        "train/inference_kld": 0.03,
    })
    monkeypatch.setattr(train_module, "flush_timing", lambda: {"perf/step_time": 1.0})
    monkeypatch.setattr(train_module, "compute_pp_recommendation", lambda *args, **kwargs: SimpleNamespace(recommended_prompts_per_step=3))
    _OrigInfraConfig = train_module.InfraConfig
    monkeypatch.setattr(
        train_module, "InfraConfig",
        lambda **kwargs: _OrigInfraConfig(ref_training_shape_id="ts-qwen3-4b-smoke-v1", **kwargs),
    )
    monkeypatch.setattr(train_module.time, "sleep", lambda seconds: events["sleeps"].append(seconds))
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: SimpleNamespace(status_code=next(readiness_status)),
    )

    train_module.main()

    assert events["resolved_shapes"] == ["ts-qwen3-4b-smoke-v1", "ts-qwen3-4b-smoke-v1"]
    assert events["deployment_shape"] == "shape/versions/7"
    assert [call["display_name"] for call in events["trainer_jobs"]] == [
        "frozen-lake-policy",
        "frozen-lake-reference",
    ]
    assert [call["row_ids"] for call in events["rollout_processor_calls"]] == [
        ["seed_101_0_0"],
        ["seed_101_0_1"],
    ]
    assert events["weight_sync_saves"] == [("step-0-base", "base"), ("step-1", "base")]
    assert events["weight_sync_dcp"] == []
    assert events["final_save"] == ("step-1", None)
    assert events["promotions"] == [
        ("policy-job", "step-1-session", "promoted-rl-model"),
    ]
    assert "fwd_bwd_call" in events
    assert events["build_loss_fn_args"].ratio_log_cap == 9.0
    # FrozenLakeConfig defaults to loss_path='client' (always-safe path).
    # Builtin kernels do not consume ref_logprobs and would silently drop
    # the KL penalty -- validate_loss_path guards against that explicitly,
    # but the default already matches what kl_beta>0 needs.
    assert cfg.loss_path == "client"
    assert cfg.kl_beta > 0
    assert events["fwd_bwd_call"]["loss_fn"] == "loss-fn"
    assert "loss_fn_config" not in events["fwd_bwd_call"]
    assert events["deleted_jobs"] == ["reference-job", "policy-job"]
    assert events["deleted_deployments"] == []
    assert events["wandb_finished"] == 1
