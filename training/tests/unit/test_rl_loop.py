from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import transformers

import training.recipes.rl_loop as module
from training.utils.rl.losses import PromptGroup


def test_extract_answer_reads_digits_from_answer_block():
    assert module.extract_answer("<answer> 42 apples </answer>") == "42"
    assert module.extract_answer("no answer block") is None


def test_reward_fn_requires_matching_numeric_answer():
    assert module.reward_fn("<answer>7</answer>", {"ground_truth": "<answer>7</answer>"}) == 1.0
    assert module.reward_fn("<answer>8</answer>", {"ground_truth": "<answer>7</answer>"}) == 0.0
    assert module.reward_fn("missing", {"ground_truth": "<answer>7</answer>"}) == 0.0


def test_should_accept_requires_reward_variance():
    same_rewards = PromptGroup(data=[], advantages=[], ref_logprobs=[], prompt_len=0, rewards=[0.0, 0.0])
    varied_rewards = PromptGroup(data=[], advantages=[], ref_logprobs=[], prompt_len=0, rewards=[0.0, 1.0])

    assert module.should_accept(same_rewards) is False
    assert module.should_accept(varied_rewards) is True


def test_dump_trajectory_writes_one_record_per_completion(tmp_path):
    prompt_groups = [
        PromptGroup(
            data=[],
            advantages=[0.5, -0.5],
            ref_logprobs=[],
            prompt_len=4,
            rewards=[1.0, 0.0],
            completion_lens=[3, 4],
            truncated=[False, True],
            prompt=[{"role": "user", "content": "Solve"}],
            completions=["<answer>1</answer>", "<answer>2</answer>"],
            row_meta={"ground_truth": "<answer>1</answer>"},
        )
    ]

    module._dump_trajectory(str(tmp_path), step=3, prompt_groups=prompt_groups)

    path = tmp_path / "step_0003.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert records == [
        {
            "step": 3,
            "prompt_group": 0,
            "completion_index": 0,
            "prompt": [{"role": "user", "content": "Solve"}],
            "completion": "<answer>1</answer>",
            "reward": 1.0,
            "advantage": 0.5,
            "completion_len": 3,
            "truncated": False,
            "ground_truth": "<answer>1</answer>",
        },
        {
            "step": 3,
            "prompt_group": 0,
            "completion_index": 1,
            "prompt": [{"role": "user", "content": "Solve"}],
            "completion": "<answer>2</answer>",
            "reward": 0.0,
            "advantage": -0.5,
            "completion_len": 4,
            "truncated": True,
            "ground_truth": "<answer>1</answer>",
        },
    ]


def test_main_requires_deployment_tokenizer_model(monkeypatch):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(
        dataset="/tmp/prompts.jsonl",
        deployment=module.DeployConfig(tokenizer_model=""),
    )

    with pytest.raises(ValueError, match="deployment.tokenizer_model"):
        module.main(cfg)


def test_main_bootstraps_without_reference_and_cleans_up(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "create_trainer_job": [],
        "wandb_logs": [],
        "deleted_jobs": [],
        "scaled_deployments": [],
        "wandb_finished": 0,
    }

    class FakeRlorMgr:
        def resolve_training_profile(self, shape_id):
            return SimpleNamespace(
                deployment_shape="dep-shape-v1",
                pipeline_parallelism=1,
                max_supported_context_length=128,
            )

        def delete(self, job_id):
            events["deleted_jobs"].append(job_id)

    class FakeDeployMgr:
        inference_url = "https://inference.unit.test"
        boot_time_s = 3.5

        def scale_to_zero(self, deployment_id):
            events["scaled_deployments"].append(deployment_id)

    class FakePolicyClient:
        def __init__(self, *args, **kwargs):
            self.inner = object()

        def save_state(self, name, timeout=None):
            events["saved_state"] = (name, timeout)
            return SimpleNamespace(path=f"tinker://unit/state/{name}")

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(path=f"tinker://unit/sampler/{name}")

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return f"tinker://unit/state/{name}"

    class FakeWeightSyncer:
        def __init__(self, **kwargs):
            events["weight_syncer_init"] = kwargs

    class FakeSampler:
        def __init__(self, **kwargs):
            events["sampler_init"] = kwargs

    async def fake_run_rl_loop(**kwargs):
        events["run_loop_kwargs"] = kwargs
        return 0

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: events.__setitem__("wandb_finished", 1))
    monkeypatch.setattr(module, "wandb_log", lambda payload, step=0: events["wandb_logs"].append((step, payload)))
    monkeypatch.setattr(module, "setup_deployment", lambda *args, **kwargs: SimpleNamespace(inference_model="accounts/test/models/deployed"))
    monkeypatch.setattr(
        module,
        "create_trainer_job",
        lambda *args, **kwargs: events["create_trainer_job"].append(kwargs) or SimpleNamespace(job_id="policy-job"),
    )
    monkeypatch.setattr(module, "ReconnectableClient", FakePolicyClient)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "DeploymentSampler", FakeSampler)
    monkeypatch.setattr(module, "WeightSyncer", FakeWeightSyncer)
    monkeypatch.setattr(module, "load_jsonl_dataset", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "build_loss_fn", lambda **kwargs: ("loss-builder", kwargs))
    monkeypatch.setattr(module, "run_rl_loop", fake_run_rl_loop)

    cfg = module.Config(
        base_model="accounts/test/models/qwen3-4b",
        dataset="/tmp/prompts.jsonl",
        kl_beta=0.0,
        deployment=module.DeployConfig(
            deployment_id="dep-123",
            tokenizer_model="Qwen/Qwen3-4B",
        ),
        infra=module.InfraConfig(training_shape_id="ts-qwen3-4b-smoke-v1"),
    )

    result = module.main(
        cfg,
        rlor_mgr=FakeRlorMgr(),
        deploy_mgr=FakeDeployMgr(),
        cleanup_on_exit=True,
    )

    assert result is None
    assert cfg.max_seq_len == 128
    assert cfg.deployment.deployment_shape == "dep-shape-v1"
    assert len(events["create_trainer_job"]) == 1
    assert events["create_trainer_job"][0]["display_name"] == "grpo-policy"
    assert events["create_trainer_job"][0]["hot_load_deployment_id"] == "dep-123"
    assert events["sampler_init"]["model"] == "accounts/test/models/deployed"
    assert events["weight_syncer_init"]["deployment_id"] == "dep-123"
    assert events["run_loop_kwargs"]["prompt_groups_per_step"] == cfg.prompt_groups_per_step
    assert events["run_loop_kwargs"]["completions_per_prompt"] == cfg.completions_per_prompt
    assert events["deleted_jobs"] == ["policy-job"]
    assert events["scaled_deployments"] == ["dep-123"]
    assert events["wandb_finished"] == 1


def test_main_runs_sampling_and_training_with_reference(monkeypatch, tmp_path):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "create_trainer_job": [],
        "deleted_jobs": [],
        "scaled_deployments": [],
        "wandb_logs": [],
        "weight_sync_saves": [],
        "weight_sync_dcp": [],
        "build_loss_fn_calls": [],
        "routing_matrix_calls": [],
        "sampler_calls": [],
    }

    class FakeRlorMgr:
        def resolve_training_profile(self, shape_id):
            if shape_id == "ref-shape":
                return SimpleNamespace(
                    deployment_shape=None,
                    deployment_shape_version=None,
                    pipeline_parallelism=1,
                    max_supported_context_length=96,
                )
            return SimpleNamespace(
                deployment_shape="dep-shape-v2",
                deployment_shape_version=None,
                pipeline_parallelism=2,
                max_supported_context_length=96,
            )

        def delete(self, job_id):
            events["deleted_jobs"].append(job_id)

    class FakeDeployMgr:
        inference_url = "https://inference.unit.test"
        boot_time_s = 1.5

        def scale_to_zero(self, deployment_id):
            events["scaled_deployments"].append(deployment_id)

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
            if self.job_id == "reference-job":
                return SimpleNamespace(
                    loss_fn_outputs=[
                        {"logprobs": SimpleNamespace(data=[-0.11, -0.12])}
                        for _ in data
                    ]
                )
            return SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": SimpleNamespace(data=[-0.21, -0.22])}
                    for _ in data
                ]
            )

        def forward_backward_custom(self, data, loss_fn):
            events["fwd_bwd_call"] = {"data": data, "loss_fn": loss_fn}
            return SimpleNamespace(metrics={"loss": 1.0})

        def optim_step(self, _params):
            events["optim_step_called"] = True
            return SimpleNamespace(metrics={"optimizer/lr": 1e-4})

        def save_state(self, name, timeout=None):
            events["saved_state"] = (name, timeout)
            return SimpleNamespace(path=f"tinker://unit/state/{name}")

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(path=f"tinker://unit/sampler/{name}")

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return f"tinker://unit/state/{name}"

    class FakeWeightSyncer:
        def __init__(self, **kwargs):
            events["weight_syncer_init"] = kwargs

        def save_and_hotload(self, name, checkpoint_type="base"):
            events["weight_sync_saves"].append((name, checkpoint_type))

        def save_dcp(self, name):
            events["weight_sync_dcp"].append(name)

    class FakeSampler:
        def __init__(self, **kwargs):
            events["sampler_init"] = kwargs

        def sample_with_tokens(self, **kwargs):
            events["sampler_calls"].append(kwargs)
            return [
                SimpleNamespace(
                    text="<answer>7</answer>",
                    full_tokens=[10, 11, 12, 13],
                    prompt_len=2,
                    inference_logprobs=[-0.5, -0.6],
                    logprobs_echoed=False,
                    finish_reason="stop",
                    routing_matrices=["rm-a"],
                ),
                SimpleNamespace(
                    text="<answer>8</answer>",
                    full_tokens=[20, 21, 22, 23],
                    prompt_len=2,
                    inference_logprobs=[-0.7, -0.8],
                    logprobs_echoed=False,
                    finish_reason="length",
                    routing_matrices=["rm-b"],
                ),
            ]

    async def fake_run_rl_loop(**kwargs):
        events["run_loop_kwargs"] = kwargs
        sample_iter = iter(kwargs["sample_fns"])
        pg = await next(sample_iter)
        assert pg is not None
        assert kwargs["dynamic_filter_fn"](pg) is True
        kwargs["minibatch_fns"].ref_forward_batch([pg])
        fwd_bwd_result = kwargs["minibatch_fns"].fwd_bwd_one([pg])
        step, metrics = kwargs["minibatch_fns"].finish_step(
            1,
            [pg],
            [fwd_bwd_result],
            1,
            {
                "valid_prompt_groups": 1,
                "total_sampled": 1,
                "filter_drops": 0,
                "sample_fails": 0,
                "sample_wait_time": 0.1,
                "step_wall_time": 0.2,
                "all_raw_rewards": list(pg.rewards),
                "fwd_bwd_group_counts": [1],
            },
        )
        kwargs["metrics_callback"]({"train/step": step, "rollout/sample_fail_count": 0})
        events["finish_metrics"] = metrics
        return step

    def fake_create_trainer_job(*args, **kwargs):
        events["create_trainer_job"].append(kwargs)
        display_name = kwargs["display_name"]
        job_id = "policy-job" if display_name == "grpo-policy" else "reference-job"
        return SimpleNamespace(job_id=job_id)

    def fake_build_loss_fn(**kwargs):
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

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: events.__setitem__("wandb_finished", 1))
    monkeypatch.setattr(module, "wandb_log", lambda payload, step=0: events["wandb_logs"].append((step, payload)))
    monkeypatch.setattr(module, "log_metrics_json", lambda step, **kwargs: events.setdefault("metrics_logs", []).append((step, kwargs)))
    monkeypatch.setattr(module, "setup_deployment", lambda *args, **kwargs: SimpleNamespace(inference_model="accounts/test/models/deployed"))
    monkeypatch.setattr(module, "ThreadPoolExecutor", FakeThreadPoolExecutor)
    monkeypatch.setattr(module, "create_trainer_job", fake_create_trainer_job)
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "DeploymentSampler", FakeSampler)
    monkeypatch.setattr(module, "WeightSyncer", FakeWeightSyncer)
    from training.utils.checkpoint_utils import ResumeInfo
    monkeypatch.setattr(module, "resolve_resume", lambda *args, **kwargs: ResumeInfo(step=0))
    monkeypatch.setattr(module, "load_jsonl_dataset", lambda *args, **kwargs: [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Solve"},
                {"role": "assistant", "content": "prior"},
            ],
            "ground_truth": "<answer>7</answer>",
        }
    ])
    monkeypatch.setattr(module, "build_loss_fn", fake_build_loss_fn)
    monkeypatch.setattr(module, "run_rl_loop", fake_run_rl_loop)
    monkeypatch.setattr(module, "compute_pp_recommendation", lambda *args, **kwargs: SimpleNamespace(recommended_prompts_per_step=3))
    monkeypatch.setattr(module, "compute_step_metrics", lambda **kwargs: {
        "rollout/reward": 0.5,
        "rollout/accuracy": 0.5,
        "train/mean_kl": 0.02,
    })
    monkeypatch.setattr(module, "flush_timing", lambda: {"perf/step_time": 1.0})
    monkeypatch.setattr(module, "build_r3_routing_matrices", lambda *args, **kwargs: events["routing_matrix_calls"].append((args, kwargs)) or {"rm": True})
    monkeypatch.setattr(
        module.tinker.ModelInput,
        "from_ints",
        lambda ints, routing_matrices=None: SimpleNamespace(tokens=list(ints), routing_matrices=routing_matrices),
    )
    monkeypatch.setattr(
        module.tinker,
        "TensorData",
        lambda data, dtype, shape: SimpleNamespace(data=data, dtype=dtype, shape=shape),
    )
    monkeypatch.setattr(
        module.tinker,
        "Datum",
        lambda model_input, loss_fn_inputs: SimpleNamespace(model_input=model_input, loss_fn_inputs=loss_fn_inputs),
    )

    cfg = module.Config(
        base_model="accounts/test/models/qwen3-4b",
        dataset="/tmp/prompts.jsonl",
        kl_beta=0.1,
        completions_per_prompt=2,
        prompt_groups_per_step=1,
        router_replay=True,
        trajectory_dir=str(tmp_path),
        hotload=module.HotloadConfig(hot_load_interval=1, dcp_save_interval=1, hot_load_before_training=True),
        deployment=module.DeployConfig(
            deployment_id="dep-123",
            tokenizer_model="Qwen/Qwen3-4B",
        ),
        infra=module.InfraConfig(training_shape_id="shape-a", ref_training_shape_id="ref-shape"),
    )

    result = module.main(
        cfg,
        rlor_mgr=FakeRlorMgr(),
        deploy_mgr=FakeDeployMgr(),
        cleanup_on_exit=True,
    )

    assert result == {
        "steps": 2,
        "policy_job_id": "policy-job",
        "reference_job_id": "reference-job",
    }
    assert cfg.max_seq_len == 96
    assert cfg.deployment.deployment_shape == "dep-shape-v2"
    assert [call["display_name"] for call in events["create_trainer_job"]] == [
        "grpo-policy",
        "grpo-reference",
    ]
    assert events["sampler_calls"][0]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Solve"},
    ]
    assert events["sampler_calls"][0]["include_routing_matrix"] is True
    assert len(events["routing_matrix_calls"]) == 2
    assert events["weight_sync_saves"][0] == ("step-0-base", "base")
    hotload_names = [name for name, _ in events["weight_sync_saves"]]
    assert "step-2" in hotload_names
    assert "step-2" in events["weight_sync_dcp"]
    assert len(events["build_loss_fn_calls"]) == 1
    advantages = events["build_loss_fn_calls"][0]["advantages"]
    assert len(advantages) == 2
    assert advantages[0] > 0
    assert advantages[1] < 0
    assert events["deleted_jobs"] == ["policy-job", "reference-job"]
    assert events["scaled_deployments"] == ["dep-123"]
