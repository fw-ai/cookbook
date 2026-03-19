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
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        deployment=module.DeployConfig(tokenizer_model=""),
    )

    with pytest.raises(ValueError, match="deployment.tokenizer_model"):
        module.main(cfg)


def test_main_bootstraps_without_reference_and_cleans_up(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
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
        log_path="/tmp/rl_test_logs",
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
    assert events["deleted_jobs"] == ["policy-job"]
    assert events["scaled_deployments"] == ["dep-123"]
    assert events["wandb_finished"] == 0


def test_main_raises_when_builtin_loss_with_pp(monkeypatch):
    """Builtin policy_loss + PP>1 should raise immediately."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    class FakeRlorMgr:
        def resolve_training_profile(self, shape_id):
            return SimpleNamespace(
                deployment_shape="dep-shape",
                deployment_shape_version=None,
                pipeline_parallelism=4,
                max_supported_context_length=128,
            )

        def delete(self, job_id):
            pass

    class FakeDeployMgr:
        inference_url = "https://inference.unit.test"
        boot_time_s = 1.0

        def scale_to_zero(self, deployment_id):
            pass

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "setup_deployment", lambda *args, **kwargs: SimpleNamespace(inference_model="m"))
    monkeypatch.setattr(module, "create_trainer_job", lambda *args, **kwargs: SimpleNamespace(job_id="j"))
    monkeypatch.setattr(module, "ReconnectableClient", lambda *a, **kw: SimpleNamespace(inner=object()))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(module, "DeploymentSampler", lambda **kw: None)
    monkeypatch.setattr(module, "WeightSyncer", lambda **kw: None)
    monkeypatch.setattr(module, "build_loss_fn", lambda **kw: None)
    monkeypatch.setattr(module, "load_jsonl_dataset", lambda *a, **kw: [])
    from training.utils.checkpoint_utils import ResumeInfo
    monkeypatch.setattr(module, "resolve_resume", lambda *a, **kw: ResumeInfo(step=0))

    cfg = module.Config(
        log_path="/tmp/pp_test",
        base_model="accounts/test/models/m",
        dataset="/tmp/d.jsonl",
        policy_loss="grpo",
        deployment=module.DeployConfig(deployment_id="dep", tokenizer_model="T"),
        infra=module.InfraConfig(training_shape_id="shape-pp4"),
    )

    with pytest.raises(ValueError, match="Pipeline parallelism.*PP=4.*not supported"):
        module.main(cfg, rlor_mgr=FakeRlorMgr(), deploy_mgr=FakeDeployMgr())


def test_main_runs_sampling_and_training_with_reference(monkeypatch, tmp_path):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
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
        "promotions": [],
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
                pipeline_parallelism=1,
                max_supported_context_length=96,
            )

        def delete(self, job_id):
            events["deleted_jobs"].append(job_id)

        def promote_checkpoint(self, job_id, checkpoint_id, output_model_id):
            events["promotions"].append((job_id, checkpoint_id, output_model_id))

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

        def forward_backward(self, data, loss_fn="cross_entropy", loss_fn_config=None):
            events["fwd_bwd_call"] = {"data": data, "loss_fn": loss_fn, "loss_fn_config": loss_fn_config}
            return SimpleNamespace(metrics={"loss": 1.0})

        def forward_backward_custom(self, data, loss_fn):
            events["fwd_bwd_call"] = {"data": data, "loss_fn": loss_fn}
            return SimpleNamespace(metrics={"loss": 1.0})

        def optim_step(self, _params, **kwargs):
            events["optim_step_called"] = True
            return SimpleNamespace(metrics={"optimizer/lr": 1e-4})

        def save_state(self, name, timeout=None):
            events["saved_state"] = (name, timeout)
            return SimpleNamespace(path=f"tinker://unit/state/{name}")

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(
                path=f"tinker://unit/sampler/{name}-session",
                snapshot_name=f"{name}-session",
            )

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

        async def sample_with_tokens(self, **kwargs):
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
        step, metrics = kwargs["train_fns"].train_step(
            1,
            [pg],
            {
                "valid_prompt_groups": 1,
                "total_sampled": 1,
                "filter_drops": 0,
                "sample_fails": 0,
                "sample_wait_time": 0.1,
                "step_wall_time": 0.2,
                "all_raw_rewards": list(pg.rewards),
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
        events["build_loss_fn_kwargs"] = kwargs
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
        log_path=str(tmp_path / "rl_logs"),
        base_model="accounts/test/models/qwen3-4b",
        dataset="/tmp/prompts.jsonl",
        kl_beta=0.1,
        ratio_log_cap=13.0,
        completions_per_prompt=2,
        prompt_groups_per_step=1,
        router_replay=True,
        trajectory_dir=str(tmp_path),
        weight_sync=module.WeightSyncConfig(weight_sync_interval=1, dcp_save_interval=1, weight_sync_before_training=True),
        deployment=module.DeployConfig(
            deployment_id="dep-123",
            tokenizer_model="Qwen/Qwen3-4B",
        ),
        infra=module.InfraConfig(training_shape_id="shape-a", ref_training_shape_id="ref-shape"),
        output_model_id="promoted-rl-model",
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
    weight_sync_names = [name for name, _ in events["weight_sync_saves"]]
    assert "step-2" in weight_sync_names
    assert events.get("saved_state") == ("step-2", None)
    assert events["promotions"] == [
        ("policy-job", "step-2-session", "promoted-rl-model"),
    ]
    assert "fwd_bwd_call" in events
    assert events["build_loss_fn_kwargs"]["ratio_log_cap"] == 13.0
    from training.utils.rl.losses import get_builtin_loss_config
    expected_kernel, expected_config = get_builtin_loss_config(
        cfg.policy_loss,
        ratio_log_cap=cfg.ratio_log_cap,
        eps_clip=cfg.eps_clip,
        eps_clip_high=cfg.eps_clip_high,
    )
    assert events["fwd_bwd_call"]["loss_fn"] == expected_kernel
    assert events["fwd_bwd_call"]["loss_fn_config"] == expected_config
    assert events["deleted_jobs"] == ["reference-job", "policy-job"]
    assert events["scaled_deployments"] == ["dep-123"]


def test_custom_policy_loss_falls_back_to_two_pass(monkeypatch, tmp_path):
    """When policy_loss is not in the builtin registry, fwd_bwd_one should
    call forward_backward_custom (two-pass) and invoke the loss builder."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "fwd_bwd_method": None,
        "build_loss_fn_calls": [],
        "deleted_jobs": [],
        "scaled_deployments": [],
        "wandb_logs": [],
        "weight_sync_saves": [],
        "weight_sync_dcp": [],
    }

    class FakeRlorMgr:
        def resolve_training_profile(self, shape_id):
            return SimpleNamespace(
                deployment_shape="dep-shape-v2",
                deployment_shape_version=None,
                pipeline_parallelism=1,
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
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
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
                    {"logprobs": SimpleNamespace(data=[-0.21, -0.22])}
                    for _ in data
                ]
            )

        def forward_backward(self, data, loss_fn="cross_entropy", loss_fn_config=None):
            events["fwd_bwd_method"] = "forward_backward"
            return SimpleNamespace(metrics={"loss": 1.0})

        def forward_backward_custom(self, data, loss_fn):
            events["fwd_bwd_method"] = "forward_backward_custom"
            return SimpleNamespace(metrics={"loss": 1.0})

        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={"optimizer/lr": 1e-4})

        def save_state(self, name, timeout=None):
            return SimpleNamespace(path=name)

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(path=name)

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return name

    class FakeWeightSyncer:
        def __init__(self, **kwargs):
            pass

        def save_and_hotload(self, name, checkpoint_type="base"):
            events["weight_sync_saves"].append(name)

        def save_dcp(self, name):
            events["weight_sync_dcp"].append(name)

    class FakeSampler:
        def __init__(self, **kwargs):
            pass

        async def sample_with_tokens(self, **kwargs):
            return [
                SimpleNamespace(
                    text="<answer>7</answer>",
                    full_tokens=[10, 11, 12, 13],
                    prompt_len=2,
                    inference_logprobs=[-0.5, -0.6],
                    logprobs_echoed=False,
                    finish_reason="stop",
                    routing_matrices=None,
                ),
                SimpleNamespace(
                    text="<answer>8</answer>",
                    full_tokens=[20, 21, 22, 23],
                    prompt_len=2,
                    inference_logprobs=[-0.7, -0.8],
                    logprobs_echoed=False,
                    finish_reason="length",
                    routing_matrices=None,
                ),
            ]

    async def fake_run_rl_loop(**kwargs):
        sample_iter = iter(kwargs["sample_fns"])
        pg = await next(sample_iter)
        step, _ = kwargs["train_fns"].train_step(0, [pg])
        return step

    def fake_create_trainer_job(*args, **kwargs):
        return SimpleNamespace(job_id="policy-job")

    def fake_build_loss_fn(**kwargs):
        events["build_loss_fn_kwargs"] = kwargs
        def _builder(adv, ref_lp, prompt_lens, inf_lp, prox_lp):
            events["build_loss_fn_calls"].append(True)
            return "custom-loss-fn"

        return _builder

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_metrics_json", lambda *args, **kwargs: None)
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
        {"messages": [{"role": "user", "content": "Solve"}], "ground_truth": "<answer>7</answer>"}
    ])
    monkeypatch.setattr(module, "build_loss_fn", fake_build_loss_fn)
    monkeypatch.setattr(module, "run_rl_loop", fake_run_rl_loop)
    monkeypatch.setattr(module, "compute_step_metrics", lambda **kwargs: {"rollout/reward": 0.5, "rollout/accuracy": 0.5, "train/mean_kl": 0.02})
    monkeypatch.setattr(module, "flush_timing", lambda: {})
    monkeypatch.setattr(
        module.tinker.ModelInput, "from_ints",
        lambda ints, routing_matrices=None: SimpleNamespace(tokens=list(ints), routing_matrices=routing_matrices),
    )
    monkeypatch.setattr(module.tinker, "TensorData", lambda data, dtype, shape: SimpleNamespace(data=data, dtype=dtype, shape=shape))
    monkeypatch.setattr(module.tinker, "Datum", lambda model_input, loss_fn_inputs: SimpleNamespace(model_input=model_input, loss_fn_inputs=loss_fn_inputs))

    cfg = module.Config(
        log_path=str(tmp_path / "rl_logs"),
        base_model="accounts/test/models/qwen3-4b",
        dataset="/tmp/prompts.jsonl",
        policy_loss="my_custom_loss",
        ratio_log_cap=17.0,
        completions_per_prompt=2,
        prompt_groups_per_step=1,
        deployment=module.DeployConfig(deployment_id="dep-123", tokenizer_model="Qwen/Qwen3-4B"),
        infra=module.InfraConfig(training_shape_id="shape-a"),
    )

    module.main(cfg, rlor_mgr=FakeRlorMgr(), deploy_mgr=FakeDeployMgr())

    assert events["fwd_bwd_method"] == "forward_backward_custom", (
        "Custom policy_loss not in builtin registry should use two-pass forward_backward_custom"
    )
    assert events["build_loss_fn_kwargs"]["ratio_log_cap"] == 17.0
    assert len(events["build_loss_fn_calls"]) == 1, (
        "Two-pass path should invoke the loss builder from build_loss_fn"
    )


# ---------------------------------------------------------------------------
# Async rollout tests
# ---------------------------------------------------------------------------


def test_async_rollout_config_defaults():
    cfg = module.Config(log_path="/tmp/test")
    assert cfg.async_rollout is False
    assert cfg.valid_prompt_groups_per_step is None
    assert cfg.max_head_offpolicy_versions == 2


def test_async_rollout_injects_hot_load_flag(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {"wandb_logs": []}

    class FakeRlorMgr:
        def resolve_training_profile(self, shape_id):
            return SimpleNamespace(
                deployment_shape="dep-shape-v1",
                pipeline_parallelism=1,
                max_supported_context_length=128,
            )

        def delete(self, job_id):
            pass

    class FakeDeployMgr:
        inference_url = "https://inference.unit.test"
        boot_time_s = 1.0

        def scale_to_zero(self, deployment_id):
            pass

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.inner = object()
            self.job_id = "policy-job"

        def forward(self, data, loss_fn):
            return SimpleNamespace(
                loss_fn_outputs=[{"logprobs": SimpleNamespace(data=[-0.1])} for _ in data]
            )

        def forward_backward_custom(self, data, loss_fn):
            return SimpleNamespace(metrics={"loss": 0.5})

        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={"optimizer/lr": 1e-4})

        def save_state(self, name, timeout=None):
            return SimpleNamespace(path=name)

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(path=name)

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return name

    class FakeWeightSyncer:
        def __init__(self, **kwargs):
            self.saves = []

        def save_and_hotload(self, name, checkpoint_type="base"):
            self.saves.append(name)

    class FakeSampler:
        def __init__(self, **kwargs):
            pass

        async def sample_with_tokens(self, **kwargs):
            return [
                SimpleNamespace(
                    text="<answer>7</answer>",
                    full_tokens=[10, 11, 12, 13],
                    prompt_len=2,
                    inference_logprobs=[-0.5, -0.6],
                    logprobs_echoed=False,
                    finish_reason="stop",
                    routing_matrices=None,
                ),
                SimpleNamespace(
                    text="<answer>8</answer>",
                    full_tokens=[20, 21, 22, 23],
                    prompt_len=2,
                    inference_logprobs=[-0.7, -0.8],
                    logprobs_echoed=False,
                    finish_reason="length",
                    routing_matrices=None,
                ),
            ]

    monkeypatch.setattr(module, "setup_wandb", lambda *a, **kw: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda payload, step=0: events["wandb_logs"].append((step, payload)))
    monkeypatch.setattr(module, "log_metrics_json", lambda *a, **kw: None)
    monkeypatch.setattr(module, "setup_deployment", lambda *a, **kw: SimpleNamespace(inference_model="m"))
    monkeypatch.setattr(module, "create_trainer_job", lambda *a, **kw: SimpleNamespace(job_id="policy-job"))
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(module, "DeploymentSampler", FakeSampler)
    monkeypatch.setattr(module, "WeightSyncer", FakeWeightSyncer)
    monkeypatch.setattr(module, "build_loss_fn", lambda **kw: lambda *a: "loss")
    monkeypatch.setattr(module, "load_jsonl_dataset", lambda *a, **kw: [
        {"messages": [{"role": "user", "content": "Q1"}], "ground_truth": "<answer>7</answer>"},
        {"messages": [{"role": "user", "content": "Q2"}], "ground_truth": "<answer>8</answer>"},
    ])
    from training.utils.checkpoint_utils import ResumeInfo
    monkeypatch.setattr(module, "resolve_resume", lambda *a, **kw: ResumeInfo(step=0))
    monkeypatch.setattr(module, "compute_step_metrics", lambda **kw: {
        "rollout/reward": 0.5, "rollout/accuracy": 0.5, "train/mean_kl": 0.01,
    })
    monkeypatch.setattr(module, "flush_timing", lambda: {})
    monkeypatch.setattr(
        module.tinker.ModelInput, "from_ints",
        lambda ints, routing_matrices=None: SimpleNamespace(tokens=list(ints)),
    )
    monkeypatch.setattr(module.tinker, "TensorData", lambda data, dtype, shape: SimpleNamespace(data=data, dtype=dtype, shape=shape))
    monkeypatch.setattr(module.tinker, "Datum", lambda model_input, loss_fn_inputs: SimpleNamespace(model_input=model_input, loss_fn_inputs=loss_fn_inputs))

    cfg = module.Config(
        log_path="/tmp/async_test",
        base_model="accounts/test/models/m",
        dataset="/tmp/d.jsonl",
        completions_per_prompt=2,
        prompt_groups_per_step=1,
        policy_loss="test_custom_loss",
        async_rollout=True,
        max_head_offpolicy_versions=2,
        deployment=module.DeployConfig(deployment_id="dep-1", tokenizer_model="T"),
        infra=module.InfraConfig(training_shape_id="shape-a"),
    )

    assert cfg.deployment.deployment_extra_args is None
    module.main(cfg, rlor_mgr=FakeRlorMgr(), deploy_mgr=FakeDeployMgr())

    # deployment_shape is set by resolve_training_profile, so auto-injection
    # of --hot-load-async-transition is skipped (shape owns the args)
    assert cfg.deployment.deployment_extra_args is None

    step_metrics = [m for s, m in events["wandb_logs"] if isinstance(m, dict) and "async/version" in m]
    assert len(step_metrics) >= 1, "Async loop should log async/version metric"


def test_sync_default_still_routes_to_run_rl_loop(monkeypatch):
    """async_rollout=False (default) should use run_rl_loop, not the async path."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {"sync_called": False}

    class FakeRlorMgr:
        def resolve_training_profile(self, shape_id):
            return SimpleNamespace(
                deployment_shape="dep-shape", pipeline_parallelism=1,
                max_supported_context_length=128,
            )

        def delete(self, job_id):
            pass

    class FakeDeployMgr:
        inference_url = "https://i"
        boot_time_s = 0.0

        def scale_to_zero(self, d):
            pass

    async def fake_run_rl_loop(**kwargs):
        events["sync_called"] = True
        return 0

    monkeypatch.setattr(module, "setup_wandb", lambda *a, **kw: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *a, **kw: None)
    monkeypatch.setattr(module, "setup_deployment", lambda *a, **kw: SimpleNamespace(inference_model="m"))
    monkeypatch.setattr(module, "create_trainer_job", lambda *a, **kw: SimpleNamespace(job_id="j"))
    monkeypatch.setattr(module, "ReconnectableClient", lambda *a, **kw: SimpleNamespace(inner=object()))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(module, "DeploymentSampler", lambda **kw: None)
    monkeypatch.setattr(module, "WeightSyncer", lambda **kw: None)
    monkeypatch.setattr(module, "build_loss_fn", lambda **kw: None)
    monkeypatch.setattr(module, "load_jsonl_dataset", lambda *a, **kw: [])
    monkeypatch.setattr(module, "run_rl_loop", fake_run_rl_loop)

    cfg = module.Config(
        log_path="/tmp/sync_test",
        async_rollout=False,
        deployment=module.DeployConfig(deployment_id="dep", tokenizer_model="T"),
        infra=module.InfraConfig(training_shape_id="shape"),
    )

    module.main(cfg, rlor_mgr=FakeRlorMgr(), deploy_mgr=FakeDeployMgr())
    assert events["sync_called"] is True


def test_async_rollout_validates_step_target(monkeypatch):
    monkeypatch.setattr(module, "setup_wandb", lambda *a, **kw: None)
    monkeypatch.setattr(module, "validate_config", lambda *a, **kw: None)

    cfg = module.Config(
        log_path="/tmp/test",
        async_rollout=True,
        valid_prompt_groups_per_step=0,
        deployment=module.DeployConfig(tokenizer_model="T"),
    )
    with pytest.raises(ValueError, match="valid_prompt_groups_per_step must be >= 1"):
        module.main(cfg)


def test_async_rollout_validates_max_offpolicy(monkeypatch):
    monkeypatch.setattr(module, "setup_wandb", lambda *a, **kw: None)
    monkeypatch.setattr(module, "validate_config", lambda *a, **kw: None)

    cfg = module.Config(
        log_path="/tmp/test",
        async_rollout=True,
        max_head_offpolicy_versions=-1,
        deployment=module.DeployConfig(tokenizer_model="T"),
    )
    with pytest.raises(ValueError, match="max_head_offpolicy_versions must be >= 0"):
        module.main(cfg)
