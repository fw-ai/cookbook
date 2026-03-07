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

        def save_state(self, name, timeout):
            events["saved_state"] = (name, timeout)

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
