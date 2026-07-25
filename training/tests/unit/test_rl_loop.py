from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import SimpleNamespace

import pytest
import tinker

import training.recipes.rl_loop as module
from training.utils.rl.losses import PromptGroup
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message, Renderer


class _StopAfterRenderer(RuntimeError):
    pass


class _StopAfterProvisioning(RuntimeError):
    pass


def test_config_has_no_runner_state() -> None:
    cfg = module.Config(log_path="gs://logs")

    assert not hasattr(cfg, "runner")
    assert not hasattr(module, "RunnerIO")
    assert "write_running_progress" not in inspect.getsource(module.main)
    assert cfg.kl_beta == 0.001


def test_config_excludes_async_and_advanced_sync_scheduling_knobs() -> None:
    cfg = module.Config(log_path="gs://logs")

    assert cfg.router_replay is True
    assert cfg.router_replay_completion_only is True
    for field_name in (
        "concurrency",
        "emit_grad_norm_metrics",
        "lora_alpha",
        "pipeline_chunks_per_step",
        "ppo_n_minibatches",
        "step_timeout",
        "trajectory_dir",
        "warm_start_from_adapter",
        "weight_sync_before_training",
        "weight_sync_interval",
    ):
        assert not hasattr(cfg, field_name)


def test_main_has_direct_client_grpo_customization_boundary() -> None:
    source = inspect.getsource(module.main)

    assert "make_grpo_loss_fn(" in source
    assert "policy.forward_backward_custom(" in source
    assert "raw_inf_logprobs=raw_inference_logprobs" in source
    assert "sampled_completion_to_rollout_run(" in source
    assert "rollout_to_prompt_group(" in source
    assert "To switch to built-in PPO or another loss" in source
    assert "skills/fireworks-training/references/rl-custom-loss.md" in source
    assert "build_loss_fn" not in source
    assert "loss_path" not in source


def test_main_is_one_strictly_synchronous_optimizer_batch_loop() -> None:
    source = inspect.getsource(module.main)

    assert "collect_prompt_groups(" in source
    assert source.count("policy.forward_backward_custom(") == 1
    assert source.count("policy.optim_step(") == 1
    assert "run_batched_training_loop" not in source
    assert "TrainStepFns" not in source
    assert "weight_sync_interval" not in source
    assert "pipeline_chunks_per_step" not in source


@pytest.mark.parametrize(
    "trainer",
    [
        module.TrainerConfig(reference_training_shape_id="ref-shape"),
        module.TrainerConfig(reference_job_id="ref-job"),
    ],
)
def test_main_rejects_unused_reference_trainer_config(trainer) -> None:
    cfg = module.Config(log_path="gs://logs", kl_beta=0, trainer=trainer)

    with pytest.raises(ValueError, match="require kl_beta > 0"):
        module.main(cfg)


@pytest.mark.parametrize(
    ("config_overrides", "message"),
    [
        ({"eps_clip": -0.1}, "eps_clip"),
        ({"eps_clip_high": -0.1}, "eps_clip"),
        ({"kl_beta": -0.1}, "kl_beta"),
    ],
)
def test_main_rejects_invalid_grpo_config(config_overrides, message) -> None:
    cfg = module.Config(log_path="gs://logs", **config_overrides)

    with pytest.raises(ValueError, match=message):
        module.main(cfg)


def test_response_text_for_grading_uses_renderer_parse_response():
    class _Renderer:
        def parse_response(self, tokens):
            assert tokens == [10, 11, 12]
            return Message(role="assistant", content="<answer>42</answer>"), True

    sampled = SimpleNamespace(
        full_tokens=[1, 2, 3, 10, 11, 12],
        prompt_len=3,
        text="raw completion with reasoning",
    )
    assert (
        module._response_text_for_grading(_Renderer(), sampled) == "<answer>42</answer>"
    )


def test_response_text_for_grading_does_not_fallback_on_parse_failure():
    class _Renderer:
        def parse_response(self, tokens):
            raise ValueError("parse failed")

    sampled = SimpleNamespace(full_tokens=[1, 2, 3], prompt_len=1, text="fallback text")
    with pytest.raises(ValueError, match="parse failed"):
        module._response_text_for_grading(_Renderer(), sampled)


def test_all_cookbook_renderer_classes_implement_parse_response():
    import training.renderer

    assert callable(get_text_content)
    get_text_content(Message(role="assistant", content="hello"))

    missing: list[str] = []
    for modinfo in pkgutil.walk_packages(
        training.renderer.__path__, training.renderer.__name__ + "."
    ):
        if ".verifier" in modinfo.name:
            continue
        mod = importlib.import_module(modinfo.name)
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if not issubclass(obj, Renderer) or obj is Renderer:
                continue
            if inspect.isabstract(obj):
                continue
            if not callable(getattr(obj, "parse_response", None)):
                missing.append(f"{modinfo.name}.{obj.__name__}")

    assert missing == []


def test_extract_answer_reads_digits_from_answer_block():
    assert module.extract_answer("<answer> 42 apples </answer>") == "42"
    assert module.extract_answer("no answer block") is None


def test_reward_fn_requires_matching_numeric_answer():
    assert (
        module.reward_fn("<answer>7</answer>", {"ground_truth": "<answer>7</answer>"})
        == 1.0
    )
    assert (
        module.reward_fn("<answer>8</answer>", {"ground_truth": "<answer>7</answer>"})
        == 0.0
    )
    assert module.reward_fn("missing", {"ground_truth": "<answer>7</answer>"}) == 0.0


def test_should_accept_requires_reward_variance():
    same_rewards = PromptGroup(
        data=[], advantages=[], ref_logprobs=[], prompt_len=0, rewards=[0.0, 0.0]
    )
    varied_rewards = PromptGroup(
        data=[], advantages=[], ref_logprobs=[], prompt_len=0, rewards=[0.0, 1.0]
    )

    assert module.should_accept(same_rewards) is False
    assert module.should_accept(varied_rewards) is True


def test_main_requires_deployment_tokenizer_model(monkeypatch):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        deployment=module.DeployConfig(tokenizer_model=""),
    )

    with pytest.raises(ValueError, match="deployment.tokenizer_model"):
        module.main(cfg)


async def _external_sample_prompt_fn(_row, *, cursor_index: int):
    return None


def _build_service_kwargs(monkeypatch, cfg, *, sample_prompt_fn=None):
    calls = []

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "validate_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module, "resolve_router_replay_enabled", lambda **kwargs: kwargs["requested"]
    )
    monkeypatch.setattr(
        module, "load_deployment_tokenizer", lambda *args, **kwargs: object()
    )

    def fake_build_service_client(**kwargs):
        calls.append(kwargs)
        raise _StopAfterProvisioning

    monkeypatch.setattr(module, "build_service_client", fake_build_service_client)

    with pytest.raises(_StopAfterProvisioning):
        module.main(cfg, sample_prompt_fn=sample_prompt_fn)

    assert len(calls) == 1
    return calls[0]


def test_main_requests_cleanup_for_sdk_created_resources(monkeypatch):
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        deployment=module.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is True
    assert (
        kwargs["cleanup_deployment_on_close"]
        == module.CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO
    )


def test_main_can_disable_cleanup_on_exit(monkeypatch):
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        cleanup_on_exit=False,
        deployment=module.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is False
    assert kwargs["cleanup_deployment_on_close"] is None


def test_main_uses_sdk_default_lora_alpha(monkeypatch):
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        lora_rank=64,
        deployment=module.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["lora_rank"] == 64
    assert "lora_alpha" not in kwargs


def test_main_delegates_trainer_cleanup_for_existing_id_to_sdk(monkeypatch):
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        trainer=module.TrainerConfig(job_id="existing-job"),
        deployment=module.DeployConfig(
            deployment_id="requested-rollout-id",
            tokenizer_model="Qwen/Qwen3-1.7B",
        ),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is True
    assert (
        kwargs["cleanup_deployment_on_close"]
        == module.CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO
    )


def test_main_requests_trainer_cleanup_for_empty_job_id(monkeypatch):
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        trainer=module.TrainerConfig(job_id=""),
        deployment=module.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is True


def test_main_sample_prompt_fn_defaults_to_rollout_deployment(monkeypatch):
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        deployment=module.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(
        monkeypatch,
        cfg,
        sample_prompt_fn=_external_sample_prompt_fn,
    )

    assert kwargs["deployment"] is cfg.deployment
    assert (
        kwargs["cleanup_deployment_on_close"]
        == module.CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO
    )


def test_main_collects_trains_and_hotloads_before_next_batch(monkeypatch) -> None:
    events: list[str] = []

    class _Policy:
        def save_weights_for_sampler(self, name, checkpoint_type=None):
            events.append(f"save:{name}:{checkpoint_type}")
            return SimpleNamespace(path=f"/{name}")

        def forward(self, data, _loss):
            events.append(f"old-policy:{len(data)}")
            return SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": SimpleNamespace(data=[-0.1, -0.2])}
                    for _ in data
                ]
            )

        def forward_backward_custom(self, data, _loss_fn):
            events.append(f"fwd-bwd:{len(data)}")
            return SimpleNamespace(metrics={})

        def optim_step(self, _params, **_kwargs):
            events.append("optim")
            return SimpleNamespace(metrics={})

    policy = _Policy()

    class _Service:
        trainer_job_id = "trainer"
        reference_client_job_id = None
        reference_trainer_job_id = None
        deployment_id = "deployment"
        max_context_length = 4096
        accelerator_type = "GPU"
        accelerator_count = 1

        def create_training_client(self, *_args, **_kwargs):
            return policy

        def hotload_sampler_snapshot(self, path):
            events.append(f"hotload:{path}")

        def close(self):
            events.append("close")

    class _Checkpoints:
        def __init__(self, *_args, **_kwargs):
            pass

        def resume(self, **_kwargs):
            return None

        def save(self, name, **_kwargs):
            events.append(f"checkpoint:{name}")

    async def sample_prompt(_row, *, cursor_index: int):
        events.append(f"sample:{cursor_index}")

        def datum():
            return tinker.Datum(
                model_input=tinker.ModelInput.from_ints([1, 2]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=[2, 3],
                        dtype="int64",
                        shape=[2],
                    )
                },
            )

        return PromptGroup(
            data=[datum(), datum()],
            advantages=[-1.0, 1.0],
            ref_logprobs=None,
            prompt_len=1,
            rewards=[0.0, 1.0],
            inf_logprobs=[[-0.1, -0.2], [-0.1, -0.2]],
            completion_lens=[1, 1],
            truncated=[False, False],
        )

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(module, "setup_wandb", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "validate_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "load_deployment_tokenizer",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "build_service_client",
        lambda **_kwargs: _Service(),
    )
    monkeypatch.setattr(
        module,
        "ReconnectableClient",
        SimpleNamespace(
            from_training_client=lambda client, **_kwargs: client,
        ),
    )
    monkeypatch.setattr(module, "TrainingCheckpoints", _Checkpoints)
    monkeypatch.setattr(module, "log_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "log_metrics_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "should_accept", lambda _group: True)

    result = module.main(
        module.Config(
            log_path="/tmp/rl-test",
            kl_beta=0,
            completions_per_prompt=2,
            prompt_groups_per_step=2,
            save_final_checkpoint=False,
            deployment=module.DeployConfig(tokenizer_model="tokenizer"),
        ),
        sample_prompt_fn=sample_prompt,
        rows=[{"id": 0}, {"id": 1}, {"id": 2}],
    )

    assert result["steps"] == 2
    assert events.index("hotload:/step-0") < events.index("sample:0")
    assert events.index("sample:1") < events.index("optim")
    assert events.index("hotload:/step-1") < events.index("sample:2")
    assert events.count("optim") == 2
    assert events.count("hotload:/step-1") == 1
    assert events.count("hotload:/step-2") == 1


def test_main_passes_renderer_name_to_rollout_renderer(monkeypatch):
    """cfg.renderer_name must reach build_renderer, not just exist on Config.

    The shape-validation lane pins renderer_name on its glm_moe_dsa shapes; if
    the recipe accepts the field but never forwards it, those runs silently roll
    out with a tokenizer-inferred renderer instead and nothing fails loudly.
    """
    captured: dict[str, str] = {}

    class _Service:
        trainer_job_id = "trainer"
        reference_client_job_id = None
        max_context_length = 4096

        def create_training_client(self, *_args, **_kwargs):
            return object()

        def create_deployment_sampler(self, *_args, **_kwargs):
            return object()

        def close(self):
            pass

    def fake_build_renderer(_tokenizer, tokenizer_model, renderer_name):
        captured["tokenizer_model"] = tokenizer_model
        captured["renderer_name"] = renderer_name
        raise _StopAfterRenderer

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "validate_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "resolve_router_replay_enabled", lambda **_kwargs: False)
    monkeypatch.setattr(module, "load_deployment_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "build_service_client", lambda **_kwargs: _Service())
    monkeypatch.setattr(
        module,
        "ReconnectableClient",
        SimpleNamespace(from_training_client=lambda client, **_kwargs: client),
    )
    monkeypatch.setattr(module, "build_renderer", fake_build_renderer)

    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        kl_beta=0,
        renderer_name="glm_moe_dsa",
        deployment=module.DeployConfig(tokenizer_model="Qwen/Qwen3.5-9B"),
    )

    with pytest.raises(_StopAfterRenderer):
        module.main(cfg)

    assert captured == {
        "tokenizer_model": "Qwen/Qwen3.5-9B",
        "renderer_name": "glm_moe_dsa",
    }
