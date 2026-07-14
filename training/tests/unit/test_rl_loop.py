from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from types import SimpleNamespace

import pytest

import training.recipes.rl_loop as module
from training.utils.rl.losses import PromptGroup
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message, Renderer


class _StopAfterProvisioning(RuntimeError):
    pass


def test_build_adam_params_threads_grad_norm_telemetry_opt_in() -> None:
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        emit_grad_norm_metrics="basic",
    )

    params = module._build_adam_params(cfg)

    assert params.emit_grad_norm_metrics == "basic"
    assert params.learning_rate == cfg.learning_rate


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
    assert module._response_text_for_grading(_Renderer(), sampled) == "<answer>42</answer>"


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


async def _external_sample_prompt_fn(_row, *, cursor_index: int):
    return None


def _build_service_kwargs(monkeypatch, cfg, *, sample_prompt_fn=None):
    calls = []

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "validate_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "validate_warm_start_config", lambda *args, **kwargs: None)

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


def test_main_forwards_lora_alpha_to_service_client(monkeypatch):
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        lora_rank=64,
        lora_alpha=128,
        deployment=module.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["lora_rank"] == 64
    assert kwargs["lora_alpha"] == 128


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
