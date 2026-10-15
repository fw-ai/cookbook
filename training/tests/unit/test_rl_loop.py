from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import pkgutil
import signal
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import training.recipes.rl_loop as module
from fireworks.training import sdk
from fireworks.training.sdk.routing import RoutingReferences
from fireworks.training.sdk.sampling import SampledCompletion
from training.renderer import get_text_content
from training.renderer.base import Message, Renderer
from training.utils.rl.losses import PromptGroup
from training.utils.supervised import has_non_text_chunks
from training.utils.termination import TerminatedBySignal


class _StopAfterRenderer(RuntimeError):
    pass


class _StopAfterProvisioning(RuntimeError):
    pass


class _StopAfterSampling(RuntimeError):
    pass


def _grpo_result(rows: int) -> SimpleNamespace:
    """A built-in GRPO result whose trainer logprobs match the rollout anchor."""
    return SimpleNamespace(
        metrics={},
        loss_fn_outputs=[
            {"logprobs": sdk.TensorData(data=[-0.1, -0.2], dtype="float32", shape=[2])} for _ in range(rows)
        ],
    )


def test_config_excludes_async_and_advanced_sync_scheduling_knobs() -> None:
    cfg = module.Config(log_path="gs://logs")

    for field_name in (
        "anchor_logp",
        "grad_accumulation_normalization",
        "kl_beta",
        "router_replay",
        "router_replay_completion_only",
        "concurrency",
        "emit_grad_norm_metrics",
        "pipeline_chunks_per_step",
        "ppo_n_minibatches",
        "step_timeout",
        "trajectory_dir",
        "weight_sync_before_training",
        "weight_sync_interval",
    ):
        assert not hasattr(cfg, field_name)


def test_main_threads_checkpoint_resume_options(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _Service:
        trainer_job_id = "trainer"

        def create_training_client(self, *_args, **_kwargs):
            return SimpleNamespace()

        def close(self):
            pass

    class _Checkpoints:
        def __init__(self, *_args, **_kwargs):
            pass

        def resume(self, **kwargs):
            captured.update(kwargs)
            raise _StopAfterProvisioning

    _stub_provisioning_dependencies(monkeypatch)
    monkeypatch.setattr(module, "build_service_client", lambda **_kwargs: _Service())
    monkeypatch.setattr(
        module, "ReconnectableClient", SimpleNamespace(from_training_client=lambda client, **_kwargs: client)
    )
    monkeypatch.setattr(module, "TrainingCheckpoints", _Checkpoints)
    monkeypatch.setattr(module, "make_weight_sync", lambda *_args: lambda *_a, **_kw: None)

    cfg = module.Config(
        log_path="/tmp/rl-test",
        init_from_checkpoint="step-16",
        deployment=module.DeployConfig(tokenizer_model="tokenizer"),
    )
    with pytest.raises(_StopAfterProvisioning):
        module.main(cfg, rows=[{"id": 0}])

    assert captured == {"init_from_checkpoint": "step-16", "warm_start_from_adapter": None}


@pytest.mark.parametrize(
    ("config_overrides", "message"),
    [
        ({"eps_clip": -0.1}, "eps_clip"),
        ({"eps_clip_high": -0.1}, "eps_clip"),
    ],
)
def test_main_rejects_invalid_grpo_config(config_overrides, message) -> None:
    cfg = module.Config(log_path="gs://logs", **config_overrides)

    with pytest.raises(ValueError, match=message):
        module.main(cfg)


def test_all_cookbook_renderer_classes_implement_parse_response():
    import training.renderer

    assert callable(get_text_content)
    get_text_content(Message(role="assistant", content="hello"))

    missing: list[str] = []
    for modinfo in pkgutil.walk_packages(training.renderer.__path__, training.renderer.__name__ + "."):
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


def test_main_requires_deployment_tokenizer_model(monkeypatch):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        deployment=module.DeployConfig(tokenizer_model=""),
    )

    with pytest.raises(ValueError, match="deployment.tokenizer_model"):
        module.main(cfg)


def _stub_provisioning_dependencies(monkeypatch: Any) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "validate_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "resolve_router_replay_enabled", lambda **kwargs: kwargs["requested"])
    monkeypatch.setattr(module, "load_deployment_tokenizer", lambda *args, **kwargs: object())


def _build_service_kwargs(monkeypatch, cfg):
    calls = []
    _stub_provisioning_dependencies(monkeypatch)

    def fake_build_service_client(**kwargs):
        calls.append(kwargs)
        raise _StopAfterProvisioning

    monkeypatch.setattr(module, "build_service_client", fake_build_service_client)

    with pytest.raises(_StopAfterProvisioning):
        module.main(cfg)

    assert len(calls) == 1
    return calls[0]


def _lora_creation_kwargs(
    monkeypatch: Any,
    cfg: module.Config,
) -> tuple[dict[str, Any], tuple[tuple[Any, ...], dict[str, Any]]]:
    service_calls = []
    training_calls = []
    _stub_provisioning_dependencies(monkeypatch)

    class FakeService:
        def close(self) -> None:
            pass

        def create_training_client(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            training_calls.append((args, kwargs))
            raise _StopAfterProvisioning

    def fake_build_service_client(**kwargs: Any) -> FakeService:
        service_calls.append(kwargs)
        return FakeService()

    monkeypatch.setattr(module, "build_service_client", fake_build_service_client)

    with pytest.raises(_StopAfterProvisioning):
        module.main(cfg)

    assert len(service_calls) == 1
    assert len(training_calls) == 1
    return service_calls[0], training_calls[0]


def test_main_requests_cleanup_for_sdk_created_resources(monkeypatch):
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        deployment=module.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is True
    assert kwargs["cleanup_deployment_on_close"] == module.CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO


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


@pytest.mark.parametrize("lora_alpha", [32, 128, None])
def test_main_forwards_lora_alpha(monkeypatch, lora_alpha):
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        lora_rank=64,
        lora_alpha=lora_alpha,
        deployment=module.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    service_kwargs, (_, training_kwargs) = _lora_creation_kwargs(monkeypatch, cfg)

    assert service_kwargs["max_lora_rank"] == 64
    assert "lora_alpha" not in service_kwargs
    assert training_kwargs["lora_rank"] == 64
    assert training_kwargs["lora_alpha"] == lora_alpha


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
    assert kwargs["cleanup_deployment_on_close"] == module.CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO


def test_main_requests_trainer_cleanup_for_empty_job_id(monkeypatch):
    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        trainer=module.TrainerConfig(job_id=""),
        deployment=module.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is True


def _datum() -> sdk.Datum:
    return sdk.Datum(
        model_input=sdk.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": sdk.TensorData(data=[2, 3], dtype="int64", shape=[2])},
    )


def _group(**overrides: Any) -> PromptGroup:
    values = dict(
        data=[_datum(), _datum()],
        advantages=[-1.0, 1.0],
        prompt_len=1,
        rewards=[0.0, 1.0],
        inf_logprobs=[[-0.1, -0.2], [-0.1, -0.2]],
        completion_lens=[2, 4],
        truncated=[False, True],
    )
    values.update(overrides)
    return PromptGroup(**values)


class _Sampler:
    def __init__(self, name: str, events: list[str], records: list[Any] | None = None):
        self.name = name
        self.base_url = "https://api.example"
        self.model = f"model-{name}"
        self._events = events
        self._records = list(records or [])

    def drain_metrics(self):
        records, self._records = self._records, []
        return records

    def close(self):
        self._events.append(f"close:{self.name}")


def _run_loop(
    monkeypatch,
    *,
    policy: Any,
    events: list[str],
    rows: list[dict],
    records: list[Any] | None = None,
    logged: list[tuple[int, dict[str, Any]]] | None = None,
    group_for_row=None,
    checkpoint_fn=None,
    **main_kwargs: Any,
) -> dict[str, Any]:
    samplers = iter(["train", "eval"])

    class _Service:
        trainer_job_id = "trainer"
        deployment_id = "deployment"
        max_context_length = 4096
        accelerator_type = "GPU"
        accelerator_count = 1

        def create_training_client(self, *_args, **_kwargs):
            return policy

        def create_deployment_sampler(self, **_kwargs):
            return _Sampler(next(samplers), events, records)

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
            if checkpoint_fn is not None:
                checkpoint_fn()
            events.append(f"checkpoint:{name}")

    def factory(setup):
        events.append(f"factory:{setup.sampler.name}")

        async def rollout_fn(row, *, cursor_index=None, rollout_idx=None, evaluation=False):
            if evaluation:
                events.append(f"eval-sample:{setup.sampler.name}")
            elif rollout_idx == 0:
                events.append(f"sample:{cursor_index}")
            return row

        return rollout_fn

    def to_group(rollout, advantage_fn):
        row = rollout.runs[0]
        return group_for_row(row) if group_for_row else _group()

    _stub_provisioning_dependencies(monkeypatch)
    monkeypatch.setattr(module, "build_service_client", lambda **_kwargs: _Service())
    monkeypatch.setattr(
        module, "ReconnectableClient", SimpleNamespace(from_training_client=lambda client, **_kwargs: client)
    )
    monkeypatch.setattr(module, "TrainingCheckpoints", _Checkpoints)
    monkeypatch.setattr(module, "rollout_to_prompt_group", to_group)
    monkeypatch.setattr(
        module,
        "log_metrics",
        lambda metrics, *, step: (logged if logged is not None else []).append((step, dict(metrics))),
    )
    monkeypatch.setattr(module, "log_metrics_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda *_args, **_kwargs: None)
    main_kwargs.setdefault("dynamic_filter_fn", None)
    return module.main(
        module.Config(
            log_path="/tmp/rl-test",
            completions_per_prompt=2,
            prompt_groups_per_step=main_kwargs.pop("prompt_groups_per_step", 1),
            save_final_checkpoint=False,
            dcp_save_interval=main_kwargs.pop("dcp_save_interval", 0),
            deployment=module.DeployConfig(tokenizer_model="tokenizer"),
        ),
        rollout_fn_factory=factory,
        rows=rows,
        **main_kwargs,
    )


class _Policy:
    default_timeout = 3600

    def __init__(self, events: list[str]):
        self.events = events
        self.datums: list[Any] = []

    def save_weights_for_sampler(self, name, checkpoint_type=None):
        self.events.append(f"save:{name}:{checkpoint_type}")
        return SimpleNamespace(path=f"/{name}")

    def forward_backward(self, data, loss_fn, loss_fn_config=None):
        assert loss_fn == "ppo"
        assert loss_fn_config == {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
        self.datums.extend(data)
        self.events.append(f"fwd-bwd:{len(data)}")
        result = _grpo_result(len(data))
        result.metrics.update({"loss:sum": 1.0, "trainer/forward_backward_duration_s:sum": 0.5})
        return result

    def submit_forward_backward(self, data, loss_fn, loss_fn_config=None):
        result = self.forward_backward(data, loss_fn, loss_fn_config)

        async def result_async(timeout=None):
            return result

        return SimpleNamespace(result_async=result_async)

    def optim_step(self, _params, **kwargs):
        assert kwargs["grad_accumulation_normalization"] == module.objective.NORMALIZATION
        self.events.append("optim")
        return SimpleNamespace(metrics={"grad_norm": 3.0, "trainer/optim_step_duration_s:sum": 0.1})


def test_main_collects_trains_and_hotloads_before_next_batch(monkeypatch) -> None:
    events: list[str] = []
    policy = _Policy(events)

    result = _run_loop(
        monkeypatch,
        policy=policy,
        events=events,
        rows=[{"id": 0}, {"id": 1}, {"id": 2}],
        prompt_groups_per_step=2,
    )

    assert result == {"steps": 2, "policy_job_id": "trainer", "deployment_id": "deployment"}
    assert events.index("hotload:/step-0") < events.index("sample:0")
    assert events.index("sample:1") < events.index("optim")
    assert events.index("hotload:/step-1") < events.index("sample:2")
    assert events.count("optim") == 2
    assert "close:train" in events
    # Rollout logprobs are the anchor; no snapshot forward runs.
    for datum in policy.datums:
        assert datum.loss_fn_inputs["logprobs"].data == pytest.approx([-0.1, -0.2])


def test_main_logs_one_record_per_step_from_metric_helpers(monkeypatch) -> None:
    events: list[str] = []
    logged: list[tuple[int, dict[str, Any]]] = []

    _run_loop(
        monkeypatch,
        policy=_Policy(events),
        events=events,
        rows=[{"id": 0}],
        records=[SimpleNamespace(client_latency=1.0, http_status_code=200, retry_attempt=1)],
        logged=logged,
    )

    [(step, metrics)] = [record for record in logged if record[0] == 1 and "train/step" in record[1]]
    assert step == 1
    assert metrics["sampler/requests"] == 1.0
    assert metrics["train/loss:sum"] == 1.0
    assert metrics["train/grad_norm"] == 3.0
    assert metrics["train/ppo_ratio_mean"] == pytest.approx(1.0)
    assert metrics["train/raw_inference_logprob_coverage"] == 0.0
    assert metrics["perf/trainer_fwd_bwd_time"] == 0.5
    assert metrics["perf/rollout_batch_wall_time"] >= 0
    assert metrics["rollout/completion_tokens_mean"] == 3.0
    assert metrics["rollout/truncated_ratio"] == 0.5


def test_prepare_batch_fn_rewrites_the_batch_before_the_update(monkeypatch) -> None:
    events: list[str] = []
    policy = _Policy(events)
    seen: list[Any] = []

    def prepare(policy_arg, batch):
        seen.append(policy_arg)
        return replace(batch, token_advantages=[[0.0, 0.5], [0.0, -0.5]]), {"igpo/mean_ig": 0.25}

    logged: list[tuple[int, dict[str, Any]]] = []
    _run_loop(monkeypatch, policy=policy, events=events, rows=[{"id": 0}], prepare_batch_fn=prepare, logged=logged)

    assert seen == [policy]
    assert [d.loss_fn_inputs["advantages"].data for d in policy.datums] == [[0.0, 0.5], [0.0, -0.5]]
    assert any(m.get("igpo/mean_ig") == 0.25 for _step, m in logged)


def test_evaluation_reuses_the_training_rollout(monkeypatch) -> None:
    events: list[str] = []
    logged: list[tuple[int, dict[str, Any]]] = []

    async def evaluation_fn(step, rollout_fn):
        await rollout_fn({"id": "held-out"}, row_index=0, sample_index=0)
        return {"eval/accuracy": float(step)}

    _run_loop(
        monkeypatch,
        policy=_Policy(events),
        events=events,
        rows=[{"id": 0}, {"id": 1}],
        evaluation_fn=evaluation_fn,
        evaluation_interval=2,
        records=[SimpleNamespace(client_latency=1.0)],
        logged=logged,
    )

    eval_records = [(step, m) for step, m in logged if "eval/failed" in m]
    assert [step for step, _m in eval_records] == [0, 2]
    # One rollout object: stateful rollouts keep one concurrency budget.
    assert events.count("factory:train") == 1 and "factory:eval" not in events
    assert events.count("eval-sample:train") == 2
    assert all(m["eval/failed"] == 0.0 for _s, m in eval_records)
    # The sync loop drains between steps, so evaluation requests stay out of sampler/*.
    assert eval_records[0][1]["eval/sampler/requests"] == 1.0


def test_main_passes_renderer_name_to_the_default_rollout(monkeypatch):
    """cfg.renderer_name must reach build_renderer, not just exist on Config.

    The shape-validation lane pins renderer_name on its glm_moe_dsa shapes; if
    the recipe accepts the field but never forwards it, those runs silently roll
    out with a tokenizer-inferred renderer instead and nothing fails loudly.
    """
    captured: dict[str, str] = {}

    class _Service:
        trainer_job_id = "trainer"
        max_context_length = 4096

        def create_training_client(self, *_args, **_kwargs):
            return SimpleNamespace()

        def create_deployment_sampler(self, *_args, **_kwargs):
            return SimpleNamespace(close=lambda: None, base_url="https://api.example", model="m")

        def close(self):
            pass

    def fake_build_renderer(_tokenizer, tokenizer_model, renderer_name):
        captured["tokenizer_model"] = tokenizer_model
        captured["renderer_name"] = renderer_name
        raise _StopAfterRenderer

    _stub_provisioning_dependencies(monkeypatch)
    monkeypatch.setattr(module, "build_service_client", lambda **_kwargs: _Service())
    monkeypatch.setattr(
        module, "ReconnectableClient", SimpleNamespace(from_training_client=lambda client, **_kwargs: client)
    )
    monkeypatch.setattr(module, "make_weight_sync", lambda *_args: lambda *_a, **_kw: None)
    monkeypatch.setattr(
        module, "TrainingCheckpoints", lambda *_args, **_kwargs: SimpleNamespace(resume=lambda **_kw: None)
    )
    monkeypatch.setattr(module, "build_renderer", fake_build_renderer)

    cfg = module.Config(
        log_path="/tmp/rl_test_logs",
        dataset="/tmp/prompts.jsonl",
        renderer_name="glm_moe_dsa",
        deployment=module.DeployConfig(tokenizer_model="Qwen/Qwen3.5-9B"),
    )
    with pytest.raises(_StopAfterRenderer):
        module.main(cfg)

    assert captured == {"tokenizer_model": "Qwen/Qwen3.5-9B", "renderer_name": "glm_moe_dsa"}


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("routing_format", [None, "base64_inline", "parquet_v1"])
def test_default_rollout_preserves_images_and_sdk_routing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, multimodal: bool, routing_format: str | None
) -> None:
    chunks = [sdk.EncodedTextChunk(tokens=[10, 11])]
    if multimodal:
        chunks.append(sdk.types.ImageChunk(data=b"image", format="png", expected_tokens=4))
    chunks.append(sdk.EncodedTextChunk(tokens=[12]))
    prompt = sdk.ModelInput(chunks=chunks)
    expanded_tokens = [10, 11] + ([0] * 4 if multimodal else []) + [12]
    routes = None
    if routing_format == "base64_inline":
        routes = ["route-30", "route-40"]
    elif routing_format == "parquet_v1":
        routes = RoutingReferences(
            2,
            ({"format": "parquet_v1", "row_count": 2, "artifact_id": "capture"},),
            ({"input_token_start": 0, "count": 2, "file_index": 0, "file_row_start": 0},),
        )
    sampling_calls = []
    sampler_closed = []

    async def sample_with_prompt_tokens(tokens: list[int], **kwargs: Any) -> list[SampledCompletion]:
        sampling_calls.append((tokens, kwargs))
        return [
            SampledCompletion(
                text="answer",
                full_tokens=expanded_tokens + [30, 40],
                prompt_len=len(expanded_tokens),
                finish_reason="stop",
                sampling_logprobs=[-0.31, -0.41],
                inference_logprobs=[-0.3, -0.4],
                routing_matrices=routes,
            )
            for _ in range(kwargs["n"])
        ]

    training_clients = []
    policy = object()

    class _Service:
        trainer_job_id = "trainer"
        max_context_length = 4096

        def create_training_client(self, *_args: Any, **_kwargs: Any) -> object:
            training_clients.append(policy)
            return policy

        def create_deployment_sampler(self, **_kwargs: Any) -> SimpleNamespace:
            assert training_clients == [policy]
            return SimpleNamespace(
                sample_with_prompt_tokens=sample_with_prompt_tokens,
                close=lambda: sampler_closed.append(True),
                base_url="https://api.example",
                model="model",
            )

        def close(self) -> None:
            pass

    row = {"messages": [{"role": "user", "content": "question"}]}
    groups = []

    async def collect(_iterator: Any, *, sample_prompt: Any, **_kwargs: Any) -> None:
        groups.append(await sample_prompt(row, cursor_index=0))
        raise _StopAfterSampling

    _stub_provisioning_dependencies(monkeypatch)
    monkeypatch.setattr(module, "build_service_client", lambda **_kwargs: _Service())
    monkeypatch.setattr(
        module,
        "ReconnectableClient",
        SimpleNamespace(from_training_client=lambda client, **_kwargs: client),
    )
    monkeypatch.setattr(module, "make_weight_sync", lambda *_args: lambda *_a, **_kw: None)
    monkeypatch.setattr(
        module,
        "TrainingCheckpoints",
        lambda *_args, **_kwargs: SimpleNamespace(resume=lambda **_kw: None),
    )
    monkeypatch.setattr(
        module,
        "build_renderer",
        lambda *_args: SimpleNamespace(
            image_placeholder_token_id=99,
            build_generation_prompt=lambda _messages: prompt,
            get_stop_sequences=lambda: ["STOP"],
            parse_response=lambda _tokens: (Message(role="assistant", content="answer"), True),
        ),
    )
    monkeypatch.setattr(module, "collect_prompt_groups", collect)
    monkeypatch.setattr(module, "reward_fn", lambda _text, _row: 1.0)
    monkeypatch.setattr(module, "single_turn_renderer_rollout", module.single_turn_renderer_rollout)
    monkeypatch.setattr(module, "log_metrics", lambda *_args, **_kwargs: None)

    # Router replay follows the model: MoE models replay, dense ones do not.
    monkeypatch.setattr(
        module,
        "resolve_router_replay_enabled",
        lambda **_kwargs: routing_format is not None,
    )

    with pytest.raises(_StopAfterSampling):
        module.main(
            module.Config(
                log_path=str(tmp_path),
                completions_per_prompt=2,
                deployment=module.DeployConfig(tokenizer_model="tokenizer"),
            ),
            rows=[row],
        )

    # The recipe owns the sampler it created and closes it even on failure.
    assert sampler_closed == [True]
    # One completion per rollout_fn call; the loop fans out completions_per_prompt calls.
    assert len(sampling_calls) == 2
    tokens, kwargs = sampling_calls[0]
    assert tokens == [10, 11] + ([99] if multimodal else []) + [12]
    assert kwargs.get("images") == (["data:image/png;base64,aW1hZ2U="] if multimodal else None)
    assert kwargs["n"] == 1
    assert kwargs["stop"] == ["STOP"]
    assert kwargs["logprobs"] is True
    assert kwargs.get("include_routing_matrix", False) == (routing_format is not None)
    assert not kwargs.get("echo", False)
    assert "routing_matrix_format" not in kwargs
    [group] = groups
    assert len(group.data) == 2
    assert group.raw_inf_logprobs[0][-2:] == [-0.3, -0.4]
    for datum in group.data:
        assert datum.model_input.length == len(expanded_tokens) + 1
        assert has_non_text_chunks(datum.model_input) == multimodal
        if multimodal:
            assert datum.model_input.chunks[1].data == b"image"
        if routing_format == "parquet_v1":
            assert datum.model_input.routing_matrix_format == "parquet_v1"
            assert datum.model_input.routing_references["files"][0]["artifact_id"] == "capture"
        elif routing_format == "base64_inline":
            assert datum.model_input.routing_matrices[-1] == "route-40"


@pytest.mark.parametrize("phase", ["before_optimizer", "optimizer_wait"])
def test_sync_rl_sigterm_stops_before_publication_and_cursor_commit(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    events: list[str] = []
    loaders: list[Any] = []
    original_loader = module.CursorDataLoader
    previous_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    def track_loader(*args: Any, **kwargs: Any) -> Any:
        loader = original_loader(*args, **kwargs)
        loaders.append(loader)
        return loader

    monkeypatch.setattr(module, "CursorDataLoader", track_loader)
    if phase == "before_optimizer":
        original_diagnostics = module.inference_drift_metrics

        def stop_after_diagnostics(*args: Any, **kwargs: Any) -> Any:
            result = original_diagnostics(*args, **kwargs)
            events.append("stop-before-optimizer")
            os.kill(os.getpid(), signal.SIGTERM)
            return result

        monkeypatch.setattr(module, "inference_drift_metrics", stop_after_diagnostics)

    class Policy(_Policy):
        def optim_step(self, params: Any, **kwargs: Any) -> Any:
            result = super().optim_step(params, **kwargs)
            if phase == "optimizer_wait":
                os.kill(os.getpid(), signal.SIGTERM)
                # A blocking native result wait must finish before service close.
                events.append("optimizer-complete")
            return result

    with pytest.raises(TerminatedBySignal, match="SIGTERM"):
        _run_loop(monkeypatch, policy=Policy(events), events=events, rows=[{"id": 0}])
    assert "hotload:/step-0" in events
    assert "hotload:/step-1" not in events
    assert loaders[0].data_consumed == 0
    assert not any(event.startswith("checkpoint:") for event in events)
    if phase == "before_optimizer":
        assert "optim" not in events
    else:
        assert events.index("optimizer-complete") < events.index("close")
    assert {signum: signal.getsignal(signum) for signum in previous_handlers} == previous_handlers


def test_batch_preparation_runs_after_all_rollouts_and_before_one_training_call(monkeypatch) -> None:
    events, logged = [], []
    policy = _Policy(events)

    def make_group(row):
        n = 2 if row["id"] == 0 else 1
        return _group(
            data=[_datum()] * n,
            advantages=[1.0] * n,
            rewards=[1.0] * n,
            inf_logprobs=[[-0.1, -0.2]] * n,
            completion_lens=[2] * n,
            truncated=[False] * n,
            row_meta=row,
        )

    def prepare(policy_arg, batch):
        assert policy_arg is policy
        assert "sample:0" in events and "sample:1" in events
        assert not any(e.startswith("fwd-bwd:") for e in events)
        assert len(batch.groups) == 2
        events.append("batch-prepared")
        rows = [[v, -v] for g in batch.groups for v in [0.5 if g.row_meta["id"] == 0 else 0.25] for _ in g.data]
        return replace(batch, token_advantages=rows), {"prepare/weighted_mean": 13 / 3}

    result = _run_loop(
        monkeypatch,
        policy=policy,
        events=events,
        logged=logged,
        rows=[{"id": 0}, {"id": 1}],
        prompt_groups_per_step=2,
        group_for_row=make_group,
        prepare_batch_fn=prepare,
    )
    assert result["steps"] == 1
    assert [d.loss_fn_inputs["advantages"].data for d in policy.datums] == [[0.5, -0.5], [0.5, -0.5], [0.25, -0.25]]
    assert events.index("batch-prepared") < events.index("fwd-bwd:3") < events.index("optim")
    metrics = [m for step, m in logged if step == 1][0]
    assert metrics["prepare/weighted_mean"] == pytest.approx(13 / 3)
    assert metrics["train/forward_backward_requests"] == events.count("optim") == 1


@pytest.mark.parametrize("stage", ["prepare", "backward"])
def test_sync_batch_failure_never_optimizes_or_publishes(monkeypatch, stage):
    events = []

    class Policy(_Policy):
        def forward_backward(self, *args, **kwargs):
            raise RuntimeError("batch failed")

    def prepare(*args):
        raise RuntimeError("batch failed")

    with pytest.raises(RuntimeError, match="batch failed"):
        _run_loop(
            monkeypatch,
            policy=Policy(events),
            events=events,
            rows=[{"id": 0}],
            **({"prepare_batch_fn": prepare} if stage == "prepare" else {}),
        )
    assert "optim" not in events and "hotload:/step-1" not in events
    assert not any(e.startswith("checkpoint:") for e in events)
    assert "close" in events


def test_all_ready_groups_share_one_request_and_one_optimizer(monkeypatch) -> None:
    events = []
    logged = []
    policy = _Policy(events)
    result = _run_loop(
        monkeypatch,
        policy=policy,
        events=events,
        logged=logged,
        rows=[{"id": i} for i in range(6)],
        prompt_groups_per_step=6,
    )
    assert result["steps"] == 1
    assert len(policy.datums) == 12
    metrics = [metrics for step, metrics in logged if step == 1][0]
    assert metrics["train/forward_backward_requests"] == 1
    assert metrics["train/groups_per_request_mean"] == metrics["train/groups_per_request_max"] == 6
    assert events.count("optim") == 1


def test_periodic_checkpoint_wait_leaves_event_loop_responsive(monkeypatch) -> None:
    heartbeat = threading.Event()
    main_thread = threading.get_ident()
    from training.utils import termination

    original_run = termination.run_async_with_signals

    def run_with_timer(awaitable):
        async def monitor():
            asyncio.get_running_loop().call_later(0.05, heartbeat.set)
            return await awaitable

        return original_run(monitor())

    def checkpoint():
        assert threading.get_ident() != main_thread
        assert heartbeat.wait(timeout=2), "event-loop timer stalled during checkpoint"

    monkeypatch.setattr(termination, "run_async_with_signals", run_with_timer)
    events = []
    _run_loop(
        monkeypatch,
        policy=_Policy(events),
        events=events,
        rows=[{"id": 0}],
        dcp_save_interval=1,
        checkpoint_fn=checkpoint,
    )
    assert "checkpoint:step-1" in events
