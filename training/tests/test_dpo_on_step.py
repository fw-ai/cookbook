from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import training.recipes.dpo_loop as module


def test_main_signature_accepts_on_step() -> None:
    sig = inspect.signature(module.main)
    param = sig.parameters["on_step"]
    assert param.default is None
    assert param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def test_train_loop_signature_accepts_on_step() -> None:
    sig = inspect.signature(module._train_loop)
    param = sig.parameters["on_step"]
    assert param.default is None
    assert param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def test_step_callback_type_alias_exists() -> None:
    assert hasattr(module, "StepCallback")


def test_on_step_called_with_correct_args(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = MagicMock()

    monkeypatch.setattr(
        module,
        "_forward_backward_pairs",
        lambda batch, policy, beta, microbatch_sizes=None: SimpleNamespace(
            metrics={"dpo_loss": 0.5, "margin": 0.1, "accuracy": 0.8}
        ),
    )
    monkeypatch.setattr(module, "flush_timing", lambda: {})
    monkeypatch.setattr(module, "log_metrics_json", lambda step, **kw: None)
    monkeypatch.setattr(module, "wandb_log", lambda payload, step: None)

    class FakePolicy:
        def optim_step(self, _params: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(metrics={})

    class FakeWeightSyncer:
        def save_and_hotload(self, name: str) -> None:
            pass

        def save_dcp(self, name: str) -> None:
            pass

    ref_cache = {
        0: {
            "chosen_datum": {"id": "c0"},
            "rejected_datum": {"id": "r0"},
            "chosen_tokens": [1, 2, 3],
            "rejected_tokens": [4, 5],
            "ref_chosen": [-0.1],
            "ref_rejected": [-0.2],
            "response_start": 2,
        },
        1: {
            "chosen_datum": {"id": "c1"},
            "rejected_datum": {"id": "r1"},
            "chosen_tokens": [6, 7],
            "rejected_tokens": [8, 9, 10],
            "ref_chosen": [-0.3],
            "ref_rejected": [-0.4],
            "response_start": 1,
        },
    }
    cfg = module.Config(
        log_path="/tmp/dpo_test_on_step",
        beta=0.1,
        epochs=1,
        batch_size=1,
        grad_accum=2,
        weight_sync=module.WeightSyncConfig(weight_sync_interval=0),
    )

    step = asyncio.run(
        module._train_loop(
            ref_cache,
            [0, 1],
            FakePolicy(),
            adam_params={"lr": 1e-4},
            weight_syncer=FakeWeightSyncer(),
            cfg=cfg,
            step_offset=0,
            on_step=callback,
        )
    )

    assert step == 1
    assert callback.call_count == 1
    call_args = callback.call_args[0]
    assert len(call_args) == 4

    step_arg, total_steps_arg, step_tokens_arg, step_metrics_arg = call_args
    assert isinstance(step_arg, int)
    assert isinstance(total_steps_arg, int)
    assert isinstance(step_tokens_arg, int)
    assert isinstance(step_metrics_arg, dict)
    assert step_arg == 1
    assert total_steps_arg == 1
    assert step_tokens_arg > 0
    assert step_tokens_arg == (3 + 2) + (2 + 3)


def test_on_step_none_does_not_break(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "_forward_backward_pairs",
        lambda batch, policy, beta, microbatch_sizes=None: SimpleNamespace(
            metrics={"dpo_loss": 0.5, "margin": 0.1, "accuracy": 0.8}
        ),
    )
    monkeypatch.setattr(module, "flush_timing", lambda: {})
    monkeypatch.setattr(module, "log_metrics_json", lambda step, **kw: None)
    monkeypatch.setattr(module, "wandb_log", lambda payload, step: None)

    class FakePolicy:
        def optim_step(self, _params: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(metrics={})

    class FakeWeightSyncer:
        def save_and_hotload(self, name: str) -> None:
            pass

        def save_dcp(self, name: str) -> None:
            pass

    ref_cache = {
        0: {
            "chosen_datum": {"id": "c0"},
            "rejected_datum": {"id": "r0"},
            "chosen_tokens": [1, 2],
            "rejected_tokens": [3, 4],
            "ref_chosen": [-0.1],
            "ref_rejected": [-0.2],
            "response_start": 1,
        },
    }
    cfg = module.Config(
        log_path="/tmp/dpo_test_on_step",
        beta=0.1,
        epochs=1,
        batch_size=1,
        grad_accum=1,
        weight_sync=module.WeightSyncConfig(weight_sync_interval=0),
    )

    step = asyncio.run(
        module._train_loop(
            ref_cache,
            [0],
            FakePolicy(),
            adam_params={"lr": 1e-4},
            weight_syncer=FakeWeightSyncer(),
            cfg=cfg,
            step_offset=0,
            on_step=None,
        )
    )

    assert step == 1


def test_step_tokens_is_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    received_tokens: list[int] = []

    def capture_callback(step: int, total_steps: int, step_tokens: int, metrics: dict[str, Any]) -> None:
        received_tokens.append(step_tokens)

    monkeypatch.setattr(
        module,
        "_forward_backward_pairs",
        lambda batch, policy, beta, microbatch_sizes=None: SimpleNamespace(
            metrics={"dpo_loss": 0.5, "margin": 0.1, "accuracy": 0.8}
        ),
    )
    monkeypatch.setattr(module, "flush_timing", lambda: {})
    monkeypatch.setattr(module, "log_metrics_json", lambda step, **kw: None)
    monkeypatch.setattr(module, "wandb_log", lambda payload, step: None)

    class FakePolicy:
        def optim_step(self, _params: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(metrics={})

    class FakeWeightSyncer:
        def save_and_hotload(self, name: str) -> None:
            pass

        def save_dcp(self, name: str) -> None:
            pass

    ref_cache = {
        0: {
            "chosen_datum": {"id": "c0"},
            "rejected_datum": {"id": "r0"},
            "chosen_tokens": [1, 2, 3, 4, 5],
            "rejected_tokens": [6, 7, 8],
            "ref_chosen": [-0.1],
            "ref_rejected": [-0.2],
            "response_start": 2,
        },
    }
    cfg = module.Config(
        log_path="/tmp/dpo_test_on_step",
        beta=0.1,
        epochs=1,
        batch_size=1,
        grad_accum=1,
        weight_sync=module.WeightSyncConfig(weight_sync_interval=0),
    )

    asyncio.run(
        module._train_loop(
            ref_cache,
            [0],
            FakePolicy(),
            adam_params={"lr": 1e-4},
            weight_syncer=FakeWeightSyncer(),
            cfg=cfg,
            step_offset=0,
            on_step=capture_callback,
        )
    )

    assert len(received_tokens) == 1
    assert received_tokens[0] > 0
    assert received_tokens[0] == 8


def test_main_passes_on_step_to_train_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_on_step: list[Any] = []

    async def fake_train_loop(
        ref_cache: Any,
        valid_indices: Any,
        policy: Any,
        adam_params: Any,
        weight_syncer: Any,
        cfg: Any,
        step_offset: int,
        on_step: Any = None,
    ) -> int:
        captured_on_step.append(on_step)
        return 1

    import transformers

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")
    monkeypatch.setattr(module, "setup_wandb", lambda *a, **kw: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *a, **kw: None)
    monkeypatch.setattr(module, "validate_config", lambda *a, **kw: None)
    monkeypatch.setattr(module, "setup_deployment", lambda *a, **kw: None)
    monkeypatch.setattr(module, "load_preference_dataset", lambda *a, **kw: [{"chosen": {}, "rejected": {}}])
    monkeypatch.setattr(module, "build_renderer", lambda *a, **kw: object())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *a, **kw: "test")
    monkeypatch.setattr(module, "_train_loop", fake_train_loop)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())

    async def fake_cache(*a: Any, **kw: Any) -> tuple[dict[int, dict[str, Any]], int]:
        return ({0: {"chosen_datum": {}, "rejected_datum": {}}}, 0)

    monkeypatch.setattr(module, "_cache_ref_logprobs", fake_cache)

    class FakeRlorMgr:
        def resolve_training_profile(self, shape_id: str) -> SimpleNamespace:
            return SimpleNamespace(max_supported_context_length=128)

        def delete(self, job_id: str) -> None:
            pass

    class FakeDeployMgr:
        pass

    class FakeFuture:
        def __init__(self, value: Any) -> None:
            self._value = value

        def result(self) -> Any:
            return self._value

    class FakeThreadPoolExecutor:
        def __init__(self, max_workers: int) -> None:
            pass

        def __enter__(self) -> FakeThreadPoolExecutor:
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> FakeFuture:
            return FakeFuture(fn(*args, **kwargs))

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.job_id = "policy-job"
            self.inner = object()

        def load_state_with_optimizer(self, path: str) -> None:
            pass

        def resolve_checkpoint_path(self, name: str, source_job_id: str | None = None) -> str:
            return f"tinker://test/{name}"

    class FakeWeightSyncer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def save_dcp(self, name: str) -> None:
            pass

        def save_and_hotload(self, name: str) -> None:
            pass

    def fake_create_trainer_job(*args: Any, **kwargs: Any) -> SimpleNamespace:
        display_name = kwargs.get("display_name", "")
        job_id = "policy-job" if display_name == "dpo-policy" else "reference-job"
        return SimpleNamespace(job_id=job_id)

    monkeypatch.setattr(module, "ThreadPoolExecutor", FakeThreadPoolExecutor)
    monkeypatch.setattr(module, "create_trainer_job", fake_create_trainer_job)
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(module, "WeightSyncer", FakeWeightSyncer)

    sentinel = lambda s, ts, tok, m: None
    cfg = module.Config(
        log_path="/tmp/dpo_test_on_step",
        dataset="/tmp/pairs.jsonl",
        tokenizer_model="Qwen/Qwen3-1.7B",
        infra=module.InfraConfig(
            training_shape_id="ts-test",
            ref_training_shape_id="ts-test",
        ),
        weight_sync=module.WeightSyncConfig(weight_sync_interval=0),
    )

    module.main(cfg, rlor_mgr=FakeRlorMgr(), deploy_mgr=FakeDeployMgr(), on_step=sentinel)

    assert len(captured_on_step) == 1
    assert captured_on_step[0] is sentinel


def test_main_calls_on_trainers_created(monkeypatch: pytest.MonkeyPatch) -> None:
    """on_trainers_created is called with correct IDs before training starts."""
    events: list[tuple[str, Any]] = []

    def on_trainers_created_cb(policy_job_id: str, reference_job_id: str | None) -> None:
        events.append(("on_trainers_created", (policy_job_id, reference_job_id)))

    async def fake_train_loop(
        ref_cache: Any,
        valid_indices: Any,
        policy: Any,
        adam_params: Any,
        weight_syncer: Any,
        cfg: Any,
        step_offset: int,
        on_step: Any = None,
    ) -> int:
        events.append(("_train_loop", None))
        return 1

    import transformers

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")
    monkeypatch.setattr(module, "setup_wandb", lambda *a, **kw: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *a, **kw: None)
    monkeypatch.setattr(module, "validate_config", lambda *a, **kw: None)
    monkeypatch.setattr(module, "setup_deployment", lambda *a, **kw: None)
    monkeypatch.setattr(module, "load_preference_dataset", lambda *a, **kw: [{"chosen": {}, "rejected": {}}])
    monkeypatch.setattr(module, "build_renderer", lambda *a, **kw: object())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *a, **kw: "test")
    monkeypatch.setattr(module, "_train_loop", fake_train_loop)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())

    async def fake_cache(*a: Any, **kw: Any) -> tuple[dict[int, dict[str, Any]], int]:
        return ({0: {"chosen_datum": {}, "rejected_datum": {}}}, 0)

    monkeypatch.setattr(module, "_cache_ref_logprobs", fake_cache)

    class FakeRlorMgr:
        def resolve_training_profile(self, shape_id: str) -> SimpleNamespace:
            return SimpleNamespace(max_supported_context_length=128)

        def delete(self, job_id: str) -> None:
            pass

    class FakeDeployMgr:
        pass

    class FakeFuture:
        def __init__(self, value: Any) -> None:
            self._value = value

        def result(self) -> Any:
            return self._value

    class FakeThreadPoolExecutor:
        def __init__(self, max_workers: int) -> None:
            pass

        def __enter__(self) -> FakeThreadPoolExecutor:
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> FakeFuture:
            return FakeFuture(fn(*args, **kwargs))

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.job_id = "policy-job"
            self.inner = object()

        def load_state_with_optimizer(self, path: str) -> None:
            pass

        def resolve_checkpoint_path(self, name: str, source_job_id: str | None = None) -> str:
            return f"tinker://test/{name}"

    class FakeWeightSyncer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def save_dcp(self, name: str) -> None:
            pass

        def save_and_hotload(self, name: str) -> None:
            pass

    def fake_create_trainer_job(*args: Any, **kwargs: Any) -> SimpleNamespace:
        display_name = kwargs.get("display_name", "")
        job_id = "policy-job" if display_name == "dpo-policy" else "reference-job"
        return SimpleNamespace(job_id=job_id)

    monkeypatch.setattr(module, "ThreadPoolExecutor", FakeThreadPoolExecutor)
    monkeypatch.setattr(module, "create_trainer_job", fake_create_trainer_job)
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(module, "WeightSyncer", FakeWeightSyncer)

    cfg = module.Config(
        log_path="/tmp/dpo_test_on_trainers_created",
        dataset="/tmp/pairs.jsonl",
        tokenizer_model="Qwen/Qwen3-1.7B",
        infra=module.InfraConfig(
            training_shape_id="ts-test",
            ref_training_shape_id="ts-test",
        ),
        weight_sync=module.WeightSyncConfig(weight_sync_interval=0),
    )

    module.main(
        cfg,
        rlor_mgr=FakeRlorMgr(),
        deploy_mgr=FakeDeployMgr(),
        on_trainers_created=on_trainers_created_cb,
    )

    assert len(events) >= 2
    assert events[0] == ("on_trainers_created", ("policy-job", "reference-job"))
    assert events[1] == ("_train_loop", None)
