from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
import torch

import training.recipes.sft_loop as module


class StubRenderer:
    def __init__(self, tokens: list[int], weights: list[float]):
        self.tokens = torch.tensor(tokens, dtype=torch.int64)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.calls: list = []

    def build_supervised_example(self, messages, train_on_what):
        self.calls.append((messages, train_on_what))
        return self.tokens, self.weights


def _write_dataset(tmp_path, rows):
    dataset_path = tmp_path / "sft.jsonl"
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows))
    return dataset_path


def _make_stub_render_worker_state(renderer=None, tokenizer=None):
    """Return a populate_render_worker_state replacement that skips HF tokenizer / renderer load."""
    def stub(state, *, tokenizer_model, renderer_name, max_seq_len, **extras):
        state.update(
            tokenizer=tokenizer if tokenizer is not None else object(),
            renderer=renderer if renderer is not None else object(),
            max_seq_len=max_seq_len,
            **extras,
        )
    return stub


def _fake_profile(shape_id: str = "accounts/test/trainingShapes/sft"):
    """Stub for ``rlor_mgr.resolve_training_profile()`` returns."""
    return SimpleNamespace(
        max_supported_context_length=48,
        training_shape_version=f"{shape_id}/versions/1",
        trainer_image_tag="trainer:1",
        accelerator_type=None,
        accelerator_count=None,
        node_count=None,
        deployment_shape_version=None,
        pipeline_parallelism=1,
    )


def _test_datum(test_id: str):
    return SimpleNamespace(
        model_input=SimpleNamespace(chunks=[SimpleNamespace(tokens=[1, 2, 3])]),
        loss_fn_inputs={
            "target_tokens": SimpleNamespace(data=[0, 0]),
            "weights": SimpleNamespace(data=[1.0, 1.0]),
        },
        _test_id=test_id,
    )


def test_main_rejects_adapter_plus_init_from_checkpoint(tmp_path, monkeypatch):
    """warm_start_from_adapter and init_from_checkpoint are mutually exclusive."""
    dataset_path = _write_dataset(
        tmp_path,
        [{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}],
    )
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)

    cfg = module.Config(
        log_path=str(tmp_path / "logs"),
        dataset=str(dataset_path),
        tokenizer_model="Qwen/Qwen3-1.7B",
        max_seq_len=32,
        lora_rank=16,
        warm_start_from_adapter="gs://bucket/adapter-dir",
        init_from_checkpoint="gs://bucket/dcp-dir",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        module.main(cfg)


def test_main_rejects_adapter_with_zero_lora_rank(tmp_path, monkeypatch):
    """warm_start_from_adapter requires lora_rank > 0 (LoRA training)."""
    dataset_path = _write_dataset(
        tmp_path,
        [{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}],
    )
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)

    cfg = module.Config(
        log_path=str(tmp_path / "logs"),
        dataset=str(dataset_path),
        tokenizer_model="Qwen/Qwen3-1.7B",
        max_seq_len=32,
        lora_rank=0,
        warm_start_from_adapter="gs://bucket/adapter-dir",
    )

    with pytest.raises(ValueError, match="lora_rank > 0"):
        module.main(cfg)


def test_main_requires_tokenizer_model(tmp_path, monkeypatch):
    dataset_path = _write_dataset(
        tmp_path,
        [{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}],
    )
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)

    cfg = module.Config(log_path=str(tmp_path / "logs"), dataset=str(dataset_path), tokenizer_model="", max_seq_len=32)

    with pytest.raises(ValueError, match="tokenizer_model"):
        module.main(cfg)


def test_main_raises_when_all_examples_are_filtered(tmp_path, monkeypatch):
    dataset_path = _write_dataset(
        tmp_path,
        [{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}],
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    deleted_jobs: list[str] = []

    class FakeMgr:
        def create(self, config):
            return SimpleNamespace(job_id="job-sft", job_name="jobs/job-sft")

        def wait_for_ready(self, job_id, **kwargs):
            return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}", base_url="https://unit.test")

        def cancel(self, job_id):
            deleted_jobs.append(job_id)

        def delete(self, job_id):
            self.cancel(job_id)

        def resolve_training_profile(self, shape_id):
            return _fake_profile(shape_id)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "populate_render_worker_state", _make_stub_render_worker_state())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(
        module,
        "auto_select_training_shape",
        lambda *args, **kwargs: "accounts/test/trainingShapes/sft",
    )
    monkeypatch.setattr(
        module,
        "render_messages_to_datum",
        lambda *args, **kwargs: SimpleNamespace(token_ids=[1], datum={"id": "too-short"}),
    )
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)

    cfg = module.Config(
        log_path=str(tmp_path / "logs"),
        base_model="accounts/test/models/custom-sft",
        dataset=str(dataset_path),
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
        render_workers=1,
    )

    with pytest.raises(RuntimeError, match="No valid training examples"):
        module.main(cfg, rlor_mgr=FakeMgr())

    assert deleted_jobs == ["job-sft"]


def test_main_reduces_batch_size_when_examples_fewer_than_batch_size(tmp_path, monkeypatch):
    """When training examples < batch_size, reduce effective batch_size so training proceeds."""
    dataset_path = _write_dataset(
        tmp_path,
        [{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}],
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {"batches": [], "deleted_jobs": []}

    class FakeMgr:
        def create(self, config):
            return SimpleNamespace(job_id="job-sft", job_name="jobs/job-sft")

        def wait_for_ready(self, job_id, **kwargs):
            return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}", base_url="https://unit.test")

        def cancel(self, job_id):
            events["deleted_jobs"].append(job_id)

        def delete(self, job_id):
            self.cancel(job_id)

        def resolve_training_profile(self, shape_id):
            return _fake_profile(shape_id)

    class FakeClient:
        job_id = "job-sft"

        def __init__(self, *args, **kwargs):
            pass

        def forward_backward(self, batch, loss_fn="cross_entropy", loss_fn_config=None):
            events["batches"].append(list(batch))
            return SimpleNamespace(
                metrics={"loss:sum": 1.0, "ce_loss_sum": 1.0, "response_tokens": 1}
            )

        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={})

        def save_state(self, name):
            return SimpleNamespace(path=name)

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(path=f"{name}-sampler")

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return name

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_metrics_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "populate_render_worker_state", _make_stub_render_worker_state())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(
        module,
        "auto_select_training_shape",
        lambda *args, **kwargs: "accounts/test/trainingShapes/sft",
    )
    monkeypatch.setattr(
        module,
        "render_messages_to_datum",
        lambda *args, **kwargs: SimpleNamespace(
            token_ids=[1, 2, 3],
            datum=SimpleNamespace(
                model_input=SimpleNamespace(chunks=[SimpleNamespace(tokens=[1, 2, 3])]),
                loss_fn_inputs={
                    "target_tokens": SimpleNamespace(data=[2, 3]),
                    "weights": SimpleNamespace(data=[1.0, 1.0]),
                },
            ),
        ),
    )
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)

    cfg = module.Config(
        log_path=str(tmp_path / "logs"),
        base_model="accounts/test/models/custom-sft",
        dataset=str(dataset_path),
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
        batch_size=999,
        epochs=1,
    )

    result = module.main(cfg, rlor_mgr=FakeMgr())

    assert result["steps"] == 1
    assert len(events["batches"]) == 1
    assert len(events["batches"][0]) == 1
    assert events["deleted_jobs"] == ["job-sft"]


def test_main_infers_documented_training_shape_for_supported_model(tmp_path, monkeypatch):
    dataset_path = _write_dataset(
        tmp_path,
        [{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}],
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {"deleted_jobs": [], "selection_requests": []}

    class FakeMgr:
        def create(self, config):
            events["created_config"] = config
            return SimpleNamespace(job_id="job-sft", job_name="jobs/job-sft")

        def wait_for_ready(self, job_id, **kwargs):
            return SimpleNamespace(
                job_id=job_id,
                job_name=f"jobs/{job_id}",
                base_url="https://unit.test",
            )

        def delete(self, job_id):
            events["deleted_jobs"].append(job_id)

        def cancel(self, job_id):
            events["deleted_jobs"].append(job_id)

        def resolve_training_profile(self, shape_id):
            return _fake_profile(shape_id)

    class FakeClient:
        job_id = "job-sft"

        def __init__(self, *args, **kwargs):
            pass

        def forward_backward(self, batch, loss_fn="cross_entropy", loss_fn_config=None):
            return SimpleNamespace(
                metrics={"loss:sum": 1.0, "ce_loss_sum": 1.0, "response_tokens": 1}
            )

        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={})

        def save_state(self, name):
            return SimpleNamespace(path=name)

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(path=f"{name}-sampler")

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return name

    def _record_auto_select(_mgr, **kwargs):
        events["selection_requests"].append(kwargs)
        return "accounts/test/trainingShapes/sft"

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_metrics_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "populate_render_worker_state", _make_stub_render_worker_state())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(module, "auto_select_training_shape", _record_auto_select)
    monkeypatch.setattr(
        module,
        "render_messages_to_datum",
        lambda *args, **kwargs: SimpleNamespace(
            token_ids=[1, 2, 3],
            datum=SimpleNamespace(
                model_input=SimpleNamespace(
                    chunks=[SimpleNamespace(tokens=[1, 2, 3])],
                ),
                loss_fn_inputs={
                    "target_tokens": SimpleNamespace(data=[2, 3]),
                    "weights": SimpleNamespace(data=[1.0, 1.0]),
                },
            ),
        ),
    )
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)

    cfg = module.Config(
        log_path=str(tmp_path / "logs"),
        base_model="accounts/fireworks/models/qwen3-8b",
        dataset=str(dataset_path),
        tokenizer_model="Qwen/Qwen3-8B",
        max_seq_len=None,
        epochs=1,
        batch_size=1,
    )

    result = module.main(cfg, rlor_mgr=FakeMgr())

    assert result["steps"] == 1
    assert cfg.max_seq_len == 48
    assert cfg.infra.training_shape_id == "accounts/test/trainingShapes/sft"
    assert len(events["selection_requests"]) == 1
    assert events["created_config"].training_shape_ref == "accounts/test/trainingShapes/sft/versions/1"
    assert events["created_config"].custom_image_tag is None
    assert events["created_config"].accelerator_type is None
    assert events["created_config"].accelerator_count is None
    assert events["created_config"].node_count is None
    assert events["deleted_jobs"] == ["job-sft"]


def test_main_uses_real_renderer_and_trains(tmp_path, monkeypatch):
    """Verify multi-turn rendering, training loop execution, and checkpoint save."""
    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "a2"},
                ]
            }
        ],
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {"batches": [], "deleted_jobs": [], "lifecycle": []}
    renderer = StubRenderer(
        tokens=[100, 101, 102, 103, 104, 105, 106],
        weights=[0, 0, 1, 1, 0, 1, 1],
    )

    class FakeMgr:
        def create(self, config):
            return SimpleNamespace(job_id="job-sft", job_name="jobs/job-sft")

        def wait_for_ready(self, job_id, **kwargs):
            return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}", base_url="https://unit.test")

        def cancel(self, job_id):
            events["lifecycle"].append(("delete", job_id))
            events["deleted_jobs"].append(job_id)

        def delete(self, job_id):
            self.cancel(job_id)

        def resolve_training_profile(self, shape_id):
            return _fake_profile(shape_id)

    class FakeClient:
        job_id = "job-sft"

        def __init__(self, *args, **kwargs):
            pass

        def forward_backward(self, batch, loss_fn="cross_entropy", loss_fn_config=None):
            events["batches"].append(batch)
            return SimpleNamespace(metrics={"loss:sum": 2.0, "ce_loss_sum": 2.0, "response_tokens": 4})

        def forward_backward_custom(self, batch, loss_fn):
            events["batches"].append(batch)
            return SimpleNamespace(metrics={"ce_loss_sum": 2.0, "response_tokens": 4})

        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={"optimizer/lr": 1e-4})

        def save_state(self, name):
            return SimpleNamespace(path=name)

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(path=f"{name}-sampler")

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return f"cross_job://{source_job_id}/{name}" if source_job_id else name

        def close(self):
            events["lifecycle"].append(("close", self.job_id))

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_metrics_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "populate_render_worker_state", _make_stub_render_worker_state(renderer=renderer))
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(
        module,
        "auto_select_training_shape",
        lambda *args, **kwargs: "accounts/test/trainingShapes/sft",
    )
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)

    cfg = module.Config(
        dataset=str(dataset_path),
        base_model="accounts/test/models/custom-sft",
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
        epochs=1,
        batch_size=1,
        grad_accum=1,
        log_path=str(tmp_path / "sft_logs"),
        render_workers=1,
    )

    result = module.main(cfg, rlor_mgr=FakeMgr())

    assert result["steps"] == 1
    assert result["job_id"] == "job-sft"

    normalized_messages, train_on_what = renderer.calls[0]
    assert [m["role"] for m in normalized_messages] == ["user", "assistant", "user", "assistant"]
    assert train_on_what.value == "all_assistant_messages"

    assert len(events["batches"]) == 1
    datum = events["batches"][0][0]
    assert datum.loss_fn_inputs["target_tokens"].data == [101, 102, 103, 104, 105, 106]
    assert datum.loss_fn_inputs["weights"].data == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    assert events["deleted_jobs"] == ["job-sft"]
    assert events["lifecycle"] == [("close", "job-sft"), ("delete", "job-sft")]


def test_each_batch_triggers_its_own_optim_step(tmp_path, monkeypatch):
    """With grad_accum removed, each batch gets its own forward_backward + optim_step."""
    dataset_path = _write_dataset(
        tmp_path,
        [
            {"messages": [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]},
            {"messages": [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]},
        ],
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {"batches": [], "optim_steps": 0, "deleted_jobs": []}

    class FakeMgr:
        def create(self, config):
            return SimpleNamespace(job_id="job-sft", job_name="jobs/job-sft")

        def wait_for_ready(self, job_id, **kwargs):
            return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}", base_url="https://unit.test")

        def cancel(self, job_id):
            events["deleted_jobs"].append(job_id)

        def delete(self, job_id):
            self.cancel(job_id)

        def resolve_training_profile(self, shape_id):
            return _fake_profile(shape_id)

    class FakeClient:
        job_id = "job-sft"

        def __init__(self, *args, **kwargs):
            pass

        def forward_backward(self, batch, loss_fn="cross_entropy", loss_fn_config=None):
            events["batches"].append(list(batch))
            return SimpleNamespace(metrics={"loss:sum": 4.0, "ce_loss_sum": 4.0, "response_tokens": 8})

        def optim_step(self, _params, **kwargs):
            events["optim_steps"] += 1
            return SimpleNamespace(metrics={})

        def save_state(self, name):
            return SimpleNamespace(path=name)

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(path=f"{name}-sampler")

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return f"cross_job://{source_job_id}/{name}" if source_job_id else name

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_metrics_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "populate_render_worker_state", _make_stub_render_worker_state())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(
        module,
        "auto_select_training_shape",
        lambda *args, **kwargs: "accounts/test/trainingShapes/sft",
    )
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)
    def _fake_render(messages, **kwargs):
        content = messages[-1]["content"]
        datum = SimpleNamespace(
            model_input=SimpleNamespace(chunks=[SimpleNamespace(tokens=[1, 2, 3])]),
            loss_fn_inputs={
                "target_tokens": SimpleNamespace(data=[0, 0]),
                "weights": SimpleNamespace(data=[1.0, 1.0]),
            },
            _test_id=content,
        )
        return SimpleNamespace(token_ids=[1, 2, 3], datum=datum)

    monkeypatch.setattr(module, "render_messages_to_datum", _fake_render)

    cfg = module.Config(
        dataset=str(dataset_path),
        base_model="accounts/test/models/custom-sft",
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
        epochs=1,
        batch_size=1,
        log_path=str(tmp_path / "sft_logs"),
        render_workers=1,
    )

    result = module.main(cfg, rlor_mgr=FakeMgr())

    assert result["steps"] == 2
    assert events["optim_steps"] == 2
    assert len(events["batches"]) == 2
    assert sorted(d._test_id for batch in events["batches"] for d in batch) == ["a1", "a2"]
    assert events["deleted_jobs"] == ["job-sft"]


def test_main_resume_preserves_epoch_zero_batch_order(tmp_path, monkeypatch):
    """Resume should replay the same epoch-0 shuffle and skip into it deterministically."""
    dataset_path = _write_dataset(
        tmp_path,
        [
            {"messages": [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]}
            for i in range(6)
        ],
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    runs: dict[str, list[list[str]]] = {"fresh": [], "resume": []}
    active = {"name": "fresh"}

    class FakeMgr:
        def create(self, config):
            job_id = f"job-{active['name']}"
            return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}")

        def wait_for_ready(self, job_id, **kwargs):
            return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}", base_url="https://unit.test")

        def cancel(self, job_id):
            pass

        def delete(self, job_id):
            pass

        def resolve_training_profile(self, shape_id):
            return _fake_profile(shape_id)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.job_id = kwargs.get("job_id", "job-sft")

        def forward_backward(self, batch, loss_fn="cross_entropy", loss_fn_config=None):
            runs[active["name"]].append([d._test_id for d in batch])
            return SimpleNamespace(
                metrics={"loss:sum": 1.0, "ce_loss_sum": 1.0, "response_tokens": len(batch)}
            )

        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={})

        def save_state(self, name):
            return SimpleNamespace(path=name)

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(path=f"{name}-sampler")

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return name

        def close(self):
            pass

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_metrics_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "populate_render_worker_state", _make_stub_render_worker_state())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(
        module,
        "auto_select_training_shape",
        lambda *args, **kwargs: "accounts/test/trainingShapes/sft",
    )
    monkeypatch.setattr(
        module,
        "render_messages_to_datum",
        lambda messages, **kwargs: SimpleNamespace(
            token_ids=[1, 2],
            datum=SimpleNamespace(
                model_input=SimpleNamespace(chunks=[SimpleNamespace(tokens=[1, 2])]),
                loss_fn_inputs={
                    "target_tokens": SimpleNamespace(data=[0, 0]),
                    "weights": SimpleNamespace(data=[1.0, 1.0]),
                },
                _test_id=messages[0]["content"],
            ),
        ),
    )
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)

    cfg = module.Config(
        log_path=str(tmp_path / "logs"),
        base_model="accounts/test/models/custom-sft",
        dataset=str(dataset_path),
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
        batch_size=2,
        epochs=1,
        seed=123,
        render_workers=1,
    )

    monkeypatch.setattr(module, "resolve_resume", lambda *args, **kwargs: None)
    module.main(cfg, rlor_mgr=FakeMgr())

    active["name"] = "resume"
    monkeypatch.setattr(
        module,
        "resolve_resume",
        lambda *args, **kwargs: SimpleNamespace(step=1, data_consumed=2),
    )
    module.main(cfg, rlor_mgr=FakeMgr())

    assert len(runs["fresh"]) == 3
    assert runs["resume"] == runs["fresh"][1:]


def test_main_resume_uses_raw_row_cursor_when_filtering_shrinks_batches(tmp_path, monkeypatch):
    """Resume should skip raw batches even when filtering makes a step smaller."""
    dataset_path = _write_dataset(
        tmp_path,
        [
            {"messages": [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]}
            for i in range(4)
        ],
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    runs: dict[str, list[list[str]]] = {"fresh": [], "resume": []}
    active = {"name": "fresh"}

    class FakeMgr:
        def create(self, config):
            job_id = f"job-{active['name']}"
            return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}")

        def wait_for_ready(self, job_id, **kwargs):
            return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}", base_url="https://unit.test")

        def cancel(self, job_id):
            pass

        def delete(self, job_id):
            pass

        def resolve_training_profile(self, shape_id):
            return _fake_profile(shape_id)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.job_id = kwargs.get("job_id", "job-sft")

        def forward_backward(self, batch, loss_fn="cross_entropy", loss_fn_config=None):
            runs[active["name"]].append([d._test_id for d in batch])
            return SimpleNamespace(
                metrics={"loss:sum": 1.0, "ce_loss_sum": 1.0, "response_tokens": len(batch)}
            )

        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={})

        def close(self):
            pass

    raw_batches = [
        [_test_datum("step-1-only")],
        [_test_datum("step-2-a"), _test_datum("step-2-b")],
    ]

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_metrics_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "populate_render_worker_state", _make_stub_render_worker_state())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(
        module,
        "auto_select_training_shape",
        lambda *args, **kwargs: "accounts/test/trainingShapes/sft",
    )
    monkeypatch.setattr(module, "make_render_dataloader", lambda *args, **kwargs: raw_batches)
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)

    cfg = module.Config(
        log_path=str(tmp_path / "logs"),
        base_model="accounts/test/models/custom-sft",
        dataset=str(dataset_path),
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
        batch_size=2,
        epochs=1,
        seed=123,
        render_workers=1,
        save_final_checkpoint=False,
    )

    monkeypatch.setattr(module, "resolve_resume", lambda *args, **kwargs: None)
    module.main(cfg, rlor_mgr=FakeMgr())

    active["name"] = "resume"
    monkeypatch.setattr(
        module,
        "resolve_resume",
        lambda *args, **kwargs: SimpleNamespace(
            step=1,
            data_consumed=1,
            raw_rows_consumed=2,
        ),
    )
    module.main(cfg, rlor_mgr=FakeMgr())

    assert runs["fresh"] == [["step-1-only"], ["step-2-a", "step-2-b"]]
    assert runs["resume"] == runs["fresh"][1:]


def test_completed_status_reports_actual_steps_when_filtering_drops_raw_batches(tmp_path, monkeypatch):
    """Final runner status should report 100% even when filtered raw batches reduce steps."""
    dataset_path = _write_dataset(
        tmp_path,
        [
            {"messages": [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]}
            for i in range(8)
        ],
    )
    status_path = tmp_path / "status.json"
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    class FakeMgr:
        def create(self, config):
            return SimpleNamespace(job_id="job-sft", job_name="jobs/job-sft")

        def wait_for_ready(self, job_id, **kwargs):
            return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}", base_url="https://unit.test")

        def cancel(self, job_id):
            pass

        def delete(self, job_id):
            pass

        def resolve_training_profile(self, shape_id):
            return _fake_profile(shape_id)

    class FakeClient:
        job_id = "job-sft"

        def __init__(self, *args, **kwargs):
            pass

        def forward_backward(self, batch, loss_fn="cross_entropy", loss_fn_config=None):
            return SimpleNamespace(
                metrics={"loss:sum": 1.0, "ce_loss_sum": 1.0, "response_tokens": len(batch)}
            )

        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={})

        def close(self):
            pass

    raw_batches = [
        [],
        [_test_datum("step-1")],
        [],
        [_test_datum("step-2")],
    ]

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_metrics_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "populate_render_worker_state", _make_stub_render_worker_state())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(
        module,
        "auto_select_training_shape",
        lambda *args, **kwargs: "accounts/test/trainingShapes/sft",
    )
    monkeypatch.setattr(module, "make_render_dataloader", lambda *args, **kwargs: raw_batches)
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(module, "resolve_resume", lambda *args, **kwargs: None)

    cfg = module.Config(
        log_path=str(tmp_path / "logs"),
        base_model="accounts/test/models/custom-sft",
        dataset=str(dataset_path),
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
        batch_size=2,
        epochs=1,
        render_workers=1,
        save_final_checkpoint=False,
        runner=module.RunnerConfig(status_file=str(status_path)),
    )

    result = module.main(cfg, rlor_mgr=FakeMgr())
    status = json.loads(status_path.read_text())

    assert result["steps"] == 2
    assert status["message"] == "done"
    assert status["details"][0]["percent"] == 100


# ---------------------------------------------------------------------------
# Eval auto carve-out tests
# ---------------------------------------------------------------------------


class TestComputeEvalCarveout:
    def test_basic_carveout(self):
        # 1000 samples, 10% ratio, max 100 → carve out 100
        assert module.compute_eval_carveout(1000) == 100

    def test_ratio_smaller_than_max(self):
        # 50 samples, 10% → 5 (ratio wins over max_seqs=100)
        assert module.compute_eval_carveout(50) == 5

    def test_max_seqs_cap(self):
        # 5000 samples, 10% = 500, but max_seqs=100 caps it
        assert module.compute_eval_carveout(5000) == 100

    def test_custom_ratio_and_max(self):
        assert module.compute_eval_carveout(200, max_ratio=0.2, max_seqs=50) == 40
        assert module.compute_eval_carveout(200, max_ratio=0.5, max_seqs=30) == 30

    def test_empty_dataset(self):
        assert module.compute_eval_carveout(0) == 0

    def test_single_sample(self):
        assert module.compute_eval_carveout(1) == 0

    def test_too_small_for_split(self):
        # 2 samples, 100% ratio, max_seqs=100 → carveout=2 >= total → 0
        assert module.compute_eval_carveout(2, max_ratio=1.0, max_seqs=100) == 0

    def test_minimal_split(self):
        # 10 samples, 10% → 1 eval, 9 training
        assert module.compute_eval_carveout(10) == 1

    def test_carveout_never_consumes_all(self):
        for n in range(0, 20):
            carveout = module.compute_eval_carveout(n)
            assert carveout < n or carveout == 0


def test_eval_auto_carveout_splits_data_and_runs_eval(tmp_path, monkeypatch):
    """Verify auto carve-out removes eval samples from training and runs eval after each epoch."""
    # Create 10 examples so carveout = min(10 * 0.1, 100) = 1
    rows = [
        {"messages": [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(10)
    ]
    dataset_path = _write_dataset(tmp_path, rows)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "train_batches": [],
        "eval_batches": [],
        "deleted_jobs": [],
    }

    class FakeMgr:
        def create(self, config):
            return SimpleNamespace(job_id="job-sft", job_name="jobs/job-sft")

        def wait_for_ready(self, job_id, **kwargs):
            return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}", base_url="https://unit.test")

        def cancel(self, job_id):
            events["deleted_jobs"].append(job_id)

        def delete(self, job_id):
            self.cancel(job_id)

        def resolve_training_profile(self, shape_id):
            return _fake_profile(shape_id)

    class FakeClient:
        job_id = "job-sft"

        def __init__(self, *args, **kwargs):
            pass

        def forward_backward(self, batch, loss_fn="cross_entropy", loss_fn_config=None):
            events["train_batches"].append(list(batch))
            return SimpleNamespace(metrics={"loss:sum": 1.0, "ce_loss_sum": 1.0, "response_tokens": 2})

        def forward_backward_custom(self, batch, loss_fn):
            events["eval_batches"].append(list(batch))
            return SimpleNamespace(metrics={"ce_loss_sum": 0.5, "response_tokens": 2})

        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={})

        def save_state(self, name):
            return SimpleNamespace(path=name)

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(path=f"{name}-sampler")

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return name

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_metrics_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "populate_render_worker_state", _make_stub_render_worker_state())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(
        module,
        "auto_select_training_shape",
        lambda *args, **kwargs: "accounts/test/trainingShapes/sft",
    )
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)

    def _fake_render(messages, **kwargs):
        row_idx = int(messages[0]["content"][1:])
        datum = SimpleNamespace(
            model_input=SimpleNamespace(chunks=[SimpleNamespace(tokens=[1, 2, 3])]),
            loss_fn_inputs={
                "target_tokens": SimpleNamespace(data=[0, 0]),
                "weights": SimpleNamespace(data=[1.0, 1.0]),
            },
            _test_id=f"example-{row_idx}",
        )
        return SimpleNamespace(token_ids=[1, 2, 3], datum=datum)

    monkeypatch.setattr(module, "render_messages_to_datum", _fake_render)

    cfg = module.Config(
        dataset=str(dataset_path),
        base_model="accounts/test/models/custom-sft",
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
        epochs=1,
        batch_size=9,  # All 9 training examples in one batch
        log_path=str(tmp_path / "sft_logs"),
        eval_auto_carveout=True,
        render_workers=1,
    )

    result = module.main(cfg, rlor_mgr=FakeMgr())

    # 10 examples, 10% carveout → 1 eval, 9 training
    assert result["steps"] == 1

    # Training should get 9 examples (not 10)
    train_ids = {d._test_id for batch in events["train_batches"] for d in batch}
    assert len(train_ids) == 9

    # Eval should get 1 example
    eval_ids = {d._test_id for batch in events["eval_batches"] for d in batch}
    assert len(eval_ids) == 1

    # They should be disjoint
    assert train_ids.isdisjoint(eval_ids)

    # Eval example comes from the 10-row dataset. We don't pin which specific
    # row lands in eval: auto-carveout now uses a seeded RNG so the eval set is
    # representative of the full distribution instead of biased toward the
    # first N rows of the input file.
    all_ids = {f"example-{i}" for i in range(10)}
    assert eval_ids.issubset(all_ids)


def test_eval_auto_carveout_eval_set_is_stable_across_epochs(tmp_path, monkeypatch):
    """Eval-loss curves are only meaningful if the eval set does not drift."""
    rows = [
        {"messages": [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(100)
    ]
    dataset_path = _write_dataset(tmp_path, rows)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "train_batches": [],
        "eval_batches": [],
        "deleted_jobs": [],
    }

    class FakeMgr:
        def create(self, config):
            return SimpleNamespace(job_id="job-sft", job_name="jobs/job-sft")

        def wait_for_ready(self, job_id, **kwargs):
            return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}", base_url="https://unit.test")

        def cancel(self, job_id):
            events["deleted_jobs"].append(job_id)

        def delete(self, job_id):
            self.cancel(job_id)

        def resolve_training_profile(self, shape_id):
            return _fake_profile(shape_id)

    class FakeClient:
        job_id = "job-sft"

        def __init__(self, *args, **kwargs):
            pass

        def forward_backward(self, batch, loss_fn="cross_entropy", loss_fn_config=None):
            events["train_batches"].append(list(batch))
            return SimpleNamespace(metrics={"loss:sum": 1.0, "ce_loss_sum": 1.0, "response_tokens": 2})

        def forward_backward_custom(self, batch, loss_fn):
            events["eval_batches"].append(list(batch))
            return SimpleNamespace(metrics={"ce_loss_sum": 0.5, "response_tokens": 2})

        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={})

        def save_state(self, name):
            return SimpleNamespace(path=name)

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(path=f"{name}-sampler")

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return name

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_metrics_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "populate_render_worker_state", _make_stub_render_worker_state())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(
        module,
        "auto_select_training_shape",
        lambda *args, **kwargs: "accounts/test/trainingShapes/sft",
    )
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)

    def _fake_render(messages, **kwargs):
        row_idx = int(messages[0]["content"][1:])
        datum = SimpleNamespace(
            model_input=SimpleNamespace(chunks=[SimpleNamespace(tokens=[1, 2, 3])]),
            loss_fn_inputs={
                "target_tokens": SimpleNamespace(data=[0, 0]),
                "weights": SimpleNamespace(data=[1.0, 1.0]),
            },
            _test_id=f"example-{row_idx}",
        )
        return SimpleNamespace(token_ids=[1, 2, 3], datum=datum)

    monkeypatch.setattr(module, "render_messages_to_datum", _fake_render)

    cfg = module.Config(
        dataset=str(dataset_path),
        base_model="accounts/test/models/custom-sft",
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
        epochs=3,
        batch_size=5,
        log_path=str(tmp_path / "sft_logs"),
        eval_auto_carveout=True,
        seed=7,
        render_workers=1,
    )

    module.main(cfg, rlor_mgr=FakeMgr())

    eval_batches = events["eval_batches"]
    eval_batches_per_epoch = 2
    per_epoch_eval_ids: list[list[str]] = []
    for epoch_idx in range(3):
        start = epoch_idx * eval_batches_per_epoch
        end = start + eval_batches_per_epoch
        epoch_ids = [d._test_id for batch in eval_batches[start:end] for d in batch]
        per_epoch_eval_ids.append(epoch_ids)

    assert per_epoch_eval_ids[0] == per_epoch_eval_ids[1]
    assert per_epoch_eval_ids[1] == per_epoch_eval_ids[2]

    train_ids = {d._test_id for batch in events["train_batches"] for d in batch}
    eval_ids = set(per_epoch_eval_ids[0])
    assert train_ids.isdisjoint(eval_ids)
    assert len(eval_ids) == 10


# ---------------------------------------------------------------------------
# _render_eagerly + _prepare_datasets: dataset-prep helpers extracted from
# main() so the three eval branches (none / explicit / auto-carveout) and
# the empty-dataset error are unit-testable without standing up the full
# trainer / SDK stack.
# ---------------------------------------------------------------------------


def _write_jsonl_at(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def _row(i: int) -> dict:
    return {"messages": [{"role": "user", "content": f"u{i}"}]}


def _expected_eval_indices(total_rows: int, carveout_count: int, seed: int) -> list[int]:
    indices = list(range(total_rows))
    random.Random(seed).shuffle(indices)
    return indices[:carveout_count]


def test_render_eagerly_drops_none(tmp_path, monkeypatch):
    """_render_eagerly materialises the first n rows and skips Nones."""
    path = _write_jsonl_at(tmp_path / "data.jsonl", [_row(i) for i in range(4)])
    monkeypatch.setattr(
        module,
        "_render_one_worker",
        lambda r: None if int(r["messages"][0]["content"][1:]) % 2 else f"d{r['messages'][0]['content'][1:]}",
    )
    ds = module.JsonlRenderDataset(str(path), module._render_one_worker)

    assert module._render_eagerly(ds, 4) == ["d0", "d2"]
    assert module._render_eagerly(ds, 0) == []


class TestPrepareDatasets:
    """Direct tests for the _prepare_datasets helper."""

    @staticmethod
    def _cfg(tmp_path, dataset_path, **overrides):
        return module.Config(
            log_path=str(tmp_path / "logs"),
            dataset=str(dataset_path),
            tokenizer_model="Qwen/Qwen3-1.7B",
            max_seq_len=32,
            **overrides,
        )

    def test_no_eval_branch(self, tmp_path, monkeypatch):
        path = _write_jsonl_at(tmp_path / "train.jsonl", [_row(i) for i in range(5)])
        monkeypatch.setattr(
            module, "_render_one_worker",
            lambda r: f"d-{r['messages'][0]['content']}",
        )

        train_ds, eval_data = module._prepare_datasets(self._cfg(tmp_path, path))

        assert len(train_ds) == 5
        assert eval_data == []

    def test_raises_on_empty_dataset(self, tmp_path, monkeypatch):
        path = _write_jsonl_at(tmp_path / "empty.jsonl", [])
        monkeypatch.setattr(module, "_render_one_worker", lambda r: r)

        with pytest.raises(RuntimeError, match="No examples found"):
            module._prepare_datasets(self._cfg(tmp_path, path))

    def test_explicit_eval_dataset(self, tmp_path, monkeypatch):
        train = _write_jsonl_at(tmp_path / "train.jsonl", [_row(i) for i in range(5)])
        eval_path = _write_jsonl_at(tmp_path / "eval.jsonl", [_row(100 + i) for i in range(3)])
        monkeypatch.setattr(
            module, "_render_one_worker",
            lambda r: f"d-{r['messages'][0]['content']}",
        )

        cfg = self._cfg(tmp_path, train, evaluation_dataset=str(eval_path))
        train_ds, eval_data = module._prepare_datasets(cfg)

        assert len(train_ds) == 5
        assert eval_data == ["d-u100", "d-u101", "d-u102"]

    def test_auto_carveout_slices_training_dataset(self, tmp_path, monkeypatch):
        # 20 rows, 10% ratio capped at max_eval_seqs=3 → 2 eval, 18 train.
        path = _write_jsonl_at(tmp_path / "train.jsonl", [_row(i) for i in range(20)])
        monkeypatch.setattr(
            module, "_render_one_worker",
            lambda r: f"d-{r['messages'][0]['content']}",
        )

        cfg = self._cfg(tmp_path, path, eval_auto_carveout=True, max_eval_seqs=3)
        train_ds, eval_data = module._prepare_datasets(cfg)

        eval_indices = _expected_eval_indices(total_rows=20, carveout_count=2, seed=cfg.seed)
        assert eval_data == [f"d-u{i}" for i in eval_indices]
        assert len(train_ds) == 18
        train_ids = {train_ds[i] for i in range(len(train_ds))}
        assert train_ids == {f"d-u{i}" for i in range(20) if i not in set(eval_indices)}

    def test_auto_carveout_drops_none_rendered_rows(self, tmp_path, monkeypatch):
        # 20 rows; whichever sampled eval rows render to None should be dropped.
        path = _write_jsonl_at(tmp_path / "train.jsonl", [_row(i) for i in range(20)])

        def render(r):
            content = r["messages"][0]["content"]
            return None if content == "u1" else f"d-{content}"

        monkeypatch.setattr(module, "_render_one_worker", render)

        cfg = self._cfg(tmp_path, path, eval_auto_carveout=True, max_eval_seqs=3)
        train_ds, eval_data = module._prepare_datasets(cfg)

        eval_indices = _expected_eval_indices(total_rows=20, carveout_count=2, seed=cfg.seed)
        expected_eval = [f"d-u{i}" for i in eval_indices if i != 1]
        assert eval_data == expected_eval
        assert len(train_ds) == 18

    def test_auto_carveout_uses_cfg_seed(self, tmp_path, monkeypatch):
        path = _write_jsonl_at(tmp_path / "train.jsonl", [_row(i) for i in range(20)])
        monkeypatch.setattr(
            module, "_render_one_worker",
            lambda r: f"d-{r['messages'][0]['content']}",
        )

        cfg_a = self._cfg(tmp_path, path, eval_auto_carveout=True, max_eval_seqs=3, seed=7)
        cfg_b = self._cfg(tmp_path, path, eval_auto_carveout=True, max_eval_seqs=3, seed=7)
        cfg_c = self._cfg(tmp_path, path, eval_auto_carveout=True, max_eval_seqs=3, seed=8)

        train_a, eval_a = module._prepare_datasets(cfg_a)
        train_b, eval_b = module._prepare_datasets(cfg_b)
        train_c, eval_c = module._prepare_datasets(cfg_c)

        assert eval_a == eval_b
        assert [train_a[i] for i in range(len(train_a))] == [train_b[i] for i in range(len(train_b))]
        assert eval_a != eval_c

    def test_auto_carveout_skipped_when_dataset_too_small(self, tmp_path, monkeypatch, caplog):
        path = _write_jsonl_at(tmp_path / "train.jsonl", [_row(0)])
        monkeypatch.setattr(module, "_render_one_worker", lambda r: r)

        cfg = self._cfg(tmp_path, path, eval_auto_carveout=True)
        with caplog.at_level("WARNING"):
            train_ds, eval_data = module._prepare_datasets(cfg)

        assert eval_data == []
        assert len(train_ds) == 1
        assert "too small for auto carve-out" in caplog.text
