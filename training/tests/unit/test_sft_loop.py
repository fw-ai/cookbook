from __future__ import annotations

import json
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


def test_main_requires_tokenizer_model(tmp_path, monkeypatch):
    dataset_path = _write_dataset(
        tmp_path,
        [{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}],
    )
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)

    cfg = module.Config(dataset=str(dataset_path), tokenizer_model="", max_seq_len=32)

    with pytest.raises(ValueError, match="tokenizer_model"):
        module.main(cfg)


def test_main_uses_training_shape_and_trains_batches(tmp_path, monkeypatch):
    dataset_path = _write_dataset(
        tmp_path,
        [
            {"messages": [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]},
            {"messages": [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]},
        ],
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "save_state": [],
        "save_weights": [],
        "batches": [],
        "optim_steps": 0,
        "wandb_logs": [],
        "metrics_logs": [],
        "deleted_jobs": [],
        "wandb_finished": 0,
    }

    class FakeMgr:
        def __init__(self):
            self.resolved_shapes: list[str] = []

        def resolve_training_profile(self, shape_id):
            self.resolved_shapes.append(shape_id)
            return SimpleNamespace(max_supported_context_length=64)

        def delete(self, job_id):
            events["deleted_jobs"].append(job_id)

    class FakeInner:
        def save_state(self, name):
            events["save_state"].append(name)

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            events["save_weights"].append((name, checkpoint_type))
            return SimpleNamespace(path=f"gs://unit/{name}")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.inner = FakeInner()

        def forward_backward_custom(self, batch, loss_fn):
            events["batches"].append((list(batch), loss_fn))
            return SimpleNamespace(metrics={"ce_loss_sum": 2.0, "response_tokens": 4})

        def optim_step(self, _params):
            events["optim_steps"] += 1
            return SimpleNamespace(metrics={"optimizer/lr": 1e-4})

    rendered = iter(
        [
            SimpleNamespace(token_ids=[1, 2, 3], datum={"id": "datum-1"}),
            SimpleNamespace(token_ids=[4, 5, 6], datum={"id": "datum-2"}),
        ]
    )

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: events.__setitem__("wandb_finished", 1))
    monkeypatch.setattr(module, "wandb_log", lambda payload, step: events["wandb_logs"].append((step, payload)))
    monkeypatch.setattr(module, "log_metrics_json", lambda step, **kwargs: events["metrics_logs"].append((step, kwargs)))
    monkeypatch.setattr(module.transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "build_renderer", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(module, "render_messages_to_datum", lambda *args, **kwargs: next(rendered))
    monkeypatch.setattr(module, "create_trainer_job", lambda *args, **kwargs: SimpleNamespace(job_id="job-sft"))
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)

    mgr = FakeMgr()
    cfg = module.Config(
        dataset=str(dataset_path),
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=None,
        epochs=1,
        batch_size=1,
        grad_accum=1,
        dcp_save_interval=2,
        infra=module.InfraConfig(training_shape_id="ts-qwen3-4b-smoke-v1"),
    )

    result = module.main(cfg, rlor_mgr=mgr)

    assert result == {"steps": 2, "job_id": "job-sft"}
    assert cfg.max_seq_len == 64
    assert mgr.resolved_shapes == ["ts-qwen3-4b-smoke-v1"]
    assert [batch for batch, _loss_fn in events["batches"]] == [
        [{"id": "datum-1"}],
        [{"id": "datum-2"}],
    ]
    assert all(callable(loss_fn) for _batch, loss_fn in events["batches"])
    assert events["optim_steps"] == 2
    assert events["save_state"] == ["step-2", "final-step-2"]
    assert events["save_weights"] == [("final-step-2", "base")]
    assert events["deleted_jobs"] == ["job-sft"]
    assert events["wandb_finished"] == 1
    assert [step for step, _ in events["metrics_logs"]] == [1, 2]


def test_main_raises_when_all_examples_are_filtered(tmp_path, monkeypatch):
    dataset_path = _write_dataset(
        tmp_path,
        [{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}],
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    deleted_jobs: list[str] = []

    class FakeMgr:
        def delete(self, job_id):
            deleted_jobs.append(job_id)

    class FakeInner:
        def save_state(self, _name):
            raise AssertionError("save_state should not be called when no examples are valid")

        def save_weights_for_sampler_ext(self, *_args, **_kwargs):
            raise AssertionError("save_weights_for_sampler_ext should not be called")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.inner = FakeInner()

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module.transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "build_renderer", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(
        module,
        "render_messages_to_datum",
        lambda *args, **kwargs: SimpleNamespace(token_ids=[1], datum={"id": "too-short"}),
    )
    monkeypatch.setattr(module, "create_trainer_job", lambda *args, **kwargs: SimpleNamespace(job_id="job-sft"))
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)

    cfg = module.Config(
        dataset=str(dataset_path),
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
    )

    with pytest.raises(RuntimeError, match="No valid training examples"):
        module.main(cfg, rlor_mgr=FakeMgr())

    assert deleted_jobs == ["job-sft"]


def test_main_uses_real_multi_turn_renderer_path(tmp_path, monkeypatch):
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
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "batches": [],
        "deleted_jobs": [],
    }
    renderer = StubRenderer(
        tokens=[100, 101, 102, 103, 104, 105, 106],
        weights=[0, 0, 1, 1, 0, 1, 1],
    )

    class FakeMgr:
        def delete(self, job_id):
            events["deleted_jobs"].append(job_id)

    class FakeInner:
        def save_state(self, _name):
            return None

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(path=f"gs://unit/{name}")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.inner = FakeInner()

        def forward_backward_custom(self, batch, loss_fn):
            events["batches"].append(batch)
            return SimpleNamespace(metrics={"ce_loss_sum": 2.0, "response_tokens": 4})

        def optim_step(self, _params):
            return SimpleNamespace(metrics={"optimizer/lr": 1e-4})

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_metrics_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "build_renderer", lambda *args, **kwargs: renderer)
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(module, "create_trainer_job", lambda *args, **kwargs: SimpleNamespace(job_id="job-sft"))
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)

    cfg = module.Config(
        dataset=str(dataset_path),
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
        epochs=1,
        batch_size=1,
        grad_accum=1,
    )

    module.main(cfg, rlor_mgr=FakeMgr())

    normalized_messages, train_on_what = renderer.calls[0]
    assert [m["role"] for m in normalized_messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in normalized_messages] == ["u1", "a1", "u2", "a2"]
    assert train_on_what.value == "all_assistant_messages"

    datum = events["batches"][0][0]
    assert datum.loss_fn_inputs["target_tokens"].data == [101, 102, 103, 104, 105, 106]
    assert datum.loss_fn_inputs["weights"].data == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    assert events["deleted_jobs"] == ["job-sft"]
