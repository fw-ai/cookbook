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

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

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
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)

    cfg = module.Config(
        log_path=str(tmp_path / "logs"),
        dataset=str(dataset_path),
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
    )

    with pytest.raises(RuntimeError, match="No valid training examples"):
        module.main(cfg, rlor_mgr=FakeMgr())

    assert deleted_jobs == ["job-sft"]


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
    monkeypatch.setattr(module.transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "build_renderer", lambda *args, **kwargs: renderer)
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)

    cfg = module.Config(
        dataset=str(dataset_path),
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
        epochs=1,
        batch_size=1,
        grad_accum=1,
        log_path=str(tmp_path / "sft_logs"),
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
    monkeypatch.setattr(module.transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "build_renderer", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
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
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=32,
        epochs=1,
        batch_size=1,
        log_path=str(tmp_path / "sft_logs"),
    )

    result = module.main(cfg, rlor_mgr=FakeMgr())

    assert result["steps"] == 2
    assert events["optim_steps"] == 2
    assert len(events["batches"]) == 2
    assert [d._test_id for d in events["batches"][0]] == ["a1"]
    assert [d._test_id for d in events["batches"][1]] == ["a2"]
    assert events["deleted_jobs"] == ["job-sft"]
