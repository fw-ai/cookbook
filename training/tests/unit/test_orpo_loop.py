from __future__ import annotations

from types import SimpleNamespace

import pytest
import transformers

import training.recipes.orpo_loop as module


def test_main_rejects_invalid_base_model(monkeypatch):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(log_path="/tmp/orpo_test_logs", base_model="qwen3-4b", dataset="/tmp/pairs.jsonl", tokenizer_model="Qwen/Qwen3-4B")

    with pytest.raises(RuntimeError, match="Invalid base_model"):
        module.main(cfg)


def test_main_rejects_invalid_output_model_id(monkeypatch):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(
        log_path="/tmp/orpo_test_logs",
        base_model="accounts/test/models/qwen3-4b",
        dataset="/tmp/pairs.jsonl",
        tokenizer_model="Qwen/Qwen3-4B",
        output_model_id="bad_name",
    )

    with pytest.raises(RuntimeError, match="Invalid output_model_id"):
        module.main(cfg)


def test_main_uses_profile_and_trains_pairs(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "forward_batches": [],
        "optim_steps": 0,
        "save_weights": [],
        "deleted_jobs": [],
        "wandb_finished": 0,
        "metrics_logs": [],
        "wandb_logs": [],
        "promotions": [],
    }

    class FakeMgr:
        def __init__(self):
            self.resolved_shapes: list[str] = []

        def resolve_training_profile(self, shape_id):
            self.resolved_shapes.append(shape_id)
            return SimpleNamespace(max_supported_context_length=48)

        def delete(self, job_id):
            events["deleted_jobs"].append(job_id)

    class FakeInner:
        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            events["save_weights"].append((name, checkpoint_type))
            return SimpleNamespace(path=f"gs://unit/{name}-session", snapshot_name=f"{name}-session")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.inner = FakeInner()

        def forward_backward_custom(self, batch, loss_fn):
            events["forward_batches"].append((list(batch), loss_fn))
            return SimpleNamespace(
                metrics={
                    "orpo_loss": 1.2,
                    "sft_loss": 0.7,
                    "or_loss": 0.5,
                    "log_odds_ratio": 0.1,
                    "accuracy": 0.75,
                }
            )

        def optim_step(self, _params, **kwargs):
            events["optim_steps"] += 1
            return SimpleNamespace()

        def save_state(self, name):
            return SimpleNamespace(path=f"tinker://unit/state/{name}")

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            events["save_weights"].append((name, checkpoint_type))
            return SimpleNamespace(
                path=f"tinker://unit/sampler/{name}-session",
                snapshot_name=f"{name}-session",
            )

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return f"tinker://unit/state/{name}"

    pair_outputs = iter(
        [
            SimpleNamespace(
                chosen_tokens=[1, 2, 3],
                rejected_tokens=[1, 2, 4],
                response_start=2,
                chosen_datum={"id": "chosen-0"},
                rejected_datum={"id": "rejected-0"},
            ),
            SimpleNamespace(
                chosen_tokens=[5, 6, 7],
                rejected_tokens=[5, 6, 8],
                response_start=2,
                chosen_datum={"id": "chosen-1"},
                rejected_datum={"id": "rejected-1"},
            ),
        ]
    )

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: events.__setitem__("wandb_finished", 1))
    monkeypatch.setattr(module, "wandb_log", lambda payload, step: events["wandb_logs"].append((step, payload)))
    monkeypatch.setattr(module, "log_metrics_json", lambda step, **kwargs: events["metrics_logs"].append((step, kwargs)))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "build_renderer", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(
        module,
        "load_preference_dataset",
        lambda *args, **kwargs: [
            {"chosen": {"messages": []}, "rejected": {"messages": []}},
            {"chosen": {"messages": []}, "rejected": {"messages": []}},
        ],
    )
    monkeypatch.setattr(module, "render_preference_pair", lambda *args, **kwargs: next(pair_outputs))
    monkeypatch.setattr(module, "create_trainer_job", lambda *args, **kwargs: SimpleNamespace(job_id="job-orpo"))
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(module, "make_batch_orpo_loss_fn", lambda response_starts, orpo_lambda: ("loss", response_starts, orpo_lambda))
    monkeypatch.setattr(module.random, "shuffle", lambda seq: None)
    monkeypatch.setattr(
        "training.utils.checkpoint_utils.promote_checkpoint",
        lambda mgr, job_id, checkpoint_id, output_model_id: events["promotions"].append(
            (job_id, checkpoint_id, output_model_id)
        ),
    )

    mgr = FakeMgr()
    cfg = module.Config(
        log_path="/tmp/orpo_test_logs",
        base_model="accounts/test/models/qwen3-4b",
        dataset="/tmp/pairs.jsonl",
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=None,
        epochs=1,
        grad_accum=1,
        infra=module.InfraConfig(training_shape_id="ts-qwen3-4b-smoke-v1"),
        output_model_id="promoted-orpo-model",
    )

    result = module.main(cfg, rlor_mgr=mgr)

    assert result == {"steps": 2, "job_id": "job-orpo"}
    assert cfg.max_seq_len == 48
    assert mgr.resolved_shapes == ["ts-qwen3-4b-smoke-v1"]
    assert [batch for batch, _loss_fn in events["forward_batches"]] == [
        [{"id": "chosen-0"}, {"id": "rejected-0"}],
        [{"id": "chosen-1"}, {"id": "rejected-1"}],
    ]
    assert events["optim_steps"] == 2
    assert events["save_weights"] == [("final-step-2", "base")]
    assert events["promotions"] == [
        ("job-orpo", "final-step-2-session", "promoted-orpo-model"),
    ]
    assert events["deleted_jobs"] == ["job-orpo"]
    assert events["wandb_finished"] == 1
    assert [step for step, _ in events["metrics_logs"]] == [1, 2]


def test_main_batches_pairs_per_optimizer_step(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "forward_batches": [],
        "optim_steps": 0,
        "deleted_jobs": [],
    }

    class FakeMgr:
        def resolve_training_profile(self, shape_id):
            return SimpleNamespace(max_supported_context_length=48)

        def delete(self, job_id):
            events["deleted_jobs"].append(job_id)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.inner = object()

        def forward_backward_custom(self, batch, loss_fn):
            events["forward_batches"].append(list(batch))
            return SimpleNamespace(
                metrics={
                    "orpo_loss": 1.2,
                    "sft_loss": 0.7,
                    "or_loss": 0.5,
                    "log_odds_ratio": 0.1,
                    "accuracy": 0.75,
                }
            )

        def optim_step(self, _params):
            events["optim_steps"] += 1
            return SimpleNamespace()

        def save_state(self, name):
            return SimpleNamespace(path=f"tinker://unit/state/{name}")

        def save_weights_for_sampler_ext(self, name, checkpoint_type="base"):
            return SimpleNamespace(path=f"tinker://unit/sampler/{name}", snapshot_name=f"{name}-session")

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return f"tinker://unit/state/{name}"

    pair_outputs = iter(
        [
            SimpleNamespace(
                chosen_tokens=[1, 2, 3],
                rejected_tokens=[1, 2, 4],
                response_start=2,
                chosen_datum={"id": "chosen-0"},
                rejected_datum={"id": "rejected-0"},
            ),
            SimpleNamespace(
                chosen_tokens=[5, 6, 7],
                rejected_tokens=[5, 6, 8],
                response_start=2,
                chosen_datum={"id": "chosen-1"},
                rejected_datum={"id": "rejected-1"},
            ),
        ]
    )

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: None)
    monkeypatch.setattr(module, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_metrics_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "build_renderer", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(
        module,
        "load_preference_dataset",
        lambda *args, **kwargs: [
            {"chosen": {"messages": []}, "rejected": {"messages": []}},
            {"chosen": {"messages": []}, "rejected": {"messages": []}},
        ],
    )
    monkeypatch.setattr(module, "render_preference_pair", lambda *args, **kwargs: next(pair_outputs))
    monkeypatch.setattr(module, "create_trainer_job", lambda *args, **kwargs: SimpleNamespace(job_id="job-orpo"))
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(module, "make_batch_orpo_loss_fn", lambda response_starts, orpo_lambda: ("loss", response_starts, orpo_lambda))
    monkeypatch.setattr(module.random, "shuffle", lambda seq: None)

    cfg = module.Config(
        log_path="/tmp/orpo_test_logs",
        base_model="accounts/test/models/qwen3-4b",
        dataset="/tmp/pairs.jsonl",
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=None,
        epochs=1,
        grad_accum=2,
        infra=module.InfraConfig(training_shape_id="ts-qwen3-4b-smoke-v1"),
    )

    result = module.main(cfg, rlor_mgr=FakeMgr())

    assert result == {"steps": 1, "job_id": "job-orpo"}
    assert events["forward_batches"] == [[
        {"id": "chosen-0"},
        {"id": "rejected-0"},
        {"id": "chosen-1"},
        {"id": "rejected-1"},
    ]]
    assert events["optim_steps"] == 1
    assert events["deleted_jobs"] == ["job-orpo"]
