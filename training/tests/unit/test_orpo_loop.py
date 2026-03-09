from __future__ import annotations

from types import SimpleNamespace

import pytest
import transformers

import training.recipes.orpo_loop as module


def test_main_rejects_invalid_base_model(monkeypatch):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(base_model="qwen3-4b", dataset="/tmp/pairs.jsonl", tokenizer_model="Qwen/Qwen3-4B")

    with pytest.raises(ValueError, match="Invalid base_model"):
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
            return SimpleNamespace(path=f"gs://unit/{name}")

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

        def optim_step(self, _params):
            events["optim_steps"] += 1
            return SimpleNamespace()

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
    monkeypatch.setattr(module, "make_orpo_loss_fn", lambda response_start, orpo_lambda: ("loss", response_start, orpo_lambda))
    monkeypatch.setattr(module.random, "shuffle", lambda seq: None)

    mgr = FakeMgr()
    cfg = module.Config(
        base_model="accounts/test/models/qwen3-4b",
        dataset="/tmp/pairs.jsonl",
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=None,
        epochs=1,
        grad_accum=1,
        infra=module.InfraConfig(training_shape_id="ts-qwen3-4b-smoke-v1"),
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
    assert events["deleted_jobs"] == ["job-orpo"]
    assert events["wandb_finished"] == 1
    assert [step for step, _ in events["metrics_logs"]] == [1, 2]
