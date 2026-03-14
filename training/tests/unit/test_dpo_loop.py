from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import transformers
import torch

import training.recipes.dpo_loop as module


class SequenceRenderer:
    def __init__(self, outputs: list[tuple[list[int], list[float]]]):
        self.outputs = [
            (torch.tensor(tokens, dtype=torch.int64), torch.tensor(weights, dtype=torch.float32))
            for tokens, weights in outputs
        ]
        self.calls: list = []

    def build_supervised_example(self, messages, train_on_what):
        self.calls.append((messages, train_on_what))
        return self.outputs[len(self.calls) - 1]


def test_tokenize_pair_handles_invalid_filtered_and_valid(monkeypatch):
    example = {"chosen": {"messages": []}, "rejected": {"messages": []}}

    monkeypatch.setattr(module, "render_preference_pair", lambda *args, **kwargs: None)
    assert module._tokenize_pair(example, tokenizer=None, renderer=None, max_seq_len=8) is None

    long_pair = SimpleNamespace(
        chosen_tokens=[1] * 9,
        rejected_tokens=[2, 3],
        response_start=3,
        chosen_datum={"kind": "chosen"},
        rejected_datum={"kind": "rejected"},
    )
    monkeypatch.setattr(module, "render_preference_pair", lambda *args, **kwargs: long_pair)
    assert module._tokenize_pair(example, tokenizer=None, renderer=None, max_seq_len=8) == "filtered"

    valid_pair = SimpleNamespace(
        chosen_tokens=[1, 2, 3, 4],
        rejected_tokens=[1, 2, 9],
        response_start=2,
        chosen_datum={"kind": "chosen"},
        rejected_datum={"kind": "rejected"},
    )
    monkeypatch.setattr(module, "render_preference_pair", lambda *args, **kwargs: valid_pair)
    assert module._tokenize_pair(example, tokenizer=None, renderer=None, max_seq_len=8) == {
        "chosen_tokens": [1, 2, 3, 4],
        "rejected_tokens": [1, 2, 9],
        "response_start": 2,
        "chosen_datum": {"kind": "chosen"},
        "rejected_datum": {"kind": "rejected"},
    }


def test_cache_ref_logprobs_batches_results(monkeypatch):
    raw_data = [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]
    tokenized_results = iter(
        [
            {
                "chosen_tokens": [1, 2, 3],
                "rejected_tokens": [1, 2, 4],
                "response_start": 2,
                "chosen_datum": {"pair": 0, "kind": "chosen"},
                "rejected_datum": {"pair": 0, "kind": "rejected"},
            },
            "filtered",
            None,
            {
                "chosen_tokens": [5, 6, 7],
                "rejected_tokens": [5, 6, 8],
                "response_start": 2,
                "chosen_datum": {"pair": 3, "kind": "chosen"},
                "rejected_datum": {"pair": 3, "kind": "rejected"},
            },
        ]
    )

    monkeypatch.setattr(
        module,
        "_tokenize_pair",
        lambda *args, **kwargs: next(tokenized_results),
    )

    class FakeReference:
        def __init__(self):
            self.calls = []

        def forward(self, datums, loss_fn):
            self.calls.append((datums, loss_fn))
            return SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": SimpleNamespace(data=[-0.1, -0.2])},
                    {"logprobs": SimpleNamespace(data=[-0.3])},
                    {"logprobs": SimpleNamespace(data=[-0.4])},
                    {"logprobs": SimpleNamespace(data=[-0.5, -0.6])},
                ]
            )

    reference = FakeReference()

    ref_cache, filtered_count = asyncio.run(
        module._cache_ref_logprobs(
            raw_data,
            reference,
            tokenizer=None,
            renderer=None,
            max_seq_len=32,
            concurrency=2,
            batch_size=2,
        )
    )

    assert filtered_count == 1
    assert reference.calls == [
        (
            [
                {"pair": 0, "kind": "chosen"},
                {"pair": 0, "kind": "rejected"},
                {"pair": 3, "kind": "chosen"},
                {"pair": 3, "kind": "rejected"},
            ],
            "cross_entropy",
        )
    ]
    assert ref_cache == {
        0: {
            "chosen_tokens": [1, 2, 3],
            "rejected_tokens": [1, 2, 4],
            "chosen_datum": {"pair": 0, "kind": "chosen"},
            "rejected_datum": {"pair": 0, "kind": "rejected"},
            "ref_chosen": [-0.1, -0.2],
            "ref_rejected": [-0.3],
            "response_start": 2,
        },
        3: {
            "chosen_tokens": [5, 6, 7],
            "rejected_tokens": [5, 6, 8],
            "chosen_datum": {"pair": 3, "kind": "chosen"},
            "rejected_datum": {"pair": 3, "kind": "rejected"},
            "ref_chosen": [-0.4],
            "ref_rejected": [-0.5, -0.6],
            "response_start": 2,
        },
    }


def test_cache_ref_logprobs_preserves_multi_turn_preference_history():
    raw_data = [
        {
            "chosen": {
                "messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "chosen"},
                ]
            },
            "rejected": {
                "messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "rejected"},
                ]
            },
        }
    ]
    renderer = SequenceRenderer(
        outputs=[
            ([1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 1, 1, 1]),
            ([1, 2, 3, 4, 8, 9], [0, 0, 0, 0, 1, 1]),
        ]
    )

    class FakeReference:
        def forward(self, datums, loss_fn):
            return SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": SimpleNamespace(data=[-0.1, -0.2, -0.3])},
                    {"logprobs": SimpleNamespace(data=[-0.4, -0.5])},
                ]
            )

    ref_cache, filtered_count = asyncio.run(
        module._cache_ref_logprobs(
            raw_data,
            FakeReference(),
            tokenizer=None,
            renderer=renderer,
            max_seq_len=32,
            concurrency=1,
            batch_size=1,
        )
    )

    assert filtered_count == 0
    assert list(ref_cache) == [0]
    assert ref_cache[0]["response_start"] == 4
    chosen_messages, _ = renderer.calls[0]
    rejected_messages, _ = renderer.calls[1]
    assert [m["role"] for m in chosen_messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in chosen_messages] == ["u1", "a1", "u2", "chosen"]
    assert [m["role"] for m in rejected_messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in rejected_messages] == ["u1", "a1", "u2", "rejected"]


def test_flush_batch_interleaves_pairs_and_builds_loss_fn(monkeypatch):
    captured = {}

    def fake_make_batch_dpo_loss_fn(ref_chosen, ref_rejected, response_starts, beta, **kwargs):
        captured["ref_chosen"] = ref_chosen
        captured["ref_rejected"] = ref_rejected
        captured["response_starts"] = response_starts
        captured["beta"] = beta
        return "loss-fn"

    class FakePolicy:
        def forward_backward_custom(self, datums, loss_fn):
            captured["datums"] = datums
            captured["loss_fn"] = loss_fn
            return "result"

    monkeypatch.setattr(module, "make_batch_dpo_loss_fn", fake_make_batch_dpo_loss_fn)

    batch_pairs = [
        {
            "chosen_datum": {"id": "chosen-0"},
            "rejected_datum": {"id": "rejected-0"},
            "ref_chosen": [-0.1],
            "ref_rejected": [-0.2],
            "response_start": 3,
        },
        {
            "chosen_datum": {"id": "chosen-1"},
            "rejected_datum": {"id": "rejected-1"},
            "ref_chosen": [-0.3],
            "ref_rejected": [-0.4],
            "response_start": 5,
        },
    ]

    result = module._flush_batch(batch_pairs, FakePolicy(), beta=0.25)

    assert result == "result"
    assert captured["datums"] == [
        {"id": "chosen-0"},
        {"id": "rejected-0"},
        {"id": "chosen-1"},
        {"id": "rejected-1"},
    ]
    assert captured["loss_fn"] == "loss-fn"
    assert captured["ref_chosen"] == [[-0.1], [-0.3]]
    assert captured["ref_rejected"] == [[-0.2], [-0.4]]
    assert captured["response_starts"] == [3, 5]
    assert captured["beta"] == 0.25


def test_main_requires_tokenizer_model(monkeypatch):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(log_path="/tmp/dpo_test_logs", dataset="/tmp/pairs.jsonl", tokenizer_model="")

    with pytest.raises(ValueError, match="tokenizer_model"):
        module.main(cfg)


def test_main_uses_profile_and_runs_training(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "create_trainer_job": [],
        "deleted_jobs": [],
        "setup_deployment": [],
        "weight_syncer_saves": [],
        "wandb_finished": 0,
    }

    class FakeRlorMgr:
        def resolve_training_profile(self, shape_id):
            return SimpleNamespace(max_supported_context_length=96)

        def delete(self, job_id):
            events["deleted_jobs"].append(job_id)

    class FakeDeployMgr:
        pass

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeThreadPoolExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return FakeFuture(fn(*args, **kwargs))

    class FakeClient:
        def __init__(self, _rlor_mgr, job_id, *_args, **_kwargs):
            self.job_id = job_id
            self.inner = object()

        def load_state_with_optimizer(self, path):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return f"tinker://unit/state/{name}"

    class FakeWeightSyncer:
        def __init__(self, **kwargs):
            events["weight_syncer_init"] = kwargs

        def save_and_hotload(self, name):
            events["weight_syncer_saves"].append(name)

        def save_dcp(self, name):
            events.setdefault("dcp_saves", []).append(name)

    async def fake_cache_ref_logprobs(*args, **kwargs):
        events["cache_args"] = {"args": args, "kwargs": kwargs}
        return (
            {
                0: {
                    "chosen_datum": {"id": "chosen"},
                    "rejected_datum": {"id": "rejected"},
                    "ref_chosen": [-0.1],
                    "ref_rejected": [-0.2],
                    "response_start": 3,
                }
            },
            1,
        )

    async def fake_train_loop(ref_cache, valid_indices, policy, adam_params, weight_syncer, cfg, step_offset):
        events["train_loop"] = {
            "ref_cache": ref_cache,
            "valid_indices": valid_indices,
            "policy_job_id": policy.job_id,
            "cfg": cfg,
            "step_offset": step_offset,
        }
        return 2

    def fake_create_trainer_job(*args, **kwargs):
        events["create_trainer_job"].append(kwargs)
        display_name = kwargs["display_name"]
        job_id = "policy-job" if display_name == "dpo-policy" else "reference-job"
        return SimpleNamespace(job_id=job_id)

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: events.__setitem__("wandb_finished", 1))
    monkeypatch.setattr(module, "setup_deployment", lambda *args, **kwargs: events["setup_deployment"].append((args, kwargs)))
    monkeypatch.setattr(module, "ThreadPoolExecutor", FakeThreadPoolExecutor)
    monkeypatch.setattr(module, "create_trainer_job", fake_create_trainer_job)
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(module, "WeightSyncer", FakeWeightSyncer)
    monkeypatch.setattr(module, "_cache_ref_logprobs", fake_cache_ref_logprobs)
    monkeypatch.setattr(module, "_train_loop", fake_train_loop)
    monkeypatch.setattr(module, "load_preference_dataset", lambda *args, **kwargs: [{"chosen": {}, "rejected": {}}])
    monkeypatch.setattr(module, "build_renderer", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())

    cfg = module.Config(
        log_path="/tmp/dpo_test_logs",
        base_model="accounts/test/models/qwen3-4b",
        dataset="/tmp/pairs.jsonl",
        tokenizer_model="Qwen/Qwen3-4B",
        max_seq_len=None,
        infra=module.InfraConfig(training_shape_id="ts-qwen3-4b-smoke-v1", ref_training_shape_id="ts-qwen3-4b-smoke-v1", extra_args=["--foo"]),
        deployment=module.DeployConfig(deployment_id="dep-123"),
        weight_sync=module.WeightSyncConfig(weight_sync_interval=1),
    )

    result = module.main(
        cfg,
        rlor_mgr=FakeRlorMgr(),
        deploy_mgr=FakeDeployMgr(),
    )

    assert result == {
        "steps": 2,
        "policy_job_id": "policy-job",
        "reference_job_id": "reference-job",
    }
    assert cfg.max_seq_len == 96
    assert len(events["setup_deployment"]) == 1
    assert [call["display_name"] for call in events["create_trainer_job"]] == [
        "dpo-policy",
        "dpo-reference",
    ]
    assert events["create_trainer_job"][0]["hot_load_deployment_id"] == "dep-123"
    assert events["create_trainer_job"][1]["forward_only"] is True
    assert events["cache_args"]["kwargs"] == {
        "concurrency": cfg.ref_cache_concurrency,
        "batch_size": cfg.ref_cache_batch_size,
    }
    assert events["train_loop"]["valid_indices"] == [0]
    assert events["train_loop"]["policy_job_id"] == "policy-job"
    assert events["weight_syncer_saves"] == ["final-step-2"]
    assert events["deleted_jobs"] == ["reference-job", "policy-job"]
    assert events["wandb_finished"] == 1


def test_train_loop_runs_accumulation_and_weight_sync(monkeypatch):
    events: dict[str, object] = {
        "flush_batches": [],
        "optim_steps": 0,
        "weight_syncs": [],
        "dcp_saves": [],
        "metrics_logs": [],
        "wandb_logs": [],
    }

    monkeypatch.setattr(
        module,
        "_flush_batch",
        lambda batch, policy, beta, **kwargs: events["flush_batches"].append((list(batch), beta)) or SimpleNamespace(
            metrics={"dpo_loss": 1.5, "margin": 0.25, "accuracy": 0.75}
        ),
    )
    monkeypatch.setattr(module, "flush_timing", lambda: {"perf/fwd_bwd_time": 1.0})
    monkeypatch.setattr(module, "log_metrics_json", lambda step, **kwargs: events["metrics_logs"].append((step, kwargs)))
    monkeypatch.setattr(module, "wandb_log", lambda payload, step: events["wandb_logs"].append((step, payload)))

    class FakePolicy:
        def optim_step(self, _params, **kwargs):
            events["optim_steps"] += 1
            return SimpleNamespace(metrics={"optimizer/lr": 1e-4})

    class FakeWeightSyncer:
        def save_and_hotload(self, name):
            events["weight_syncs"].append(name)

        def save_dcp(self, name):
            events["dcp_saves"].append(name)

    ref_cache = {
        0: {"chosen_datum": {"id": "c0"}, "rejected_datum": {"id": "r0"}, "ref_chosen": [-0.1], "ref_rejected": [-0.2], "response_start": 3},
        1: {"chosen_datum": {"id": "c1"}, "rejected_datum": {"id": "r1"}, "ref_chosen": [-0.3], "ref_rejected": [-0.4], "response_start": 4},
    }
    cfg = module.Config(
        log_path="/tmp/dpo_test_logs",
        beta=0.2,
        epochs=1,
        batch_size=1,
        grad_accum=2,
        weight_sync=module.WeightSyncConfig(weight_sync_interval=1, dcp_save_interval=1),
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
        )
    )

    assert step == 1
    assert events["flush_batches"] == [
        ([ref_cache[0]], 0.2),
        ([ref_cache[1]], 0.2),
    ]
    assert events["optim_steps"] == 1
    assert events["weight_syncs"] == ["step-1"]
    assert events["dcp_saves"] == ["step-1"]
    assert events["metrics_logs"] == [(1, {"dpo_loss": 1.5, "margin": 0.25, "accuracy": 0.75})]
    assert len(events["wandb_logs"]) == 1
