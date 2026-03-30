from __future__ import annotations

import asyncio
import time
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


def test_tokenize_pairs_filters_and_collects(monkeypatch):
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

    tokenized, filtered_count = module._tokenize_pairs(
        raw_data, tokenizer=None, renderer=None, max_seq_len=32,
    )

    assert filtered_count == 1
    assert len(tokenized) == 2
    assert tokenized[0][0] == 0
    assert tokenized[0][1]["chosen_tokens"] == [1, 2, 3]
    assert tokenized[1][0] == 3
    assert tokenized[1][1]["chosen_tokens"] == [5, 6, 7]


def test_ref_forward_batch_computes_logprobs():
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
    pairs = [
        (0, {
            "chosen_tokens": [1, 2, 3],
            "rejected_tokens": [1, 2, 4],
            "response_start": 2,
            "chosen_datum": {"pair": 0, "kind": "chosen"},
            "rejected_datum": {"pair": 0, "kind": "rejected"},
        }),
        (3, {
            "chosen_tokens": [5, 6, 7],
            "rejected_tokens": [5, 6, 8],
            "response_start": 2,
            "chosen_datum": {"pair": 3, "kind": "chosen"},
            "rejected_datum": {"pair": 3, "kind": "rejected"},
        }),
    ]

    sem = asyncio.Semaphore(2)
    enriched = asyncio.run(
        module._ref_forward_batch(pairs, reference, sem, ref_batch_size=2)
    )

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
    assert len(enriched) == 2
    assert enriched[0] == (0, {
        "chosen_tokens": [1, 2, 3],
        "rejected_tokens": [1, 2, 4],
        "chosen_datum": {"pair": 0, "kind": "chosen"},
        "rejected_datum": {"pair": 0, "kind": "rejected"},
        "ref_chosen": [-0.1, -0.2],
        "ref_rejected": [-0.3],
        "response_start": 2,
    })
    assert enriched[1] == (3, {
        "chosen_tokens": [5, 6, 7],
        "rejected_tokens": [5, 6, 8],
        "chosen_datum": {"pair": 3, "kind": "chosen"},
        "rejected_datum": {"pair": 3, "kind": "rejected"},
        "ref_chosen": [-0.4],
        "ref_rejected": [-0.5, -0.6],
        "response_start": 2,
    })


def test_tokenize_pairs_preserves_multi_turn_preference_history():
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

    tokenized, filtered_count = module._tokenize_pairs(
        raw_data, tokenizer=None, renderer=renderer, max_seq_len=32,
    )

    assert filtered_count == 0
    assert len(tokenized) == 1
    idx, pair_data = tokenized[0]
    assert idx == 0
    assert pair_data["response_start"] == 4
    chosen_messages, _ = renderer.calls[0]
    rejected_messages, _ = renderer.calls[1]
    assert [m["role"] for m in chosen_messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in chosen_messages] == ["u1", "a1", "u2", "chosen"]
    assert [m["role"] for m in rejected_messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in rejected_messages] == ["u1", "a1", "u2", "rejected"]


def test_forward_backward_pairs_interleaves_and_builds_loss_fn(monkeypatch):
    captured = {}

    def fake_make_batch_dpo_loss_fn(ref_chosen, ref_rejected, response_starts, beta):
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

    result = module._forward_backward_pairs(batch_pairs, FakePolicy(), beta=0.25)

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
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "deleted_jobs": [],
        "lifecycle": [],
        "setup_deployment": [],
        "weight_syncer_saves": [],
        "wandb_finished": 0,
    }

    _DPO_JOB_IDS = {"dpo-policy": "policy-job", "dpo-reference": "reference-job"}

    class FakeRlorMgr:
        def resolve_training_profile(self, shape_id):
            return SimpleNamespace(
                max_supported_context_length=96,
                training_shape_version="accounts/test/trainingShapes/dpo/versions/1",
            )

        def create(self, config):
            events.setdefault("created_configs", []).append(config)
            jid = _DPO_JOB_IDS.get(config.display_name, "job-unknown")
            return SimpleNamespace(job_id=jid, job_name=f"jobs/{jid}")

        def wait_for_ready(self, job_id, **kwargs):
            return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}", base_url="https://unit.test")

        def cancel(self, job_id):
            events["lifecycle"].append(("delete", job_id))
            events["deleted_jobs"].append(job_id)

        def delete(self, job_id):
            self.cancel(job_id)

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

        def save_state(self, name, timeout=None):
            pass

        def save_weights_for_sampler_ext(self, name, checkpoint_type=None, timeout=None):
            return SimpleNamespace(snapshot_name=f"{name}-session")

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return f"tinker://unit/state/{name}"

        def close(self):
            events["lifecycle"].append(("close", self.job_id))

    class FakeWeightSyncer:
        def __init__(self, **kwargs):
            events["weight_syncer_init"] = kwargs

        def save_and_hotload(self, name):
            events["weight_syncer_saves"].append(name)

        def hotload(self, snapshot_name):
            events.setdefault("hotload_calls", []).append(snapshot_name)
            return True

    def fake_tokenize_pairs(*args, **kwargs):
        events["tokenize_args"] = args
        return (
            [(0, {
                "chosen_datum": {"id": "chosen"},
                "rejected_datum": {"id": "rejected"},
                "chosen_tokens": [1, 2],
                "rejected_tokens": [3, 4],
                "ref_chosen": [-0.1],
                "ref_rejected": [-0.2],
                "response_start": 3,
            })],
            1,
        )

    async def fake_train_loop(tokenized_pairs, reference, policy, adam_params,
                              weight_syncer, cfg, step_offset, on_ref_done=None, runner=None):
        events["train_loop"] = {
            "tokenized_pairs": tokenized_pairs,
            "reference_job_id": reference.job_id,
            "policy_job_id": policy.job_id,
            "cfg": cfg,
            "step_offset": step_offset,
        }
        if on_ref_done is not None:
            on_ref_done()
        return 2

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: events.__setitem__("wandb_finished", 1))
    monkeypatch.setattr(module, "setup_deployment", lambda *args, **kwargs: events["setup_deployment"].append((args, kwargs)))
    monkeypatch.setattr(module, "ThreadPoolExecutor", FakeThreadPoolExecutor)
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(module, "WeightSyncer", FakeWeightSyncer)
    monkeypatch.setattr(module, "_tokenize_pairs", fake_tokenize_pairs)
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
        "reference_job_id": None,
    }
    assert cfg.max_seq_len == 96
    assert len(events["setup_deployment"]) == 1
    assert [cfg.display_name for cfg in events["created_configs"]] == [
        "dpo-policy",
        "dpo-reference",
    ]
    assert events["created_configs"][0].hot_load_deployment_id == "dep-123"
    assert events["created_configs"][1].forward_only is True
    assert events["train_loop"]["reference_job_id"] == "reference-job"
    assert events["train_loop"]["policy_job_id"] == "policy-job"
    assert events["weight_syncer_saves"] == ["final-step-2"]

    ref_del_idx = events["deleted_jobs"].index("reference-job")
    pol_del_idx = events["deleted_jobs"].index("policy-job")
    assert ref_del_idx < pol_del_idx, "reference must be deleted before policy"
    assert events["deleted_jobs"].count("reference-job") == 1, "reference deleted exactly once"
    assert events["deleted_jobs"].count("policy-job") == 1, "policy deleted exactly once"
    ref_close_idx = events["lifecycle"].index(("close", "reference-job"))
    ref_delete_idx = events["lifecycle"].index(("delete", "reference-job"))
    pol_close_idx = events["lifecycle"].index(("close", "policy-job"))
    pol_delete_idx = events["lifecycle"].index(("delete", "policy-job"))
    assert ref_close_idx < ref_delete_idx
    assert pol_close_idx < pol_delete_idx
    assert events["wandb_finished"] == 1


def test_main_promotes_final_base_checkpoint(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://unit.test")

    events: dict[str, object] = {
        "deleted_jobs": [],
        "setup_deployment": [],
        "hotload_calls": [],
        "save_and_hotload_calls": [],
        "promotions": [],
        "wandb_finished": 0,
    }

    _DPO_JOB_IDS2 = {"dpo-policy": "policy-job", "dpo-reference": "reference-job"}

    class FakeRlorMgr:
        def resolve_training_profile(self, shape_id):
            return SimpleNamespace(
                max_supported_context_length=96,
                training_shape_version="accounts/test/trainingShapes/dpo/versions/1",
            )

        def create(self, config):
            jid = _DPO_JOB_IDS2.get(config.display_name, "job-unknown")
            return SimpleNamespace(job_id=jid, job_name=f"jobs/{jid}")

        def wait_for_ready(self, job_id, **kwargs):
            return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}", base_url="https://unit.test")

        def cancel(self, job_id):
            events["deleted_jobs"].append(job_id)

        def delete(self, job_id):
            self.cancel(job_id)

        def promote_checkpoint(self, job_id, checkpoint_id, output_model_id):
            events["promotions"].append((job_id, checkpoint_id, output_model_id))

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

        def save_state(self, name, timeout=None):
            pass

        def save_weights_for_sampler_ext(self, name, checkpoint_type=None, timeout=None):
            return SimpleNamespace(snapshot_name=f"{name}-session")

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return f"tinker://unit/state/{name}"

    class FakeWeightSyncer:
        def __init__(self, **kwargs):
            events["weight_syncer_init"] = kwargs

        def save_and_hotload(self, name, checkpoint_type=None):
            events["save_and_hotload_calls"].append((name, checkpoint_type))
            return f"{name}-session"

        def hotload(self, snapshot_name):
            events["hotload_calls"].append(snapshot_name)
            return True

    def fake_tokenize_pairs(*args, **kwargs):
        return (
            [(0, {
                "chosen_datum": {"id": "chosen"},
                "rejected_datum": {"id": "rejected"},
                "chosen_tokens": [1, 2],
                "rejected_tokens": [3, 4],
                "ref_chosen": [-0.1],
                "ref_rejected": [-0.2],
                "response_start": 3,
            })],
            0,
        )

    async def fake_train_loop(tokenized_pairs, reference, policy, adam_params,
                              weight_syncer, cfg, step_offset, on_ref_done=None, runner=None):
        events["train_loop"] = {
            "tokenized_pairs": tokenized_pairs,
            "policy_job_id": policy.job_id,
        }
        if on_ref_done is not None:
            on_ref_done()
        return 2

    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "wandb_finish", lambda: events.__setitem__("wandb_finished", 1))
    monkeypatch.setattr(module, "setup_deployment", lambda *args, **kwargs: events["setup_deployment"].append((args, kwargs)))
    monkeypatch.setattr(module, "ThreadPoolExecutor", FakeThreadPoolExecutor)
    monkeypatch.setattr(module, "ReconnectableClient", FakeClient)
    monkeypatch.setattr(module, "WeightSyncer", FakeWeightSyncer)
    monkeypatch.setattr(module, "_tokenize_pairs", fake_tokenize_pairs)
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
        infra=module.InfraConfig(training_shape_id="ts-qwen3-4b-smoke-v1", ref_training_shape_id="ts-qwen3-4b-smoke-v1"),
        deployment=module.DeployConfig(deployment_id="dep-123"),
        weight_sync=module.WeightSyncConfig(weight_sync_interval=1),
        output_model_id="promoted-dpo-model",
    )

    result = module.main(
        cfg,
        rlor_mgr=FakeRlorMgr(),
        deploy_mgr=FakeDeployMgr(),
    )

    assert result == {
        "steps": 2,
        "policy_job_id": "policy-job",
        "reference_job_id": None,
    }
    assert events["hotload_calls"] == ["step-2-session"]
    assert events["save_and_hotload_calls"] == []
    assert events["promotions"] == [
        ("policy-job", "step-2-session", "promoted-dpo-model"),
    ]

    ref_del_idx = events["deleted_jobs"].index("reference-job")
    pol_del_idx = events["deleted_jobs"].index("policy-job")
    assert ref_del_idx < pol_del_idx, "reference must be deleted before policy"
    assert events["deleted_jobs"].count("reference-job") == 1, "reference deleted exactly once"
    assert events["deleted_jobs"].count("policy-job") == 1, "policy deleted exactly once"
    assert events["wandb_finished"] == 1


def test_train_loop_pipeline_and_weight_sync(monkeypatch):
    """Test the pipelined _train_loop with real ref forward + training."""
    events: dict[str, object] = {
        "flush_batches": [],
        "optim_steps": 0,
        "weight_syncs": [],
        "dcp_saves": [],
        "metrics_logs": [],
        "wandb_logs": [],
        "ref_done_called": False,
    }

    monkeypatch.setattr(
        module,
        "_forward_backward_pairs",
        lambda batch, policy, beta: events["flush_batches"].append(
            (list(batch), beta)
        ) or SimpleNamespace(
            metrics={"dpo_loss": 1.5, "margin": 0.25, "accuracy": 0.75}
        ),
    )
    monkeypatch.setattr(module, "flush_timing", lambda: {"perf/fwd_bwd_time": 1.0})
    monkeypatch.setattr(module, "log_metrics_json", lambda step, **kwargs: events["metrics_logs"].append((step, kwargs)))
    monkeypatch.setattr(module, "wandb_log", lambda payload, step: events["wandb_logs"].append((step, payload)))

    class FakePolicy:
        job_id = "fake-policy-job"

        def optim_step(self, _params, **kwargs):
            events["optim_steps"] += 1
            return SimpleNamespace(metrics={"optimizer/lr": 1e-4})

        def save_state(self, name, timeout=None):
            pass

        def resolve_checkpoint_path(self, name, source_job_id=None):
            return f"tinker://unit/state/{name}"

    class FakeReference:
        def forward(self, datums, loss_fn):
            n_pairs = len(datums) // 2
            return SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": SimpleNamespace(data=[-0.1 * (i + 1)])}
                    for i in range(len(datums))
                ]
            )

    class FakeWeightSyncer:
        def save_and_hotload(self, name):
            events["weight_syncs"].append(name)

    tokenized_pairs = [
        (0, {"chosen_datum": {"id": "c0"}, "rejected_datum": {"id": "r0"},
             "chosen_tokens": [1, 2, 3], "rejected_tokens": [1, 2, 4], "response_start": 3}),
        (1, {"chosen_datum": {"id": "c1"}, "rejected_datum": {"id": "r1"},
             "chosen_tokens": [5, 6], "rejected_tokens": [5, 7], "response_start": 4}),
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_log_path:
        cfg = module.Config(
            log_path=tmp_log_path,
            beta=0.2,
            epochs=1,
            batch_size=2,
            weight_sync=module.WeightSyncConfig(weight_sync_interval=1, dcp_save_interval=1),
        )

        def _on_ref_done():
            events["ref_done_called"] = True

        step = asyncio.run(
            module._train_loop(
                tokenized_pairs,
                FakeReference(),
                FakePolicy(),
                adam_params={"lr": 1e-4},
                weight_syncer=FakeWeightSyncer(),
                cfg=cfg,
                step_offset=0,
                on_ref_done=_on_ref_done,
            )
        )

        assert step == 1
        assert events["ref_done_called"]
        assert len(events["flush_batches"]) == 1
        assert events["flush_batches"][0][1] == 0.2
        trained_pairs = events["flush_batches"][0][0]
        assert len(trained_pairs) == 2
        assert "ref_chosen" in trained_pairs[0]
        assert "ref_rejected" in trained_pairs[0]
        assert events["optim_steps"] == 1
        assert events["weight_syncs"] == ["step-1"]
        # DCP save now goes through save_checkpoint -> checkpoints.jsonl
        import json, os
        cp_file = os.path.join(tmp_log_path, "checkpoints.jsonl")
        assert os.path.exists(cp_file)
        with open(cp_file) as f:
            cp = json.loads(f.readline())
        assert cp["name"] == "step-1"
        assert cp["step"] == 1
        assert events["metrics_logs"][0][0] == 1
        assert events["metrics_logs"][0][1]["dpo_loss"] == 1.5
        assert len(events["wandb_logs"]) == 1


def test_pipeline_overlap_ref_freed_before_training_done():
    """Verify the producer finishes (and on_ref_done fires) while training
    is still in progress — the core benefit of the pipeline."""
    timeline = []

    class SlowPolicy:
        def optim_step(self, _params, **kwargs):
            return SimpleNamespace(metrics={})

    class FastReference:
        def forward(self, datums, loss_fn):
            return SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": SimpleNamespace(data=[-0.1])}
                    for _ in datums
                ]
            )

    def slow_fwd_bwd(batch, policy, beta):
        time.sleep(0.15)
        return SimpleNamespace(
            metrics={"dpo_loss": 0.5, "margin": 0.1, "accuracy": 0.9}
        )

    def on_ref_done():
        timeline.append(("ref_done", time.monotonic()))

    cfg = module.Config(
        log_path="/tmp/dpo_test_logs",
        beta=0.1,
        epochs=1,
        batch_size=1,
        ref_cache_concurrency=4,
    )

    import training.recipes.dpo_loop as mod

    orig_fwd = mod._forward_backward_pairs
    mod._forward_backward_pairs = slow_fwd_bwd
    orig_flush = mod.flush_timing
    mod.flush_timing = lambda: {}
    orig_log = mod.log_metrics_json
    mod.log_metrics_json = lambda *a, **kw: None
    orig_wandb = mod.wandb_log
    mod.wandb_log = lambda *a, **kw: None

    try:
        tokenized = [
            (i, {"chosen_datum": {"id": f"c{i}"}, "rejected_datum": {"id": f"r{i}"},
                 "chosen_tokens": [1, 2], "rejected_tokens": [3, 4], "response_start": 1})
            for i in range(4)
        ]

        class FakeWS:
            def save_and_hotload(self, name): pass

        t0 = time.monotonic()
        step = asyncio.run(
            mod._train_loop(
                tokenized, FastReference(), SlowPolicy(),
                adam_params={"lr": 1e-4},
                weight_syncer=FakeWS(),
                cfg=cfg,
                step_offset=0,
                on_ref_done=on_ref_done,
            )
        )
        t_end = time.monotonic()

        assert step == 4
        assert len(timeline) == 1
        ref_done_t = timeline[0][1] - t0
        total_t = t_end - t0
        assert ref_done_t < total_t * 0.8, (
            f"ref_done should fire well before training finishes "
            f"(ref_done={ref_done_t:.2f}s, total={total_t:.2f}s)"
        )
    finally:
        mod._forward_backward_pairs = orig_fwd
        mod.flush_timing = orig_flush
        mod.log_metrics_json = orig_log
        mod.wandb_log = orig_wandb
