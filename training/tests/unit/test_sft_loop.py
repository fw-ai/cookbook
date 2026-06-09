from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

import training.recipes.sft_loop as module
from training.utils.checkpoints import TrainingCheckpoints


def _patch_resume(monkeypatch, resume_info):
    """Replace TrainingCheckpoints.resume on the class for unit tests.

    ``resume_info`` is what ``ckpt.resume()`` should return: ``None`` for
    a fresh start, or a SimpleNamespace with ``step`` / ``data_consumed``
    fields. After PR #385 ``data_consumed`` carries SFT's
    ``raw_rows_consumed`` semantically (single load-bearing counter).
    """
    monkeypatch.setattr(
        TrainingCheckpoints, "resume", lambda self, **kwargs: resume_info,
    )


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


def test_init_render_worker_forwards_tokenizer_revision(monkeypatch):
    captured: dict = {}

    def fake_populate_render_worker_state(state, **kwargs):
        captured["state"] = state
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        module, "populate_render_worker_state", fake_populate_render_worker_state
    )

    module._init_render_worker(
        "moonshotai/Kimi-K2.6", "kimi_k25", "all_assistant_messages", 4096, "2755962"
    )

    assert captured["state"] is module._worker_state
    assert captured["kwargs"]["tokenizer_model"] == "moonshotai/Kimi-K2.6"
    assert captured["kwargs"]["tokenizer_revision"] == "2755962"
    assert captured["kwargs"]["renderer_name"] == "kimi_k25"
    assert captured["kwargs"]["max_seq_len"] == 4096


def test_configure_render_sample_state_does_not_repopulate_renderer(tmp_path, monkeypatch):
    calls = {"populate": 0}
    tokenizer = object()

    def fake_populate_render_worker_state(state, **kwargs):
        calls["populate"] += 1
        state.update(
            tokenizer=tokenizer,
            renderer=object(),
            max_seq_len=kwargs["max_seq_len"],
            train_on_what=kwargs["train_on_what"],
        )

    monkeypatch.setattr(
        module, "populate_render_worker_state", fake_populate_render_worker_state
    )
    monkeypatch.setattr(module, "resolve_renderer_name", lambda *args, **kwargs: "unit-renderer")

    module._worker_state.clear()
    module._init_render_worker("model", "renderer", "all_assistant_messages", 128)
    module._configure_render_sample_state(str(tmp_path), 3)

    assert calls["populate"] == 1
    assert module._worker_state["tokenizer"] is tokenizer
    assert module._worker_state["resolved_renderer_name"] == "unit-renderer"
    assert module._worker_state["render_samples_local_dir"] == str(tmp_path)
    assert module._worker_state["render_samples_limit"] == 3
    assert module._worker_state["render_samples_written"] == 0


def test_resolve_render_samples_limit(monkeypatch):
    monkeypatch.delenv(module.RENDER_SAMPLE_LIMIT_ENV, raising=False)

    assert module._resolve_render_samples_limit(None) == module.DEFAULT_RENDER_SAMPLE_LIMIT
    assert module._resolve_render_samples_limit(7) == 7
    assert module._resolve_render_samples_limit(-1) is None

    monkeypatch.setenv(module.RENDER_SAMPLE_LIMIT_ENV, "0")
    assert module._resolve_render_samples_limit(None) == 0

    monkeypatch.setenv(module.RENDER_SAMPLE_LIMIT_ENV, "full")
    assert module._resolve_render_samples_limit(3) is None

    monkeypatch.setenv(module.RENDER_SAMPLE_LIMIT_ENV, "not-an-int")
    assert module._resolve_render_samples_limit(3) == module.DEFAULT_RENDER_SAMPLE_LIMIT


def test_write_render_samples_captures_token_debug_payload(tmp_path):
    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            return f"tok-{token_ids[0]}"

    module._worker_state.clear()
    module._worker_state.update(
        tokenizer=FakeTokenizer(),
        render_samples_local_dir=str(tmp_path),
        render_samples_limit=1,
        render_samples_written=0,
        worker_id=2,
        resolved_renderer_name="unit-renderer",
        train_on_what_str="all_assistant_messages",
    )
    datum = SimpleNamespace(
        loss_fn_inputs={
            "target_tokens": SimpleNamespace(data=[11, 12]),
            "weights": SimpleNamespace(data=[0.0, 1.0]),
        }
    )
    rendered = SimpleNamespace(
        token_ids=[10, 11, 12],
        token_weights=[0.0, 0.0, 1.0],
        datum=datum,
    )
    extra_rendered = SimpleNamespace(token_ids=[13], token_weights=[1.0], datum=datum)

    module._write_render_samples(
        {module.JSONL_ROW_INDEX_KEY: 4},
        [rendered, extra_rendered],
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "render_samples.worker-2.jsonl").read_text().splitlines()
    ]
    assert records == [
        {
            "source_jsonl_row_index": 4,
            "source_jsonl_line_number": 5,
            "split_index": 0,
            "worker_id": 2,
            "renderer": "unit-renderer",
            "train_on_what": "all_assistant_messages",
            "token_ids": [10, 11, 12],
            "decoded_tokens": ["tok-10", "tok-11", "tok-12"],
            "token_weights": [0.0, 0.0, 1.0],
            "training_target_token_ids": [11, 12],
            "training_loss_weights": [0.0, 1.0],
        }
    ]
    assert module._worker_state["render_samples_written"] == 1


def test_finalize_render_samples_uploads_worker_jsonl(tmp_path, monkeypatch):
    (tmp_path / "render_samples.worker-0.jsonl").write_text('{"worker":0}\n')
    (tmp_path / "render_samples.worker-1.jsonl").write_text('{"worker":1}\n')
    uploaded: dict[str, bytes] = {}
    monkeypatch.setattr(
        module.fileio,
        "write_bytes",
        lambda path, data: uploaded.setdefault(path, data),
    )

    module._finalize_render_samples(
        str(tmp_path),
        "gs://bucket/job/render_samples.jsonl",
        render_samples_limit=None,
    )

    assert uploaded == {
        "gs://bucket/job/render_samples.jsonl": b'{"worker":0}\n{"worker":1}\n',
    }


def test_finalize_render_samples_enforces_global_limit_round_robin(tmp_path, monkeypatch):
    (tmp_path / "render_samples.worker-0.jsonl").write_text(
        '{"worker":0,"seq":0}\n{"worker":0,"seq":1}\n'
    )
    (tmp_path / "render_samples.worker-1.jsonl").write_text(
        '{"worker":1,"seq":0}\n{"worker":1,"seq":1}\n'
    )
    uploaded: dict[str, bytes] = {}
    monkeypatch.setattr(
        module.fileio,
        "write_bytes",
        lambda path, data: uploaded.setdefault(path, data),
    )

    module._finalize_render_samples(
        str(tmp_path),
        "gs://bucket/job/render_samples.jsonl",
        render_samples_limit=3,
    )

    assert uploaded == {
        "gs://bucket/job/render_samples.jsonl": (
            b'{"worker":0,"seq":0}\n'
            b'{"worker":1,"seq":0}\n'
            b'{"worker":0,"seq":1}\n'
        ),
    }


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
        warm_start_from_adapter="accounts/test-account/models/promoted-lora",
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
        warm_start_from_adapter="accounts/test-account/models/promoted-lora",
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
