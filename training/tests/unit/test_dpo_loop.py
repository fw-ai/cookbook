"""Unit tests for the streaming DPO loop.

Covers the public-ish surface of ``training.recipes.dpo_loop``:

  * ``_render_pair_worker`` -- per-row render fn used by the JsonlRenderDataset
    (schema normalisation + render + over-length filtering, with all drop
    reasons collapsed into a ``None`` return).
  * ``_ref_forward_batch`` -- enriches pairs with ``array.array('f')``
    reference logprobs and preserves input order.
  * ``_forward_backward_pairs`` -- arranges datums and ref logprobs for
    the policy update.
  * ``_train_loop`` -- end-to-end pipeline for the single-epoch path
    (no ref_cache_log) and the multi-epoch path (``AppendOnlyPickleLog``
    captures pairs in producer order so epochs 1+ stream from disk).
  * ``iter_preference_examples`` -- streaming JSONL reader that
    normalises the three on-disk preference schemas.

The streaming refactor mirrors the SFT v2 fix (fw-ai/cookbook#371).
"""

from __future__ import annotations

import array
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

import training.recipes.dpo_loop as module
from training.utils import AppendOnlyPickleLog, JsonlRenderDataset
from training.utils.data import iter_preference_examples, load_preference_dataset


# ---------------------------------------------------------------------------
# _render_pair_worker
# ---------------------------------------------------------------------------


def _setup_pair_worker(monkeypatch, max_seq_len: int = 8) -> None:
    monkeypatch.setitem(module._pair_worker_state, "renderer", "R")
    monkeypatch.setitem(module._pair_worker_state, "tokenizer", "T")
    monkeypatch.setitem(module._pair_worker_state, "max_seq_len", max_seq_len)


class TestRenderPairWorker:
    def test_returns_none_when_schema_unrecognized(self, monkeypatch):
        _setup_pair_worker(monkeypatch)
        assert module._render_pair_worker({"unknown": "schema"}) is None

    def test_returns_none_when_render_fails(self, monkeypatch):
        _setup_pair_worker(monkeypatch)
        monkeypatch.setattr(module, "render_preference_pair", lambda *a, **k: None)
        assert module._render_pair_worker(
            {"chosen": {"messages": []}, "rejected": {"messages": []}}
        ) is None

    def test_returns_none_when_too_long(self, monkeypatch):
        _setup_pair_worker(monkeypatch)
        long_pair = SimpleNamespace(
            chosen_tokens=[1] * 9, rejected_tokens=[2, 3], response_start=3,
            chosen_datum={"kind": "chosen"}, rejected_datum={"kind": "rejected"},
        )
        monkeypatch.setattr(module, "render_preference_pair", lambda *a, **k: long_pair)
        assert module._render_pair_worker(
            {"chosen": {}, "rejected": {}}
        ) is None

    def test_returns_dict_with_lengths_only(self, monkeypatch):
        """Stored dict carries lengths, not token lists -- saves O(N*L) RAM."""
        _setup_pair_worker(monkeypatch)
        valid_pair = SimpleNamespace(
            chosen_tokens=[1, 2, 3, 4], rejected_tokens=[1, 2, 9],
            response_start=2,
            chosen_datum={"kind": "chosen"}, rejected_datum={"kind": "rejected"},
        )
        monkeypatch.setattr(module, "render_preference_pair", lambda *a, **k: valid_pair)
        result = module._render_pair_worker(
            {"chosen": {}, "rejected": {}}
        )
        assert result == {
            "chosen_tokens_len": 4,
            "rejected_tokens_len": 3,
            "response_start": 2,
            "chosen_datum": {"kind": "chosen"},
            "rejected_datum": {"kind": "rejected"},
        }
        assert "chosen_tokens" not in result
        assert "rejected_tokens" not in result

    def test_normalises_samples_schema(self, monkeypatch):
        """Render fn does the schema normalisation step itself (no helper needed)."""
        _setup_pair_worker(monkeypatch)
        captured: dict = {}

        def fake_render(chosen, rejected, **kwargs):
            captured["chosen"] = chosen
            captured["rejected"] = rejected
            return SimpleNamespace(
                chosen_tokens=[1], rejected_tokens=[1], response_start=0,
                chosen_datum={}, rejected_datum={},
            )

        monkeypatch.setattr(module, "render_preference_pair", fake_render)
        row = {"samples": [
            {"text": "good", "evals": {"score": 1.0}},
            {"text": "bad", "evals": {"score": 0.0}},
        ]}
        assert module._render_pair_worker(row) is not None
        assert captured["chosen"]["text"] == "good"
        assert captured["rejected"]["text"] == "bad"


# ---------------------------------------------------------------------------
# _ref_forward_batch
# ---------------------------------------------------------------------------


class TestRefForwardBatch:
    def test_enriches_pairs_in_input_order(self):
        """Returned dicts carry ``array.array('f')`` logprobs and preserve order."""

        class FakeReference:
            def __init__(self):
                self.calls = []

            def forward(self, datums, loss_fn):
                self.calls.append((datums, loss_fn))
                return SimpleNamespace(loss_fn_outputs=[
                    {"logprobs": SimpleNamespace(data=[-0.1, -0.2])},
                    {"logprobs": SimpleNamespace(data=[-0.3])},
                    {"logprobs": SimpleNamespace(data=[-0.4])},
                    {"logprobs": SimpleNamespace(data=[-0.5, -0.6])},
                ])

        reference = FakeReference()
        pairs = [
            {"chosen_tokens_len": 3, "rejected_tokens_len": 3, "response_start": 2,
             "chosen_datum": {"id": "c0"}, "rejected_datum": {"id": "r0"}},
            {"chosen_tokens_len": 3, "rejected_tokens_len": 3, "response_start": 2,
             "chosen_datum": {"id": "c1"}, "rejected_datum": {"id": "r1"}},
        ]

        sem = asyncio.Semaphore(2)
        enriched = asyncio.run(
            module._ref_forward_batch(pairs, reference, sem, ref_batch_size=2)
        )

        assert reference.calls == [(
            [{"id": "c0"}, {"id": "r0"}, {"id": "c1"}, {"id": "r1"}],
            "cross_entropy",
        )]
        assert len(enriched) == 2
        assert isinstance(enriched[0]["ref_chosen"], array.array)
        assert enriched[0]["ref_chosen"].typecode == "f"
        assert list(enriched[0]["ref_chosen"]) == pytest.approx([-0.1, -0.2])
        assert list(enriched[0]["ref_rejected"]) == pytest.approx([-0.3])
        assert list(enriched[1]["ref_chosen"]) == pytest.approx([-0.4])
        assert list(enriched[1]["ref_rejected"]) == pytest.approx([-0.5, -0.6])
        assert enriched[0]["chosen_datum"] == {"id": "c0"}
        assert enriched[0]["rejected_datum"] == {"id": "r0"}

    def test_sub_batches_split_correctly(self):
        """``ref_batch_size`` controls how many pairs go in each forward call."""

        class FakeReference:
            def __init__(self):
                self.calls = []

            def forward(self, datums, loss_fn):
                self.calls.append(len(datums))
                return SimpleNamespace(loss_fn_outputs=[
                    {"logprobs": SimpleNamespace(data=[-0.1])} for _ in datums
                ])

        reference = FakeReference()
        pairs = [
            {"chosen_tokens_len": 1, "rejected_tokens_len": 1, "response_start": 0,
             "chosen_datum": {"i": i}, "rejected_datum": {"i": -i}}
            for i in range(5)
        ]
        asyncio.run(
            module._ref_forward_batch(
                pairs, reference, asyncio.Semaphore(4), ref_batch_size=2,
            )
        )
        # 5 pairs / 2 per batch -> [2, 2, 1] pairs -> [4, 4, 2] datums
        assert sorted(reference.calls) == [2, 4, 4]


# ---------------------------------------------------------------------------
# _forward_backward_pairs
# ---------------------------------------------------------------------------


def test_forward_backward_pairs_interleaves_and_builds_loss_fn(monkeypatch):
    captured: dict = {}

    def fake_make_loss(ref_chosen, ref_rejected, response_starts, beta):
        captured.update(
            ref_chosen=ref_chosen, ref_rejected=ref_rejected,
            response_starts=response_starts, beta=beta,
        )
        return "loss-fn"

    class FakePolicy:
        def forward_backward_custom(self, datums, loss_fn):
            captured["datums"] = datums
            captured["loss_fn"] = loss_fn
            return "result"

    monkeypatch.setattr(module, "make_batch_dpo_loss_fn", fake_make_loss)

    batch_pairs = [
        {"chosen_datum": {"id": "c0"}, "rejected_datum": {"id": "r0"},
         "ref_chosen": array.array("f", [-0.1]), "ref_rejected": array.array("f", [-0.2]),
         "response_start": 3, "chosen_tokens_len": 1, "rejected_tokens_len": 1},
        {"chosen_datum": {"id": "c1"}, "rejected_datum": {"id": "r1"},
         "ref_chosen": array.array("f", [-0.3]), "ref_rejected": array.array("f", [-0.4]),
         "response_start": 5, "chosen_tokens_len": 1, "rejected_tokens_len": 1},
    ]
    assert module._forward_backward_pairs(batch_pairs, FakePolicy(), beta=0.25) == "result"

    assert captured["datums"] == [
        {"id": "c0"}, {"id": "r0"}, {"id": "c1"}, {"id": "r1"},
    ]
    assert all(isinstance(r, array.array) for r in captured["ref_chosen"])
    assert captured["response_starts"] == [3, 5]
    assert captured["beta"] == 0.25


# ---------------------------------------------------------------------------
# _train_loop  (DataLoader → ref forward → train, ref-cache log on multi-epoch)
# ---------------------------------------------------------------------------


def _make_pair(idx: int) -> dict:
    """Tokenized pair dict matching the new (lengths-only) shape."""
    return {
        "chosen_datum": {"id": f"c{idx}"},
        "rejected_datum": {"id": f"r{idx}"},
        "chosen_tokens_len": 3,
        "rejected_tokens_len": 3,
        "response_start": 2,
    }


def _test_render_pair(row: dict) -> dict:
    """Module-level render_fn for JsonlRenderDataset in tests."""
    return _make_pair(row["i"])


def _make_pair_dataset(tmp_path, n: int) -> JsonlRenderDataset:
    path = tmp_path / "pairs.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"i": i}) + "\n")
    return JsonlRenderDataset(str(path), _test_render_pair)


class _FakeReference:
    def forward(self, datums, loss_fn):
        return SimpleNamespace(loss_fn_outputs=[
            {"logprobs": SimpleNamespace(data=[-0.1 * (j + 1)])}
            for j in range(len(datums))
        ])


class _FakePolicy:
    job_id = "fake-policy-job"

    def __init__(self):
        self.optim_step_count = 0

    def optim_step(self, _params, **kwargs):
        self.optim_step_count += 1
        return SimpleNamespace(metrics={"optimizer/lr": 1e-4})

    def save_state(self, name, timeout=None):
        pass

    def resolve_checkpoint_path(self, name, source_job_id=None):
        return f"tinker://unit/state/{name}"


def _stub_train_step_deps(monkeypatch, events: dict):
    """Common stubs so _train_loop runs without metrics/wandb/checkpoint side effects."""
    monkeypatch.setattr(
        module, "_forward_backward_pairs",
        lambda batch, policy, beta: events.setdefault("flush_batches", []).append(
            (list(batch), beta)
        ) or SimpleNamespace(metrics={"dpo_loss": 1.5, "margin": 0.25, "accuracy": 0.75}),
    )
    monkeypatch.setattr(module, "flush_timing", lambda: {})
    monkeypatch.setattr(module, "log_metrics_json", lambda *a, **kw: None)
    monkeypatch.setattr(module, "wandb_log", lambda *a, **kw: None)


class TestTrainLoop:
    def test_single_epoch_no_ref_cache_log(self, tmp_path, monkeypatch):
        events: dict = {}
        _stub_train_step_deps(monkeypatch, events)

        ds = _make_pair_dataset(tmp_path, n=4)
        cfg = module.Config(
            log_path=str(tmp_path), beta=0.2, epochs=1, batch_size=2,
            render_workers=0,
        )
        ref_done = []
        step = asyncio.run(
            module._train_loop(
                ds, None,
                _FakeReference(), _FakePolicy(),
                adam_params={"lr": 1e-4},
                cfg=cfg,
                step_offset=0,
                on_ref_done=lambda: ref_done.append(True),
            )
        )
        # 4 pairs / batch_size=2 -> 2 train steps
        assert step == 2
        assert ref_done == [True]
        assert len(events["flush_batches"]) == 2
        for batch, beta in events["flush_batches"]:
            assert beta == 0.2
            for pair in batch:
                assert "ref_chosen" in pair and "ref_rejected" in pair

    def test_multi_epoch_uses_ref_cache_log(self, tmp_path, monkeypatch):
        events: dict = {}
        _stub_train_step_deps(monkeypatch, events)

        ds = _make_pair_dataset(tmp_path, n=4)
        ref_cache_path = str(tmp_path / "ref_cache.pkl")
        ref_cache = AppendOnlyPickleLog(ref_cache_path)
        try:
            cfg = module.Config(
                log_path=str(tmp_path), beta=0.1, epochs=3, batch_size=2,
                render_workers=0,
            )
            step = asyncio.run(
                module._train_loop(
                    ds, ref_cache,
                    _FakeReference(), _FakePolicy(),
                    adam_params={"lr": 1e-4},
                    cfg=cfg, step_offset=0,
                )
            )
            # 4 pairs * 3 epochs / batch_size=2 = 6 steps
            assert step == 6
            assert len(events["flush_batches"]) == 6

            # Ref cache must contain all 4 pairs in producer order; epochs 1+
            # stream through it sequentially (validated by replaying it now).
            assert len(ref_cache) == 4
            cached = list(ref_cache)
            assert [p["chosen_datum"]["id"] for p in cached] == ["c0", "c1", "c2", "c3"]
            for c in cached:
                assert isinstance(c["ref_chosen"], array.array)
                assert isinstance(c["ref_rejected"], array.array)
        finally:
            ref_cache.close()

    def test_multi_epoch_requires_ref_cache_log(self, tmp_path):
        ds = _make_pair_dataset(tmp_path, n=2)
        cfg = module.Config(
            log_path=str(tmp_path), epochs=2, batch_size=1, render_workers=0,
        )
        with pytest.raises(ValueError, match="ref_cache_log"):
            asyncio.run(
                module._train_loop(
                    ds, None,
                    _FakeReference(), _FakePolicy(),
                    adam_params={"lr": 1e-4},
                    cfg=cfg, step_offset=0,
                )
            )

    def test_pipeline_overlap_ref_freed_before_training_done(self, tmp_path, monkeypatch):
        """Producer must finish (and on_ref_done fire) while training is still
        in progress -- the core benefit of the pipeline."""
        timeline: list = []

        monkeypatch.setattr(module, "flush_timing", lambda: {})
        monkeypatch.setattr(module, "log_metrics_json", lambda *a, **kw: None)
        monkeypatch.setattr(module, "wandb_log", lambda *a, **kw: None)

        def slow_fwd_bwd(batch, policy, beta):
            time.sleep(0.15)
            return SimpleNamespace(metrics={"dpo_loss": 0.5, "margin": 0.1, "accuracy": 0.9})

        monkeypatch.setattr(module, "_forward_backward_pairs", slow_fwd_bwd)

        ds = _make_pair_dataset(tmp_path, n=4)
        cfg = module.Config(
            log_path=str(tmp_path), beta=0.1, epochs=1, batch_size=1,
            ref_cache_concurrency=4, render_workers=0,
        )
        t0 = time.monotonic()
        step = asyncio.run(
            module._train_loop(
                ds, None,
                _FakeReference(), _FakePolicy(),
                adam_params={"lr": 1e-4},
                cfg=cfg, step_offset=0,
                on_ref_done=lambda: timeline.append(time.monotonic()),
            )
        )
        t_end = time.monotonic()

        assert step == 4
        assert len(timeline) == 1
        ref_done_t = timeline[0] - t0
        total_t = t_end - t0
        assert ref_done_t < total_t * 0.8, (
            f"ref_done should fire well before training finishes "
            f"(ref_done={ref_done_t:.2f}s, total={total_t:.2f}s)"
        )

    def test_raises_when_all_rows_render_to_none(self, tmp_path, monkeypatch):
        """All-filtered datasets must fail loudly instead of finishing at step 0."""
        events: dict = {}
        _stub_train_step_deps(monkeypatch, events)

        # Render every row to None: the loader yields empty batches that the
        # producer must skip without calling reference.forward.
        path = tmp_path / "pairs.jsonl"
        with open(path, "w") as f:
            for i in range(4):
                f.write(json.dumps({"i": i}) + "\n")
        ds = JsonlRenderDataset(str(path), lambda row: None)

        ref_calls = []

        class CountingReference:
            def forward(self, datums, loss_fn):
                ref_calls.append(len(datums))
                return SimpleNamespace(loss_fn_outputs=[])

        cfg = module.Config(
            log_path=str(tmp_path), epochs=1, batch_size=2, render_workers=0,
        )
        with pytest.raises(RuntimeError, match="No valid pairs after tokenization"):
            asyncio.run(
                module._train_loop(
                    ds, None, CountingReference(), _FakePolicy(),
                    adam_params={"lr": 1e-4}, cfg=cfg, step_offset=0,
                )
            )
        assert ref_calls == []
        assert events.get("flush_batches", []) == []

    def test_max_pairs_caps_valid_pairs_after_filtering(self, tmp_path, monkeypatch):
        """``max_pairs`` counts valid rendered pairs, not raw JSONL rows."""
        events: dict = {}
        _stub_train_step_deps(monkeypatch, events)

        path = tmp_path / "pairs.jsonl"
        with open(path, "w") as f:
            for i in range(5):
                f.write(json.dumps({"i": i}) + "\n")

        def render_some_none(row: dict) -> dict | None:
            if row["i"] in {0, 3}:
                return None
            return _make_pair(row["i"])

        ds = JsonlRenderDataset(str(path), render_some_none)
        cfg = module.Config(
            log_path=str(tmp_path),
            epochs=1,
            batch_size=2,
            render_workers=0,
            max_pairs=2,
        )

        step = asyncio.run(
            module._train_loop(
                ds, None, _FakeReference(), _FakePolicy(),
                adam_params={"lr": 1e-4}, cfg=cfg, step_offset=0,
            )
        )
        assert step == 1
        assert len(events["flush_batches"]) == 1
        trained_ids = [pair["chosen_datum"]["id"] for pair in events["flush_batches"][0][0]]
        assert trained_ids == ["c1", "c2"]

    def test_total_steps_matches_actual_step_count(self, tmp_path, monkeypatch) -> None:
        """The runner-facing total_steps should use ceil(valid_pairs / batch_size)."""
        executed = {"count": 0}
        reported_total_steps = {"value": None}

        def fake_fwd_bwd(batch, policy, beta):
            executed["count"] += 1
            return SimpleNamespace(metrics={"dpo_loss": 0.0, "margin": 0.0, "accuracy": 0.5})

        class RecordingRunner:
            def start_training(self):
                pass

            def write_status(self, *args, **kwargs):
                total = kwargs.get("total_steps")
                if total is not None:
                    reported_total_steps["value"] = total

            def write_metadata(self, *args, **kwargs):
                pass

            def append_metrics(self, *args, **kwargs):
                pass

            def report_rendering_progress(self, *args, **kwargs):
                pass

            def set_accelerator_info(self, *args, **kwargs):
                pass

        monkeypatch.setattr(module, "_forward_backward_pairs", fake_fwd_bwd)
        monkeypatch.setattr(module, "flush_timing", lambda: {})
        monkeypatch.setattr(module, "log_metrics_json", lambda *a, **kw: None)
        monkeypatch.setattr(module, "wandb_log", lambda *a, **kw: None)

        ds = _make_pair_dataset(tmp_path, n=5)
        cfg = module.Config(
            log_path=str(tmp_path),
            beta=0.1,
            epochs=1,
            batch_size=2,
            ref_cache_concurrency=4,
            render_workers=0,
        )

        asyncio.run(
            module._train_loop(
                ds,
                None,
                _FakeReference(),
                _FakePolicy(),
                adam_params={"lr": 1e-4},
                cfg=cfg,
                step_offset=0,
                runner=RecordingRunner(),
            )
        )

        assert executed["count"] == 3
        assert reported_total_steps["value"] == 3


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_requires_tokenizer_model(monkeypatch):
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    cfg = module.Config(
        log_path="/tmp/dpo_test_logs", dataset="/tmp/pairs.jsonl", tokenizer_model="",
    )
    with pytest.raises(ValueError, match="tokenizer_model"):
        module.main(cfg)


# ---------------------------------------------------------------------------
# iter_preference_examples (training/utils/data.py) -- still used by callers
# outside the streaming path; kept for back-compat.
# ---------------------------------------------------------------------------


def _write_jsonl(path, rows) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestIterPreferenceExamples:
    def test_chosen_rejected_format_passthrough(self, tmp_path):
        path = tmp_path / "pairs.jsonl"
        _write_jsonl(path, [
            {"chosen": {"messages": [{"role": "user", "content": "x"}]},
             "rejected": {"messages": [{"role": "user", "content": "y"}]}},
        ])
        out = list(iter_preference_examples(str(path)))
        assert len(out) == 1
        assert out[0]["chosen"]["messages"][0]["content"] == "x"

    def test_samples_format_with_score_evals(self, tmp_path):
        path = tmp_path / "samples.jsonl"
        _write_jsonl(path, [
            {"samples": [
                {"text": "good", "evals": {"score": 1.0}},
                {"text": "bad", "evals": {"score": 0.0}},
            ]},
            {"samples": [{"text": "lonely", "evals": {"score": 1.0}}]},
        ])
        out = list(iter_preference_examples(str(path)))
        assert len(out) == 1
        assert out[0]["chosen"]["text"] == "good"
        assert out[0]["rejected"]["text"] == "bad"

    def test_preferred_output_format(self, tmp_path):
        path = tmp_path / "pref.jsonl"
        _write_jsonl(path, [{
            "input": {"messages": [{"role": "user", "content": "q"}]},
            "preferred_output": [{"role": "assistant", "content": "yes"}],
            "non_preferred_output": "no",
        }])
        out = list(iter_preference_examples(str(path)))
        assert len(out) == 1
        assert out[0]["chosen"]["messages"] == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "yes"},
        ]
        assert out[0]["rejected"]["messages"] == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "no"},
        ]

    def test_max_pairs_caps_valid_pairs(self, tmp_path):
        path = tmp_path / "pairs.jsonl"
        _write_jsonl(path, [
            {"chosen": {"text": str(i)}, "rejected": {"text": "_"}}
            for i in range(5)
        ])
        assert len(list(iter_preference_examples(str(path), max_pairs=3))) == 3

    def test_skips_blank_and_unrecognized_rows(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"chosen": {"t": 1}, "rejected": {"t": 2}}) + "\n")
            f.write("\n")
            f.write(json.dumps({"unknown": "schema"}) + "\n")
            f.write(json.dumps({"chosen": {"t": 3}, "rejected": {"t": 4}}) + "\n")
        out = list(iter_preference_examples(str(path)))
        assert len(out) == 2

    def test_is_lazy(self, tmp_path):
        path = tmp_path / "pairs.jsonl"
        _write_jsonl(path, [
            {"chosen": {"i": i}, "rejected": {"i": -i}} for i in range(100)
        ])
        it = iter_preference_examples(str(path), max_pairs=2)
        first = next(it)
        second = next(it)
        assert first["chosen"]["i"] == 0
        assert second["chosen"]["i"] == 1
        with pytest.raises(StopIteration):
            next(it)


def _write_preference_jsonl(tmp_path, rows):
    path = tmp_path / "pref.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


class TestLoadPreferenceDatasetValidation:
    def test_rejects_non_binary_scores(self, tmp_path) -> None:
        rows = [
            {
                "samples": [
                    {"messages": [{"role": "assistant", "content": "a"}], "score": 0.5},
                    {"messages": [{"role": "assistant", "content": "b"}], "score": 0.8},
                ],
            },
        ]
        path = _write_preference_jsonl(tmp_path, rows)

        with pytest.raises(ValueError, match="score"):
            load_preference_dataset(path)

    def test_rejects_rows_without_both_chosen_and_rejected(self, tmp_path) -> None:
        rows = [
            {
                "samples": [
                    {"messages": [{"role": "assistant", "content": "a"}], "score": 1.0},
                ],
            },
        ]
        path = _write_preference_jsonl(tmp_path, rows)

        with pytest.raises(ValueError, match="rejected"):
            load_preference_dataset(path)

    def test_rejects_duplicate_chosen_samples(self, tmp_path) -> None:
        rows = [
            {
                "samples": [
                    {"messages": [{"role": "assistant", "content": "a"}], "score": 1.0},
                    {"messages": [{"role": "assistant", "content": "b"}], "score": 1.0},
                    {"messages": [{"role": "assistant", "content": "c"}], "score": 0.0},
                ],
            },
        ]
        path = _write_preference_jsonl(tmp_path, rows)

        with pytest.raises(ValueError, match="ambiguous|multiple"):
            load_preference_dataset(path)

    def test_rejects_unknown_row_format(self, tmp_path) -> None:
        rows = [
            {"foo": "bar"},
            {"chosen": {"messages": [{"role": "assistant", "content": "a"}]}},
        ]
        path = _write_preference_jsonl(tmp_path, rows)

        with pytest.raises(ValueError, match="supported preference format"):
            load_preference_dataset(path)

    def test_rejects_non_list_samples(self, tmp_path) -> None:
        path = _write_preference_jsonl(tmp_path, [{"samples": {"not": "a list"}}])

        with pytest.raises(ValueError, match="list"):
            load_preference_dataset(path)

    def test_rejects_non_dict_sample_entry(self, tmp_path) -> None:
        rows = [
            {
                "samples": [
                    "not a dict",
                    {"messages": [{"role": "assistant", "content": "b"}], "score": 0.0},
                ],
            },
        ]
        path = _write_preference_jsonl(tmp_path, rows)

        with pytest.raises(ValueError, match="dict"):
            load_preference_dataset(path)
