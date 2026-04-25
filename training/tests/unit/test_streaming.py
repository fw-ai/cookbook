"""Tests for ``training.utils.streaming``.

Covers the JSONL → render → DataLoader pipeline:

* ``_scan_jsonl_offsets`` -- byte offsets for non-blank lines.
* ``JsonlRenderDataset`` -- map-style indexing, ``max_examples`` cap,
  ``with_indices`` views, and ``None`` passthrough on filter.
* ``make_render_dataloader`` -- single-process collate + filter.

Multi-worker DataLoader behaviour (spawn workers, ``worker_init_fn``,
``persistent_workers``) is delegated to PyTorch's own test suite.
Anything more elaborate than what's exercised here belongs in an
integration test against ``sft_loop.main``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from training.utils.streaming import (
    AppendOnlyPickleLog,
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_RENDER_WORKERS,
    JsonlRenderDataset,
    _scan_jsonl_offsets,
    make_render_dataloader,
    setup_render_worker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# _scan_jsonl_offsets
# ---------------------------------------------------------------------------


def test_scan_offsets_skips_blank_lines(tmp_path):
    path = str(tmp_path / "data.jsonl")
    with open(path, "w") as f:
        f.write('{"a": 1}\n')
        f.write("\n")
        f.write('{"a": 2}\n')
        f.write("   \n")
        f.write('{"a": 3}\n')

    offsets = _scan_jsonl_offsets(path)

    assert len(offsets) == 3
    # Reading from each offset should yield the matching JSON object.
    with open(path) as f:
        seen = []
        for off in offsets:
            f.seek(off)
            seen.append(json.loads(f.readline()))
    assert seen == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_scan_offsets_max_examples(tmp_path):
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i} for i in range(10)])
    assert len(_scan_jsonl_offsets(path, max_examples=4)) == 4
    assert len(_scan_jsonl_offsets(path)) == 10


def test_scan_offsets_empty_file(tmp_path):
    path = str(tmp_path / "empty.jsonl")
    open(path, "w").close()
    assert _scan_jsonl_offsets(path) == []


# ---------------------------------------------------------------------------
# JsonlRenderDataset
# ---------------------------------------------------------------------------


def _identity_render(row: dict) -> dict:
    return {"id": row["i"], "doubled": row["i"] * 2}


def _filter_odd_render(row: dict) -> dict | None:
    if row["i"] % 2 == 1:
        return None
    return {"id": row["i"]}


def test_dataset_basic_indexing(tmp_path):
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i} for i in range(5)])

    ds = JsonlRenderDataset(path, _identity_render)

    assert len(ds) == 5
    assert ds.num_underlying_rows == 5
    assert ds[0] == {"id": 0, "doubled": 0}
    assert ds[3] == {"id": 3, "doubled": 6}
    # Out-of-order access works (verifies offset-based seek).
    assert ds[4] == {"id": 4, "doubled": 8}
    assert ds[1] == {"id": 1, "doubled": 2}


def test_dataset_max_examples_caps_underlying_rows(tmp_path):
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i} for i in range(10)])

    ds = JsonlRenderDataset(path, _identity_render, max_examples=3)

    assert len(ds) == 3
    assert ds.num_underlying_rows == 3
    assert [ds[i]["id"] for i in range(3)] == [0, 1, 2]


def test_dataset_render_fn_can_return_none(tmp_path):
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i} for i in range(4)])

    ds = JsonlRenderDataset(path, _filter_odd_render)

    # The dataset still has 4 entries; filtering is the loader's job.
    assert len(ds) == 4
    assert [ds[i] for i in range(4)] == [
        {"id": 0}, None, {"id": 2}, None,
    ]


def test_dataset_with_indices_carve_view(tmp_path):
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i} for i in range(10)])

    full = JsonlRenderDataset(path, _identity_render)
    tail = full.with_indices(list(range(3, 10)))

    assert len(full) == 10  # original is unchanged
    assert len(tail) == 7
    assert tail[0] == {"id": 3, "doubled": 6}
    assert tail[6] == {"id": 9, "doubled": 18}


def test_dataset_init_with_explicit_indices(tmp_path):
    """Constructor ``indices=...`` lets callers reorder / sub-select rows."""
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i} for i in range(10)])

    ds = JsonlRenderDataset(path, _identity_render, indices=[7, 2, 9])

    assert len(ds) == 3
    assert ds.num_underlying_rows == 10  # full file still scanned
    assert [ds[i]["id"] for i in range(3)] == [7, 2, 9]


# ---------------------------------------------------------------------------
# make_render_dataloader (single-process path)
# ---------------------------------------------------------------------------


def test_dataloader_drops_none_and_returns_lists(tmp_path):
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i} for i in range(6)])
    ds = JsonlRenderDataset(path, _filter_odd_render)

    loader = make_render_dataloader(
        ds,
        batch_size=3,
        num_workers=0,
        shuffle=False,
    )

    batches = list(loader)
    # Two batches of 3 -> after dropping None, sizes [2, 1] (rows 0,2 and 4).
    assert [len(b) for b in batches] == [2, 1]
    flat = [item["id"] for batch in batches for item in batch]
    assert flat == [0, 2, 4]
    # Each batch is a plain python list (not stacked).
    assert all(isinstance(b, list) for b in batches)


def test_dataloader_num_workers_one_uses_in_process_path(tmp_path):
    """``num_workers <= 1`` falls back to in-process rendering.

    Verified by passing a closure as render_fn -- closures aren't
    picklable, so this would raise ``PicklingError`` if a spawn worker
    were started. sft_loop's unit tests rely on this escape hatch to
    monkey-patch the renderer.
    """
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i} for i in range(4)])
    captured = {"calls": 0}

    def render_with_closure(row):  # closure -> not picklable
        captured["calls"] += 1
        return {"id": row["i"]}

    ds = JsonlRenderDataset(path, render_with_closure)
    loader = make_render_dataloader(ds, batch_size=4, num_workers=1, shuffle=False)

    batches = list(loader)
    assert batches == [[{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]]
    assert captured["calls"] == 4  # rendered in the test process


def test_dataloader_shuffle_is_deterministic_per_iteration(tmp_path):
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i} for i in range(8)])
    ds = JsonlRenderDataset(path, _identity_render)

    import torch  # local import: torch is heavy
    g1 = torch.Generator().manual_seed(0)
    loader = make_render_dataloader(
        ds, batch_size=8, num_workers=0, shuffle=True, generator=g1,
    )
    first = [item["id"] for item in next(iter(loader))]

    g2 = torch.Generator().manual_seed(0)
    loader = make_render_dataloader(
        ds, batch_size=8, num_workers=0, shuffle=True, generator=g2,
    )
    second = [item["id"] for item in next(iter(loader))]

    assert first == second
    assert sorted(first) == list(range(8))


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_default_constants_are_sane():
    assert DEFAULT_RENDER_WORKERS >= 1
    assert DEFAULT_PREFETCH_FACTOR >= 1


# ---------------------------------------------------------------------------
# AppendOnlyPickleLog
# ---------------------------------------------------------------------------


class TestAppendOnlyPickleLog:
    def test_round_trip_preserves_order(self, tmp_path):
        path = str(tmp_path / "log.pkl")
        log = AppendOnlyPickleLog(path)
        for i in range(5):
            log.append({"i": i, "data": list(range(i + 1))})
        assert len(log) == 5
        log.close_write()

        assert list(log) == [{"i": i, "data": list(range(i + 1))} for i in range(5)]

    def test_iter_before_close_write_raises(self, tmp_path):
        log = AppendOnlyPickleLog(str(tmp_path / "log.pkl"))
        log.append({"x": 1})
        with pytest.raises(RuntimeError, match="close_write"):
            list(log)
        log.close()

    def test_append_after_close_raises(self, tmp_path):
        log = AppendOnlyPickleLog(str(tmp_path / "log.pkl"))
        log.append({"x": 1})
        log.close_write()
        with pytest.raises(RuntimeError, match="closed"):
            log.append({"x": 2})

    def test_context_manager_auto_closes(self, tmp_path):
        path = str(tmp_path / "log.pkl")
        with AppendOnlyPickleLog(path) as log:
            log.append({"hello": "world"})
        # After __exit__ the log is closed and iterable.
        assert list(log) == [{"hello": "world"}]

    def test_disk_size_grows_with_appends(self, tmp_path):
        log = AppendOnlyPickleLog(str(tmp_path / "log.pkl"))
        log.append({"big": "x" * 4096})
        size_after_one = log.disk_size_bytes()
        log.append({"big": "x" * 4096})
        log.close_write()
        assert log.disk_size_bytes() > size_after_one
        assert size_after_one > 4096

    def test_empty_log_iterates_empty(self, tmp_path):
        log = AppendOnlyPickleLog(str(tmp_path / "log.pkl"))
        log.close_write()
        assert list(log) == []
        assert len(log) == 0

    def test_close_is_idempotent(self, tmp_path):
        log = AppendOnlyPickleLog(str(tmp_path / "log.pkl"))
        log.append({"x": 1})
        log.close()
        log.close()  # second call no-ops
        log.close_write()  # also safe


# ---------------------------------------------------------------------------
# setup_render_worker
# ---------------------------------------------------------------------------


class TestSetupRenderWorker:
    def test_seeds_parent_and_returns_partial(self):
        seen: list[tuple] = []

        def init_fn(a, b, _worker_id=None):
            seen.append((a, b, _worker_id))

        worker_init_fn = setup_render_worker(init_fn, "x", 7)

        # Parent invocation seeds in-process state immediately.
        assert seen == [("x", 7, None)]

        # Returned partial is a DataLoader-shaped worker_init_fn that takes
        # only the worker id.
        worker_init_fn(0)
        worker_init_fn(1)
        assert seen[1:] == [("x", 7, 0), ("x", 7, 1)]

    def test_returned_partial_is_picklable(self):
        """Picklability is required for spawn workers."""
        import pickle

        # Must use a top-level (importable) function for pickle, not a closure.
        worker_init_fn = setup_render_worker(_recording_init_fn, "tok", "ren", 8)

        roundtripped = pickle.loads(pickle.dumps(worker_init_fn))
        assert callable(roundtripped)


# Module-level so pickle can resolve it by qualified name.
def _recording_init_fn(*args, _worker_id=None):
    pass
