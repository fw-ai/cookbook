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
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_RENDER_WORKERS,
    JsonlRenderDataset,
    _scan_jsonl_offsets,
    make_render_dataloader,
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


def test_dataloader_shuffle_is_deterministic_per_iteration(tmp_path):
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i} for i in range(8)])
    ds = JsonlRenderDataset(path, _identity_render)

    import torch  # local import: torch is heavy
    torch.manual_seed(0)
    loader = make_render_dataloader(ds, batch_size=8, num_workers=0, shuffle=True)
    first = [item["id"] for item in next(iter(loader))]

    torch.manual_seed(0)
    loader = make_render_dataloader(ds, batch_size=8, num_workers=0, shuffle=True)
    second = [item["id"] for item in next(iter(loader))]

    assert first == second
    assert sorted(first) == list(range(8))


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_default_constants_are_sane():
    assert DEFAULT_RENDER_WORKERS >= 1
    assert DEFAULT_PREFETCH_FACTOR >= 1
