"""Tests for the streaming dataset rendering utilities.

Covers ``training.utils.streaming``:

  * ``iter_jsonl_rows`` / ``count_jsonl_rows`` -- lazy JSONL helpers.
  * ``DiskBackedDatumStore`` -- append-only pickle store with random reads.
  * ``stream_render_to_store`` -- single-process and spawn-pool paths.

These are the building blocks behind the SFT v2 streaming render fix
(see fw-ai/cookbook#371). Anything more elaborate than what's exercised
here would belong in an integration test against ``sft_loop.main``.
"""

from __future__ import annotations

import json
import os
import pickle

import pytest

from training.utils.streaming import (
    DEFAULT_RENDER_CHUNKSIZE,
    DEFAULT_RENDER_WORKERS,
    DiskBackedDatumStore,
    count_jsonl_rows,
    iter_jsonl_rows,
    stream_render_to_store,
)


# ---------------------------------------------------------------------------
# Module-level helpers (must be picklable for the spawn-pool tests)
# ---------------------------------------------------------------------------


def _double(row: dict) -> dict | None:
    """Render fn that doubles ``row['n']`` and filters out negatives."""
    n = row["n"]
    if n < 0:
        return None
    return {"n": n * 2}


def _identity(row: dict) -> dict | None:
    return row


_OFFSET_STATE: dict = {}


def _init_offset(offset: int) -> None:
    _OFFSET_STATE["offset"] = offset


def _render_with_offset(row: dict) -> dict:
    return {"n": row["n"] + _OFFSET_STATE["offset"]}


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path, rows, *, with_blank_lines: bool = False) -> None:
    parts = []
    for row in rows:
        parts.append(json.dumps(row))
        if with_blank_lines:
            parts.append("")
    path.write_text("\n".join(parts) + "\n")


class TestIterJsonlRows:
    def test_yields_each_row_in_order(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [{"n": i} for i in range(5)])
        assert list(iter_jsonl_rows(str(path))) == [{"n": i} for i in range(5)]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [{"n": i} for i in range(3)], with_blank_lines=True)
        assert list(iter_jsonl_rows(str(path))) == [{"n": i} for i in range(3)]

    def test_max_examples_caps_output(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [{"n": i} for i in range(10)])
        assert list(iter_jsonl_rows(str(path), max_examples=3)) == [
            {"n": 0}, {"n": 1}, {"n": 2},
        ]

    def test_is_lazy(self, tmp_path):
        """Iterator must not slurp the whole file into memory at construction."""
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [{"n": i} for i in range(100)])
        it = iter_jsonl_rows(str(path), max_examples=2)
        assert next(it) == {"n": 0}
        assert next(it) == {"n": 1}
        with pytest.raises(StopIteration):
            next(it)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text("")
        assert list(iter_jsonl_rows(str(path))) == []


class TestCountJsonlRows:
    def test_matches_iter(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [{"n": i} for i in range(7)])
        assert count_jsonl_rows(str(path)) == 7

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [{"n": i} for i in range(4)], with_blank_lines=True)
        assert count_jsonl_rows(str(path)) == 4

    def test_max_examples_short_circuits(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [{"n": i} for i in range(100)])
        assert count_jsonl_rows(str(path), max_examples=5) == 5

    def test_empty_file(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text("")
        assert count_jsonl_rows(str(path)) == 0


# ---------------------------------------------------------------------------
# DiskBackedDatumStore
# ---------------------------------------------------------------------------


class TestDiskBackedDatumStore:
    def test_round_trip_random_access(self, tmp_path):
        path = str(tmp_path / "store.bin")
        items = [{"n": i, "payload": "x" * i} for i in range(5)]
        with DiskBackedDatumStore(path) as store:
            for item in items:
                store.append(item)
            store.close_write()
            assert len(store) == 5
            assert store[0] == items[0]
            assert store[4] == items[4]
            assert store[2] == items[2]  # out-of-order read

    def test_sequential_iter(self, tmp_path):
        path = str(tmp_path / "store.bin")
        items = [{"n": i} for i in range(3)]
        with DiskBackedDatumStore(path) as store:
            for item in items:
                store.append(item)
            store.close_write()
            assert list(store) == items

    def test_disk_size_bytes_grows_with_appends(self, tmp_path):
        path = str(tmp_path / "store.bin")
        with DiskBackedDatumStore(path) as store:
            store.append({"n": 0})
            store.close_write()  # flush so on-disk size is accurate
            size_after_one = store.disk_size_bytes()

        with DiskBackedDatumStore(path) as store2:
            for _ in range(10):
                store2.append({"n": 0})
            store2.close_write()
            assert store2.disk_size_bytes() > size_after_one

    def test_append_after_close_raises(self, tmp_path):
        path = str(tmp_path / "store.bin")
        store = DiskBackedDatumStore(path)
        with store:
            store.append({"n": 1})
            store.close_write()
            with pytest.raises(RuntimeError, match="not open for writing"):
                store.append({"n": 2})

    def test_close_write_is_idempotent(self, tmp_path):
        path = str(tmp_path / "store.bin")
        with DiskBackedDatumStore(path) as store:
            store.append({"n": 1})
            store.close_write()
            store.close_write()  # must not raise
            assert store[0] == {"n": 1}

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "store.bin"
        with DiskBackedDatumStore(str(nested)) as store:
            store.append({"n": 1})
            store.close_write()
            assert os.path.exists(nested)

    def test_path_property_exposes_backing_file(self, tmp_path):
        path = str(tmp_path / "store.bin")
        with DiskBackedDatumStore(path) as store:
            assert store.path == path

    def test_payload_can_be_arbitrary_picklable(self, tmp_path):
        path = str(tmp_path / "store.bin")
        payloads = [
            {"a": 1},
            ("tuple", 2),
            [1, 2, 3],
            "string",
            42,
        ]
        with DiskBackedDatumStore(path) as store:
            for p in payloads:
                store.append(p)
            store.close_write()
            assert list(store) == payloads

    def test_on_disk_format_is_concatenated_pickles(self, tmp_path):
        """Sanity-check the wire format: writes must equal raw pickle bytes."""
        path = str(tmp_path / "store.bin")
        items = [{"n": i} for i in range(3)]
        with DiskBackedDatumStore(path) as store:
            for item in items:
                store.append(item)
            store.close_write()

        with open(path, "rb") as f:
            raw = f.read()
        expected = b"".join(pickle.dumps(i, protocol=pickle.HIGHEST_PROTOCOL) for i in items)
        assert raw == expected


# ---------------------------------------------------------------------------
# stream_render_to_store
# ---------------------------------------------------------------------------


class TestStreamRenderToStore:
    def test_single_process_basic(self, tmp_path):
        path = str(tmp_path / "store.bin")
        rows = [{"n": i} for i in range(5)]
        with DiskBackedDatumStore(path) as store:
            filtered = stream_render_to_store(
                rows, render_fn=_identity, store=store, num_workers=1,
            )
            store.close_write()
            assert filtered == 0
            assert len(store) == 5
            assert list(store) == rows

    def test_single_process_filtering(self, tmp_path):
        path = str(tmp_path / "store.bin")
        rows = [{"n": i} for i in range(-2, 5)]  # 7 rows, 2 negative
        with DiskBackedDatumStore(path) as store:
            filtered = stream_render_to_store(
                rows, render_fn=_double, store=store, num_workers=1,
            )
            store.close_write()
            assert filtered == 2
            assert len(store) == 5
            assert list(store) == [{"n": i * 2} for i in range(0, 5)]

    def test_on_progress_called_for_every_item(self, tmp_path):
        path = str(tmp_path / "store.bin")
        rows = [{"n": i} for i in range(4)]
        seen = []

        def _on_progress(i: int, item) -> None:
            seen.append((i, item))

        with DiskBackedDatumStore(path) as store:
            stream_render_to_store(
                rows, render_fn=_identity, store=store,
                num_workers=1, on_progress=_on_progress,
            )
            store.close_write()

        # progress index is 1-based and called for every input row
        assert [i for i, _ in seen] == [1, 2, 3, 4]
        assert [item for _, item in seen] == rows

    def test_on_progress_receives_none_for_filtered_items(self, tmp_path):
        path = str(tmp_path / "store.bin")
        rows = [{"n": -1}, {"n": 1}]
        seen = []
        with DiskBackedDatumStore(path) as store:
            stream_render_to_store(
                rows, render_fn=_double, store=store,
                num_workers=1, on_progress=lambda i, x: seen.append((i, x)),
            )
            store.close_write()
        assert seen == [(1, None), (2, {"n": 2})]

    def test_empty_input(self, tmp_path):
        path = str(tmp_path / "store.bin")
        with DiskBackedDatumStore(path) as store:
            filtered = stream_render_to_store(
                iter([]), render_fn=_identity, store=store, num_workers=1,
            )
            store.close_write()
            assert filtered == 0
            assert len(store) == 0

    def test_parallel_spawn_basic(self, tmp_path):
        """Spawn pool path: render_fn must be picklable; results stream out
        in input order via ``imap``."""
        path = str(tmp_path / "store.bin")
        rows = [{"n": i} for i in range(20)]
        with DiskBackedDatumStore(path) as store:
            filtered = stream_render_to_store(
                rows, render_fn=_double, store=store,
                num_workers=2, chunksize=4,
            )
            store.close_write()
            assert filtered == 0
            assert len(store) == 20
            # imap preserves input order
            assert list(store) == [{"n": i * 2} for i in range(20)]

    def test_parallel_spawn_with_initializer(self, tmp_path):
        """Initializer + initargs must be applied before the first task runs."""
        path = str(tmp_path / "store.bin")
        rows = [{"n": i} for i in range(8)]
        with DiskBackedDatumStore(path) as store:
            stream_render_to_store(
                rows, render_fn=_render_with_offset, store=store,
                num_workers=2, chunksize=2,
                initializer=_init_offset, initargs=(100,),
            )
            store.close_write()
            assert list(store) == [{"n": i + 100} for i in range(8)]

    def test_parallel_spawn_filtering(self, tmp_path):
        path = str(tmp_path / "store.bin")
        rows = [{"n": i} for i in range(-3, 4)]
        with DiskBackedDatumStore(path) as store:
            filtered = stream_render_to_store(
                rows, render_fn=_double, store=store,
                num_workers=2, chunksize=2,
            )
            store.close_write()
            assert filtered == 3
            assert list(store) == [{"n": i * 2} for i in range(0, 4)]


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_default_render_workers_is_positive():
    assert DEFAULT_RENDER_WORKERS >= 1


def test_default_render_chunksize_is_positive():
    assert DEFAULT_RENDER_CHUNKSIZE >= 1
