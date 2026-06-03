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

import pytest

from training.utils.streaming import (
    AppendOnlyPickleLog,
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_RENDER_WORKERS,
    JSONL_ROW_INDEX_KEY,
    JsonlRenderDataset,
    LengthGroupedBatchSampler,
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


def test_dataset_init_with_explicit_indices(tmp_path):
    """Constructor ``indices=...`` lets callers reorder / sub-select rows."""
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i} for i in range(10)])

    ds = JsonlRenderDataset(path, _identity_render, indices=[7, 2, 9])

    assert len(ds) == 3
    assert ds.num_underlying_rows == 10  # full file still scanned
    assert [ds[i]["id"] for i in range(3)] == [7, 2, 9]


def test_dataset_passes_source_row_index_to_render_fn(tmp_path):
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i} for i in range(5)])

    def render(row: dict) -> dict:
        return {"id": row["i"], "source_index": row[JSONL_ROW_INDEX_KEY]}

    ds = JsonlRenderDataset(
        path,
        render,
        indices=[4, 1, 3],
        row_index_key=JSONL_ROW_INDEX_KEY,
    )

    assert [ds[i] for i in range(3)] == [
        {"id": 4, "source_index": 4},
        {"id": 1, "source_index": 1},
        {"id": 3, "source_index": 3},
    ]


def test_dataset_does_not_add_source_row_index_by_default(tmp_path):
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": 0}])

    def render(row: dict) -> dict:
        return row

    ds = JsonlRenderDataset(path, render)

    assert ds[0] == {"i": 0}


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
# approx_row_sizes
# ---------------------------------------------------------------------------


def test_approx_row_sizes_tracks_content_length(tmp_path):
    path = str(tmp_path / "data.jsonl")
    # Rows with deliberately increasing payload size.
    _write_jsonl(path, [{"i": i, "pad": "x" * (i * 100)} for i in range(5)])

    ds = JsonlRenderDataset(path, _identity_render)
    sizes = ds.approx_row_sizes()

    assert len(sizes) == 5
    # Byte length is strictly increasing with payload size and is a usable
    # proxy (each entry at least covers its own pad).
    assert sizes == sorted(sizes)
    assert sizes[4] > sizes[0]


def test_approx_row_sizes_aligns_with_view_indexing(tmp_path):
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i, "pad": "x" * (i * 50)} for i in range(10)])

    full = JsonlRenderDataset(path, _identity_render)
    full_sizes = full.approx_row_sizes()

    view = full.with_indices([7, 2, 9])
    view_sizes = view.approx_row_sizes()

    # Sizes follow the view's index map, not the underlying file order.
    assert view_sizes == [full_sizes[7], full_sizes[2], full_sizes[9]]


# ---------------------------------------------------------------------------
# LengthGroupedBatchSampler
# ---------------------------------------------------------------------------


def _make_sampler(sizes, batch_size, *, seed=0, shuffle=True, group_factor=2):
    import torch

    gen = torch.Generator().manual_seed(seed)
    return LengthGroupedBatchSampler(
        sizes,
        batch_size,
        shuffle=shuffle,
        generator=gen,
        group_factor=group_factor,
    )


def test_length_grouped_sampler_covers_every_index_once():
    sizes = list(range(23))  # n not divisible by batch_size
    sampler = _make_sampler(sizes, batch_size=4, group_factor=3)
    batches = list(sampler)
    flat = [idx for batch in batches for idx in batch]
    assert sorted(flat) == list(range(23))
    # No duplicates, nothing dropped.
    assert len(flat) == 23


def test_length_grouped_sampler_batch_count_matches_ceil():
    sizes = list(range(23))
    sampler = _make_sampler(sizes, batch_size=4, group_factor=3)
    assert len(sampler) == 6  # ceil(23 / 4)
    assert len(list(sampler)) == 6


def test_length_grouped_sampler_remainder_is_last():
    sizes = list(range(23))
    sampler = _make_sampler(sizes, batch_size=4, group_factor=3)
    batches = list(sampler)
    # Exactly one short batch and it must be emitted last so the recipe's
    # positional cursor/resume math (batch i == i*batch_size rows) holds.
    short = [i for i, b in enumerate(batches) if len(b) < 4]
    assert short == [len(batches) - 1]
    assert len(batches[-1]) == 23 % 4


def test_length_grouped_sampler_is_deterministic_per_seed():
    sizes = [i % 7 for i in range(40)]
    a = list(_make_sampler(sizes, batch_size=4, seed=123))
    b = list(_make_sampler(sizes, batch_size=4, seed=123))
    c = list(_make_sampler(sizes, batch_size=4, seed=124))
    assert a == b
    assert a != c  # different seed -> different bucketing/order


def test_length_grouped_sampler_reduces_within_batch_spread():
    import random
    import torch

    rng = random.Random(0)
    sizes = [rng.randint(1, 10_000) for _ in range(256)]

    grouped = list(_make_sampler(sizes, batch_size=8, group_factor=8, seed=0))

    gen = torch.Generator().manual_seed(0)
    rand_order = torch.randperm(len(sizes), generator=gen).tolist()
    random_batches = [rand_order[i : i + 8] for i in range(0, len(sizes), 8)]

    def mean_spread(batches):
        spreads = [max(sizes[i] for i in b) - min(sizes[i] for i in b) for b in batches]
        return sum(spreads) / len(spreads)

    # Grouping should dramatically shrink the average max-min length spread
    # within a batch (this is what cuts padding / CP cost).
    assert mean_spread(grouped) < 0.5 * mean_spread(random_batches)


def test_length_grouped_sampler_no_shuffle_is_sorted_within_mega_batch():
    sizes = [5, 1, 4, 2, 3, 0]
    sampler = LengthGroupedBatchSampler(
        sizes, batch_size=2, shuffle=False, generator=None, group_factor=10
    )
    batches = list(sampler)
    # One mega-batch (group_factor large): sorted desc by size, chunked by 2.
    assert batches == [[0, 2], [4, 3], [1, 5]]


def test_length_grouped_sampler_empty():
    sampler = LengthGroupedBatchSampler([], batch_size=4)
    assert list(sampler) == []
    assert len(sampler) == 0


# ---------------------------------------------------------------------------
# make_render_dataloader (group_by_length path)
# ---------------------------------------------------------------------------


def test_dataloader_group_by_length_requires_sizes(tmp_path):
    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i} for i in range(4)])
    ds = JsonlRenderDataset(path, _identity_render)
    with pytest.raises(ValueError, match="requires `sizes`"):
        make_render_dataloader(
            ds, batch_size=2, num_workers=0, group_by_length=True, sizes=None
        )


def test_dataloader_group_by_length_covers_all_rows(tmp_path):
    import torch

    path = str(tmp_path / "data.jsonl")
    # Increasing payloads so byte length is a meaningful sort key.
    _write_jsonl(path, [{"i": i, "pad": "x" * (i * 20)} for i in range(10)])
    ds = JsonlRenderDataset(path, _identity_render)

    gen = torch.Generator().manual_seed(0)
    loader = make_render_dataloader(
        ds,
        batch_size=3,
        num_workers=0,
        shuffle=True,
        generator=gen,
        group_by_length=True,
        length_group_factor=2,
        sizes=ds.approx_row_sizes(),
    )
    batches = list(loader)
    flat = sorted(item["id"] for batch in batches for item in batch)
    assert flat == list(range(10))
    assert len(batches) == 4  # ceil(10 / 3)


# ---------------------------------------------------------------------------
# LengthGroupedBatchSampler -- proof tests (this is the behavior we want)
# ---------------------------------------------------------------------------


def test_length_grouped_sampler_cuts_total_padding():
    """The throughput-relevant metric: pad+stack waste (B*max_len - sum(len)).

    Random batching pads every row up to the batch's longest member; grouping
    similar lengths together collapses that waste. This is *the* reason the
    feature exists, so assert the padding token total drops sharply.
    """
    import random
    import torch

    rng = random.Random(7)
    sizes = [rng.randint(1, 4096) for _ in range(512)]
    bs = 8

    grouped = list(_make_sampler(sizes, batch_size=bs, group_factor=16, seed=0))

    gen = torch.Generator().manual_seed(0)
    rand_order = torch.randperm(len(sizes), generator=gen).tolist()
    random_batches = [rand_order[i : i + bs] for i in range(0, len(sizes), bs)]

    def pad_waste(batches):
        total = 0
        for b in batches:
            longest = max(sizes[i] for i in b)
            total += len(b) * longest - sum(sizes[i] for i in b)
        return total

    grouped_waste = pad_waste(grouped)
    random_waste = pad_waste(random_batches)
    # Grouping should eliminate the large majority of padding tokens.
    assert grouped_waste < 0.25 * random_waste


def test_length_grouped_sampler_per_epoch_reseed_changes_order_same_coverage():
    """SFT reseeds the generator each epoch (seed + epoch). Because the sampler
    reads the generator lazily in __iter__, a reseed must change the batch
    composition while still covering every index exactly once."""
    import torch

    sizes = [i % 11 for i in range(50)]
    gen = torch.Generator()
    sampler = LengthGroupedBatchSampler(
        sizes, batch_size=4, shuffle=True, generator=gen, group_factor=3
    )

    gen.manual_seed(100)  # "epoch 0"
    epoch0 = list(sampler)
    gen.manual_seed(101)  # "epoch 1" reseed -- picked up lazily in __iter__
    epoch1 = list(sampler)

    assert epoch0 != epoch1  # different epochs -> different batches
    flat0 = sorted(i for b in epoch0 for i in b)
    flat1 = sorted(i for b in epoch1 for i in b)
    assert flat0 == flat1 == list(range(50))  # coverage identical both epochs


# ---------------------------------------------------------------------------
# Dataloader-level proof: grouping never drops/duplicates vs the default path
# ---------------------------------------------------------------------------


def test_dataloader_group_by_length_same_coverage_as_default(tmp_path):
    """Length grouping must train on exactly the same rows (and same batch
    count) as the default shuffle path -- it only changes *which batch* a row
    lands in, never *whether* it is trained on."""
    import torch

    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i, "pad": "x" * ((i * 13) % 200)} for i in range(37)])
    ds = JsonlRenderDataset(path, _identity_render)

    default_loader = make_render_dataloader(
        ds, batch_size=5, num_workers=0, shuffle=True,
        generator=torch.Generator().manual_seed(1),
    )
    default_ids = sorted(it["id"] for b in default_loader for it in b)
    default_nbatches = len(list(default_loader))

    grouped_loader = make_render_dataloader(
        ds, batch_size=5, num_workers=0, shuffle=True,
        generator=torch.Generator().manual_seed(1),
        group_by_length=True, length_group_factor=3, sizes=ds.approx_row_sizes(),
    )
    grouped_ids = sorted(it["id"] for b in grouped_loader for it in b)

    assert grouped_ids == default_ids == list(range(37))
    assert len(list(grouped_loader)) == default_nbatches  # same step count


# ---------------------------------------------------------------------------
# Performance regression: grouping keeps the dataloader cheap (no O(n^2))
# ---------------------------------------------------------------------------


def _time_min(fn, repeats: int = 5) -> float:
    """Best-of-k wall time. ``min`` is the most stable estimator under noise."""
    import time

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def test_length_grouped_sampler_construction_scales_linearly():
    """Guard against an accidental O(n^2) regression in the sampler. With a
    fixed mega-batch (batch_size * group_factor) the work is linearithmic, so
    doubling n should roughly double the time -- never quadruple it. Generous
    slack + an absolute floor keep this robust to CI timing noise while still
    catching a real quadratic blow-up (which would be seconds at n=40k)."""
    import torch

    def builder(n):
        sizes = [(i * 2654435761) % 100_000 for i in range(n)]

        def run():
            gen = torch.Generator().manual_seed(0)
            sampler = LengthGroupedBatchSampler(
                sizes, batch_size=8, shuffle=True, generator=gen, group_factor=50
            )
            list(sampler)

        return run

    t_n = _time_min(builder(20_000))
    t_2n = _time_min(builder(40_000))

    assert t_2n < 3.0 * t_n + 0.05


def test_dataloader_group_by_length_overhead_is_bounded(tmp_path):
    """Enabling grouping must not materially slow the dataloader: per-row
    render/seek cost is identical, only the batch index math is added. Assert
    the grouped path stays within a small constant factor of the default."""
    import torch

    path = str(tmp_path / "data.jsonl")
    _write_jsonl(path, [{"i": i, "pad": "x" * ((i % 50) * 8)} for i in range(2000)])
    ds = JsonlRenderDataset(path, _identity_render)
    sizes = ds.approx_row_sizes()

    def run_default():
        loader = make_render_dataloader(
            ds, batch_size=8, num_workers=0, shuffle=True,
            generator=torch.Generator().manual_seed(0),
        )
        for _ in loader:
            pass

    def run_grouped():
        loader = make_render_dataloader(
            ds, batch_size=8, num_workers=0, shuffle=True,
            generator=torch.Generator().manual_seed(0),
            group_by_length=True, length_group_factor=16, sizes=sizes,
        )
        for _ in loader:
            pass

    t_default = _time_min(run_default, repeats=3)
    t_grouped = _time_min(run_grouped, repeats=3)

    assert t_grouped < 2.0 * t_default + 0.2


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
