"""Unit tests for ``training.utils.data`` helper utilities."""

from __future__ import annotations

import pytest

from training.utils import replicate_rows_for_epochs
from training.utils.data import (
    iter_preference_examples,
    load_jsonl_dataset,
    load_preference_dataset,
)
from training.utils.runner import DatasetError


def test_replicate_rows_for_epochs_each_row_is_independent():
    """The naive ``rows * epochs`` only multiplies list references --
    every epoch shares the same dict instances.  Any rollout function
    that mutates its input row in place (attaching scratch fields,
    normalizing prompt data, caching renders) leaks that mutation
    into every later epoch and subsequent passes train on the
    already-mutated row instead of the original dataset.
    ``replicate_rows_for_epochs`` returns ``epochs * len(rows)``
    INDEPENDENT dict instances so per-epoch mutations cannot leak.
    """
    rows = [{"prompt": "a", "scratch": []}, {"prompt": "b", "scratch": []}]
    out = replicate_rows_for_epochs(rows, epochs=3)
    assert len(out) == 6  # 2 rows x 3 epochs
    # Every output dict is a distinct object -- id() differs even for
    # the same logical row across epochs.
    assert len({id(r) for r in out}) == 6
    # The nested mutable container (``scratch``) is also independent
    # -- that's what makes ``deepcopy`` (vs shallow ``dict(r)``) the
    # right primitive here.
    assert len({id(r["scratch"]) for r in out}) == 6
    # Mutating one copy does not affect any other copy or the originals.
    out[0]["scratch"].append("epoch0-mutation")
    assert all(r["scratch"] == [] for r in rows), (
        "Mutating a replicated copy leaked back into the originals -- "
        "deepcopy expected."
    )
    assert all(out[i]["scratch"] == [] for i in range(1, 6)), (
        "Mutating one copy leaked into other copies -- sibling rows "
        "must be independent across all epoch slots."
    )


def test_replicate_rows_for_epochs_zero_epochs_returns_empty():
    """``epochs=0`` is a degenerate but valid input; the helper
    returns an empty list rather than raising."""
    rows = [{"x": 1}]
    assert replicate_rows_for_epochs(rows, epochs=0) == []


def test_replicate_rows_for_epochs_preserves_per_epoch_order():
    """Rows are emitted epoch-by-epoch so the resume slicing logic
    (which advances a raw-row cursor through ``all_rows``) sees the
    same order it did when ``rows * epochs`` was used."""
    rows = [{"i": 0}, {"i": 1}, {"i": 2}]
    out = replicate_rows_for_epochs(rows, epochs=2)
    assert [r["i"] for r in out] == [0, 1, 2, 0, 1, 2]


# ---------------------------------------------------------------------------
# Multi-shard (staged directory) datasets -- FIR2-2500
# ---------------------------------------------------------------------------


def _write_preference_shards(tmp_path):
    """2-shard preference dataset (a.jsonl then b.jsonl).

    b.jsonl mixes the chosen/rejected schema with a 'samples'-schema row so
    per-shard normalization and cross-shard ordering are both exercised.
    """
    import json
    import os

    samples_row = {
        "samples": [
            {"messages": [{"role": "assistant", "content": "good2"}], "score": 1.0},
            {"messages": [{"role": "assistant", "content": "bad2"}], "score": 0.0},
        ]
    }
    shards = (
        ("a.jsonl", [{"chosen": {"t": "a0"}, "rejected": {"t": "a0r"}},
                     {"chosen": {"t": "a1"}, "rejected": {"t": "a1r"}}]),
        ("b.jsonl", [{"chosen": {"t": "b0"}, "rejected": {"t": "b0r"}}, samples_row]),
    )
    for rel, rows in shards:
        with open(os.path.join(tmp_path, rel), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    with open(os.path.join(tmp_path, "readme.txt"), "w") as f:
        f.write("not a shard\n")
    return str(tmp_path)


def test_load_preference_dataset_multi_shard(tmp_path):
    """Sorted shard order, per-schema normalization, and max_pairs."""
    data = load_preference_dataset(_write_preference_shards(tmp_path))
    assert len(data) == 4
    assert [pair["chosen"]["t"] for pair in data[:3]] == ["a0", "a1", "b0"]
    assert data[3]["chosen"]["messages"][0]["content"] == "good2"
    assert data[3]["rejected"]["messages"][0]["content"] == "bad2"

    capped = load_preference_dataset(_write_preference_shards(tmp_path), max_pairs=3)
    assert [pair["chosen"]["t"] for pair in capped] == ["a0", "a1", "b0"]


def test_load_preference_dataset_multi_shard_error_names_shard(tmp_path):
    import os

    root = _write_preference_shards(tmp_path)
    with open(os.path.join(root, "b.jsonl"), "a") as f:
        f.write('{"chosen": {"t": "x"}\n')

    with pytest.raises(DatasetError, match=r"b\.jsonl:3: invalid JSONL"):
        load_preference_dataset(root)


def test_iter_preference_examples_multi_shard(tmp_path):
    pairs = list(iter_preference_examples(_write_preference_shards(tmp_path)))
    # The 'samples'-schema row normalizes to a chosen/rejected pair.
    assert [p["chosen"].get("t") for p in pairs[:3]] == ["a0", "a1", "b0"]
    assert pairs[3]["chosen"]["messages"][0]["content"] == "good2"

    capped = list(iter_preference_examples(_write_preference_shards(tmp_path), max_pairs=2))
    assert [p["chosen"].get("t") for p in capped] == ["a0", "a1"]


def test_load_jsonl_dataset_multi_shard(tmp_path):
    rows = load_jsonl_dataset(_write_preference_shards(tmp_path))
    # Raw rows are returned unnormalized, so the 'samples' row passes through.
    assert [row["chosen"]["t"] for row in rows[:3]] == ["a0", "a1", "b0"]
    assert "samples" in rows[3]

    capped = load_jsonl_dataset(_write_preference_shards(tmp_path), max_rows=2)
    assert [row["chosen"]["t"] for row in capped] == ["a0", "a1"]


@pytest.mark.parametrize("max_rows", [2, 3, 10, None])
def test_load_jsonl_dataset_cap_counts_nonblank_rows_across_shards(tmp_path, max_rows):
    (tmp_path / "a.jsonl").write_text('{"id": 1}\n\n  \n')
    (tmp_path / "b.jsonl").write_text('\n{"id": 2}\n\n{"id": 3}\n')
    rows = load_jsonl_dataset(str(tmp_path), max_rows=max_rows)
    assert [row["id"] for row in rows] == [1, 2, 3][:max_rows]


def test_load_jsonl_dataset_cap_stops_before_malformed_rows(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"id": 1}\n\n')
    (tmp_path / "b.jsonl").write_text('{"id": 2}\ninvalid json\n')
    (tmp_path / "c.jsonl").write_text('invalid json\n')
    assert load_jsonl_dataset(str(tmp_path), max_rows=2) == [{"id": 1}, {"id": 2}]
