"""Streaming dataset rendering via PyTorch DataLoader.

Renders JSONL rows on the fly inside DataLoader workers, with prefetch
hiding per-row tokenization behind the GPU train step. Workers are
spawned (not forked) so each gets its own tokenizer / renderer state
without copy-on-write inflating the parent's heap.

Replaces the earlier "render to a disk-backed pickle store, then read
back during training" pipeline because:

* Per-row CPU render cost is small relative to the trainer's
  ``forward_backward`` step, so DataLoader prefetch hides it.
* No disk persistence is needed for SFT -- multi-epoch simply
  re-renders the same JSONL rows (cheap on the CPU orchestrator pod).
* Eliminates ~200 LoC of bespoke worker pool / disk store / index code
  in favour of battle-tested DataLoader machinery.

Used today by ``sft_loop``. DPO/ORPO can reuse :class:`JsonlRenderDataset`
with a preference-pair ``render_fn``; their multi-epoch reference logprob
cache is a separate concern (kept on disk so the reference trainer can
be released after epoch 0).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Callable, Iterator, List

import torch
import torch.utils.data as torch_data

logger = logging.getLogger(__name__)

# Defaults validated end-to-end on a 110K-example, 256K-context SFT
# dataset on a 16 vCPU / 58 GiB orchestrator pod.
DEFAULT_RENDER_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2
JSONL_ROW_INDEX_KEY = "_fireworks_jsonl_row_index"


# ---------------------------------------------------------------------------
# JSONL → render → object dataset
# ---------------------------------------------------------------------------


def _scan_jsonl_offsets(path: str, max_examples: int | None = None) -> List[int]:
    """Return the byte offset of every non-blank line in a JSONL file."""
    offsets: List[int] = []
    with open(path, "rb") as f:
        offset = 0
        for line in f:
            if line.strip():
                offsets.append(offset)
                if max_examples is not None and len(offsets) >= max_examples:
                    break
            offset += len(line)
    return offsets


class JsonlRenderDataset(torch_data.Dataset):
    """Map-style dataset that lazily renders each JSONL row on access.

    A single linear scan at construction time builds the byte-offset
    table; each ``__getitem__(i)`` is a seek + readline + ``json.loads``
    + ``render_fn``. ``render_fn`` must be a top-level (module-level)
    function so it is picklable for spawn workers.

    ``render_fn`` may return ``None`` for rows that should be dropped
    (empty messages, over-length sequences, ...). The companion
    :func:`make_render_dataloader` wires up a collate that filters Nones.
    ``row_index_key`` opt-in attaches the original 0-based JSONL row
    index for callers that need source-file diagnostics.
    """

    def __init__(
        self,
        path: str,
        render_fn: Callable[[dict], Any | None],
        *,
        max_examples: int | None = None,
        indices: List[int] | None = None,
        row_index_key: str | None = None,
    ):
        self._path = path
        self._render_fn = render_fn
        self._offsets = _scan_jsonl_offsets(path, max_examples)
        self._index_map: List[int] = (
            list(indices)
            if indices is not None
            else list(range(len(self._offsets)))
        )
        self._row_index_key = row_index_key

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, i: int) -> Any:
        offset = self._offsets[self._index_map[i]]
        with open(self._path) as f:
            f.seek(offset)
            line = f.readline()
        row = json.loads(line)
        if self._row_index_key is not None:
            row[self._row_index_key] = self._index_map[i]
        return self._render_fn(row)

    def with_indices(self, indices: List[int]) -> "JsonlRenderDataset":
        """Return a view of this dataset restricted to ``indices``.

        Shares the underlying offset table and render_fn; only the
        index mapping differs. Used to carve out a contiguous eval slice
        from the head of the training data without rescanning the file.
        """
        view = object.__new__(type(self))
        view._path = self._path
        view._render_fn = self._render_fn
        view._offsets = self._offsets
        view._index_map = list(indices)
        view._row_index_key = self._row_index_key
        return view

    @property
    def num_underlying_rows(self) -> int:
        return len(self._offsets)

    def approx_row_sizes(self) -> List[int]:
        """Return a cheap per-item size proxy: raw JSONL byte length.

        Byte length is already derivable from the offset table built at
        construction (no rendering / tokenization needed) and correlates
        strongly with token count, so it is a good sort key for
        :class:`LengthGroupedBatchSampler`. Values are aligned with
        ``__getitem__`` indexing, i.e. they honor ``with_indices`` /
        eval-carveout views.

        It is a proxy, not an exact token count -- solid for text rows,
        weaker for base64-image multimodal rows. Keep length grouping
        opt-in for that reason.
        """
        try:
            file_size = os.path.getsize(self._path)
        except OSError:
            file_size = self._offsets[-1] if self._offsets else 0
        n = len(self._offsets)
        full_sizes = [
            (self._offsets[j + 1] if j + 1 < n else file_size) - self._offsets[j]
            for j in range(n)
        ]
        return [full_sizes[j] for j in self._index_map]


# ---------------------------------------------------------------------------
# Length-grouped batch sampler
# ---------------------------------------------------------------------------


class LengthGroupedBatchSampler(torch_data.Sampler):
    """Yield batches of similarly-sized items to cut padding / CP overhead.

    Random arrival order forces each trainer batch to pad to its longest
    member (the trainer's ``_plan_non_pp`` pad+stacks ``[B, max_len]``)
    and, when context parallel is enabled, to run at the CP degree of
    that longest member. Grouping by length makes batches
    length-homogeneous so most batches pad to ~their own length and
    (under dynamic CP) run at a low CP degree; only the long-sequence
    batches pay the high-CP / long-pad cost.

    Bucket-then-shuffle (HF-style) preserves epoch-to-epoch randomness: a
    fresh permutation is cut into mega-batches of
    ``batch_size * group_factor``; each mega-batch is sorted by size and
    chunked into batches; finally batch *order* is shuffled so length is
    not monotonic across the epoch. A single short remainder batch (when
    ``len(dataset) % batch_size != 0``) is always emitted last so the
    recipe's positional resume/cursor math stays valid (batch ``i`` still
    accounts for ``i * batch_size`` consumed rows).

    The sampler reads ``generator`` lazily in ``__iter__``, so the
    recipe's per-epoch ``generator.manual_seed(seed + epoch)`` reseed
    takes effect exactly as it does for the default shuffling path.
    """

    def __init__(
        self,
        sizes: List[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
        group_factor: int = 50,
    ) -> None:
        self._sizes = list(sizes)
        self._batch_size = max(1, int(batch_size))
        self._shuffle = bool(shuffle)
        self._generator = generator
        self._group_factor = max(1, int(group_factor))

    def __len__(self) -> int:
        n = len(self._sizes)
        return (n + self._batch_size - 1) // self._batch_size

    def __iter__(self) -> Iterator[List[int]]:
        n = len(self._sizes)
        if n == 0:
            return
        if self._shuffle:
            order = torch.randperm(n, generator=self._generator).tolist()
        else:
            order = list(range(n))

        # mega = batch_size * group_factor is a multiple of batch_size, so
        # only the final mega-batch can produce a short remainder chunk.
        mega = self._batch_size * self._group_factor
        batches: List[List[int]] = []
        for start in range(0, n, mega):
            chunk = order[start : start + mega]
            chunk.sort(key=lambda idx: self._sizes[idx], reverse=True)
            for j in range(0, len(chunk), self._batch_size):
                batches.append(chunk[j : j + self._batch_size])

        remainder = None
        if batches and len(batches[-1]) < self._batch_size:
            remainder = batches.pop()
        if self._shuffle and batches:
            perm = torch.randperm(len(batches), generator=self._generator).tolist()
            batches = [batches[k] for k in perm]
        if remainder is not None:
            batches.append(remainder)
        yield from batches


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def _drop_none_collate(batch: List[Any]) -> List[Any]:
    return [d for d in batch if d is not None]


def make_render_dataloader(
    dataset: torch_data.Dataset,
    *,
    batch_size: int,
    num_workers: int = DEFAULT_RENDER_WORKERS,
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
    worker_init_fn: Callable[[int], None] | None = None,
    group_by_length: bool = False,
    length_group_factor: int = 50,
    sizes: List[int] | None = None,
) -> torch_data.DataLoader:
    """Build a DataLoader for ``dataset`` with our spawn / collate defaults.

    * ``multiprocessing_context="spawn"`` keeps each worker's RSS
      independent of the parent (no copy-on-write inflation of the
      tokenizer / renderer state).
    * ``persistent_workers=True`` keeps the per-worker tokenizer alive
      across epochs so we pay the spawn-and-init cost once.
    * ``collate_fn`` returns a python list of rendered items, dropping
      any ``None`` entries. Tinker's ``forward_backward`` accepts
      variable-size batches so dropping is safe.

    When ``group_by_length`` is set, batches are composed from
    similarly-sized items via :class:`LengthGroupedBatchSampler` (needs
    ``sizes``, a per-item length proxy aligned with ``dataset``
    indexing). The batch *count* is unchanged, so the recipe's
    positional resume/cursor math is unaffected. ``num_workers <= 1``
    falls back to single-process rendering. Unit tests rely on this to
    monkey-patch the renderer (spawn workers can't see test-time monkey
    patches), and a single worker subprocess rarely earns its keep over
    in-process rendering anyway.
    """
    batch_sampler = None
    if group_by_length:
        if sizes is None:
            raise ValueError(
                "make_render_dataloader(group_by_length=True) requires `sizes` "
                "(a per-item length proxy, e.g. dataset.approx_row_sizes())."
            )
        batch_sampler = LengthGroupedBatchSampler(
            sizes,
            batch_size,
            shuffle=shuffle,
            generator=generator,
            group_factor=length_group_factor,
        )

    if num_workers <= 1:
        if batch_sampler is not None:
            return torch_data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=_drop_none_collate,
            )
        return torch_data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            collate_fn=_drop_none_collate,
        )

    spawn_kwargs = dict(
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_context="spawn",
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        collate_fn=_drop_none_collate,
    )
    if batch_sampler is not None:
        return torch_data.DataLoader(dataset, batch_sampler=batch_sampler, **spawn_kwargs)
    return torch_data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        **spawn_kwargs,
    )


# ---------------------------------------------------------------------------
# Append-only pickle log (DPO ref-cache)
# ---------------------------------------------------------------------------


class AppendOnlyPickleLog:
    """Sequential disk-backed object log: append in epoch 0, iterate in epochs 1+.

    Designed for DPO's reference-logprob cache: we need to spool enriched
    preference pairs to disk so the (expensive) reference trainer can be
    released after epoch 0, but epochs 1+ only ever iterate the cache
    front-to-back. That eliminates everything the previous
    ``DiskBackedDatumStore`` carried (offset index, mmap, random access,
    truncation safety) and reduces the disk store to ~30 LoC of pickle.

    Lifecycle: ``append(...)`` while writing → ``close_write()`` → iterate
    via ``for x in log``. Iterating before ``close_write()`` raises.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._fh: Any = open(path, "wb")
        self._count = 0

    def append(self, obj: Any) -> None:
        if self._fh is None:
            raise RuntimeError("AppendOnlyPickleLog is closed; cannot append")
        pickle.dump(obj, self._fh, protocol=pickle.HIGHEST_PROTOCOL)
        self._count += 1

    def close_write(self) -> None:
        """Flush + fsync + close the write handle so readers see all data."""
        if self._fh is None:
            return
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        self._fh = None

    def __len__(self) -> int:
        return self._count

    def disk_size_bytes(self) -> int:
        try:
            return os.path.getsize(self._path)
        except OSError:
            return 0

    def __iter__(self) -> Iterator[Any]:
        if self._fh is not None:
            raise RuntimeError(
                "close_write() must be called before iterating an AppendOnlyPickleLog"
            )
        with open(self._path, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    return

    def close(self) -> None:
        if self._fh is not None:
            self.close_write()

    def __enter__(self) -> "AppendOnlyPickleLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
