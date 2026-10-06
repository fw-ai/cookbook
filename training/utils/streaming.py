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

import bisect
from collections import OrderedDict
from dataclasses import dataclass
import json
import logging
import os
import pickle
from typing import Any, Callable, Iterator, List

import torch
import torch.utils.data as torch_data

from training.utils.runner import DatasetError

logger = logging.getLogger(__name__)

# Four workers is the large-dataset-tested safe fallback. Managed SFT and DPO
# jobs auto-size below the runtime CPU and memory ceilings instead of treating
# this fallback as a fixed target.
DEFAULT_RENDER_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2
JSONL_ROW_INDEX_KEY = "_fireworks_jsonl_row_index"
DEFAULT_GCS_BLOCK_SIZE = 8 * 1024 * 1024
DEFAULT_GCS_CACHE_BLOCKS = 2


# ---------------------------------------------------------------------------
# JSONL → render → object dataset
# ---------------------------------------------------------------------------


def _decode_jsonl_object(raw_line: bytes, *, path: str, line_no: int) -> dict:
    try:
        row = json.loads(raw_line.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise DatasetError(
            f"{path}:{line_no}: JSONL row must be valid UTF-8."
        ) from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}:{line_no}: invalid JSONL: {exc.msg}.") from exc
    if not isinstance(row, dict):
        raise DatasetError(
            f"{path}:{line_no}: JSONL row must be an object, got "
            f"{type(row).__name__}."
        )
    return row


def _validate_jsonl_object(raw_line: bytes, *, path: str, line_no: int) -> None:
    _decode_jsonl_object(raw_line, path=path, line_no=line_no)


def _scan_jsonl_offsets(path: str, max_examples: int | None = None) -> List[int]:
    """Return byte offsets for non-blank, valid JSON-object lines."""
    offsets: List[int] = []
    with open(path, "rb") as f:
        offset = 0
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                _validate_jsonl_object(line, path=path, line_no=line_no)
                offsets.append(offset)
                if max_examples is not None and len(offsets) >= max_examples:
                    break
            offset += len(line)
    return offsets


@dataclass(frozen=True)
class _GcsJsonlShard:
    uri: str
    bucket_name: str
    object_name: str
    generation: int | None
    size: int


@dataclass(frozen=True)
class _GcsJsonlIndex:
    shards: List[_GcsJsonlShard]
    shard_offsets: List[List[int]]
    shard_lengths: List[List[int]]
    shard_line_numbers: List[List[int]]
    shard_starts: List[int]


_GCS_JSONL_INDEX_CACHE: dict[tuple[Any, ...], _GcsJsonlIndex] = {}


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise DatasetError(f"GCS dataset path must start with gs://, got {uri!r}.")
    without_scheme = uri[len("gs://") :]
    bucket_name, _, object_name = without_scheme.partition("/")
    object_name = object_name.strip("/")
    if not bucket_name or not object_name:
        raise DatasetError(
            f"GCS dataset path must include bucket and object/prefix: {uri!r}."
        )
    return bucket_name, object_name


def _make_gcs_client() -> Any:
    # lazy: google-cloud-storage is needed only for remote managed SFT inputs.
    try:
        from google.cloud import storage as gcs
    except ImportError as exc:
        raise DatasetError(
            "GCS streaming datasets require the google-cloud-storage package."
        ) from exc

    return gcs.Client()


def _generation_value(blob: Any) -> int | None:
    generation = getattr(blob, "generation", None)
    if generation is None:
        return None
    try:
        return int(generation)
    except (TypeError, ValueError):
        return None


def _shard_from_blob(bucket_name: str, blob: Any) -> _GcsJsonlShard:
    name = str(blob.name)
    return _GcsJsonlShard(
        uri=f"gs://{bucket_name}/{name}",
        bucket_name=bucket_name,
        object_name=name,
        generation=_generation_value(blob),
        size=int(getattr(blob, "size", 0) or 0),
    )


def resolve_gcs_jsonl_shards(path: str, *, client: Any | None = None) -> List[_GcsJsonlShard]:
    """Resolve a GCS object or prefix to generation-pinned JSONL shard metadata."""
    bucket_name, object_name = _parse_gcs_uri(path)
    client = client if client is not None else _make_gcs_client()
    bucket = client.bucket(bucket_name)

    # Explicit object paths behave like local explicit file paths: honor them
    # even when they do not end in .jsonl.
    direct_blob = bucket.get_blob(object_name)
    if direct_blob is not None:
        shard = _shard_from_blob(bucket_name, direct_blob)
        if shard.size <= 0:
            raise DatasetError(f"GCS dataset object {path!r} is empty.")
        return [shard]

    prefix = object_name.rstrip("/") + "/"
    blobs = [
        blob
        for blob in client.list_blobs(bucket_name, prefix=prefix)
        if not str(blob.name).endswith("/") and str(blob.name).endswith(".jsonl")
    ]
    blobs.sort(key=lambda blob: str(blob.name))
    shards = [_shard_from_blob(bucket_name, blob) for blob in blobs]
    shards = [shard for shard in shards if shard.size > 0]
    if not shards:
        raise DatasetError(
            f"GCS dataset prefix {path!r} contains no non-empty '*.jsonl' shards."
        )
    return shards


def _download_gcs_range(
    client: Any,
    shard: _GcsJsonlShard,
    start: int,
    length: int,
) -> bytes:
    if length <= 0:
        return b""
    bucket = client.bucket(shard.bucket_name)
    blob = bucket.blob(shard.object_name, generation=shard.generation)
    return blob.download_as_bytes(start=start, end=start + length - 1)


def _scan_gcs_jsonl_offsets(
    shard: _GcsJsonlShard,
    *,
    client: Any,
    block_size: int,
    max_examples: int | None = None,
    row_callback: Callable[[str, int, dict], None] | None = None,
) -> tuple[List[int], List[int], List[int]]:
    offsets: List[int] = []
    lengths: List[int] = []
    line_numbers: List[int] = []
    pending = b""
    pending_start = 0
    line_no = 0

    for block_start in range(0, shard.size, block_size):
        block_len = min(block_size, shard.size - block_start)
        data = _download_gcs_range(client, shard, block_start, block_len)
        scan_pos = 0
        while scan_pos < len(data):
            newline = data.find(b"\n", scan_pos)
            if newline == -1:
                part = data[scan_pos:]
                if pending:
                    pending += part
                else:
                    pending = part
                    pending_start = block_start + scan_pos
                break

            end = newline + 1
            part = data[scan_pos:end]
            if pending:
                line = pending + part
                line_start = pending_start
                pending = b""
            else:
                line = part
                line_start = block_start + scan_pos
            line_no += 1
            if line.strip():
                row = _decode_jsonl_object(line, path=shard.uri, line_no=line_no)
                if row_callback is not None:
                    row_callback(shard.uri, line_no, row)
                offsets.append(line_start)
                lengths.append(len(line))
                line_numbers.append(line_no)
                if max_examples is not None and len(offsets) >= max_examples:
                    return offsets, lengths, line_numbers
            scan_pos = end

    if pending:
        line_no += 1
        if pending.strip():
            row = _decode_jsonl_object(pending, path=shard.uri, line_no=line_no)
            if row_callback is not None:
                row_callback(shard.uri, line_no, row)
            offsets.append(pending_start)
            lengths.append(len(pending))
            line_numbers.append(line_no)

    return offsets, lengths, line_numbers


def _gcs_index_cache_key(
    path: str,
    shards: List[_GcsJsonlShard],
    *,
    max_examples: int | None,
    block_size: int,
) -> tuple[Any, ...]:
    return (
        path,
        max_examples,
        block_size,
        tuple((shard.uri, shard.generation, shard.size) for shard in shards),
    )


def _gcs_shard_starts(shard_offsets: List[List[int]]) -> List[int]:
    starts: List[int] = [0]
    for offsets in shard_offsets:
        starts.append(starts[-1] + len(offsets))
    return starts


def _get_gcs_jsonl_index(
    path: str,
    *,
    client: Any,
    max_examples: int | None,
    block_size: int,
    row_callback: Callable[[str, int, dict], None] | None = None,
) -> _GcsJsonlIndex:
    shards = resolve_gcs_jsonl_shards(path, client=client)
    cache_key = _gcs_index_cache_key(
        path,
        shards,
        max_examples=max_examples,
        block_size=block_size,
    )
    cached = _GCS_JSONL_INDEX_CACHE.get(cache_key)
    if cached is not None and row_callback is None:
        return cached

    shard_offsets: List[List[int]] = []
    shard_lengths: List[List[int]] = []
    shard_line_numbers: List[List[int]] = []
    remaining = max_examples
    for shard in shards:
        if remaining is not None and remaining <= 0:
            break
        offsets, lengths, line_numbers = _scan_gcs_jsonl_offsets(
            shard,
            client=client,
            block_size=block_size,
            max_examples=remaining,
            row_callback=row_callback,
        )
        shard_offsets.append(offsets)
        shard_lengths.append(lengths)
        shard_line_numbers.append(line_numbers)
        if remaining is not None:
            remaining -= len(offsets)
            if remaining <= 0:
                break

    index = _GcsJsonlIndex(
        shards=shards,
        shard_offsets=shard_offsets,
        shard_lengths=shard_lengths,
        shard_line_numbers=shard_line_numbers,
        shard_starts=_gcs_shard_starts(shard_offsets),
    )
    _GCS_JSONL_INDEX_CACHE[cache_key] = index
    return index


def resolve_jsonl_shards(path: str) -> List[str]:
    """Resolve a dataset path to a deterministic list of JSONL shard files.

    Dataset staging (managed ``stage_gcs_dataset``) returns a *directory*
    when a BYOB dataset has multiple JSONL shards and a single file
    otherwise; both shapes must load identically downstream. Shards are
    never concatenated into one file (that would double staging storage).

    Shard policy:

    * ``path`` is a regular file -> ``[path]`` (the extension is not
      filtered, an explicit file path is always honored).
    * ``path`` is a directory -> every file whose name ends in
      ``.jsonl``, discovered recursively, in sorted POSIX
      relative-path order so the concatenated row order -- and with it
      every row index, resume cursor, and eval carve-out -- is stable
      across runs. Hidden files are included like any other shard
      (staging never writes dot-files, and skipping them would let a
      valid shard silently vanish).
    * ``path`` is missing, an empty directory, or a directory with no
      ``.jsonl`` shards -> ``DatasetError``. We never silently train on
      zero or partial data.

    This selects the same shards as managed-side ``collect_jsonl_files``
    (``firetitan.train.managed.jsonl_io``). Only the cookbook guarantees
    globally sorted relative paths: the managed validator checks all rows
    without assigning recipe row indices, and managed RFT materializes its
    own rows in directory traversal order. Their row indices are not shared
    with the SFT/DPO/ORPO recipe resume cursors or eval splits.
    """
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise DatasetError(
            f"Dataset path {path!r} does not exist or is not a file/directory."
        )
    shards: List[str] = []
    for root, _dirs, names in os.walk(path):
        for name in names:
            if name.endswith(".jsonl"):
                shards.append(os.path.join(root, name))
    shards.sort(
        key=lambda p: os.path.relpath(p, path).replace(os.sep, "/")
    )
    if not shards:
        raise DatasetError(
            f"Dataset directory {path!r} contains no '*.jsonl' shards; "
            "refusing to load an empty dataset."
        )
    return shards


class JsonlRenderDataset(torch_data.Dataset):
    """Map-style dataset that lazily renders each JSONL row on access.

    ``path`` may be a single JSONL file or a multi-shard directory (see
    :func:`resolve_jsonl_shards`); shards are read in deterministic
    sorted order without concatenating them. A linear scan per shard at
    construction time builds the byte-offset
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
        # ``path`` may be a single JSONL file or a staged multi-shard
        # directory (see :func:`resolve_jsonl_shards`). Rows are numbered
        # shard-major: global row g maps to shard ``bisect(...)`` at that
        # shard's local offset. All state is plain lists / str / the
        # picklable render_fn, so spawn DataLoader workers can rebuild it.
        self._shard_paths = resolve_jsonl_shards(path)
        self._shard_offsets: List[List[int]] = []
        remaining = max_examples
        for shard in self._shard_paths:
            if remaining is not None and remaining <= 0:
                break
            offsets = _scan_jsonl_offsets(shard, remaining)
            self._shard_offsets.append(offsets)
            if remaining is not None:
                remaining -= len(offsets)
                if remaining <= 0:
                    break
        # Cumulative shard start rows; ``self._shard_starts[k]`` is the
        # first global row index of shard k (``_shard_starts[-1]`` is the
        # total row count).
        self._shard_starts: List[int] = [0]
        for offsets in self._shard_offsets:
            self._shard_starts.append(self._shard_starts[-1] + len(offsets))
        self._index_map: List[int] = (
            list(indices)
            if indices is not None
            else list(range(self._shard_starts[-1]))
        )
        self._row_index_key = row_index_key

    def __len__(self) -> int:
        return len(self._index_map)

    def _locate(self, g: int) -> tuple[str, int]:
        """Map global row ``g`` to ``(shard_path, byte_offset)``."""
        k = bisect.bisect_right(self._shard_starts, g) - 1
        return (
            self._shard_paths[k],
            self._shard_offsets[k][g - self._shard_starts[k]],
        )

    def __getitem__(self, i: int) -> Any:
        shard_path, offset = self._locate(self._index_map[i])
        with open(shard_path, "rb") as f:
            f.seek(offset)
            line = f.readline()
        row = json.loads(line.decode("utf-8"))
        if self._row_index_key is not None:
            row[self._row_index_key] = self._index_map[i]
        return self._render_fn(row)

    def with_indices(self, indices: List[int]) -> "JsonlRenderDataset":
        """Return a view of this dataset restricted to ``indices``.

        Shares the underlying shard/offset tables and render_fn; only the
        index mapping differs. Used to carve out a contiguous eval slice
        from the head of the training data without rescanning the file.
        """
        view = object.__new__(type(self))
        view._path = self._path
        view._render_fn = self._render_fn
        view._shard_paths = self._shard_paths
        view._shard_offsets = self._shard_offsets
        view._shard_starts = self._shard_starts
        view._index_map = list(indices)
        view._row_index_key = self._row_index_key
        return view

    @property
    def num_underlying_rows(self) -> int:
        return self._shard_starts[-1]

    def approx_row_sizes(self) -> List[int]:
        """Return a cheap per-item size proxy: raw JSONL byte length.

        Byte length is already derivable from the offset tables built at
        construction (no rendering / tokenization needed) and correlates
        strongly with token count, so it is a good sort key for
        :class:`LengthGroupedBatchSampler`. Values are aligned with
        ``__getitem__`` indexing, i.e. they honor ``with_indices`` /
        eval-carveout views.

        It is a proxy, not an exact token count -- solid for text rows,
        weaker for base64-image multimodal rows. Keep length grouping
        opt-in for that reason.
        """
        full_sizes: List[int] = []
        for shard_path, offsets in zip(self._shard_paths, self._shard_offsets):
            try:
                file_size = os.path.getsize(shard_path)
            except OSError:
                file_size = offsets[-1] if offsets else 0
            n = len(offsets)
            full_sizes.extend(
                (offsets[j + 1] if j + 1 < n else file_size) - offsets[j]
                for j in range(n)
            )
        return [full_sizes[g] for g in self._index_map]


class GcsJsonlRenderDataset(torch_data.Dataset):
    """Remote-backed JSONL dataset using bounded GCS range reads.

    Construction scans each shard once to validate JSON and build byte offsets.
    ``__getitem__`` then fetches only the block(s) containing the requested row.
    This keeps orchestrator disk usage independent of the dataset payload size
    while preserving the same map-style access contract as ``JsonlRenderDataset``.
    """

    def __init__(
        self,
        path: str,
        render_fn: Callable[[dict], Any | None],
        *,
        max_examples: int | None = None,
        indices: List[int] | None = None,
        row_index_key: str | None = None,
        block_size: int = DEFAULT_GCS_BLOCK_SIZE,
        cache_blocks: int = DEFAULT_GCS_CACHE_BLOCKS,
        row_callback: Callable[[str, int, dict], None] | None = None,
    ):
        self._path = path
        self._render_fn = render_fn
        self._block_size = max(1, int(block_size))
        self._cache_blocks = max(1, int(cache_blocks))
        self._row_index_key = row_index_key
        self._client: Any | None = None
        self._block_cache: OrderedDict[tuple[int, int], bytes] = OrderedDict()

        client = self._gcs_client()
        index = _get_gcs_jsonl_index(
            path,
            client=client,
            max_examples=max_examples,
            block_size=self._block_size,
            row_callback=row_callback,
        )
        self._shards = index.shards
        self._shard_offsets = index.shard_offsets
        self._shard_lengths = index.shard_lengths
        self._shard_line_numbers = index.shard_line_numbers
        self._shard_starts = index.shard_starts
        self._index_map: List[int] = (
            list(indices)
            if indices is not None
            else list(range(self._shard_starts[-1]))
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_client"] = None
        state["_block_cache"] = OrderedDict()
        return state

    def _gcs_client(self) -> Any:
        if self._client is None:
            self._client = _make_gcs_client()
        return self._client

    def __len__(self) -> int:
        return len(self._index_map)

    def _locate(self, g: int) -> tuple[int, int, int]:
        shard_index = bisect.bisect_right(self._shard_starts, g) - 1
        local_index = g - self._shard_starts[shard_index]
        return (
            shard_index,
            self._shard_offsets[shard_index][local_index],
            self._shard_lengths[shard_index][local_index],
        )

    def _locate_with_line(self, g: int) -> tuple[int, int, int, int]:
        shard_index = bisect.bisect_right(self._shard_starts, g) - 1
        local_index = g - self._shard_starts[shard_index]
        return (
            shard_index,
            self._shard_offsets[shard_index][local_index],
            self._shard_lengths[shard_index][local_index],
            self._shard_line_numbers[shard_index][local_index],
        )

    def _read_block(self, shard_index: int, block_start: int) -> bytes:
        key = (shard_index, block_start)
        cached = self._block_cache.get(key)
        if cached is not None:
            self._block_cache.move_to_end(key)
            return cached

        shard = self._shards[shard_index]
        length = min(self._block_size, shard.size - block_start)
        data = _download_gcs_range(self._gcs_client(), shard, block_start, length)
        self._block_cache[key] = data
        self._block_cache.move_to_end(key)
        while len(self._block_cache) > self._cache_blocks:
            self._block_cache.popitem(last=False)
        return data

    def _read_range(self, shard_index: int, start: int, length: int) -> bytes:
        end = start + length
        pos = start
        chunks: list[bytes] = []
        while pos < end:
            block_start = (pos // self._block_size) * self._block_size
            block = self._read_block(shard_index, block_start)
            relative = pos - block_start
            take = min(len(block) - relative, end - pos)
            if take <= 0:
                raise DatasetError(
                    "Failed to read GCS row bytes from "
                    f"{self._shards[shard_index].uri} at offset {start}."
                )
            chunks.append(block[relative : relative + take])
            pos += take
        return b"".join(chunks)

    def __getitem__(self, i: int) -> Any:
        row_index = self._index_map[i]
        shard_index, offset, length = self._locate(row_index)
        line = self._read_range(shard_index, offset, length)
        row = json.loads(line.decode("utf-8"))
        if self._row_index_key is not None:
            row[self._row_index_key] = row_index
        return self._render_fn(row)

    def iter_raw_rows(self) -> Iterator[tuple[str, int, dict]]:
        """Yield ``(uri, line_number, row)`` for validation without staging."""
        for shard_index, shard in enumerate(self._shards[: len(self._shard_offsets)]):
            row_count = len(self._shard_offsets[shard_index])
            for local_index in range(row_count):
                offset = self._shard_offsets[shard_index][local_index]
                length = self._shard_lengths[shard_index][local_index]
                line_no = self._shard_line_numbers[shard_index][local_index]
                line = self._read_range(shard_index, offset, length)
                row = _decode_jsonl_object(line, path=shard.uri, line_no=line_no)
                yield shard.uri, line_no, row

    def with_indices(self, indices: List[int]) -> "GcsJsonlRenderDataset":
        view = object.__new__(type(self))
        view._path = self._path
        view._render_fn = self._render_fn
        view._block_size = self._block_size
        view._cache_blocks = self._cache_blocks
        view._row_index_key = self._row_index_key
        view._client = None
        view._block_cache = OrderedDict()
        view._shards = self._shards
        view._shard_offsets = self._shard_offsets
        view._shard_lengths = self._shard_lengths
        view._shard_line_numbers = self._shard_line_numbers
        view._shard_starts = self._shard_starts
        view._index_map = list(indices)
        return view

    @property
    def num_underlying_rows(self) -> int:
        return self._shard_starts[-1]

    def approx_row_sizes(self) -> List[int]:
        full_sizes: List[int] = []
        for lengths in self._shard_lengths:
            full_sizes.extend(lengths)
        return [full_sizes[g] for g in self._index_map]


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


def dataset_error_for_rendered_row(row: dict, exc: BaseException) -> DatasetError:
    """Return a dataset error whose message survives a DataLoader worker.

    Workers must return this instead of raising. PyTorch rebuilds a raised
    worker exception around the traceback, which would leak into the public
    training status.
    """
    message = str(exc)
    row_index = row.get(JSONL_ROW_INDEX_KEY)
    if isinstance(row_index, int) and not isinstance(row_index, bool):
        prefix = f"row {row_index}: "
        if not message.startswith(prefix):
            message = prefix + message
    if isinstance(exc, DatasetError) and str(exc) == message:
        return exc
    return DatasetError(message)


def raise_rendered_dataset_errors(batch: List[Any]) -> None:
    """Re-raise dataset errors that render workers returned in ``batch``."""
    for item in batch:
        if isinstance(item, DatasetError):
            raise item
        if isinstance(item, list):
            raise_rendered_dataset_errors(item)


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
