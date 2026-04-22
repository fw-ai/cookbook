"""Streaming dataset rendering utilities.

These utilities decouple the *render* phase from RAM. The eager pattern of:

    raw = list(json.loads(line) for line in f)            # ~10 GiB
    rendered = pool.map(render_fn, raw)                   # ~400+ GiB

does not fit on a single CPU node (allocatable RAM ~120 GiB on m5.8xlarge)
for realistic SFT/DPO datasets at long context. Instead this module exposes:

  * ``iter_jsonl_rows`` / ``count_jsonl_rows`` -- lazy reads of raw JSONL.
  * ``DiskBackedDatumStore`` -- append-only pickle store for rendered
    objects, with random read access via an in-memory ``(offset, length)``
    index (~16 B/row).
  * ``stream_render_to_store`` -- glue that runs a worker pool and spills
    each rendered item to a store as it completes (``Pool.imap``).

Peak RAM becomes O(num_workers * per_worker_render_footprint) instead of
O(num_examples * avg_seq_len * bytes_per_token).

Used today by ``sft_loop``. Designed so DPO/ORPO can adopt the same
pattern with a different ``render_fn`` and store payload type.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import pickle
from typing import IO, Any, Callable, Dict, Iterable, Iterator, List, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Disk-backed append-only store
# ---------------------------------------------------------------------------


class DiskBackedDatumStore:
    """Pickle-backed append-only store for arbitrary picklable objects.

    Items are appended during the rendering (write) phase and random-accessed
    by integer index during training (read) phase. All payload bytes live on
    local disk; only a small ``(offset, length)`` index is held in RAM.

    Usage::

        with DiskBackedDatumStore(path) as store:
            for item in producer:
                store.append(item)
            store.close_write()
            ...
            datum = store[i]            # random read
            for d in store: ...         # sequential read

    The store is named ``DiskBackedDatumStore`` for historical reasons but
    accepts any picklable payload (tinker.Datum, dict, dataclass, ...).
    """

    def __init__(self, path: str):
        self._path = path
        self._offsets: List[Tuple[int, int]] = []
        self._write_fh: IO[bytes] | None = None
        self._read_fh: IO[bytes] | None = None

    def __enter__(self) -> "DiskBackedDatumStore":
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._write_fh = open(self._path, "wb")
        return self

    def __exit__(self, *_exc) -> None:
        self.close_write()
        if self._read_fh is not None:
            self._read_fh.close()
            self._read_fh = None

    def append(self, item: Any) -> None:
        if self._write_fh is None:
            raise RuntimeError("DiskBackedDatumStore is not open for writing")
        blob = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
        offset = self._write_fh.tell()
        self._write_fh.write(blob)
        self._offsets.append((offset, len(blob)))

    def close_write(self) -> None:
        """Flush and close the write handle. Idempotent.

        Subsequent ``__getitem__`` calls reopen the file for reading.
        """
        if self._write_fh is not None:
            self._write_fh.flush()
            try:
                os.fsync(self._write_fh.fileno())
            except OSError:
                pass
            self._write_fh.close()
            self._write_fh = None

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Any:
        offset, length = self._offsets[idx]
        if self._read_fh is None:
            self._read_fh = open(self._path, "rb")
        self._read_fh.seek(offset)
        return pickle.loads(self._read_fh.read(length))

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self._offsets)):
            yield self[i]

    def disk_size_bytes(self) -> int:
        try:
            return os.path.getsize(self._path)
        except OSError:
            return 0

    @property
    def path(self) -> str:
        return self._path


# ---------------------------------------------------------------------------
# Lazy JSONL reads
# ---------------------------------------------------------------------------


def iter_jsonl_rows(
    path: str, max_examples: int | None = None,
) -> Iterator[Dict[str, Any]]:
    """Stream JSON rows from a JSONL file without materialising the list.

    Skips blank lines. Stops after ``max_examples`` valid rows when set.
    """
    yielded = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            yielded += 1
            if max_examples is not None and yielded >= max_examples:
                return


def count_jsonl_rows(path: str, max_examples: int | None = None) -> int:
    """Count non-blank lines in a JSONL file without parsing JSON."""
    n = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            if max_examples is not None and n >= max_examples:
                break
    return n


# ---------------------------------------------------------------------------
# Worker pool that spills directly to a store
# ---------------------------------------------------------------------------


# Conservative defaults validated end-to-end on a 110K-example, 256K-context
# SFT dataset:
#   workers=4   ~28 GiB spawn baseline (vs 56 GiB at 8 workers)
#   chunksize=10 cuts result-buffer bursts ~10x vs the imap default
DEFAULT_RENDER_WORKERS = 4
DEFAULT_RENDER_CHUNKSIZE = 10


def stream_render_to_store(
    rows: Iterable[Dict[str, Any]],
    *,
    render_fn: Callable[[Dict[str, Any]], T | None],
    store: DiskBackedDatumStore,
    num_workers: int = DEFAULT_RENDER_WORKERS,
    chunksize: int = DEFAULT_RENDER_CHUNKSIZE,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
    on_progress: Callable[[int, T | None], None] | None = None,
) -> int:
    """Render ``rows`` in parallel and append non-None results to ``store``.

    Returns the number of rows that were filtered out (``render_fn`` returned
    ``None``). The total number of rows processed equals
    ``len(store) + filtered_count`` after this returns.

    Uses the ``spawn`` multiprocessing context to avoid Copy-on-Write
    inflation that would otherwise multiply the parent's heap across
    workers. ``Pool.imap`` is used (not ``map``) so results stream out
    one chunk at a time instead of accumulating in the pool's internal
    queue.

    ``on_progress`` is called with ``(processed_count_one_indexed, item)``
    after each item; use it to drive progress logging or memory tracing.
    Pass ``num_workers=1`` to fall back to the single-process path (useful
    when ``initializer`` cannot be pickled or for unit tests).
    """
    filtered = 0

    if num_workers <= 1:
        for i, row in enumerate(rows):
            datum = render_fn(row)
            if datum is None:
                filtered += 1
            else:
                store.append(datum)
            if on_progress is not None:
                on_progress(i + 1, datum)
        return filtered

    spawn_ctx = multiprocessing.get_context("spawn")
    with spawn_ctx.Pool(
        processes=num_workers,
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        for i, datum in enumerate(pool.imap(render_fn, rows, chunksize=chunksize)):
            if datum is None:
                filtered += 1
            else:
                store.append(datum)
            if on_progress is not None:
                on_progress(i + 1, datum)
    return filtered
