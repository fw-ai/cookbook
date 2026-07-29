"""Retrieval evaluation for embedding models served on Fireworks.

Embeds a corpus and a query set through ``/v1/embeddings``, runs exact
nearest-neighbour search, and scores the run with ``pytrec_eval`` -- the same
library MTEB uses, so the numbers are comparable to published benchmarks.

Qwen3-Embedding is *asymmetric*: queries carry an instruction prefix, documents
never do. ``training/recipes/embedding_loop.py`` applies that prefix at training
time (see ``_build_batch_datums``)::

    query_texts = _QWEN3_INSTRUCTION_TEMPLATE.format(query_instruction) + q
    doc_texts   = d                       # no prefix

Evaluation has to match, or it measures a different input distribution than the
model trained on. This module reads the recipe's own template so the two cannot
drift apart.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytrec_eval
import requests

_FALLBACK_TEMPLATE = "Instruct: {}\nQuery: "


def instruction_template() -> str:
    """The query-side template, read from the recipe *at call time*.

    Resolving this dynamically rather than binding it at import means a caller can
    monkeypatch ``embedding_loop._QWEN3_INSTRUCTION_TEMPLATE`` once and have both
    training and evaluation pick up the same value -- they cannot drift apart.
    """
    try:
        import training.recipes.embedding_loop as _el

        return _el._QWEN3_INSTRUCTION_TEMPLATE
    except Exception:  # pragma: no cover
        return _FALLBACK_TEMPLATE

# pytrec_eval measure names -> the labels we print.
METRIC_KEYS = {
    "ndcg_cut_10": "nDCG@10",
    "ndcg_cut_100": "nDCG@100",
    "recall_10": "Recall@10",
    "recall_100": "Recall@100",
    "recip_rank": "MRR",
    "map_cut_100": "MAP@100",
}


def format_query(text: str, instruction: str) -> str:
    """Apply the Qwen3 query-side instruction prefix. Documents get none."""
    return instruction_template().format(instruction) + text


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingClient:
    """Batching + retry wrapper over the Fireworks embeddings endpoint."""

    model: str
    api_key: str = field(default_factory=lambda: os.environ["FIREWORKS_API_KEY"])
    base_url: str = field(
        default_factory=lambda: os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    )
    batch_size: int = 32
    max_workers: int = 8
    max_retries: int = 6
    timeout: int = 300

    def __post_init__(self):
        self._url = f"{self.base_url.rstrip('/')}/inference/v1/embeddings"
        self._session = requests.Session()

    def _post(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        delay, last = 2.0, None
        for _ in range(self.max_retries):
            try:
                r = self._session.post(
                    self._url,
                    headers=headers,
                    json={"model": self.model, "input": texts},
                    timeout=self.timeout,
                )
                if r.status_code == 200:
                    data = r.json()["data"]
                    return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
                # 429 and 5xx are transient; other 4xx will not fix themselves.
                if r.status_code != 429 and r.status_code < 500:
                    raise RuntimeError(f"embeddings {r.status_code}: {r.text[:300]}")
                last = f"{r.status_code}: {r.text[:200]}"
            except requests.RequestException as e:
                last = str(e)
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise RuntimeError(f"embeddings failed after {self.max_retries} retries: {last}")

    def embed(self, texts: list[str], desc: str = "") -> np.ndarray:
        """Embed ``texts`` in order -> L2-normalized ``[N, D]`` float32 array."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        chunks = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        out: list[list[list[float]] | None] = [None] * len(chunks)
        t0, done = time.monotonic(), 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._post, c): i for i, c in enumerate(chunks)}
            for fut in as_completed(futures):
                out[futures[fut]] = fut.result()
                done += 1
                if desc and (done % 20 == 0 or done == len(chunks)):
                    frac = done / len(chunks)
                    el = time.monotonic() - t0
                    print(
                        f"\r  {desc}: {done}/{len(chunks)} batches ({frac:5.1%})"
                        f" eta {el / frac - el:5.0f}s",
                        end="",
                        flush=True,
                    )
        if desc:
            print()

        vecs = np.asarray([v for chunk in out for v in chunk], dtype=np.float32)
        # The endpoint already returns normalized vectors; re-normalizing is cheap
        # and makes "dot product == cosine" hold unconditionally below.
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------
# Retrieval + scoring
# ---------------------------------------------------------------------------


def search(
    query_vecs: np.ndarray,
    doc_vecs: np.ndarray,
    query_ids: list[str],
    doc_ids: list[str],
    top_k: int = 100,
    block: int = 1024,
) -> dict[str, dict[str, float]]:
    """Exact cosine top-k -> a pytrec_eval run dict.

    Blocked over queries so a large corpus never materializes a full
    ``n_queries x n_docs`` score matrix.
    """
    k = min(top_k, len(doc_ids))
    doc_ids_arr = np.asarray(doc_ids)
    run: dict[str, dict[str, float]] = {}

    for start in range(0, len(query_ids), block):
        sims = query_vecs[start : start + block] @ doc_vecs.T
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(sims.shape[0])[:, None]
        top = sims[rows, idx]
        order = np.argsort(-top, axis=1)
        idx, top = idx[rows, order], top[rows, order]
        for i, qid in enumerate(query_ids[start : start + block]):
            run[qid] = {str(d): float(s) for d, s in zip(doc_ids_arr[idx[i]], top[i])}
    return run


def score(run: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """Mean metrics over judged queries."""
    judged = {q: d for q, d in run.items() if q in qrels}
    per_query = pytrec_eval.RelevanceEvaluator(qrels, set(METRIC_KEYS)).evaluate(judged)
    if not per_query:
        raise RuntimeError("No overlap between run and qrels -- check id types (str vs int).")
    out = {label: float(np.mean([m[raw] for m in per_query.values()]))
           for raw, label in METRIC_KEYS.items()}
    out["#queries"] = len(per_query)
    return out


def evaluate_model(
    model: str,
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    instruction: str,
    *,
    batch_size: int = 32,
    max_workers: int = 8,
    top_k: int = 100,
    label: str = "",
    cache_dir: str | Path | None = None,
) -> dict[str, float]:
    """Embed corpus + queries, retrieve, score.

    Corpus vectors are cached on disk when ``cache_dir`` is set: the corpus is the
    expensive side and gets re-embedded once per model.
    """
    tag = label or model.rsplit("/", 1)[-1]
    client = EmbeddingClient(model=model, batch_size=batch_size, max_workers=max_workers)
    doc_ids = list(corpus)

    cache_path = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"docvecs_{model.replace('/', '_')}_{len(doc_ids)}.npy"

    if cache_path and cache_path.exists():
        doc_vecs = np.load(cache_path)
        print(f"  [{tag}] loaded {len(doc_ids):,} cached doc vectors")
    else:
        doc_vecs = client.embed([corpus[d] for d in doc_ids], desc=f"[{tag}] docs")
        if cache_path:
            np.save(cache_path, doc_vecs)

    query_ids = list(queries)
    query_vecs = client.embed(
        [format_query(queries[q], instruction) for q in query_ids], desc=f"[{tag}] queries"
    )
    return score(search(query_vecs, doc_vecs, query_ids, doc_ids, top_k=top_k), qrels)


# ---------------------------------------------------------------------------
# Dataset IO -- the on-disk contract shared by every experiment
# ---------------------------------------------------------------------------


def write_jsonl(path: str | Path, rows) -> int:
    with Path(path).open("w") as f:
        n = 0
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_qrels(path: str | Path, qrels: dict[str, dict[str, int]]) -> int:
    """Standard TREC 4-column qrels: ``qid 0 docno rel``."""
    with Path(path).open("w") as f:
        n = 0
        for qid, docs in qrels.items():
            for docno, rel in docs.items():
                f.write(f"{qid}\t0\t{docno}\t{rel}\n")
                n += 1
    return n


def read_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with Path(path).open() as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                qrels.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    return qrels


def load_experiment(data_dir: str | Path):
    """Read the four files every prepare step emits -> (corpus, queries, qrels)."""
    d = Path(data_dir)
    corpus = {r["id"]: r["text"] for r in read_jsonl(d / "corpus.jsonl")}
    queries = {r["id"]: r["text"] for r in read_jsonl(d / "queries.jsonl")}
    return corpus, queries, read_qrels(d / "qrels.tsv")


def comparison_table(base: dict[str, float], tuned: dict[str, float]) -> str:
    """Markdown base-vs-fine-tuned table with relative gains."""
    rows = ["| Metric | Base | Fine-tuned | Relative gain |", "|---|---|---|---|"]
    for label in METRIC_KEYS.values():
        b, t = base.get(label), tuned.get(label)
        if b is not None and t is not None:
            rows.append(f"| {label} | {b:.3f} | {t:.3f} | {(t - b) / b:+.0%} |" if b
                        else f"| {label} | {b:.3f} | {t:.3f} | n/a |")
    return "\n".join(rows)
