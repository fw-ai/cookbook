"""Prepare the BIRD text-to-SQL data for agentic RL.

Reads `dev.json` + `dev_databases/` (you place them in this folder) and:

1. Builds a persistent Chroma evidence store from the BIRD `evidence` field,
   embedded with Fireworks embeddings (FIREWORKS_API_KEY only).
2. For a sampled subset of questions, executes the gold `SQL` to cache the
   gold result set (so the DB-less evaluator can score answer correctness).
3. Writes `text2sql_train.jsonl` / `text2sql_holdout.jsonl` as eval-protocol
   rows (system + user messages, gold cached in `input_metadata.dataset_info`).

Usage:
    python prepare_data.py --max-rows 80 --holdout 10
    python prepare_data.py --skip-chroma        # rebuild rows only
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from sql_reward import build_evaluation_row, split_train_holdout, write_jsonl

HERE = Path(__file__).resolve().parent


def _load_dev(dev_json: Path) -> List[Dict[str, Any]]:
    with open(dev_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _available_dbs(db_root: Path) -> List[str]:
    if not db_root.is_dir():
        return []
    return sorted(
        p.name for p in db_root.iterdir()
        if p.is_dir() and (p / f"{p.name}.sqlite").exists()
    )


def build_chroma(dev_data: List[Dict[str, Any]], chroma_dir: Path, collection: str) -> int:
    """Build/persist the Chroma evidence store (one doc per evidence string)."""
    import chromadb
    from chromadb.config import Settings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    from sql_tools import make_embeddings

    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
    try:
        client.delete_collection(collection)
    except Exception:  # noqa: BLE001
        pass
    store = Chroma(
        client=client,
        collection_name=collection,
        embedding_function=make_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )

    docs: List[Document] = []
    for item in dev_data:
        evidences = [e.strip() for e in (item.get("evidence") or "").split(";") if e.strip()]
        for ev in evidences:
            docs.append(Document(
                page_content=ev,
                metadata={"db_id": item["db_id"], "question": item["question"], "sql": item.get("SQL", "")},
            ))
    if docs:
        # Batch to keep embedding requests reasonable.
        for i in range(0, len(docs), 256):
            store.add_documents(docs[i:i + 256])
    return len(docs)


def _gold_result(db_root: Path, db_id: str, sql: str, cap: int) -> Any:
    path = db_root / db_id / f"{db_id}.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return [list(r) for r in cur.fetchall()[:cap]]
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-json", default=str(HERE / "dev.json"))
    ap.add_argument("--db-root", default=str(HERE / "dev_databases"))
    ap.add_argument("--chroma-dir", default=str(HERE / "chromadb_text2sql"))
    ap.add_argument("--collection", default="sql_examples")
    ap.add_argument("--max-rows", type=int, default=80, help="total questions to keep (train + holdout)")
    ap.add_argument("--holdout", type=int, default=10, help="number of holdout rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--result-cap", type=int, default=200, help="max gold rows cached per question")
    ap.add_argument("--skip-chroma", action="store_true", help="don't rebuild the evidence store")
    args = ap.parse_args()

    dev_json, db_root, chroma_dir = Path(args.dev_json), Path(args.db_root), Path(args.chroma_dir)
    assert dev_json.exists(), f"dev.json not found at {dev_json} (place the BIRD dev.json here)"
    assert db_root.is_dir(), f"dev_databases not found at {db_root} (place the BIRD databases here)"

    dev_data = _load_dev(dev_json)
    available_dbs = _available_dbs(db_root)
    assert available_dbs, f"no SQLite databases found under {db_root}"
    print(f"loaded {len(dev_data)} dev questions; {len(available_dbs)} databases available")

    if not args.skip_chroma:
        n_docs = build_chroma(dev_data, chroma_dir, args.collection)
        print(f"built Chroma evidence store: {n_docs} docs -> {chroma_dir}")

    # Keep only questions whose db is present locally; sample deterministically.
    usable = [d for d in dev_data if d["db_id"] in set(available_dbs)]
    import random
    random.Random(args.seed).shuffle(usable)

    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(usable):
        if len(rows) >= args.max_rows:
            break
        db_id, sql = item["db_id"], item.get("SQL", "")
        if not sql:
            continue
        try:
            gold = _gold_result(db_root, db_id, sql, args.result_cap)
        except Exception as e:  # noqa: BLE001 - skip questions whose gold SQL errors
            print(f"  skip {db_id} q{i}: gold SQL failed ({str(e)[:80]})")
            continue
        rid = f"q{item.get('question_id', i)}"
        ev = build_evaluation_row(
            row_id=rid,
            question=item["question"],
            db_id=db_id,
            gold_sql=sql,
            gold_result=gold,
            available_dbs=available_dbs,
        )
        d = json.loads(ev.model_dump_json())
        d["messages"] = [{k: v for k, v in m.items() if v is not None} for m in d.get("messages", [])]
        rows.append(d)

    print(f"built {len(rows)} eval rows with cached gold results")

    train, holdout = split_train_holdout(rows, holdout_ratio=args.holdout / max(len(rows), 1), seed=args.seed)
    # Respect the exact holdout count when possible.
    if len(rows) > args.holdout:
        holdout, train = rows[: args.holdout], rows[args.holdout:]

    write_jsonl(train, HERE / "text2sql_train.jsonl")
    write_jsonl(holdout, HERE / "text2sql_holdout.jsonl")
    print(f"wrote {len(train)} train -> text2sql_train.jsonl")
    print(f"wrote {len(holdout)} holdout -> text2sql_holdout.jsonl")


if __name__ == "__main__":
    main()
