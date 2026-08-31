"""CLI tool to embed documents and insert them into pgvector.

Usage:
    python scripts/ingest.py --file docs.jsonl

Each line of the JSONL file should be:
    {"doc_id": "...", "title": "...", "text": "..."}

Optional fields:
    {"metadata": {"source": "...", "author": "..."}}
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import asyncpg

# Allow running from repo root: python scripts/ingest.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.embeddings import FireworksEmbedder
from app.retrieval import init_db, insert_document


async def ingest_file(path: Path, batch_size: int = 32) -> None:
    lines = path.read_text().splitlines()
    docs = [json.loads(line) for line in lines if line.strip()]

    print(f"Loaded {len(docs)} documents from {path}")

    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=5)
    await init_db(pool)

    embedder = FireworksEmbedder(
        api_key=settings.fireworks_api_key,
        model=settings.fireworks_embedding_model,
    )

    texts = [d["text"] for d in docs]
    print(f"Embedding {len(texts)} texts (batch_size={batch_size})...")
    embeddings = await embedder.embed_batch(texts, batch_size=batch_size)

    for doc, embedding in zip(docs, embeddings):
        await insert_document(
            pool,
            doc_id=doc["doc_id"],
            title=doc["title"],
            chunk_text=doc["text"],
            embedding=embedding,
            metadata=doc.get("metadata", {}),
        )
        print(f"  Inserted: {doc['doc_id']}")

    await pool.close()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into pgvector")
    parser.add_argument("--file", required=True, help="Path to JSONL file")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(ingest_file(path, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
