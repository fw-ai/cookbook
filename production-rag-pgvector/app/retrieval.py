"""pgvector retrieval logic using asyncpg."""

import json

import asyncpg

from app.models import Citation


CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id        SERIAL PRIMARY KEY,
    doc_id    TEXT UNIQUE NOT NULL,
    title     TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding  vector(768) NOT NULL,
    metadata   JSONB DEFAULT '{}'
)
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
"""

UPSERT_DOCUMENT = """
INSERT INTO documents (doc_id, title, chunk_text, embedding, metadata)
VALUES ($1, $2, $3, $4::vector, $5)
ON CONFLICT (doc_id) DO UPDATE
    SET title      = EXCLUDED.title,
        chunk_text = EXCLUDED.chunk_text,
        embedding  = EXCLUDED.embedding,
        metadata   = EXCLUDED.metadata
"""

RETRIEVE_SIMILAR = """
SELECT doc_id,
       title,
       chunk_text,
       1 - (embedding <=> $1::vector) AS similarity
FROM documents
ORDER BY embedding <=> $1::vector
LIMIT $2
"""


async def init_db(pool: asyncpg.Pool) -> None:
    """Create the pgvector extension, documents table, and index."""
    async with pool.acquire() as conn:
        await conn.execute(CREATE_EXTENSION)
        await conn.execute(CREATE_TABLE)
        await conn.execute(CREATE_INDEX)


async def retrieve(
    pool: asyncpg.Pool,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[Citation]:
    """Return the top-k documents by cosine similarity to *query_embedding*."""
    vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
    async with pool.acquire() as conn:
        rows = await conn.fetch(RETRIEVE_SIMILAR, vec_str, top_k)
    return [
        Citation(
            doc_id=row["doc_id"],
            title=row["title"],
            chunk_text=row["chunk_text"],
            similarity=min(1.0, max(0.0, float(row["similarity"]))),
        )
        for row in rows
    ]


async def insert_document(
    pool: asyncpg.Pool,
    doc_id: str,
    title: str,
    chunk_text: str,
    embedding: list[float],
    metadata: dict | None = None,
) -> None:
    """Upsert a document with its embedding."""
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
    meta_json = json.dumps(metadata or {})
    async with pool.acquire() as conn:
        await conn.execute(
            UPSERT_DOCUMENT, doc_id, title, chunk_text, vec_str, meta_json
        )
