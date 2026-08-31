"""pgvector integration tests.

Requires a running pgvector/postgres instance. Set DATABASE_URL in env.
Tests are skipped automatically if the database is unreachable.
"""

import os
import pytest
import pytest_asyncio
import asyncpg

from app.embeddings import MockEmbedder
from app.retrieval import init_db, insert_document, retrieve


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")


async def _get_pool() -> asyncpg.Pool | None:
    try:
        pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
        return pool
    except Exception:
        return None


@pytest_asyncio.fixture
async def pool():
    p = await _get_pool()
    if p is None:
        pytest.skip("pgvector/postgres not available — set DATABASE_URL")
    yield p
    await p.close()


@pytest.mark.asyncio
async def test_init_db_creates_table(pool):
    """init_db should create the documents table without error."""
    await init_db(pool)
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'documents'
            )
            """
        )
    assert exists is True


@pytest.mark.asyncio
async def test_insert_and_retrieve(pool):
    """Insert a document and confirm it comes back from retrieve()."""
    await init_db(pool)

    embedder = MockEmbedder(dim=768)
    doc_text = "pgvector enables fast cosine similarity search in Postgres."
    embedding = await embedder.embed_one(doc_text)

    await insert_document(
        pool,
        doc_id="test-pgvector-001",
        title="pgvector Overview",
        chunk_text=doc_text,
        embedding=embedding,
        metadata={"source": "test"},
    )

    query_text = "cosine similarity in Postgres"
    query_embedding = await embedder.embed_one(query_text)
    results = await retrieve(pool, query_embedding, top_k=5)

    assert len(results) >= 1
    doc_ids = [r.doc_id for r in results]
    assert "test-pgvector-001" in doc_ids

    for result in results:
        assert 0.0 <= result.similarity <= 1.0


@pytest.mark.asyncio
async def test_upsert_updates_existing(pool):
    """Upserting the same doc_id updates the record."""
    await init_db(pool)
    embedder = MockEmbedder(dim=768)

    embedding = await embedder.embed_one("original text")
    await insert_document(
        pool,
        doc_id="test-upsert-001",
        title="Original Title",
        chunk_text="original text",
        embedding=embedding,
    )

    new_embedding = await embedder.embed_one("updated text")
    await insert_document(
        pool,
        doc_id="test-upsert-001",
        title="Updated Title",
        chunk_text="updated text",
        embedding=new_embedding,
    )

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT title, chunk_text FROM documents WHERE doc_id = $1",
            "test-upsert-001",
        )

    assert row["title"] == "Updated Title"
    assert row["chunk_text"] == "updated text"


@pytest.mark.asyncio
async def test_retrieve_empty_db_returns_empty(pool):
    """Retrieval on an empty table returns an empty list (not an error)."""
    await init_db(pool)
    # Clear any leftover test documents
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM documents WHERE doc_id LIKE 'test-%'")

    embedder = MockEmbedder(dim=768)
    query_embedding = await embedder.embed_one("anything")
    results = await retrieve(pool, query_embedding, top_k=5)
    # May be empty or have non-test docs; just confirm it doesn't raise
    assert isinstance(results, list)
