"""Pydantic model validation tests — no external services required."""

import pytest
from pydantic import ValidationError

from app.models import Citation, IngestRequest, QueryRequest, RAGResponse


def test_citation_similarity_valid():
    c = Citation(
        doc_id="doc-1", title="Test Doc", chunk_text="Some text.", similarity=0.85
    )
    assert c.similarity == 0.85
    assert c.doc_id == "doc-1"


def test_citation_similarity_boundaries():
    low = Citation(doc_id="x", title="T", chunk_text="t", similarity=0.0)
    high = Citation(doc_id="x", title="T", chunk_text="t", similarity=1.0)
    assert low.similarity == 0.0
    assert high.similarity == 1.0


def test_citation_similarity_too_high():
    with pytest.raises(ValidationError):
        Citation(doc_id="x", title="T", chunk_text="t", similarity=1.5)


def test_citation_similarity_too_low():
    with pytest.raises(ValidationError):
        Citation(doc_id="x", title="T", chunk_text="t", similarity=-0.1)


def test_query_request_defaults():
    q = QueryRequest(query="What is RAG?")
    assert q.top_k == 5
    assert q.include_citations is True


def test_query_request_custom():
    q = QueryRequest(query="Explain pgvector", top_k=10, include_citations=False)
    assert q.top_k == 10
    assert q.include_citations is False


def test_query_request_empty_fails():
    with pytest.raises(ValidationError):
        QueryRequest(query="")


def test_query_request_too_long_fails():
    with pytest.raises(ValidationError):
        QueryRequest(query="x" * 2001)


def test_query_request_top_k_bounds():
    with pytest.raises(ValidationError):
        QueryRequest(query="hello", top_k=0)
    with pytest.raises(ValidationError):
        QueryRequest(query="hello", top_k=21)


def test_ingest_request_valid():
    r = IngestRequest(doc_id="doc-1", title="My Doc", text="Content here.")
    assert r.metadata == {}


def test_ingest_request_with_metadata():
    r = IngestRequest(
        doc_id="doc-2",
        title="Another Doc",
        text="More content.",
        metadata={"source": "web", "author": "alice"},
    )
    assert r.metadata["source"] == "web"


def test_rag_response_structure():
    citations = [
        Citation(doc_id="d1", title="Title", chunk_text="chunk", similarity=0.9)
    ]
    r = RAGResponse(
        answer="The answer is 42.",
        citations=citations,
        model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        tokens_used=128,
    )
    assert len(r.citations) == 1
    assert r.tokens_used == 128
