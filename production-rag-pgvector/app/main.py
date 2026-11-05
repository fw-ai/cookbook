"""FastAPI RAG service: Fireworks AI embeddings + pgvector retrieval."""

from contextlib import asynccontextmanager

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException, Request

from app.config import settings
from app.embeddings import FireworksAuthError, FireworksEmbedder
from app.models import (
    HealthResponse,
    IngestRequest,
    QueryRequest,
    RAGResponse,
)
from app.retrieval import init_db, insert_document, retrieve


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        pool = await asyncpg.create_pool(settings.database_url, min_size=2, max_size=10)
        await init_db(pool)
        app.state.pool = pool
        app.state.db_ok = True
    except Exception as exc:
        app.state.pool = None
        app.state.db_ok = False
        app.state.db_error = str(exc)

    app.state.embedder = FireworksEmbedder(
        api_key=settings.fireworks_api_key,
        model=settings.fireworks_embedding_model,
    )
    yield

    # Shutdown
    if app.state.pool:
        await app.state.pool.close()


app = FastAPI(
    title="Production RAG with pgvector + Fireworks AI",
    description=(
        "FastAPI service for retrieval-augmented generation using "
        "Fireworks AI embeddings and pgvector for similarity search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    db_status = "connected" if request.app.state.db_ok else "unavailable"
    return HealthResponse(status="ok", db=db_status)


@app.post("/query", response_model=RAGResponse)
async def query(body: QueryRequest, request: Request) -> RAGResponse:
    if not request.app.state.db_ok:
        raise HTTPException(status_code=503, detail="Database unavailable")

    embedder: FireworksEmbedder = request.app.state.embedder
    pool: asyncpg.Pool = request.app.state.pool

    # Embed the query
    try:
        query_embedding = await embedder.embed_one(body.query)
    except FireworksAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    # Retrieve relevant documents
    top_k = body.top_k
    citations = await retrieve(pool, query_embedding, top_k)

    # Build prompt with retrieved context
    context_blocks = "\n\n".join(
        f"[{c.doc_id}] {c.title}\n{c.chunk_text}" for c in citations
    )
    system_prompt = (
        "You are a helpful assistant. Answer the question using only the "
        "provided context. Cite sources by their doc_id in brackets."
    )
    user_prompt = f"Context:\n{context_blocks}\n\nQuestion: {body.query}"

    # Call Fireworks chat completion
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.fireworks_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.fireworks_chat_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.1,
                },
            )
        if resp.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="Fireworks API key rejected by chat endpoint.",
            )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502, detail=f"Fireworks chat error: {exc}"
        ) from exc

    data = resp.json()
    answer = data["choices"][0]["message"]["content"]
    tokens_used = data.get("usage", {}).get("total_tokens", 0)

    return RAGResponse(
        answer=answer,
        citations=citations if body.include_citations else [],
        model=settings.fireworks_chat_model,
        tokens_used=tokens_used,
    )


@app.post("/ingest", status_code=201)
async def ingest(body: IngestRequest, request: Request) -> dict:
    """Embed and store a document. Upserts on doc_id."""
    if not request.app.state.db_ok:
        raise HTTPException(status_code=503, detail="Database unavailable")

    embedder: FireworksEmbedder = request.app.state.embedder
    pool: asyncpg.Pool = request.app.state.pool

    try:
        embedding = await embedder.embed_one(body.text)
    except FireworksAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    await insert_document(
        pool,
        doc_id=body.doc_id,
        title=body.title,
        chunk_text=body.text,
        embedding=embedding,
        metadata=body.metadata,
    )
    return {"doc_id": body.doc_id, "status": "ingested"}
