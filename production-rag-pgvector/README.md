# Production RAG with pgvector + Fireworks AI

A production-shaped retrieval-augmented generation (RAG) service using
[Fireworks AI](https://fireworks.ai) embeddings and
[pgvector](https://github.com/pgvector/pgvector) for fast cosine similarity
search in Postgres. The service exposes a FastAPI HTTP API, handles async
batch embedding, and returns structured citations alongside each answer.

## Directory Structure

```
production-rag-pgvector/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── app/
│   ├── config.py        # Pydantic-settings configuration
│   ├── embeddings.py    # Fireworks embedding client (async batch) + MockEmbedder
│   ├── main.py          # FastAPI app with lifespan, /health, /query, /ingest
│   ├── models.py        # Pydantic v2 request/response models
│   └── retrieval.py     # pgvector schema init + cosine retrieval (asyncpg)
├── scripts/
│   └── ingest.py        # CLI: embed + bulk-insert documents from JSONL
├── tests/
│   ├── test_models.py   # Pydantic model validation (no external services)
│   └── test_retrieval.py # pgvector integration tests (skips if DB absent)
└── cost_comparison.md   # Fireworks vs OpenAI pricing breakdown
```

## Architecture

```
Client
  │
  ▼ POST /query
FastAPI (app/main.py)
  │
  ├─► FireworksEmbedder.embed_one()
  │       └── POST api.fireworks.ai/inference/v1/embeddings
  │
  ├─► retrieve(pool, query_embedding, top_k)
  │       └── asyncpg → SELECT ... ORDER BY embedding <=> $1::vector LIMIT $2
  │               (pgvector IVFFlat cosine index)
  │
  └─► POST api.fireworks.ai/inference/v1/chat/completions
          (context = top-k chunks, model = llama-v3p1-70b-instruct)
  │
  ▼ RAGResponse(answer, citations[], model, tokens_used)
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- Python 3.11+
- [Fireworks AI API key](https://fireworks.ai)

## Quick Start

**1. Clone and enter the directory**

```bash
git clone https://github.com/fw-ai/cookbook.git
cd cookbook/production-rag-pgvector
```

**2. Set environment variables**

```bash
cp .env.example .env
# Edit .env — set FIREWORKS_API_KEY
```

**3. Start the stack**

```bash
docker-compose up --build
```

Postgres + pgvector starts first (health-checked), then the app starts on port 8000.

**4. Ingest a document**

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "fireworks-intro",
    "title": "Fireworks AI Overview",
    "text": "Fireworks AI provides fast, cost-efficient inference for open-source models including Llama 3.1, Mixtral, and embedding models like nomic-embed-text."
  }'
```

**5. Query**

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What models does Fireworks support?", "top_k": 3}' | python3 -m json.tool
```

**6. Health check**

```bash
curl -s http://localhost:8000/health
# {"status":"ok","db":"connected"}
```

## Configuration

All settings are loaded from environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `FIREWORKS_API_KEY` | _(required)_ | Your Fireworks API key |
| `FIREWORKS_EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model |
| `FIREWORKS_CHAT_MODEL` | `accounts/fireworks/models/llama-v3p1-70b-instruct` | Chat model |
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag` | Postgres connection string |
| `EMBEDDING_DIM` | `768` | Vector dimension |
| `RETRIEVAL_TOP_K` | `5` | Default number of results |

## API Reference

### `GET /health`
Returns service and database status.

```json
{"status": "ok", "db": "connected"}
```

### `POST /query`
Run a RAG query against the document store.

**Request body:**
```json
{
  "query": "string (1–2000 chars)",
  "top_k": 5,
  "include_citations": true
}
```

**Response:**
```json
{
  "answer": "string",
  "citations": [
    {"doc_id": "...", "title": "...", "chunk_text": "...", "similarity": 0.87}
  ],
  "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
  "tokens_used": 312
}
```

### `POST /ingest`
Embed and store a document (upserts on `doc_id`).

**Request body:**
```json
{
  "doc_id": "unique-id",
  "title": "Document Title",
  "text": "Document content...",
  "metadata": {"source": "web"}
}
```

## Bulk Ingestion

Use the CLI script to embed and insert documents from a JSONL file:

```bash
# Each line: {"doc_id": "...", "title": "...", "text": "..."}
python scripts/ingest.py --file docs.jsonl --batch-size 32
```

## Testing

```bash
# Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pydantic model tests — no external services required
pytest tests/test_models.py -v

# pgvector integration tests — requires docker-compose up postgres
DATABASE_URL=postgresql://rag:rag@localhost:5432/rag \
FIREWORKS_API_KEY=fake \
pytest tests/test_retrieval.py -v
```

The retrieval tests use `MockEmbedder` (deterministic unit vectors) so no
Fireworks API key is needed for the test suite.

## Cost

See [cost_comparison.md](cost_comparison.md) for a breakdown of Fireworks AI
vs OpenAI pricing for embeddings, generation, and vector storage.
