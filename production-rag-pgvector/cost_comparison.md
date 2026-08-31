# Cost Comparison: Fireworks AI vs OpenAI for RAG

_Prices as of July 2026. Subject to change — verify at [fireworks.ai/pricing](https://fireworks.ai/pricing) and [openai.com/pricing](https://platform.openai.com/docs/pricing)._

## Embeddings

| Provider | Model | Price per 1M tokens |
|---|---|---|
| **Fireworks AI** | `nomic-ai/nomic-embed-text-v1.5` | ~$0.008 |
| OpenAI | `text-embedding-3-small` | $0.020 |
| OpenAI | `text-embedding-3-large` | $0.130 |

Fireworks `nomic-embed-text-v1.5` is roughly **2.5x cheaper** than `text-embedding-3-small` at similar quality for retrieval tasks. The 768-dim output keeps pgvector index sizes manageable.

## Chat / Generation

| Provider | Model | Input per 1M | Output per 1M |
|---|---|---|---|
| **Fireworks AI** | `llama-v3p1-70b-instruct` | ~$0.90 | ~$0.90 |
| OpenAI | `gpt-4o` | $2.50 | $10.00 |
| OpenAI | `gpt-4o-mini` | $0.15 | $0.60 |

For high-throughput RAG workloads, Fireworks Llama 3.1 70B cuts generation costs by ~3–10x vs `gpt-4o`, with competitive quality on structured Q&A tasks.

## Vector Store

| Option | Cost |
|---|---|
| **pgvector (self-hosted)** | Compute only — no per-query or per-vector fees |
| Pinecone (Serverless) | ~$0.10/1M reads + storage |
| Weaviate Cloud | Cluster pricing, starts ~$25/month |

Self-hosting pgvector on the same Postgres instance your app already uses eliminates a managed-vector-DB line item entirely. The IVFFlat index in this example handles millions of vectors with sub-10ms retrieval at `lists=100`.

## Rough Full-Stack Cost Example

For 100K queries/month, each with 5 retrieved chunks (500 tokens context) + 300 token answer:

| Stack | Est. Monthly Cost |
|---|---|
| Fireworks embeddings + Llama 70B + pgvector | ~$130 |
| OpenAI embeddings + gpt-4o + Pinecone | ~$1,400+ |
