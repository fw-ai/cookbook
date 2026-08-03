# Voyage AI on Fireworks

Two-stage retrieval with Voyage AI embeddings and reranking, served from
Fireworks dedicated deployments and backed by MongoDB vector search.

## Notebook

[`voyage_two_stage_retrieval.ipynb`](./voyage_two_stage_retrieval.ipynb) embeds a
small corpus with a Voyage embedding model, stores the vectors in MongoDB, runs
a `$vectorSearch` recall stage, and reranks the candidates with a Voyage
reranker.

[Open in Colab](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/partners/voyage-ai/voyage_two_stage_retrieval.ipynb)

## Prerequisites

- A [Fireworks API key](https://docs.fireworks.ai/api-reference/create-api-key).
- Two [dedicated deployments](https://docs.fireworks.ai/api-reference/create-deployment):
  one Voyage embedding model and one Voyage reranker. Voyage models are not
  available on serverless.
- A MongoDB deployment with Vector Search, either
  [Atlas](https://www.mongodb.com/cloud/atlas/register) or
  [Community Edition](https://www.mongodb.com/docs/vector-search/tutorials/quick-start/?deployment-type=self&embedding=byo&interface=driver&language=python).
  On Atlas, add an IP access list entry for your client or the connection will
  hang.

## Notes

Dedicated deployments are addressed by deployment path rather than model name:

```
accounts/{ACCOUNT_ID}/deployments/{DEPLOYMENT_ID}
```

Voyage models are trained for asymmetric retrieval. Pass
`input_type="document"` when embedding a corpus and `input_type="query"` when
embedding a search query; the model prepends task-specific instructions
internally.

Deployments scale to zero when idle, so the first request after a quiet period
can return `503` while the deployment cold-starts. The notebook retries with
backoff.
