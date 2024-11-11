# Optimizing Retrieval Augmented Generation (RAG) with MongoDB Atlas and Fireworks AI

By Fireworks.ai | March 21, 2024
- Link to blog post on Fireworks: https://fireworks.ai/blog/optimizing-rag-with-mongodb-atlas-and-fireworks
- Link to announcement on MongoDB: https://www.mongodb.com/blog/post/fireworks-ai-mongodb-fastest-ai-apps-with-best-models-powered-by-your-data


---
## Table of Contents
1. [What is RAG?](#what-is-rag)
2. [Why RAG?](#why-rag)
3. [RAG Architecture](#rag-architecture)
4. [Optimizing RAG Architecture](#optimizing-rag-architecture)
5. [Prerequisites](#prerequisites)
6. [Configuring Your Environment](#configuring-your-environment)
7. [Gathering Credentials](#gathering-credentials)
8. [Initializing Fireworks and MongoDB Clients](#initializing-fireworks-and-mongodb-clients)
9. [Using Fireworks with OSS Embedding Models](#using-fireworks-with-oss-embedding-models)
10. [Generating Embeddings](#generating-embeddings)
11. [Creating an Index on MongoDB Collection](#creating-an-index-on-mongodb-collection)
12. [Generating Personalized Recommendations with Fireworks](#generating-personalized-recommendations-with-fireworks)
13. [What's Next?](#whats-next)

## What is RAG?
Retrieval Augmented Generation (RAG) combines a retrieval component to fetch relevant information from a database or vector store and a generative component (LLM) to synthesize a coherent response to the user query.

## Why RAG?
- **Data Efficiency**: Dynamically pulls relevant data not seen during training, saving time and resources compared to fine-tuning.
- **Flexibility**: Enables dynamic updates to knowledge bases without regular retraining.

## RAG Architecture
A RAG Architecture involves:
- A Large Language Model (LLM) for generating responses.
- A vector store (e.g., MongoDB Atlas) for retrieving relevant data based on embeddings.
- Fireworks AI for creating embeddings and handling LLM inference.

## Optimizing RAG Architecture
Tips for optimizing RAG:
- **Cost Reduction**: Use smaller embeddings for lower storage costs.
- **Improved Throughput**: Implement batching for efficient processing.
- **Function Calling**: Use Fireworks' function-calling models for dynamic query handling.

## Prerequisites
- MongoDB Atlas Account
- Fireworks AI Account

## Configuring Your Environment
Install necessary packages:
```bash
!pip install -q pymongo fireworks-ai tqdm openai
```

## Gathering Credentials
Replace `"FIREWORKS_API_KEY"` and `"MONGODB_URI"` with your credentials.

## Initializing Fireworks and MongoDB Clients
```python
from pymongo.mongo_client import MongoClient
import openai

uri = "MONGODB_URI"
client = MongoClient(uri)

fw_client = openai.OpenAI(api_key="FIREWORKS_API_KEY", base_url="https://api.fireworks.ai/inference/v1")
```

## Using Fireworks with OSS Embedding Models
Generate embeddings using the Fireworks embedding API:
```python
def generate_embeddings(input_texts, model_api_string):
    return fw_client.embeddings.create(input=input_texts, model=model_api_string).data[0].embedding
```

## Generating Embeddings
Process movie data through the `generate_embeddings` function:
```python
embedding_model_string = 'nomic-ai/nomic-embed-text-v1.5'
sample_output = generate_embeddings(["This is a test."], embedding_model_string)
print(f"Embedding size is: {len(sample_output)}")
```

## Creating an Index on MongoDB Collection
Define an index structure for efficient search:
```json
{
 "fields": [
   {
     "type": "vector",
     "path": "embed",
     "numDimensions": 768,
     "similarity": "dotProduct"
   }
 ]
}
```

## Generating Personalized Recommendations with Fireworks
Example query for recommendations:
```python
query = "I like Christmas movies, any recommendations?"
query_emb = generate_embeddings([query], embedding_model_string)

results = collection.aggregate([
 {
   "$vectorSearch": {
     "queryVector": query_emb,
     "path": "embed",
     "limit": 10
   }
 }
])
```

## What's Next?
Explore guides for optimizing RAG architectures, reducing embedding size, and integrating function calling for dynamic query handling.