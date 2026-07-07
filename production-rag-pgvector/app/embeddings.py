"""Fireworks AI embedding client with async batch support."""

import asyncio
import math

import httpx


FIREWORKS_EMBEDDINGS_URL = "https://api.fireworks.ai/inference/v1/embeddings"


class FireworksAuthError(Exception):
    """Raised when the Fireworks API key is missing or rejected."""


class FireworksEmbedder:
    """Async Fireworks embedding client.

    Args:
        api_key: Fireworks API key. Raises FireworksAuthError if empty.
        model: Embedding model name.
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nomic-ai/nomic-embed-text-v1.5",
        timeout: float = 30.0,
    ) -> None:
        if not api_key or api_key.strip() == "":
            raise FireworksAuthError(
                "FIREWORKS_API_KEY is not set. "
                "Export it in your environment or .env file."
            )
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def _embed_single_batch(
        self,
        client: httpx.AsyncClient,
        texts: list[str],
    ) -> list[list[float]]:
        response = await client.post(
            FIREWORKS_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._model, "input": texts},
            timeout=self._timeout,
        )
        if response.status_code == 401:
            raise FireworksAuthError(
                f"Fireworks API rejected the key (HTTP 401): {response.text}"
            )
        response.raise_for_status()
        data = response.json()
        # Sort by index to guarantee order matches input
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed a list of texts, splitting into batches of *batch_size*.

        Returns embeddings in the same order as *texts*.
        """
        if not texts:
            return []

        n_batches = math.ceil(len(texts) / batch_size)
        batches = [
            texts[i * batch_size : (i + 1) * batch_size]
            for i in range(n_batches)
        ]

        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(
                *[self._embed_single_batch(client, b) for b in batches]
            )

        # Flatten
        embeddings: list[list[float]] = []
        for batch_result in results:
            embeddings.extend(batch_result)
        return embeddings

    async def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for a single text."""
        results = await self.embed_batch([text])
        return results[0]


class MockEmbedder:
    """Deterministic unit-vector embedder for testing.

    Produces a repeatable embedding for each text based on the hash of
    the input, so identical strings always return the same vector.
    """

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim

    def _make_vector(self, text: str) -> list[float]:
        seed = hash(text) % (2**31)
        # Simple LCG for determinism without numpy
        state = seed
        values: list[float] = []
        for _ in range(self._dim):
            state = (state * 1664525 + 1013904223) % (2**32)
            values.append((state / (2**32)) * 2 - 1)
        # Normalise to unit length
        norm = math.sqrt(sum(v * v for v in values))
        if norm > 0:
            values = [v / norm for v in values]
        return values

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        return [self._make_vector(t) for t in texts]

    async def embed_one(self, text: str) -> list[float]:
        return self._make_vector(text)
