"""Small exact index for guardrail examples, without native inference or Annoy."""

import math

from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.embeddings.providers.google import GoogleEmbeddingModel
from nemoguardrails.rails.llm.config import EmbeddingsCacheConfig
from app.config import settings


class ApiEmbeddingsIndex(EmbeddingsIndex):
    def __init__(self):
        self._model = GoogleEmbeddingModel(
            "gemini-embedding-001",
            api_key=settings.GEMINI_API_KEY,
            http_options={"timeout": 30000},
        )
        self._items = []
        self._vectors = []
        self._size = 0

    @property
    def embedding_size(self):
        return self._size

    @property
    def cache_config(self):
        return EmbeddingsCacheConfig(enabled=False)

    @staticmethod
    def _normalize(vector):
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("Invalid guardrail embedding")
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            raise ValueError("Zero-length guardrail embedding")
        return [value / norm for value in vector]

    async def _get_embeddings(self, texts):
        return await self._model.encode_async(texts)

    async def add_item(self, item):
        await self.add_items([item])

    async def add_items(self, items):
        if not items:
            return
        vectors = await self._get_embeddings([item.text for item in items])
        if len(vectors) != len(items):
            raise ValueError("Guardrail embedding count mismatch")
        normalized = [self._normalize(vector) for vector in vectors]
        size = self._size or len(normalized[0])
        if any(len(vector) != size for vector in normalized):
            raise ValueError("Guardrail embedding dimension mismatch")
        self._size = size
        self._items.extend(items)
        self._vectors.extend(normalized)

    async def build(self):
        # Exact search needs no compiled index for this small example set.
        pass

    async def search(self, text, max_results=20, threshold=None):
        if not self._items or max_results <= 0:
            return []
        query = self._normalize((await self._get_embeddings([text]))[0])
        if len(query) != self._size:
            raise ValueError("Guardrail query embedding dimension mismatch")
        ranked = []
        for item, vector in zip(self._items, self._vectors):
            cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(query, vector))))
            # Match NeMo 0.22's angular-distance threshold convention.
            score = 1 - math.sqrt(max(0.0, 2 - 2 * cosine)) / 2
            if threshold is None or threshold == float("inf") or score >= threshold:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked[:max_results]]
