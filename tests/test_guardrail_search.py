import unittest
from unittest.mock import AsyncMock, patch

from nemoguardrails.embeddings.index import IndexItem
from app.guardrails.search import ApiEmbeddingsIndex


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_cosine_order_threshold_and_metadata(self):
        with patch("app.guardrails.search.GoogleEmbeddingModel") as model:
            model.return_value.encode_async = AsyncMock(side_effect=[
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                [[1.0, 0.0]], [[1.0, 0.0]],
            ])
            index = ApiEmbeddingsIndex()
            items = [IndexItem("same", {"intent": "hello"}), IndexItem("other"), IndexItem("opposite")]
            await index.add_items(items)
            await index.build()
            self.assertEqual(await index.search("query", max_results=2), items[:2])
            self.assertEqual(await index.search("query", threshold=0.9), items[:1])
            self.assertEqual(index.embedding_size, 2)

    async def test_failed_batch_does_not_modify_index(self):
        with patch("app.guardrails.search.GoogleEmbeddingModel") as model:
            model.return_value.encode_async = AsyncMock(return_value=[[0.0, 0.0]])
            index = ApiEmbeddingsIndex()
            with self.assertRaises(ValueError):
                await index.add_items([IndexItem("bad")])
            self.assertEqual(await index.search("query"), [])
