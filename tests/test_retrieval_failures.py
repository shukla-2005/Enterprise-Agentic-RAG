import unittest
from unittest.mock import patch

from app.services.retrieval import embedding, qdrant_service


class RetrievalFailureTests(unittest.TestCase):
    def test_embedding_failure_does_not_select_another_model(self):
        with patch.object(embedding, "_active_model", None), \
             patch.object(embedding, "_model_type", None), \
             patch.object(embedding, "GoogleGenerativeAIEmbeddings") as factory:
            factory.return_value.embed_query.side_effect = RuntimeError("API unavailable")
            with self.assertRaisesRegex(RuntimeError, "Gemini embeddings unavailable"):
                embedding.embed_query("Kubernetes")
            self.assertIsNone(embedding._active_model)

    def test_retrieval_failure_is_not_an_empty_success(self):
        with patch.object(qdrant_service, "embed_query", side_effect=RuntimeError("API unavailable")), \
             patch.object(qdrant_service, "client") as client:
            with self.assertRaisesRegex(RuntimeError, "Knowledge retrieval is unavailable"):
                qdrant_service.search_enterprise_knowledge("Kubernetes")
            client.query_points.assert_not_called()
