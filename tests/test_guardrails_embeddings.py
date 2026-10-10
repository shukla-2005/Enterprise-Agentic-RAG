import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nemoguardrails.embeddings.providers import init_embedding_model
from nemoguardrails.embeddings.providers.fastembed import FastEmbedEmbeddingModel
from app.guardrails import rails


class GuardrailEmbeddingTests(unittest.TestCase):
    def test_config_routes_sync_and_async_embeddings_to_google(self):
        with patch.dict(os.environ), \
             patch.object(rails.settings, "GEMINI_API_KEY", "test-key"), \
             patch.object(rails, "ChatGroq"), \
             patch.object(rails, "LLMRails") as factory, \
             patch.object(rails, "_rails"):
            rails.initialize_rails()
            config = factory.call_args.args[0]

        self.assertEqual(len(config.models), 1)
        model = config.models[0]
        self.assertEqual((model.type, model.engine), ("embeddings", "google"))
        self.assertNotIn("test-key", str(config))
        with patch("google.genai.Client") as client, \
             patch.object(FastEmbedEmbeddingModel, "__init__", side_effect=AssertionError("Local inference forbidden")):
            client.return_value.models.embed_content.return_value = SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2])]
            )
            provider = init_embedding_model(model.model, model.engine, model.parameters)
            self.assertEqual(provider.encode(["hello"]), [[0.1, 0.2]])
            self.assertEqual(asyncio.run(provider.encode_async(["hello"])), [[0.1, 0.2]])
            client.assert_called_once_with(http_options={"timeout": 30000})
            client.return_value.models.embed_content.assert_called_with(
                model="gemini-embedding-001", contents=["hello"]
            )

    def test_missing_key_prevents_initialization(self):
        with patch.object(rails.settings, "GEMINI_API_KEY", None), \
             patch.object(rails, "LLMRails") as factory:
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                rails.initialize_rails()
            factory.assert_not_called()

    def test_uninitialized_guardrails_do_not_bypass_gate(self):
        with patch.object(rails, "_rails", None):
            with self.assertRaisesRegex(RuntimeError, "not initialized"):
                rails.guard("hello")
