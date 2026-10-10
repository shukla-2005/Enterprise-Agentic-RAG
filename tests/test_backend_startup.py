import unittest
from threading import Event
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app import main


class BackendStartupTests(unittest.TestCase):
    def test_server_responds_while_initializing_and_blocks_queries(self):
        release = Event()
        finished = Event()
        original_initialize = main._initialize_backend

        def initialize(application):
            try:
                original_initialize(application)
            finally:
                finished.set()

        with patch.object(main, "initialize_backend", side_effect=lambda: release.wait(5)), \
             patch.object(main, "_initialize_backend", side_effect=initialize), \
             patch.object(main, "guard") as guard:
            with TestClient(main.app) as client:
                try:
                    self.assertEqual(client.get("/").status_code, 200)
                    health = client.get("/health")
                    self.assertEqual(health.status_code, 503)
                    self.assertEqual(health.json(), {"status": "starting"})
                    self.assertEqual(client.post("/query", json={"q": "hello"}).status_code, 503)
                    self.assertEqual(client.get("/graph").status_code, 503)
                    guard.assert_not_called()
                finally:
                    release.set()
                    self.assertTrue(finished.wait(5))
                self.assertEqual(client.get("/health").status_code, 200)

    def test_initialization_failure_is_reported_without_exposing_exception(self):
        with patch.object(main, "initialize_backend", side_effect=RuntimeError("private details")), \
             patch.object(main.logging, "exception") as log:
            # Run synchronously to assert a completed failure deterministically.
            main._initialize_backend(main.app)
            client = TestClient(main.app)
            self.assertEqual(client.get("/health").json(), {"status": "error"})
            response = client.post("/query", json={"q": "hello"})
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("private details", response.text)
            log.assert_called()

    def test_ready_backend_preserves_guardrail_response(self):
        main.app.state.backend_status = "ready"
        with patch.object(main, "guard", return_value=(True, "Hello!")), \
             patch.object(main, "rag_agent", Mock()) as agent:
            response = TestClient(main.app).post("/query", json={"q": "hello"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["answer"], "Hello!")
            agent.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
