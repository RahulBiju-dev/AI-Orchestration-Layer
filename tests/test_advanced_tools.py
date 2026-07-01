import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.api_orchestrator import api_orchestrator
from tools.automated_routine_executor import automated_routine_executor
from tools.context_memory_optimizer import context_memory_optimizer
from tools.knowledge_graph_builder import knowledge_graph_builder
from tools.reasoning_chain_debugger import reasoning_chain_debugger
from tools.run_simulation import run_simulation


class AdvancedToolTests(unittest.TestCase):
    def test_graph_inference_has_traceable_path(self):
        result = json.loads(knowledge_graph_builder(
            [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            [{"source": "a", "target": "b", "type": "causes"},
             {"source": "b", "target": "c", "type": "mitigates"}],
            {"source": "a", "target": "c"},
        ))
        inferred = result["analysis"]["inferred_paths"][0]
        self.assertEqual(inferred["inferred_effect"], "negative")
        self.assertEqual(len(inferred["path"]), 2)

    def test_graph_detects_directed_feedback_cycle(self):
        result = json.loads(knowledge_graph_builder(
            [{"id": "a"}, {"id": "b"}],
            [{"source": "a", "target": "b", "type": "causes"},
             {"source": "b", "target": "a", "type": "causes"}],
        ))
        self.assertTrue(result["analysis"]["potential_feedback_cycles"])

    def test_simulation_recurrence_and_expression_sandbox(self):
        result = json.loads(run_simulation({"stock": 10}, {"stock": "stock + 2"}, steps=3))
        self.assertEqual(result["scenarios"][0]["sample_trajectory"][-1]["stock"], 16)
        unsafe = json.loads(run_simulation({"x": 1}, {"x": "__import__('os').system('id')"}))
        self.assertIn("error", unsafe)

    def test_context_optimizer_preserves_system_and_recent(self):
        messages = [
            {"role": "system", "content": "Policy"},
            {"role": "user", "content": "Decision: use SQLite."},
            {"role": "assistant", "content": "Acknowledged."},
        ]
        result = json.loads(context_memory_optimizer(messages, target_tokens=300, preserve_recent=1))
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertEqual(result["messages"][-1]["content"], "Acknowledged.")

    def test_reasoning_audit_detects_missing_support(self):
        result = json.loads(reasoning_chain_debugger("B", [{"id": "s1", "claim": "B", "confidence": 0.95}]))
        self.assertTrue(any("Unsupported" in issue["issue"] for issue in result["issues"]))

    @patch("tools.api_orchestrator.requests.request")
    def test_api_orchestrator_retries_then_succeeds(self, request_mock):
        failed = Mock(status_code=503, headers={}, text="busy", ok=False)
        success = Mock(status_code=200, headers={"Content-Type": "application/json"}, text='{"ok":true}', ok=True)
        request_mock.side_effect = [failed, success]
        result = json.loads(api_orchestrator(
            {"url": "https://example.test/data"}, retry={"max_attempts": 2, "base_delay": 0}
        ))
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["attempts"]), 2)

    @patch("tools.api_orchestrator.requests.post")
    @patch("tools.api_orchestrator.requests.request")
    def test_api_orchestrator_refreshes_oauth_after_401(self, request_mock, post_mock):
        token_one = Mock()
        token_one.raise_for_status.return_value = None
        token_one.json.return_value = {"access_token": "first"}
        token_two = Mock()
        token_two.raise_for_status.return_value = None
        token_two.json.return_value = {"access_token": "second"}
        post_mock.side_effect = [token_one, token_two]
        unauthorized = Mock(status_code=401, headers={}, text="unauthorized", ok=False)
        success = Mock(status_code=200, headers={}, text="ok", ok=True)
        request_mock.side_effect = [unauthorized, success]
        auth = {"type": "oauth2_client_credentials", "token_url": "https://auth.test/token", "client_id_env": "CLIENT_ID", "client_secret_env": "CLIENT_SECRET"}
        with patch.dict("os.environ", {"CLIENT_ID": "id", "CLIENT_SECRET": "secret"}):
            result = json.loads(api_orchestrator({"url": "https://example.test/data"}, auth=auth, retry={"max_attempts": 2, "base_delay": 0}))
        self.assertTrue(result["ok"])
        self.assertTrue(result["attempts"][0]["auth_refreshed"])
        self.assertEqual(post_mock.call_count, 2)

    def test_routine_requires_preview_and_confirmation(self):
        import tools.automated_routine_executor as module
        with tempfile.TemporaryDirectory() as directory, patch.object(module, "STORE_PATH", Path(directory) / "routines.json"):
            definition = {"triggers": ["check it"], "actions": [{"type": "command", "argv": ["python", "--version"]}]}
            self.assertTrue(json.loads(automated_routine_executor("define", name="check", routine=definition))["ok"])
            preview = json.loads(automated_routine_executor("run", trigger="check it"))
            self.assertTrue(preview["dry_run"])
            denied = json.loads(automated_routine_executor("run", name="check", dry_run=False))
            self.assertIn("error", denied)


if __name__ == "__main__":
    unittest.main()
