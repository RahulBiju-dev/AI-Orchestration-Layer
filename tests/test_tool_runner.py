import json
import sys
import unittest

agent_module = sys.modules.get("agent")
if agent_module is not None and not hasattr(agent_module, "__path__"):
    sys.modules.pop("agent", None)
    sys.modules.pop("agent.tool_runner", None)

from agent.tool_runner import execute_tool_call, normalize_tool_calls


class TestToolRunner(unittest.TestCase):
    def test_normalize_tool_calls_accepts_json_string_arguments(self):
        specs = normalize_tool_calls([
            {"function": {"name": "read_file", "arguments": '{"file_path":"README.md","lines":"1"}'}}
        ])

        self.assertEqual(specs[0].arguments, {"file_path": "README.md", "lines": "1"})
        self.assertIsNone(specs[0].argument_error)

    def test_execute_tool_call_reports_invalid_json_arguments(self):
        specs = normalize_tool_calls([
            {"function": {"name": "read_file", "arguments": '{"file_path":'}}
        ])

        result = execute_tool_call(specs[0])
        payload = json.loads(result.content)

        self.assertIn("error", payload)
        self.assertIn("not valid JSON", payload["error"])


if __name__ == "__main__":
    unittest.main()
