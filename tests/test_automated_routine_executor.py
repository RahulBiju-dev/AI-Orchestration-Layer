import json
import unittest
from unittest.mock import patch, MagicMock
from tools.automated_routine_executor import automated_routine_executor, _validate, _run_action

class TestAutomatedRoutineExecutor(unittest.TestCase):

    def test_validate_allowed_command(self):
        routine = {
            "description": "Test routine",
            "triggers": ["test"],
            "actions": [
                {"type": "command", "argv": ["echo", "hello"]}
            ]
        }
        errors = _validate(routine)
        self.assertEqual(len(errors), 0)

    def test_validate_blocked_command(self):
        routine = {
            "description": "Test routine",
            "triggers": ["test"],
            "actions": [
                {"type": "command", "argv": ["rm", "-rf", "/"]}
            ]
        }
        errors = _validate(routine)
        self.assertEqual(len(errors), 1)
        self.assertIn("must be an allowed command", errors[0])
        self.assertIn("found 'rm'", errors[0])

    @patch('subprocess.run')
    def test_run_action_allowed_command(self, mock_subprocess_run):
        mock_completed = MagicMock()
        mock_completed.returncode = 0
        mock_completed.stdout = "hello"
        mock_completed.stderr = ""
        mock_subprocess_run.return_value = mock_completed

        action = {"type": "command", "argv": ["echo", "hello"]}
        result = _run_action(action)
        self.assertTrue(result["ok"])
        self.assertEqual(result["argv"], ["echo", "hello"])

    def test_run_action_blocked_command(self):
        action = {"type": "command", "argv": ["sh", "-c", "malicious_code"]}
        with self.assertRaises(ValueError) as context:
            _run_action(action)
        self.assertIn("Command 'sh' is not permitted", str(context.exception))

if __name__ == '__main__':
    unittest.main()
