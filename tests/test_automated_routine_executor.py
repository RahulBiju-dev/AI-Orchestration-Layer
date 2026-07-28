import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from agent.cancellation import CancellationToken, OperationCancelled
from tools import automated_routine_executor as routine_module
from tools.automated_routine_executor import (
    _run_action,
    _run_registered_tool,
    _validate,
    automated_routine_executor,
)

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

    @patch('tools.automated_routine_executor.spawn_detached')
    def test_run_action_allowed_command(self, spawn_detached):
        handle = MagicMock()
        handle.poll.return_value = 0
        handle.process.poll.return_value = 0
        spawn_detached.return_value = handle

        action = {"type": "command", "argv": ["echo", "hello"]}
        result = _run_action(action)
        self.assertTrue(result["ok"])
        self.assertEqual(result["argv"], ["echo", "hello"])
        spawn_detached.assert_called_once()

    def test_cancelled_command_stops_only_its_owned_process_tree(self):
        token = CancellationToken()
        handle = MagicMock()

        def poll():
            token.cancel("stop routine")
            return None

        handle.poll.side_effect = poll
        with (
            patch('tools.automated_routine_executor.spawn_detached', return_value=handle),
            patch('tools.automated_routine_executor.terminate_process_tree') as terminate,
            self.assertRaises(OperationCancelled),
        ):
            _run_action(
                {"type": "command", "argv": ["echo", "hello"]},
                token,
            )
        terminate.assert_called_once_with(handle)

    def test_cancelled_command_does_not_claim_unconfirmed_termination(self):
        token = CancellationToken()
        handle = MagicMock()

        def poll():
            token.cancel("stop routine")
            return None

        handle.poll.side_effect = poll
        with (
            patch('tools.automated_routine_executor.spawn_detached', return_value=handle),
            patch('tools.automated_routine_executor.terminate_process_tree', return_value=False),
            self.assertRaisesRegex(RuntimeError, "could not be confirmed"),
        ):
            _run_action(
                {"type": "command", "argv": ["echo", "hello"]},
                token,
            )

    def test_run_action_blocked_command(self):
        action = {"type": "command", "argv": ["sh", "-c", "malicious_code"]}
        with self.assertRaises(ValueError) as context:
            _run_action(action)
        self.assertIn("Command 'sh' is not permitted", str(context.exception))

    def test_fast_command_cannot_bypass_the_output_limit(self):
        with patch.object(routine_module, "MAX_COMMAND_OUTPUT_BYTES", 100):
            result = _run_action({
                "type": "command",
                "argv": ["python", "-c", "print('x' * 1000)"],
            })

        self.assertFalse(result["ok"])
        self.assertIn("output exceeded", result["error"])

    def test_nested_open_app_receives_the_approved_routine_confirmation(self):
        execution = SimpleNamespace(
            content='{"ok": true}',
            ok=True,
            status=SimpleNamespace(value="completed"),
        )
        with patch(
            "agent.tool_runner.execute_tool_call", return_value=execution
        ) as execute:
            result = _run_registered_tool("open_app", {"app_name": "Editor"})

        self.assertTrue(result["ok"])
        spec = execute.call_args.args[0]
        self.assertTrue(spec.arguments["confirmed"])

    def test_invalid_stored_action_is_revalidated_before_execution(self):
        stored = {
            "unsafe": {
                "description": "Tampered routine",
                "triggers": ["unsafe"],
                "actions": [{"type": "command", "argv": ["sh", "-c", "id"]}],
                "automatic_approved": False,
            }
        }
        with (
            patch.object(routine_module, "_load", return_value=stored),
            patch.object(routine_module, "_run_action") as run_action,
        ):
            result = json.loads(automated_routine_executor(
                action="run",
                name="unsafe",
                confirmed=True,
            ))

        self.assertIn("Stored routine is invalid", result["error"])
        self.assertTrue(result["preserved"])
        run_action.assert_not_called()

    def test_trigger_collisions_and_unapproved_replacement_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory) / "routines.json"
            legacy = Path(directory) / "legacy.json"
            routine = {
                "description": "Open an editor",
                "triggers": ["start work"],
                "actions": [{"type": "open_app", "app_name": "Editor"}],
            }
            with (
                patch.object(routine_module, "STORE_PATH", store),
                patch.object(routine_module, "LEGACY_STORE_PATH", legacy),
            ):
                first = json.loads(automated_routine_executor(
                    action="define", name="work", routine=routine
                ))
                collision = json.loads(automated_routine_executor(
                    action="define",
                    name="other",
                    routine={**routine, "description": "A second routine"},
                ))
                replacement = json.loads(automated_routine_executor(
                    action="define",
                    name="WORK",
                    routine={**routine, "description": "Replacement"},
                ))
                replaced = json.loads(automated_routine_executor(
                    action="define",
                    name="WORK",
                    routine={**routine, "description": "Replacement"},
                    overwrite=True,
                    confirmed=True,
                ))

        self.assertTrue(first["ok"])
        self.assertIn("unique", collision["error"])
        self.assertEqual(collision["conflicts"], {"work": ["start work"]})
        self.assertIn("already exists", replacement["error"])
        self.assertTrue(replacement["preserved"])
        self.assertTrue(replaced["ok"])
        self.assertEqual(replaced["defined"], "work")

    def test_tool_actions_are_schema_validated_before_save(self):
        routine = {
            "description": "Invalid spreadsheet export",
            "triggers": ["export it"],
            "actions": [{
                "type": "tool",
                "tool_name": "spreadsheet",
                "arguments": {"action": "create"},
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(routine_module, "STORE_PATH", Path(directory) / "routines.json"),
                patch.object(routine_module, "LEGACY_STORE_PATH", Path(directory) / "legacy.json"),
            ):
                result = json.loads(automated_routine_executor(
                    action="define",
                    name="export",
                    routine=routine,
                ))

        self.assertEqual(result["error"], "Invalid routine")
        self.assertIn("file_path is required", " ".join(result["details"]))

    def test_confirmed_routine_approval_reaches_nested_spreadsheet_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "routines.json"
            output = root / "routine.csv"
            routine = {
                "description": "Export rows to CSV",
                "triggers": ["export rows"],
                "actions": [{
                    "type": "tool",
                    "tool_name": "spreadsheet",
                    "arguments": {
                        "action": "create",
                        "file_path": str(output),
                        "rows": [["Name", "Score"], ["Ada", 10]],
                    },
                }],
            }
            with (
                patch.object(routine_module, "STORE_PATH", store),
                patch.object(routine_module, "LEGACY_STORE_PATH", root / "legacy.json"),
            ):
                defined = json.loads(automated_routine_executor(
                    action="define", name="export", routine=routine
                ))
                executed = json.loads(automated_routine_executor(
                    action="run", name="export", confirmed=True
                ))

            self.assertTrue(defined["ok"])
            self.assertTrue(executed["ok"])
            self.assertEqual(executed["actions_executed"], 1)
            self.assertFalse(executed["stopped_early"])
            self.assertTrue(output.is_file())

if __name__ == '__main__':
    unittest.main()
