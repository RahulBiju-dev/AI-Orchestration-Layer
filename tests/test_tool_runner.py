import json
import subprocess
import sys
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

agent_module = sys.modules.get("agent")
if agent_module is not None and not hasattr(agent_module, "__path__"):
    sys.modules.pop("agent", None)
    sys.modules.pop("agent.tool_runner", None)

import agent.tool_runner as tool_runner
from agent.cancellation import CancellationToken
from agent.tool_runner import (
    ToolResultStatus,
    build_execution_batches,
    execute_tool_call,
    execute_tool_calls,
    normalize_tool_calls,
)
from tools.registry import (
    TOOL_DISPATCH,
    TOOL_METADATA,
    TOOL_SCHEMA_BY_NAME,
    ToolMetadata,
    validate_tool_registry,
)


def _metadata(
    name,
    *,
    side_effecting=False,
    parallel_safe=False,
    cpu_heavy=False,
    gpu_heavy=False,
    supports_cancellation=False,
    timeout=1.0,
    output_limit=2_000,
):
    return ToolMetadata(
        name=name,
        read_only=not side_effecting,
        side_effecting=side_effecting,
        parallel_safe=parallel_safe,
        idempotent=not side_effecting,
        cpu_heavy=cpu_heavy,
        gpu_heavy=gpu_heavy,
        supports_cancellation=supports_cancellation,
        default_timeout_seconds=timeout,
        max_output_chars=output_limit,
    )


@contextmanager
def _registered_tool(name, handler, metadata=None, schema=None):
    metadata = metadata or _metadata(name)
    with patch.dict(TOOL_DISPATCH, {name: handler}, clear=False):
        with patch.dict(TOOL_METADATA, {name: metadata}, clear=False):
            if schema is None:
                yield
            else:
                with patch.dict(TOOL_SCHEMA_BY_NAME, {name: schema}, clear=False):
                    yield


def _spec(name, arguments=None):
    return normalize_tool_calls(
        [{"function": {"name": name, "arguments": arguments or {}}}]
    )[0]


class TestToolRunner(unittest.TestCase):
    def tearDown(self):
        tool_runner.set_tool_resource_guard(None)

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

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertEqual(payload["error_code"], "invalid_arguments")
        self.assertIn("not valid JSON", payload["error"])

    def test_unknown_tool_returns_structured_error_without_execution(self):
        result = execute_tool_call(_spec("does_not_exist", {"value": 1}))
        payload = json.loads(result.content)

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "unknown_tool")
        self.assertIn("does_not_exist", payload["error"])

    def test_unknown_argument_is_rejected_before_handler_invocation(self):
        result = execute_tool_call(_spec("get_current_datetime", {"made_up": True}))
        payload = json.loads(result.content)

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertEqual(payload["error_code"], "invalid_arguments")
        self.assertIn(
            "arguments.made_up is not allowed",
            payload["details"]["validation_errors"],
        )

    def test_plain_text_error_result_is_not_reported_as_success(self):
        metadata = _metadata("legacy_error", side_effecting=True)
        with _registered_tool("legacy_error", lambda: "Error: backend unavailable", metadata):
            result = execute_tool_call(_spec("legacy_error"))

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertEqual(result.content, "Error: backend unavailable")

    def test_schema_validation_rejects_missing_and_wrong_typed_arguments(self):
        invoked = threading.Event()

        def handler(**_arguments):
            invoked.set()
            return "should not execute"

        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "count": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        }
        with _registered_tool("schema_probe", handler, schema=schema):
            result = execute_tool_call(
                _spec("schema_probe", {"count": "three", "unexpected": True})
            )

        payload = json.loads(result.content)
        errors = payload["details"]["validation_errors"]
        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertEqual(payload["error_code"], "invalid_arguments")
        self.assertTrue(any("query is required" in error for error in errors))
        self.assertTrue(any("count must be integer" in error for error in errors))
        self.assertTrue(any("unexpected is not allowed" in error for error in errors))
        self.assertFalse(invoked.is_set())

    def test_schema_validation_rejects_non_finite_numbers(self):
        schema = {
            "type": "object",
            "properties": {"payload": {}},
            "required": ["payload"],
            "additionalProperties": False,
        }
        with _registered_tool("finite_probe", lambda **_arguments: "unexpected", schema=schema):
            result = execute_tool_call(_spec("finite_probe", {"payload": {"value": float("nan")}}))

        payload = json.loads(result.content)
        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("arguments.payload.value must be finite", payload["details"]["validation_errors"])

    def test_tool_output_is_bounded_and_marked_as_truncated(self):
        metadata = _metadata("large_output", output_limit=256)
        with _registered_tool("large_output", lambda: "x" * 5_000, metadata):
            result = execute_tool_call(_spec("large_output"))

        payload = json.loads(result.content)
        self.assertEqual(result.status, ToolResultStatus.SUCCESS)
        self.assertTrue(result.truncated)
        self.assertLessEqual(len(result.content), metadata.max_output_chars)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["original_characters"], 5_000)

    def test_non_serializable_tool_result_is_a_controlled_error(self):
        recursive = []
        recursive.append(recursive)
        with _registered_tool("recursive_result", lambda: recursive):
            result = execute_tool_call(_spec("recursive_result"))

        payload = json.loads(result.content)
        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertEqual(payload["error_code"], "invalid_tool_result")

    def test_non_finite_timeout_is_rejected_without_invoking_handler(self):
        invoked = threading.Event()
        with _registered_tool("timeout_validation", lambda: invoked.set()):
            result = execute_tool_call(
                _spec("timeout_validation"),
                timeout_seconds=float("nan"),
            )

        payload = json.loads(result.content)
        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertEqual(payload["error_code"], "invalid_timeout")
        self.assertFalse(invoked.is_set())

    def test_duplicate_calls_replay_exact_result_once_in_input_order(self):
        calls = []

        def handler(value):
            calls.append(value)
            return {"value": value, "invocation": len(calls)}

        tool_calls = [
            {"function": {"name": "dedupe_probe", "arguments": {"value": 1}}},
            {"function": {"name": "dedupe_probe", "arguments": '{"value":1}'}},
            {"function": {"name": "dedupe_probe", "arguments": {"value": 2}}},
        ]
        with _registered_tool("dedupe_probe", handler):
            results = execute_tool_calls(tool_calls)

        self.assertEqual([result.spec.index for result in results], [0, 1, 2])
        self.assertEqual(calls, [1, 2])
        self.assertEqual(results[1].duplicate_of, 0)
        self.assertEqual(results[1].content, results[0].content)
        self.assertEqual(results[1].status, results[0].status)
        self.assertEqual(results[1].truncated, results[0].truncated)

    def test_parallel_completion_is_reported_in_deterministic_input_order(self):
        def slow():
            time.sleep(0.03)
            return "slow"

        def fast():
            return "fast"

        calls = [
            {"function": {"name": "parallel_slow", "arguments": {}}},
            {"function": {"name": "parallel_fast", "arguments": {}}},
        ]
        metadata_slow = _metadata("parallel_slow", parallel_safe=True)
        metadata_fast = _metadata("parallel_fast", parallel_safe=True)
        ended = []
        with _registered_tool("parallel_slow", slow, metadata_slow):
            with _registered_tool("parallel_fast", fast, metadata_fast):
                results = execute_tool_calls(
                    calls, on_end=lambda result: ended.append(result.spec.index)
                )

        self.assertEqual([result.spec.index for result in results], [0, 1])
        self.assertEqual([result.content for result in results], ["slow", "fast"])
        self.assertEqual(ended, [0, 1])

    def test_inflight_cancellation_reaches_cooperative_handler(self):
        started = threading.Event()
        finished = threading.Event()

        def handler(cancellation_token):
            started.set()
            try:
                while not cancellation_token.cancelled:
                    time.sleep(0.002)
                cancellation_token.raise_if_cancelled()
            finally:
                finished.set()

        metadata = _metadata(
            "cancel_probe", supports_cancellation=True, timeout=1.0
        )
        owner_token = CancellationToken()
        holder = []
        with _registered_tool("cancel_probe", handler, metadata):
            thread = threading.Thread(
                target=lambda: holder.append(
                    execute_tool_call(
                        _spec("cancel_probe"),
                        cancellation_token=owner_token,
                    )
                )
            )
            thread.start()
            self.assertTrue(started.wait(0.5))
            owner_token.cancel("test cancellation")
            thread.join(0.5)
            self.assertFalse(thread.is_alive())
            self.assertTrue(finished.wait(0.5))

        self.assertEqual(holder[0].status, ToolResultStatus.CANCELLED)
        payload = json.loads(holder[0].content)
        self.assertEqual(payload["error_code"], "cancelled")
        self.assertIn("test cancellation", payload["error"])

    def test_timeout_returns_structured_status_and_eventually_releases_guard(self):
        semaphore = threading.BoundedSemaphore(1)
        guard_exits = []
        first_started = threading.Event()

        @contextmanager
        def guard(_metadata, _token):
            semaphore.acquire()
            try:
                yield
            finally:
                guard_exits.append(time.monotonic())
                semaphore.release()

        invocation = 0

        def handler():
            nonlocal invocation
            invocation += 1
            if invocation == 1:
                first_started.set()
                time.sleep(0.05)
            return f"run-{invocation}"

        tool_runner.set_tool_resource_guard(guard)
        with _registered_tool("timeout_probe", handler):
            first = execute_tool_call(
                _spec("timeout_probe"), timeout_seconds=0.005
            )
            self.assertTrue(first_started.is_set())
            payload = json.loads(first.content)
            self.assertEqual(first.status, ToolResultStatus.TIMEOUT)
            self.assertEqual(payload["error_code"], "timeout")

            deadline = time.monotonic() + 0.5
            while len(guard_exits) < 1 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(len(guard_exits), 1)

            second = execute_tool_call(
                _spec("timeout_probe"), timeout_seconds=0.5
            )

        self.assertEqual(second.status, ToolResultStatus.SUCCESS)
        self.assertEqual(second.content, "run-2")
        self.assertEqual(len(guard_exits), 2)

    def test_side_effect_after_timeout_is_blocked_and_never_invoked(self):
        first_finished = threading.Event()
        invocations = []

        def first(cancellation_token):
            invocations.append("first")
            try:
                while not cancellation_token.cancelled:
                    time.sleep(0.002)
                cancellation_token.raise_if_cancelled()
            finally:
                first_finished.set()

        def second():
            invocations.append("second")
            return "unsafe"

        first_metadata = _metadata(
            "slow_side_effect",
            side_effecting=True,
            supports_cancellation=True,
            timeout=0.01,
        )
        second_metadata = _metadata(
            "later_side_effect", side_effecting=True, timeout=0.5
        )
        calls = [
            {"function": {"name": "slow_side_effect", "arguments": {}}},
            {"function": {"name": "later_side_effect", "arguments": {}}},
        ]
        with _registered_tool("slow_side_effect", first, first_metadata):
            with _registered_tool("later_side_effect", second, second_metadata):
                results = execute_tool_calls(calls)
                self.assertTrue(first_finished.wait(0.5))

        self.assertEqual(invocations, ["first"])
        self.assertEqual(results[0].status, ToolResultStatus.TIMEOUT)
        self.assertEqual(results[1].status, ToolResultStatus.CANCELLED)
        blocked = json.loads(results[1].content)
        self.assertEqual(blocked["error_code"], "blocked_by_prior_call")
        self.assertIn("slow_side_effect", blocked["error"])

    def test_nested_registered_tool_executes_without_worker_pool_deadlock(self):
        def child(value):
            return {"value": value}

        def parent():
            nested = execute_tool_call(_spec("nested_child", {"value": 7}))
            return {"nested_status": nested.status.value, "nested": json.loads(nested.content)}

        with _registered_tool("nested_child", child):
            with _registered_tool("nested_parent", parent):
                result = execute_tool_call(_spec("nested_parent"), timeout_seconds=0.5)

        payload = json.loads(result.content)
        self.assertEqual(result.status, ToolResultStatus.SUCCESS)
        self.assertEqual(payload["nested_status"], "success")
        self.assertEqual(payload["nested"]["value"], 7)

    def test_temporal_preflight_is_ordered_before_dependent_live_call(self):
        specs = normalize_tool_calls([
            {"function": {"name": "web_search", "arguments": {"query": "today"}}},
            {"function": {"name": "get_current_datetime", "arguments": {}}},
        ])

        batches = build_execution_batches(specs)

        self.assertEqual(len(batches), 1)
        self.assertFalse(batches[0][0])
        self.assertEqual(
            [spec.name for spec in batches[0][1]],
            ["get_current_datetime", "web_search"],
        )


class TestToolRegistryContract(unittest.TestCase):
    def test_dispatch_schema_and_metadata_registry_are_in_sync(self):
        self.assertEqual(validate_tool_registry(), [])

    def test_model_tool_root_schemas_reject_unknown_parameters(self):
        for name, schema in TOOL_SCHEMA_BY_NAME.items():
            with self.subTest(tool=name):
                self.assertIs(schema.get("additionalProperties"), False)

    def test_registry_detects_schema_handler_signature_drift(self):
        broken_schema = {
            "type": "object",
            "properties": {"invented": {"type": "string"}},
            "additionalProperties": False,
        }
        with patch.dict(TOOL_SCHEMA_BY_NAME, {"get_current_datetime": broken_schema}):
            errors = validate_tool_registry()

        self.assertTrue(any("not accepted by the handler" in error for error in errors))

    def test_status_actions_do_not_acquire_heavy_resource_slots(self):
        for name in ("build_vault_notes_pdf", "codebase_indexer", "index_vault"):
            with self.subTest(tool=name):
                metadata = tool_runner.get_tool_metadata(name, {"action": "status"})
                self.assertIsNotNone(metadata)
                self.assertTrue(metadata.read_only)
                self.assertTrue(metadata.parallel_safe)
                self.assertFalse(metadata.side_effecting)
                self.assertFalse(metadata.cpu_heavy)
                self.assertFalse(metadata.gpu_heavy)

    def test_registry_import_survives_missing_optional_chromadb(self):
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "sys.modules['chromadb'] = None\n"
                    "import tools.registry\n"
                    "from tools.vault_indexer import get_chroma_client\n"
                    "assert tools.registry.validate_tool_registry() == []\n"
                    "try:\n"
                    "    get_chroma_client()\n"
                    "except RuntimeError as exc:\n"
                    "    assert 'ChromaDB is unavailable' in str(exc)\n"
                    "else:\n"
                    "    raise AssertionError('missing ChromaDB was not isolated')\n"
                ),
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_resource_heavy_tools_are_never_marked_parallel_safe(self):
        heavy = {
            name: metadata
            for name, metadata in TOOL_METADATA.items()
            if metadata.cpu_heavy or metadata.gpu_heavy
        }

        self.assertTrue(
            {"describe_image", "codebase_indexer", "index_vault", "vault_search"}
            <= set(heavy)
        )
        for name, metadata in heavy.items():
            with self.subTest(tool=name):
                self.assertFalse(metadata.parallel_safe)

    def test_every_tool_has_platform_support_classification(self):
        for name, metadata in TOOL_METADATA.items():
            with self.subTest(tool=name):
                self.assertIn(metadata.fedora_support, {"supported", "partial", "unsupported", "limited"})
                self.assertIn(metadata.windows_support, {"supported", "partial", "unsupported", "limited"})
                self.assertTrue(callable(TOOL_DISPATCH[name]))


class TestToolRunnerUncertainty(unittest.TestCase):
    def tearDown(self):
        tool_runner.set_tool_resource_guard(None)

    def test_exception_after_side_effect_blocks_later_side_effects_when_uncertain(self):
        """Non-idempotent side effects that raise still block subsequent effects."""
        invocations = []

        def first():
            invocations.append("first")
            raise RuntimeError("failed after possible side effect")

        def second():
            invocations.append("second")
            return "should-not-run"

        first_metadata = ToolMetadata(
            name="uncertain_write",
            read_only=False,
            side_effecting=True,
            parallel_safe=False,
            idempotent=False,
            default_timeout_seconds=0.5,
            max_output_chars=2_000,
        )
        second_metadata = _metadata("later_write", side_effecting=True, timeout=0.5)
        calls = [
            {"function": {"name": "uncertain_write", "arguments": {}}},
            {"function": {"name": "later_write", "arguments": {}}},
        ]
        with _registered_tool("uncertain_write", first, first_metadata):
            with _registered_tool("later_write", second, second_metadata):
                results = execute_tool_calls(calls)

        self.assertEqual(results[0].status, ToolResultStatus.ERROR)
        self.assertEqual(results[1].status, ToolResultStatus.CANCELLED)
        self.assertEqual(invocations, ["first"])
        blocked = json.loads(results[1].content)
        self.assertEqual(blocked["error_code"], "blocked_by_prior_call")

    def test_shutdown_rejects_new_work_without_deadlock(self):
        def handler():
            return "ok"

        with _registered_tool("after_shutdown", handler):
            # Soft shutdown must return promptly and refuse new work.
            started = time.monotonic()
            tool_runner.shutdown_tool_runner(wait=False)
            result = execute_tool_call(_spec("after_shutdown"), timeout_seconds=0.5)
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 1.0)
        self.assertEqual(result.status, ToolResultStatus.CANCELLED)
        payload = json.loads(result.content)
        self.assertEqual(payload["error_code"], "runner_shutdown")
        # Restore a live executor for later tests in this process.
        from concurrent.futures import ThreadPoolExecutor

        tool_runner._SHUTDOWN = False
        tool_runner._HANDLER_EXECUTOR = ThreadPoolExecutor(
            max_workers=tool_runner.MAX_PARALLEL_TOOL_WORKERS,
            thread_name_prefix="selene-tool",
        )


if __name__ == "__main__":
    unittest.main()
