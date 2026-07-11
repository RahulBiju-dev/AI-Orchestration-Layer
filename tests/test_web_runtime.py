import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.cancellation import CancellationToken
from agent.tool_runner import ToolCallResult
from agent.web_runtime import (
    ClientSessionStore,
    GenerationConflict,
    GenerationOwnershipError,
    GenerationRegistry,
    TerminalState,
)


class TestGenerationRegistry(unittest.TestCase):
    def test_saved_session_rejects_a_second_generation_across_clients(self):
        registry = GenerationRegistry()
        lease = registry.begin("conversation.json", "client-one", "generation-one")

        with self.assertRaises(GenerationConflict):
            registry.begin("conversation.json", "client-two", "generation-two")

        registry.finish(lease, TerminalState.COMPLETED)

    def test_unsaved_sessions_are_isolated_per_client(self):
        registry = GenerationRegistry()

        first = registry.begin("Active Session", "client-one", "generation-one")
        second = registry.begin("Active Session", "client-two", "generation-two")

        self.assertNotEqual(first.session_key, second.session_key)
        self.assertIs(registry.active_for_session("Active Session", "client-one"), first)
        self.assertIs(registry.active_for_session("Active Session", "client-two"), second)

    def test_only_the_owning_client_can_cancel_a_generation(self):
        registry = GenerationRegistry()
        lease = registry.begin("conversation.json", "owner", "generation-one")

        with self.assertRaises(GenerationOwnershipError):
            registry.cancel("generation-one", "other-client")
        self.assertFalse(lease.token.cancelled)

        cancelled = registry.cancel("generation-one", "owner", reason="Stop requested")
        self.assertIs(cancelled, lease)
        self.assertTrue(lease.token.cancelled)
        self.assertEqual(lease.token.reason, "Stop requested")

    def test_finish_is_idempotent_and_preserves_first_terminal_state(self):
        registry = GenerationRegistry()
        lease = registry.begin("conversation.json", "owner", "generation-one")

        first = registry.finish(lease, TerminalState.COMPLETED, "first result")
        repeated = registry.finish(lease, TerminalState.FAILED, "late failure")

        self.assertIs(repeated, first)
        self.assertEqual(repeated.state, TerminalState.COMPLETED)
        self.assertEqual(repeated.detail, "first result")
        self.assertEqual(registry.active_operations(), [])
        self.assertIs(registry.get_terminal("generation-one"), first)

    def test_title_rebind_reserves_new_saved_session_identity(self):
        registry = GenerationRegistry()
        lease = registry.begin("temporary.json", "owner", "generation-one")

        rebound = registry.rebind_generation(
            "generation-one", "owner", "semantic-title.json"
        )

        self.assertIs(rebound, lease)
        self.assertIs(
            registry.active_for_session("semantic-title.json", "another-client"),
            lease,
        )
        with self.assertRaises(GenerationConflict):
            registry.begin("semantic-title.json", "another-client", "generation-two")


class TestClientSessionStore(unittest.TestCase):
    def setUp(self):
        self.default_settings = {
            "runtime_profile": "low-vram",
            "options": {"num_ctx": 4096},
            "history": True,
        }
        self.store = ClientSessionStore(self.default_settings)

    def test_generation_commit_does_not_overwrite_a_newly_selected_session(self):
        self.store.select(
            "client-one",
            "origin.json",
            self.default_settings,
            [{"role": "user", "content": "origin"}],
        )
        generation_start = self.store.snapshot("client-one")
        selected_settings = {
            "runtime_profile": "balanced",
            "options": {"num_ctx": 8192},
            "history": True,
        }
        selected_history = [{"role": "user", "content": "selected"}]
        self.store.select(
            "client-one",
            "other.json",
            selected_settings,
            selected_history,
        )

        committed = self.store.commit_generation(
            "client-one",
            "origin.json",
            "renamed-origin.json",
            generation_start.session,
            [{"role": "assistant", "content": "stale completion"}],
            generation_start_session=generation_start.session,
        )

        self.assertFalse(committed)
        current = self.store.snapshot("client-one")
        self.assertEqual(current.active_session_name, "other.json")
        self.assertEqual(current.session, selected_settings)
        self.assertEqual(current.history, selected_history)

    def test_generation_commit_preserves_settings_changed_while_running(self):
        self.store.select(
            "client-one",
            "origin.json",
            self.default_settings,
            [{"role": "user", "content": "request"}],
        )
        generation_start = self.store.snapshot("client-one")
        updated_settings = {
            "runtime_profile": "balanced",
            "options": {"num_ctx": 8192, "num_predict": 1024},
            "history": False,
        }
        self.store.update_settings("client-one", updated_settings)
        completed_history = [
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": "answer"},
        ]

        committed = self.store.commit_generation(
            "client-one",
            "origin.json",
            "renamed-origin.json",
            generation_start.session,
            completed_history,
            generation_start_session=generation_start.session,
        )

        self.assertTrue(committed)
        current = self.store.snapshot("client-one")
        self.assertEqual(current.active_session_name, "renamed-origin.json")
        self.assertEqual(current.session, updated_settings)
        self.assertEqual(current.history, completed_history)

    def test_loading_saved_session_does_not_inherit_current_overrides(self):
        from agent import web

        self.store.select(
            "client-one",
            "current.json",
            {**self.default_settings, "options": {"num_ctx": 8192}},
            [],
        )
        saved_settings = {
            "runtime_profile": "low-vram",
            "options": {},
            "history": True,
        }
        saved_history = [{"role": "user", "content": "saved"}]
        with (
            patch.object(web, "CLIENT_SESSIONS", self.store),
            patch.object(
                web,
                "_read_session_snapshot",
                return_value=(saved_settings, saved_history),
            ),
        ):
            web.load_session("saved.json", "client-one")

        current = self.store.snapshot("client-one")
        self.assertEqual(current.session["options"], {})
        self.assertEqual(current.history, saved_history)

    def test_manual_saves_with_the_same_name_never_overwrite(self):
        from agent import web

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(web, "_SESSIONS_DIR", temporary),
            patch.object(web, "CLIENT_SESSIONS", ClientSessionStore(self.default_settings)),
        ):
            first = web.save_session("Same Name", self.default_settings, [], "client-one")
            second = web.save_session("Same Name", self.default_settings, [], "client-one")

            self.assertNotEqual(first, second)
            self.assertTrue((Path(temporary) / first).is_file())
            self.assertTrue((Path(temporary) / second).is_file())


class TestChatEventTerminalState(unittest.TestCase):
    @staticmethod
    def _run_with_implementation(implementation, *, token=None):
        from agent import web

        with (
            patch.object(web, "_generate_chat_events_impl", implementation),
            patch.object(web, "list_saved_sessions", return_value=["conversation.json"]),
        ):
            return list(
                web.generate_chat_events(
                    "hello",
                    {"options": {}},
                    [],
                    "conversation.json",
                    cancellation_token=token,
                    generation_id="generation-one",
                    publish_global=False,
                    client_id="client-one",
                )
            )

    def test_duplicate_implementation_terminal_events_become_exactly_one(self):
        def implementation(*args, **kwargs):
            payload = {
                "type": "done",
                "state": "completed",
                "history": [],
                "active_session_name": "conversation.json",
                "saved_sessions": ["conversation.json"],
            }
            yield payload
            yield dict(payload)

        events = self._run_with_implementation(implementation)
        terminal = [event for event in events if event.get("type") == "done"]

        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["state"], "completed")
        self.assertEqual(terminal[0]["generation_id"], "generation-one")

    def test_missing_implementation_terminal_event_becomes_failed_terminal(self):
        def implementation(*args, **kwargs):
            yield {"type": "status", "message": "working"}

        events = self._run_with_implementation(implementation)
        terminal = [event for event in events if event.get("type") == "done"]

        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["state"], "failed")
        self.assertIn("without a completion result", terminal[0]["error"])

    def test_cancelled_implementation_becomes_cancelled_terminal(self):
        token = CancellationToken()

        def implementation(*args, **kwargs):
            token.cancel("Cancelled by test owner")
            yield {"type": "content_chunk", "content": "must not escape"}

        events = self._run_with_implementation(implementation, token=token)
        terminal = [event for event in events if event.get("type") == "done"]

        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["state"], "cancelled")
        self.assertEqual(terminal[0]["error"], "Cancelled by test owner")
        self.assertFalse(any(event.get("type") == "content_chunk" for event in events))

    def test_tool_round_cap_never_persists_an_unmatched_tool_call(self):
        from agent import web

        tool_call = SimpleNamespace(
            function=SimpleNamespace(name="get_current_datetime", arguments={})
        )
        chunk = SimpleNamespace(
            message=SimpleNamespace(tool_calls=[tool_call], thinking="", content=""),
            prompt_eval_count=0,
            eval_count=0,
        )
        service = MagicMock()
        service.chat.side_effect = lambda **kwargs: iter([chunk])

        def execute(calls, **kwargs):
            if not calls:
                return []
            spec = web.normalize_tool_calls(calls)[0]
            return [ToolCallResult(spec, json.dumps({"ok": True, "now": "test"}))]

        session = {
            "runtime_profile": "low-vram",
            "options": {},
            "history": True,
            "system": "",
            "think": False,
        }
        history = []
        with (
            patch.object(web, "TOOL_DISPATCH", {}),
            patch.object(web, "OllamaService", return_value=service),
            patch.object(web, "execute_tool_calls", side_effect=execute),
            patch.object(web, "load_default_system_prompt", return_value=""),
            patch.object(web, "prepare_messages_for_model", side_effect=lambda messages, *args, **kwargs: list(messages)),
            patch.object(web, "tool_schemas_for_model", return_value=[]),
            patch.object(web, "guarded_options_for_call", return_value={"num_ctx": 4096, "num_predict": 768}),
            patch.object(web, "effective_session_model_options", return_value=(None, {"num_ctx": 4096, "num_predict": 768})),
            patch.object(web, "save_session_snapshot", return_value="conversation.json"),
            patch.object(web, "list_saved_sessions", return_value=["conversation.json"]),
        ):
            events = list(web._generate_chat_events_impl(
                "hello",
                session,
                history,
                "conversation.json",
                cancellation_token=CancellationToken(),
                publish_global=False,
                client_id="client-one",
            ))

        terminal_history = next(event["history"] for event in reversed(events) if event.get("type") == "done")
        assistant_calls = [message for message in terminal_history if message.get("role") == "assistant" and message.get("tool_calls")]
        tool_results = [message for message in terminal_history if message.get("role") == "tool"]
        self.assertEqual(len(assistant_calls), len(tool_results))
        self.assertNotIn("tool_calls", terminal_history[-1])

    def test_length_limited_web_segments_stream_and_persist_as_one_answer(self):
        from agent import web

        def chunk(content, *, reason, count):
            return SimpleNamespace(
                message=SimpleNamespace(tool_calls=[], thinking="", content=content),
                prompt_eval_count=20,
                eval_count=count,
                done_reason=reason,
            )

        service = MagicMock()
        service.chat.side_effect = [
            iter([chunk("Once upon a ", reason="length", count=768)]),
            iter([chunk("time.", reason="stop", count=12)]),
        ]
        session = {
            "runtime_profile": "low-vram",
            "options": {},
            "history": True,
            "system": "",
            "think": True,
            "format": "",
        }
        history = []
        with (
            patch.object(web, "TOOL_DISPATCH", {}),
            patch.object(web, "OllamaService", return_value=service),
            patch.object(web, "load_default_system_prompt", return_value=""),
            patch.object(web, "prepare_messages_for_model", side_effect=lambda messages, *args, **kwargs: list(messages)),
            patch.object(web, "tool_schemas_for_model", return_value=[]),
            patch.object(web, "guarded_options_for_call", return_value={"num_ctx": 4096, "num_predict": 768}),
            patch.object(web, "effective_session_model_options", return_value=(None, {"num_ctx": 4096, "num_predict": 768})),
            patch.object(web, "title_temporary_session", return_value=None),
            patch.object(web, "save_session_snapshot", return_value="conversation.json"),
            patch.object(web, "list_saved_sessions", return_value=["conversation.json"]),
        ):
            events = list(web._generate_chat_events_impl(
                "Write a story",
                session,
                history,
                "conversation.json",
                cancellation_token=CancellationToken(),
                publish_global=False,
                client_id="client-one",
            ))

        visible = "".join(
            event.get("text", "") for event in events
            if event.get("type") == "content_chunk"
        )
        terminal = next(event for event in reversed(events) if event.get("type") == "done")
        self.assertEqual(visible, "Once upon a time.")
        self.assertEqual(terminal["state"], "completed")
        self.assertEqual(terminal["history"][-1]["content"], "Once upon a time.")
        self.assertEqual(service.chat.call_count, 2)
        self.assertFalse(service.chat.call_args_list[1].kwargs["think"])
        self.assertNotIn("tools", service.chat.call_args_list[1].kwargs)


class TestHTTPGenerationLifecycle(unittest.TestCase):
    @staticmethod
    def _handler(generation_id, *, end_headers=None):
        from agent import web

        handler = object.__new__(web.AgentHTTPRequestHandler)
        handler.path = "/api/chat"
        handler.headers = {
            "Host": "127.0.0.1:5005",
            "X-Selene-Client-ID": "client-one",
        }
        handler.read_json_body = MagicMock(return_value={
            "client_id": "client-one",
            "generation_id": generation_id,
            "session_name": "Active Session",
            "message": "hello",
        })
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.send_json_response = MagicMock()
        handler.end_headers = end_headers or MagicMock()
        handler.wfile = MagicMock()
        return handler

    def test_sse_header_disconnect_releases_generation_ownership(self):
        from agent import web

        registry = GenerationRegistry()
        store = ClientSessionStore({"options": {}, "history": True})
        handler = self._handler(
            "generation-one",
            end_headers=MagicMock(side_effect=BrokenPipeError()),
        )
        with (
            patch.object(web, "ACTIVE_GENERATIONS", registry),
            patch.object(web, "CLIENT_SESSIONS", store),
            patch.object(web, "save_session_snapshot", return_value="conversation.json"),
        ):
            handler.do_POST()

        self.assertEqual(registry.active_operations(), [])
        self.assertEqual(
            registry.get_terminal("generation-one").state,
            TerminalState.CANCELLED,
        )

    def test_simultaneous_blank_submissions_run_only_one_generation(self):
        from agent import web

        registry = GenerationRegistry()
        store = ClientSessionStore({"options": {}, "history": True})
        first = self._handler("generation-one")
        second = self._handler("generation-two")
        started = threading.Event()
        release = threading.Event()

        def events(*args, **kwargs):
            started.set()
            release.wait(2)
            yield {
                "type": "done",
                "state": "completed",
                "history": [],
                "active_session_name": "conversation.json",
                "saved_sessions": ["conversation.json"],
            }

        with (
            patch.object(web, "ACTIVE_GENERATIONS", registry),
            patch.object(web, "CLIENT_SESSIONS", store),
            patch.object(web, "save_session_snapshot", return_value="conversation.json"),
            patch.object(web, "generate_chat_events", side_effect=events),
        ):
            thread = threading.Thread(target=first.do_POST)
            thread.start()
            self.assertTrue(started.wait(1))
            second.do_POST()
            release.set()
            thread.join(2)

        self.assertFalse(thread.is_alive())
        second.send_json_response.assert_called_once()
        self.assertEqual(second.send_json_response.call_args.args[0], 409)
        self.assertEqual(registry.active_operations(), [])


class TestBackendShutdownOwnership(unittest.TestCase):
    @staticmethod
    def _handler(owner_header: str):
        from agent import web

        handler = object.__new__(web.AgentHTTPRequestHandler)
        handler.path = "/api/shutdown"
        handler.headers = {
            "Host": "127.0.0.1:5005",
            "X-Selene-Backend-Owner": owner_header,
        }
        handler.server = MagicMock()
        handler.send_json_response = MagicMock()
        return web, handler

    def test_shutdown_rejects_non_owner(self):
        web, handler = self._handler("wrong-owner")
        with (
            patch.dict("os.environ", {"SELENE_BACKEND_OWNER": "electron-owner"}),
            patch.object(web.ACTIVE_GENERATIONS, "cancel_all") as cancel_all,
        ):
            handler.do_POST()

        handler.send_json_response.assert_called_once_with(
            403,
            {
                "status": "error",
                "error": "Backend shutdown ownership could not be verified",
            },
        )
        cancel_all.assert_not_called()

    def test_shutdown_accepts_exact_electron_owner(self):
        web, handler = self._handler("electron-owner")
        thread = MagicMock()
        with (
            patch.dict("os.environ", {"SELENE_BACKEND_OWNER": "electron-owner"}),
            patch.object(web.ACTIVE_GENERATIONS, "cancel_all") as cancel_all,
            patch.object(web.threading, "Thread", return_value=thread) as thread_factory,
        ):
            handler.do_POST()

        cancel_all.assert_called_once()
        handler.send_json_response.assert_called_once_with(202, {"status": "shutting-down"})
        thread_factory.assert_called_once_with(target=handler.server.shutdown, daemon=True)
        thread.start.assert_called_once_with()


class TestConversationTitleRename(unittest.TestCase):
    def test_temporary_session_patterns_include_uuid_suffix(self):
        from agent import web

        self.assertTrue(web.is_temporary_session_filename("session_20260711_012208.json"))
        self.assertTrue(web.is_temporary_session_filename("session_20260711_012208_123456.json"))
        self.assertTrue(
            web.is_temporary_session_filename("session_20260711_012208_123456_a1b2c3d4.json")
        )
        self.assertFalse(web.is_temporary_session_filename("Python_Tips_20260711_012208.json"))
        self.assertFalse(web.is_temporary_session_filename("Active Session"))

    def test_title_temporary_session_renames_uuid_temp_files(self):
        from agent import web

        with tempfile.TemporaryDirectory() as directory:
            temp_name = "session_20260711_012208_123456_a1b2c3d4.json"
            old_path = Path(directory) / temp_name
            old_path.write_text("{}", encoding="utf-8")
            history = [
                {"role": "user", "content": "How do I fix a Python import cycle?"},
                {"role": "assistant", "content": "Break the cycle with a local import."},
            ]
            with (
                patch.object(web, "_SESSIONS_DIR", directory),
                patch.object(web, "generate_conversation_title", return_value="Python Imports"),
            ):
                renamed = web.title_temporary_session(
                    history,
                    temp_name,
                    generation_id=None,
                    client_id="client-one",
                )

            self.assertEqual(renamed, "Python_Imports_20260711_012208_123456.json")
            self.assertFalse(old_path.exists())
            self.assertTrue((Path(directory) / renamed).is_file())


class TestRuntimeSystemPromptPayload(unittest.TestCase):
    def test_runtime_payload_exposes_default_and_active_system_prompt(self):
        from agent import web

        session = {
            "runtime_profile": "low-vram",
            "options": {},
            "system": "",
            "history": True,
            "think": True,
            "verbose": False,
            "wordwrap": True,
            "format": "",
        }
        with (
            patch.object(web, "load_default_system_prompt", return_value="DEFAULT SYSTEM POLICY"),
            patch.object(web, "get_runtime_config") as runtime_config,
            patch.object(web, "get_runtime_paths") as runtime_paths,
        ):
            runtime_config.return_value = SimpleNamespace(
                requested_profile=SimpleNamespace(value="auto"),
                profile=SimpleNamespace(value="low-vram"),
                selection_reason="test",
                warnings=[],
                ollama_options=lambda: {"num_ctx": 4096},
            )
            runtime_paths.return_value = SimpleNamespace(report=lambda: {"data_dir": "/tmp"})
            payload = web._runtime_payload(session)

        self.assertEqual(payload["default_system_prompt"], "DEFAULT SYSTEM POLICY")
        self.assertEqual(payload["active_system_prompt"], "DEFAULT SYSTEM POLICY")

        session["system"] = "  custom override  "
        with (
            patch.object(web, "load_default_system_prompt", return_value="DEFAULT SYSTEM POLICY"),
            patch.object(web, "get_runtime_config") as runtime_config,
            patch.object(web, "get_runtime_paths") as runtime_paths,
        ):
            runtime_config.return_value = SimpleNamespace(
                requested_profile=SimpleNamespace(value="auto"),
                profile=SimpleNamespace(value="low-vram"),
                selection_reason="test",
                warnings=[],
                ollama_options=lambda: {"num_ctx": 4096},
            )
            runtime_paths.return_value = SimpleNamespace(report=lambda: {"data_dir": "/tmp"})
            overridden = web._runtime_payload(session)

        self.assertEqual(overridden["default_system_prompt"], "DEFAULT SYSTEM POLICY")
        self.assertEqual(overridden["active_system_prompt"], "custom override")


if __name__ == "__main__":
    unittest.main()
