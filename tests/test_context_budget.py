import json
import unittest
from unittest.mock import patch

from agent.core import (
    CONTEXT_TOOL_LOOP_RESERVE,
    ContextWindowError,
    OUTPUT_CONTINUATION_PROMPT,
    SYSTEM_PROMPT_ANCHOR_THRESHOLD,
    TOOL_CONTINUATION_PROMPT,
    _new_vault_index_loop_state,
    _output_limit_reached,
    _should_reexecute_turn_duplicate,
    _stream_complete_response,
    _tool_loop_stop_message,
    _update_vault_index_loop_state,
    build_tool_continuation_prompt,
    _check_and_compact_history,
    _context_safety_margin,
    _estimate_messages_tokens,
    _estimate_tool_schema_tokens,
    _estimate_tokens,
    guarded_options_for_call,
    load_default_system_prompt,
    prepare_messages_for_model,
    tool_schemas_for_model,
    validate_session_options,
)
from agent.runtime_config import RuntimeConfigurationError
from tools.registry import TOOL_SCHEMAS


class TestContextBudget(unittest.TestCase):
    def test_output_limit_detection_prefers_explicit_terminal_reason(self):
        self.assertTrue(_output_limit_reached("length", 10, 768))
        self.assertTrue(_output_limit_reached("", 768, 768))
        self.assertFalse(_output_limit_reached("stop", 768, 768))

    def test_length_limited_cli_segments_become_one_response(self):
        segments = [
            {
                "role": "assistant",
                "content": "Once upon a ",
                "thinking": "initial thought",
                "_done_reason": "length",
                "_eval_count": 768,
                "_num_predict": 768,
            },
            {
                "role": "assistant",
                "content": "time.",
                "_done_reason": "stop",
                "_eval_count": 12,
                "_num_predict": 768,
            },
        ]
        session = {"options": {}, "verbose": False, "think": True, "format": ""}

        with (
            patch("agent.core._stream_thinking_response", side_effect=segments) as stream,
            patch("agent.core.prepare_messages_for_model", side_effect=lambda messages, *args, **kwargs: messages),
            patch("agent.core.effective_session_model_options", return_value=(None, {})),
        ):
            response = _stream_complete_response(
                model="selene",
                messages=[{"role": "user", "content": "Write a story"}],
                session=session,
                user_input="Write a story",
                tools=[{"type": "function", "function": {"name": "unused"}}],
            )

        self.assertEqual(response["content"], "Once upon a time.")
        self.assertEqual(response["thinking"], "initial thought")
        self.assertNotIn("_done_reason", response)
        self.assertEqual(stream.call_count, 2)
        self.assertFalse(stream.call_args_list[1].kwargs["think"])
        self.assertIsNone(stream.call_args_list[1].kwargs["tools"])

    def test_low_vram_first_turn_with_relevant_tools_fits_before_generation(self):
        session = {"runtime_profile": "low-vram", "options": {}}
        messages = [
            {"role": "system", "content": load_default_system_prompt()},
            {"role": "user", "content": "Open Spotify and play a song"},
        ]

        tools = tool_schemas_for_model(messages, session, TOOL_SCHEMAS)
        prepared = prepare_messages_for_model(messages, session, TOOL_SCHEMAS)
        options = guarded_options_for_call(
            prepared,
            {"num_ctx": 4096, "num_predict": 768},
            tools,
        )
        names = {tool["function"]["name"] for tool in tools}
        projected = (
            _estimate_messages_tokens(prepared)
            + _estimate_tool_schema_tokens(tools)
            + _context_safety_margin(4096)
            + options["num_predict"]
        )

        self.assertLessEqual(len(tools), 10)
        self.assertIn("spotify_play", names)
        self.assertLessEqual(projected, 4096)

    def test_unicode_token_estimate_is_not_ascii_underweighted(self):
        self.assertGreaterEqual(_estimate_tokens("漢" * 100), 100)

    def test_full_context_repeats_exact_system_prompt_near_latest_turn(self):
        system_prompt = "CUSTOM POLICY: always preserve this exact instruction."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "old context " * 2_000},
            {"role": "assistant", "content": "old answer " * 2_000},
            {"role": "user", "content": "What policy are you following?"},
        ]

        prepared = prepare_messages_for_model(
            messages,
            {"options": {"num_ctx": 2048, "num_predict": 256}},
            tools=None,
        )

        system_messages = [message for message in prepared if message["role"] == "system"]
        self.assertEqual(
            [message["content"] for message in system_messages],
            [system_prompt, system_prompt],
        )
        self.assertEqual(prepared[-2], {"role": "system", "content": system_prompt})
        self.assertEqual(prepared[-1], messages[-1])

    def test_short_context_does_not_duplicate_system_prompt(self):
        messages = [
            {"role": "system", "content": "system policy"},
            {"role": "user", "content": "hello"},
        ]
        session = {"options": {"num_ctx": 4096, "num_predict": 256}}

        prepared = prepare_messages_for_model(messages, session, tools=None)
        projected = (
            _estimate_messages_tokens(messages)
            + _context_safety_margin(4096)
            + 256
        )

        self.assertLess(projected, int(4096 * SYSTEM_PROMPT_ANCHOR_THRESHOLD))
        self.assertEqual(prepared, messages)

    def test_tool_continuation_tail_remains_atomic_when_trimming(self):
        tool_call = {
            "role": "assistant",
            "content": "discarded thinking text",
            "provider_metadata": {
                "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
            },
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"file_path": "README.md"}}}
            ],
        }
        tool_result = {
            "role": "tool",
            "tool_name": "read_file",
            "name": "read_file",
            "content": "bounded result",
        }
        reminder = {
            "role": "user",
            "content": TOOL_CONTINUATION_PROMPT.format(user_input="Explain the project"),
        }
        messages = [
            {"role": "system", "content": "system policy"},
            {"role": "user", "content": "old " * 4000},
            {"role": "assistant", "content": "old response " * 4000},
            tool_call,
            tool_result,
            reminder,
        ]

        prepared = prepare_messages_for_model(
            messages,
            {"options": {"num_ctx": 2048, "num_predict": 256}},
        )

        self.assertEqual(prepared[-3]["tool_calls"], tool_call["tool_calls"])
        self.assertEqual(
            prepared[-3]["provider_metadata"],
            tool_call["provider_metadata"],
        )
        self.assertEqual(prepared[-3]["content"], "")
        self.assertEqual(prepared[-2], tool_result)
        self.assertEqual(prepared[-1], reminder)
        self.assertEqual(prepared[-4], {"role": "system", "content": "system policy"})
        self.assertEqual(
            [message["content"] for message in prepared if message["role"] == "system"],
            ["system policy", "system policy"],
        )

    def test_output_continuation_keeps_latest_partial_answer_suffix(self):
        partial = "opening " + ("middle " * 3000) + "exact final sentence fragment"
        reminder = {
            "role": "user",
            "content": OUTPUT_CONTINUATION_PROMPT.format(user_input="Write a long story"),
        }
        prepared = prepare_messages_for_model(
            [
                {"role": "system", "content": "system policy"},
                {"role": "user", "content": "Write a long story"},
                {"role": "assistant", "content": partial},
                reminder,
            ],
            {"options": {"num_ctx": 2048, "num_predict": 768}},
            tools=None,
        )

        self.assertEqual(prepared[-1], reminder)
        self.assertEqual(prepared[-2]["role"], "assistant")
        self.assertEqual(prepared[-3], {"role": "system", "content": "system policy"})
        self.assertTrue(partial.endswith(prepared[-2]["content"]))
        self.assertIn("exact final sentence fragment", prepared[-2]["content"])
        self.assertLess(len(prepared[-2]["content"]), len(partial))

    def test_oversized_tool_result_yields_space_for_followup_answer(self):
        tool_call = {
            "role": "assistant",
            "content": "searching",
            "tool_calls": [
                {"function": {"name": "web_search", "arguments": {"query": "AI news"}}}
            ],
        }
        tool_result = {
            "role": "tool",
            "tool_name": "web_search",
            "name": "web_search",
            "content": "result " * 4_000,
        }
        reminder = {
            "role": "user",
            "content": TOOL_CONTINUATION_PROMPT.format(user_input="Summarize the latest AI news"),
        }
        messages = [
            {"role": "system", "content": load_default_system_prompt()},
            {"role": "user", "content": "Summarize the latest AI news"},
            tool_call,
            tool_result,
            reminder,
        ]
        session = {"runtime_profile": "low-vram", "options": {}}

        prepared = prepare_messages_for_model(
            messages,
            session,
            tools=TOOL_SCHEMAS,
            extra_reserved_tokens=CONTEXT_TOOL_LOOP_RESERVE,
        )
        runtime_tools = tool_schemas_for_model(prepared, session, TOOL_SCHEMAS)
        options = guarded_options_for_call(
            prepared,
            {"num_ctx": 4096, "num_predict": 768},
            runtime_tools,
            extra_reserved_tokens=CONTEXT_TOOL_LOOP_RESERVE,
        )

        self.assertIn("Tool result truncated", prepared[-2]["content"])
        self.assertEqual(tool_result["content"], "result " * 4_000)
        self.assertGreaterEqual(options["num_predict"], 96)
        system_messages = [message for message in prepared if message["role"] == "system"]
        self.assertEqual(len(system_messages), 1)
        self.assertEqual(system_messages[0]["content"], load_default_system_prompt())
        self.assertEqual(prepared[-4], system_messages[0])

    def test_vault_result_keeps_parseable_resume_control_under_context_pressure(self):
        file_path = "/run/media/rahulb/Rahul Files/DSA/Reference books/DSA_notes.pdf"
        call = {
            "function": {
                "name": "index_vault",
                "arguments": {
                    "action": "index",
                    "collection": "DSA",
                    "file_path": file_path,
                    "vision_mode": "all",
                    "max_pages": 20,
                },
            }
        }
        continuation_arguments = {
            "action": "index",
            "collection": "DSA",
            "file_path": file_path,
            "vision_mode": "all",
            "max_pages": 20,
            "chunk_size": 1800,
            "chunk_overlap": 250,
            "resume_page": 21,
        }
        job = {
            "complete": False,
            "next_page": 21,
            "source": "DSA_notes.pdf",
            "page_count": 110,
            "indexed_pages": 20,
            "indexed_chunks": 39,
            "vision_pages": 7,
            "vision_failed_pages": list(range(1, 51)),
            "warning_count": 50,
            "warnings": ["Ollama returned an empty response " + ("x" * 80)] * 50,
            "fingerprint": "fingerprint",
        }
        raw_result = json.dumps({
            "complete": False,
            "continuation_required": True,
            "next_page": 21,
            "continuation": {
                "tool": "index_vault",
                "arguments": continuation_arguments,
            },
            "collection": "DSA",
            "indexed_files": 0,
            "indexed_chunks": 39,
            "pdf_jobs": [job],
            "incomplete_pdf_count": 1,
            "guidance": "resume " * 100,
        })
        self.assertGreater(len(raw_result), 3_187)
        original_request = (
            "Index the large handwritten PDF into DSA with the vision model "
            "and do not finish until every page is indexed."
        )
        state = _new_vault_index_loop_state()
        tool_result = {
            "role": "tool",
            "tool_name": "index_vault",
            "name": "index_vault",
            "content": raw_result,
        }
        _update_vault_index_loop_state(state, [call], [tool_result])
        reminder = {
            "role": "user",
            "content": build_tool_continuation_prompt(original_request, state),
        }
        prepared = prepare_messages_for_model(
            [
                {"role": "system", "content": load_default_system_prompt()},
                {"role": "user", "content": "old context " * 3_000},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                tool_result,
                reminder,
            ],
            {"runtime_profile": "low-vram", "options": {}},
            tools=TOOL_SCHEMAS,
            extra_reserved_tokens=CONTEXT_TOOL_LOOP_RESERVE,
        )

        compact = json.loads(prepared[-2]["content"])
        self.assertFalse(compact["complete"])
        self.assertEqual(compact["next_page"], 21)
        self.assertEqual(compact["incomplete_pdf_count"], 1)
        self.assertEqual(compact["pdf_jobs"][0]["vision_failed_count"], 50)
        self.assertIn("Ollama returned an empty response", compact["pdf_jobs"][0]["warnings"][0])
        self.assertEqual(compact["continuation"]["arguments"], continuation_arguments)
        self.assertIn(
            json.dumps(continuation_arguments, ensure_ascii=False, separators=(",", ":")),
            prepared[-1]["content"],
        )
        self.assertIn(original_request, prepared[-1]["content"])
        self.assertNotIn("Tool result truncated", prepared[-2]["content"])

    def test_vault_rounds_exceed_cap_only_while_checkpoint_progresses(self):
        state = _new_vault_index_loop_state()
        current_call = None
        for page in range(1, 11):
            arguments = {
                "action": "index",
                "collection": "DSA",
                "file_path": "/tmp/DSA_notes.pdf",
                "vision_mode": "all",
                "max_pages": 1,
            }
            if page > 1:
                arguments["resume_page"] = page
            current_call = {"function": {"name": "index_vault", "arguments": arguments}}
            next_arguments = {
                **arguments,
                "resume_page": page + 1,
            }
            payload = {
                "complete": False,
                "continuation_required": True,
                "next_page": page + 1,
                "continuation": {"tool": "index_vault", "arguments": next_arguments},
                "collection": "DSA",
                "incomplete_pdf_count": 1,
                "pdf_jobs": [{
                    "complete": False,
                    "next_page": page + 1,
                    "fingerprint": "same-file",
                    "indexed_pages": page,
                    "indexed_chunks": page,
                    "vision_pages": page,
                    "vision_failed_count": 0,
                }],
            }
            _update_vault_index_loop_state(
                state,
                [current_call],
                [{"role": "tool", "name": "index_vault", "content": json.dumps(payload)}],
            )

        next_call = {
            "function": {
                "name": "index_vault",
                "arguments": state["expected_arguments"],
            }
        }
        self.assertIsNone(_tool_loop_stop_message(8, [next_call], state))
        generic = {"function": {"name": "read_file", "arguments": {"file_path": "README.md"}}}
        self.assertIn("Stopped after 8", _tool_loop_stop_message(8, [generic], state))

        repeated_payload = {
            "complete": False,
            "continuation_required": True,
            "next_page": 11,
            "continuation": {"tool": "index_vault", "arguments": state["expected_arguments"]},
            "collection": "DSA",
            "incomplete_pdf_count": 1,
            "pdf_jobs": [{
                "complete": False,
                "next_page": 11,
                "fingerprint": "same-file",
                "indexed_pages": 10,
                "indexed_chunks": 10,
                "vision_pages": 10,
                "vision_failed_count": 0,
            }],
        }
        _update_vault_index_loop_state(
            state,
            [next_call],
            [{"role": "tool", "name": "index_vault", "content": json.dumps(repeated_payload)}],
        )
        self.assertIn("checkpoint repeated", _tool_loop_stop_message(9, [next_call], state))

    def test_vault_cap_exemption_requires_exact_error_free_continuation(self):
        state = _new_vault_index_loop_state()
        expected = {
            "action": "index",
            "collection": "DSA",
            "file_path": "/tmp/DSA_notes.pdf",
            "vision_mode": "all",
            "max_pages": 1,
            "chunk_size": 1800,
            "chunk_overlap": 250,
            "resume_page": 2,
        }
        first_call = {
            "function": {
                "name": "index_vault",
                "arguments": {**expected, "resume_page": 1},
            }
        }
        payload = {
            "complete": False,
            "continuation_required": True,
            "next_page": 2,
            "continuation": {"tool": "index_vault", "arguments": expected},
            "collection": "DSA",
            "incomplete_pdf_count": 1,
            "pdf_jobs": [{
                "complete": False,
                "next_page": 2,
                "fingerprint": "same-file",
                "indexed_pages": 1,
                "indexed_chunks": 1,
                "vision_pages": 1,
                "vision_failed_count": 0,
            }],
        }
        _update_vault_index_loop_state(
            state,
            [first_call],
            [{"role": "tool", "name": "index_vault", "content": json.dumps(payload)}],
        )

        exact_call = {"function": {"name": "index_vault", "arguments": expected}}
        malformed_call = {
            "function": {
                "name": "index_vault",
                "arguments": {**expected, "invented": True},
            }
        }
        self.assertIsNone(_tool_loop_stop_message(8, [exact_call], state))
        self.assertIn("Stopped after 8", _tool_loop_stop_message(8, [malformed_call], state))

        error_payload = {"complete": False, "error": "embedding timeout"}
        _update_vault_index_loop_state(
            state,
            [exact_call],
            [{"role": "tool", "name": "index_vault", "content": json.dumps(error_payload)}],
        )
        self.assertIsNone(state["expected_arguments"])
        self.assertIn("embedding timeout", _tool_loop_stop_message(8, [exact_call], state))

    def test_failed_vision_recovery_can_progress_beyond_eight_rounds(self):
        state = _new_vault_index_loop_state()
        expected = {
            "action": "index",
            "collection": "DSA",
            "file_path": "/tmp/DSA_notes.pdf",
            "vision_mode": "all",
            "max_pages": 1,
            "chunk_size": 1800,
            "chunk_overlap": 250,
            "resume_page": 1,
        }
        for recovered in range(10):
            call = {"function": {"name": "index_vault", "arguments": expected}}
            next_arguments = {**expected, "resume_page": recovered + 2}
            payload = {
                "complete": False,
                "continuation_required": True,
                "next_page": recovered + 2,
                "continuation": {"tool": "index_vault", "arguments": next_arguments},
                "collection": "DSA",
                "incomplete_pdf_count": 1,
                "pdf_jobs": [{
                    "complete": False,
                    "next_page": recovered + 2,
                    "fingerprint": "same-file",
                    "indexed_pages": 110,
                    "indexed_chunks": 39,
                    "vision_pages": 38 + recovered,
                    "vision_failed_count": 72 - recovered,
                }],
            }
            _update_vault_index_loop_state(
                state,
                [call],
                [{"role": "tool", "name": "index_vault", "content": json.dumps(payload)}],
            )
            expected = next_arguments

        next_call = {"function": {"name": "index_vault", "arguments": expected}}
        self.assertIsNone(_tool_loop_stop_message(10, [next_call], state))

    def test_truncated_vault_wrapper_is_never_compacted_to_success(self):
        from agent.core import _compact_index_vault_content_for_context

        compact = json.loads(_compact_index_vault_content_for_context(json.dumps({
            "ok": True,
            "truncated": True,
            "tool": "index_vault",
            "content_preview": "{\"complete\":false",
        })))

        self.assertFalse(compact["complete"])
        self.assertTrue(compact["truncated"])
        self.assertIn("truncated", compact["error"])

    def test_source_fingerprint_change_blocks_extended_vault_loop(self):
        state = _new_vault_index_loop_state()
        arguments = {
            "action": "index",
            "collection": "DSA",
            "file_path": "/tmp/DSA_notes.pdf",
            "resume_page": 2,
        }
        call = {"function": {"name": "index_vault", "arguments": arguments}}

        def result(fingerprint, next_page, indexed_pages):
            continuation = {**arguments, "resume_page": next_page}
            return {
                "role": "tool",
                "name": "index_vault",
                "content": json.dumps({
                    "complete": False,
                    "continuation_required": True,
                    "next_page": next_page,
                    "continuation": {"tool": "index_vault", "arguments": continuation},
                    "collection": "DSA",
                    "incomplete_pdf_count": 1,
                    "pdf_jobs": [{
                        "complete": False,
                        "next_page": next_page,
                        "fingerprint": fingerprint,
                        "indexed_pages": indexed_pages,
                        "indexed_chunks": indexed_pages,
                        "vision_pages": indexed_pages,
                        "vision_failed_count": 0,
                    }],
                }),
            }

        _update_vault_index_loop_state(state, [call], [result("first", 2, 1)])
        _update_vault_index_loop_state(state, [call], [result("replacement", 2, 1)])

        self.assertIsNone(state["expected_arguments"])
        self.assertIn("source PDF changed", state["blocked_reason"])

    def test_incomplete_vault_duplicate_is_reexecuted_but_completed_one_is_cached(self):
        call = {
            "function": {
                "name": "index_vault",
                "arguments": {
                    "action": "index",
                    "collection": "DSA",
                    "file_path": "/tmp/DSA_notes.pdf",
                    "resume_page": 1,
                },
            }
        }
        incomplete = {
            "role": "tool",
            "name": "index_vault",
            "content": json.dumps({
                "complete": False,
                "continuation_required": True,
                "incomplete_pdf_count": 1,
            }),
        }
        completed = {
            **incomplete,
            "content": json.dumps({
                "complete": True,
                "continuation_required": False,
                "incomplete_pdf_count": 0,
            }),
        }

        self.assertTrue(_should_reexecute_turn_duplicate(call, incomplete))
        self.assertFalse(_should_reexecute_turn_duplicate(call, completed))

    def test_cli_executes_more_than_eight_progressing_vault_rounds(self):
        from agent import core

        arguments = {
            "action": "index",
            "collection": "DSA",
            "file_path": "/tmp/DSA_notes.pdf",
            "vision_mode": "all",
            "max_pages": 1,
        }
        assistant_messages = [{
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {"name": "index_vault", "arguments": arguments}
                }],
            }, {
            "role": "assistant",
            "content": "The full document is indexed.",
        }]
        executed = 0

        def execute(calls, cache):
            nonlocal executed
            executed += 1
            complete = executed == 10
            next_page = None if complete else executed + 1
            continuation_arguments = {
                "action": "index",
                "collection": "DSA",
                "file_path": "/tmp/DSA_notes.pdf",
                "vision_mode": "all",
                "max_pages": 1,
                "resume_page": next_page,
            } if next_page is not None else None
            payload = {
                "complete": complete,
                "continuation_required": not complete,
                "next_page": next_page,
                "continuation": (
                    {"tool": "index_vault", "arguments": continuation_arguments}
                    if continuation_arguments else None
                ),
                "collection": "DSA",
                "incomplete_pdf_count": 0 if complete else 1,
                "pdf_jobs": [{
                    "complete": complete,
                    "next_page": next_page,
                    "fingerprint": "same-file",
                    "indexed_pages": executed,
                    "indexed_chunks": executed,
                    "vision_pages": executed,
                    "vision_failed_count": 0,
                }],
            }
            return [{
                "role": "tool",
                "tool_name": "index_vault",
                "name": "index_vault",
                "content": json.dumps(payload),
            }]

        session = {
            "history": True,
            "system": "",
            "options": {},
            "verbose": False,
            "think": False,
            "format": "",
        }
        history = []
        with (
            patch.object(core, "_stream_complete_response", side_effect=assistant_messages),
            patch.object(core, "_process_tool_calls_with_turn_guard", side_effect=execute),
            patch.object(core, "prepare_messages_for_model", side_effect=lambda messages, *args, **kwargs: list(messages)),
            patch.object(core, "_check_and_compact_history"),
        ):
            core.process_user_turn(
                "Index all handwritten pages",
                session,
                history,
                default_system_prompt="",
            )

        self.assertEqual(executed, 10)
        self.assertEqual(history[-1]["content"], "The full document is indexed.")

    def test_cli_executes_verified_continuation_without_model_interruption(self):
        from agent import core

        initial_arguments = {
            "action": "index",
            "collection": "DSA",
            "file_path": "/tmp/DSA_notes.pdf",
            "vision_mode": "all",
            "max_pages": 1,
        }
        expected_arguments = {
            **initial_arguments,
            "chunk_size": 1800,
            "chunk_overlap": 250,
            "resume_page": 2,
        }
        assistant_messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {"name": "index_vault", "arguments": initial_arguments}
                }],
            },
            {"role": "assistant", "content": "The document is now fully indexed."},
        ]
        executed_arguments = []

        def execute(calls, cache):
            arguments = calls[0]["function"]["arguments"]
            executed_arguments.append(arguments)
            complete = len(executed_arguments) == 2
            payload = {
                "complete": complete,
                "continuation_required": not complete,
                "next_page": None if complete else 2,
                "continuation": None if complete else {
                    "tool": "index_vault",
                    "arguments": expected_arguments,
                },
                "collection": "DSA",
                "incomplete_pdf_count": 0 if complete else 1,
                "pdf_jobs": [{
                    "complete": complete,
                    "next_page": None if complete else 2,
                    "fingerprint": "same-file",
                    "indexed_pages": len(executed_arguments),
                    "indexed_chunks": len(executed_arguments),
                    "vision_pages": len(executed_arguments),
                    "vision_failed_count": 0,
                }],
            }
            return [{
                "role": "tool",
                "tool_name": "index_vault",
                "name": "index_vault",
                "content": json.dumps(payload),
            }]

        session = {
            "history": True,
            "system": "",
            "options": {},
            "verbose": False,
            "think": False,
            "format": "",
        }
        history = []
        with (
            patch.object(core, "_stream_complete_response", side_effect=assistant_messages),
            patch.object(core, "_process_tool_calls_with_turn_guard", side_effect=execute),
            patch.object(
                core,
                "prepare_messages_for_model",
                side_effect=lambda messages, *args, **kwargs: list(messages),
            ),
            patch.object(core, "_check_and_compact_history"),
        ):
            core.process_user_turn(
                "Index every page",
                session,
                history,
                default_system_prompt="",
            )

        self.assertEqual(executed_arguments, [initial_arguments, expected_arguments])
        self.assertEqual(history[-1]["content"], "The document is now fully indexed.")

    def test_cli_vault_error_ends_without_unmatched_tool_call(self):
        from agent import core

        call = {
            "function": {
                "name": "index_vault",
                "arguments": {
                    "action": "index",
                    "collection": "DSA",
                    "file_path": "/tmp/DSA_notes.pdf",
                },
            }
        }
        assistant_messages = [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "assistant", "content": "Indexed everything."},
        ]
        error_result = [{
            "role": "tool",
            "tool_name": "index_vault",
            "name": "index_vault",
            "content": json.dumps({
                "complete": False,
                "error": "embedding timeout",
                "incomplete_pdf_count": 1,
            }),
        }]
        session = {
            "history": True,
            "system": "",
            "options": {},
            "verbose": False,
            "think": False,
            "format": "",
        }
        history = []
        with (
            patch.object(core, "_stream_complete_response", side_effect=assistant_messages),
            patch.object(
                core,
                "_process_tool_calls_with_turn_guard",
                return_value=error_result,
            ),
            patch.object(
                core,
                "prepare_messages_for_model",
                side_effect=lambda messages, *args, **kwargs: list(messages),
            ),
            patch.object(core, "_check_and_compact_history"),
        ):
            core.process_user_turn(
                "Index every page",
                session,
                history,
                default_system_prompt="",
            )

        pending_calls = sum(bool(item.get("tool_calls")) for item in history)
        tool_results = sum(item.get("role") == "tool" for item in history)
        self.assertEqual(pending_calls, tool_results)
        self.assertNotIn("tool_calls", history[-1])
        self.assertIn("embedding timeout", history[-1]["content"])

    def test_tool_continuation_selection_uses_original_request_not_boilerplate(self):
        messages = [
            {"role": "user", "content": "Summarize the latest AI news"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {"name": "web_search", "arguments": {"query": "AI news"}}
                }],
            },
            {
                "role": "user",
                "content": TOOL_CONTINUATION_PROMPT.format(
                    user_input="Summarize the latest AI news"
                ),
            },
        ]

        selected = tool_schemas_for_model(
            messages,
            {"runtime_profile": "low-vram", "options": {}},
            TOOL_SCHEMAS,
        )
        names = {item["function"]["name"] for item in selected}

        self.assertIn("web_search", names)
        self.assertIn("web_scrape", names)
        self.assertNotIn("api_orchestrator", names)
        self.assertNotIn("context_memory_optimizer", names)

    def test_num_predict_is_capped_to_remaining_context(self):
        messages = [{"role": "user", "content": "x" * 5000}]

        options = guarded_options_for_call(
            messages,
            {"num_ctx": 2048, "num_predict": 1400},
        )

        self.assertLess(options["num_predict"], 1400)
        self.assertGreaterEqual(options["num_predict"], 96)

    def test_unavoidable_context_overflow_is_controlled(self):
        messages = [{"role": "system", "content": "x" * 12000}]

        with self.assertRaises(ContextWindowError):
            guarded_options_for_call(messages, {"num_ctx": 1024, "num_predict": 256})

    def test_invalid_session_parameter_is_rejected(self):
        with self.assertRaises(RuntimeConfigurationError):
            validate_session_options({"num_ctx": 512})

    def test_history_compaction_does_not_start_secondary_ollama_call(self):
        history = [{"role": "system", "content": "policy"}]
        for index in range(5):
            history.extend([
                {"role": "user", "content": f"request {index} " + ("x" * 1200)},
                {"role": "assistant", "content": f"answer {index} " + ("y" * 1200)},
            ])
        session = {"options": {"num_ctx": 2048}}

        from tools.context_memory_optimizer import context_memory_optimizer

        with (
            patch("ollama.chat", side_effect=AssertionError("unexpected model call")),
            patch(
                "tools.context_memory_optimizer.context_memory_optimizer",
                wraps=context_memory_optimizer,
            ) as optimize,
        ):
            _check_and_compact_history(history, session)

        self.assertEqual(optimize.call_count, 1)
        self.assertNotIn("_is_compacting", session)
        self.assertEqual(history[0]["role"], "system")
        self.assertEqual([m["content"] for m in history if m.get("role") == "user"][-2:], [
            "request 3 " + ("x" * 1200),
            "request 4 " + ("x" * 1200),
        ])


if __name__ == "__main__":
    unittest.main()
