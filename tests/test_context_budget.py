import unittest
from unittest.mock import patch

from agent.core import (
    CONTEXT_TOOL_LOOP_RESERVE,
    ContextWindowError,
    OUTPUT_CONTINUATION_PROMPT,
    TOOL_CONTINUATION_PROMPT,
    _output_limit_reached,
    _stream_complete_response,
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

    def test_tool_continuation_tail_remains_atomic_when_trimming(self):
        tool_call = {
            "role": "assistant",
            "content": "discarded thinking text",
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
        self.assertEqual(prepared[-3]["content"], "")
        self.assertEqual(prepared[-2], tool_result)
        self.assertEqual(prepared[-1], reminder)

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

        with patch("ollama.chat", side_effect=AssertionError("unexpected model call")):
            _check_and_compact_history(history, session)

        self.assertNotIn("_is_compacting", session)
        self.assertEqual(history[0]["role"], "system")
        self.assertEqual([m["content"] for m in history if m.get("role") == "user"][-2:], [
            "request 3 " + ("x" * 1200),
            "request 4 " + ("x" * 1200),
        ])


if __name__ == "__main__":
    unittest.main()
