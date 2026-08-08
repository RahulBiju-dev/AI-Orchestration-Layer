from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.runtime_config import RuntimeConfigurationError
from agent.modes import (
    AGENT_MODE_SLASH_COMMANDS,
    DEEP_RESEARCH_COMPACT_MARKER,
    compact_deep_research_messages,
    force_hard_web_search_schema,
    force_high_tool_difficulty,
    normalize_agent_mode,
    parse_research_queries,
    research_query_count,
)


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "agent" / "static" / "app.js").read_text(encoding="utf-8")
HTML = (ROOT / "agent" / "static" / "index.html").read_text(encoding="utf-8")
STYLE = (ROOT / "agent" / "static" / "style.css").read_text(encoding="utf-8")


class AgentModePolicyTests(unittest.TestCase):
    def test_mode_normalization_is_strict(self):
        self.assertEqual(normalize_agent_mode(None), "normal")
        self.assertEqual(normalize_agent_mode("Deep_Research"), "deep-research")
        with self.assertRaises(ValueError):
            normalize_agent_mode("unbounded-magic")

    def test_direct_slash_commands_map_to_the_three_modes(self):
        self.assertEqual(
            AGENT_MODE_SLASH_COMMANDS,
            {
                "/fast": "normal",
                "/ultrathink": "ultra",
                "/deepresearch": "deep-research",
            },
        )

    def test_web_slash_commands_update_conversation_mode(self):
        from agent import web

        session = web._session_from_runtime(web.get_runtime_config())
        for command, expected in AGENT_MODE_SLASH_COMMANDS.items():
            result = web.execute_command_web(command, session, [])
            self.assertEqual(session["agent_mode"], expected)
            self.assertIn("Mode", result)

    def test_terminal_slash_commands_update_conversation_mode(self):
        from agent import core

        session = core._new_session_state()
        for command in AGENT_MODE_SLASH_COMMANDS:
            self.assertIn(command, core.CLI_SLASH_COMPLETIONS)
        with patch.object(core, "_refresh_tui_runtime_meta"):
            for command, expected in AGENT_MODE_SLASH_COMMANDS.items():
                self.assertTrue(core._handle_command(command, session, []))
                self.assertEqual(session["agent_mode"], expected)

    def test_terminal_deep_research_uses_mode_prompt_and_hard_search_schema(self):
        from agent import core

        session = core._new_session_state()
        session["agent_mode"] = "deep-research"
        history = []
        response = {"role": "assistant", "content": "done"}
        with (
            patch.object(core, "_stream_complete_response", return_value=response) as stream,
            patch.object(core, "_check_and_compact_history"),
        ):
            core.process_user_turn("compare the evidence", session, history, "system")

        request = stream.call_args.kwargs
        self.assertIn("Deep Research mode is active", request["messages"][-1]["content"])
        self.assertIn("compare the evidence", request["messages"][-1]["content"])
        web_search = next(
            tool for tool in request["tools"]
            if tool["function"]["name"] == "web_search"
        )
        difficulty = web_search["function"]["parameters"]["properties"]["difficulty"]
        self.assertEqual(difficulty["enum"], ["hard"])
        self.assertEqual(history[1]["content"], "compare the evidence")

    def test_terminal_enhanced_mode_forces_easy_search_call_to_hard(self):
        from agent import core

        session = core._new_session_state()
        session["agent_mode"] = "ultra"
        first = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "web_search",
                    "arguments": {"query": "topic", "difficulty": "easy"},
                },
            }],
        }
        final = {"role": "assistant", "content": "done"}
        process_tools = MagicMock(return_value=[{
            "role": "tool",
            "tool_name": "web_search",
            "name": "web_search",
            "content": "[]",
        }])
        with (
            patch.object(
                core,
                "_stream_complete_response",
                side_effect=[first, final],
            ) as stream,
            patch.object(core, "_process_tool_calls_with_turn_guard", process_tools),
            patch.object(core, "_check_and_compact_history"),
        ):
            core.process_user_turn("research topic", session, [], "system")

        calls = process_tools.call_args.args[0]
        self.assertEqual(calls[0]["function"]["arguments"]["difficulty"], "hard")
        continuation = stream.call_args_list[1].kwargs["messages"]
        self.assertTrue(any(
            message.get("role") == "user"
            and "Ultra Thinking mode is active" in message.get("content", "")
            for message in continuation
        ))

    def test_research_breadth_scales_with_context(self):
        self.assertEqual(research_query_count(4096), 3)
        self.assertEqual(research_query_count(8192), 4)
        self.assertEqual(research_query_count(16384), 5)
        self.assertEqual(research_query_count(32768), 6)
        self.assertEqual(research_query_count(65536), 8)

    def test_planner_queries_are_deduplicated_and_filled(self):
        planned = parse_research_queries(
            '{"queries":["topic evidence","TOPIC EVIDENCE",""]}',
            "topic",
            4,
        )
        self.assertEqual(len(planned), 4)
        self.assertEqual(planned[0], "topic evidence")
        self.assertEqual(len({query.casefold() for query in planned}), 4)

    def test_high_difficulty_is_enforced_without_mutating_input(self):
        calls = [{
            "function": {
                "name": "web_search",
                "arguments": {"query": "topic", "difficulty": "easy"},
            }
        }]
        hardened = force_high_tool_difficulty(calls)
        self.assertEqual(hardened[0]["function"]["arguments"]["difficulty"], "hard")
        self.assertEqual(calls[0]["function"]["arguments"]["difficulty"], "easy")

    def test_enhanced_mode_schema_only_allows_hard_web_search(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                        },
                    },
                },
            },
        }]
        hardened = force_hard_web_search_schema(tools)
        difficulty = hardened[0]["function"]["parameters"]["properties"]["difficulty"]
        self.assertEqual(difficulty["enum"], ["hard"])
        self.assertEqual(difficulty["default"], "hard")
        self.assertEqual(
            tools[0]["function"]["parameters"]["properties"]["difficulty"]["enum"],
            ["easy", "medium", "hard"],
        )

    def test_deep_research_compaction_keeps_exact_request_and_bounded_evidence(self):
        original_request = "Compare the evidence without losing this exact request"
        prior_call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "web_search",
                    "arguments": {"query": "unrelated earlier turn"},
                }
            }],
        }
        current_calls = {
            "role": "assistant",
            "content": "searching",
            "tool_calls": [
                {
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": f"current query {index}"},
                    }
                }
                for index in range(3)
            ],
        }

        def result(query: str, index: int) -> dict:
            return {
                "role": "tool",
                "tool_name": "web_search",
                "name": "web_search",
                "content": (
                    '[{"title":"Source %d","url":"https://example.com/%d",'
                    '"snippet":"%s"}]'
                ) % (index, index, query + " evidence " * 500),
            }

        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "Earlier unrelated request"},
            prior_call,
            result("unrelated", 99),
            {"role": "user", "content": original_request},
            current_calls,
            *(result(f"current query {index}", index) for index in range(3)),
        ]
        compacted, search_count = compact_deep_research_messages(
            messages,
            original_request,
            max_checkpoint_chars=3000,
        )

        self.assertEqual(search_count, 3)
        self.assertEqual(
            sum(message.get("content") == original_request for message in compacted),
            1,
        )
        checkpoints = [
            message["content"] for message in compacted
            if DEEP_RESEARCH_COMPACT_MARKER in str(message.get("content") or "")
        ]
        self.assertEqual(len(checkpoints), 1)
        self.assertIn("3 search(es), 0 scrape(s)", checkpoints[0])
        self.assertIn("https://example.com/2", checkpoints[0])
        self.assertLess(len(checkpoints[0]), 3300)
        self.assertTrue(any(
            message.get("role") == "tool" and "example.com/99" in message.get("content", "")
            for message in compacted
        ))
        self.assertEqual(len(messages), 9)
        self.assertEqual(len(messages[-1]["content"]), len(result("current query 2", 2)["content"]))

    def test_web_session_validation_persists_only_known_modes(self):
        from agent import web

        normalized = web._normalize_session_settings({"agent_mode": "ultra"})
        self.assertEqual(normalized["agent_mode"], "ultra")
        with self.assertRaises(RuntimeConfigurationError):
            web._normalize_session_settings({"agent_mode": "invented"})


class AgentModeFrontendTests(unittest.TestCase):
    def test_composer_mode_menu_and_clear_control_are_wired(self):
        self.assertIn('id="mode-trigger"', HTML)
        self.assertIn('id="mode-menu"', HTML)
        self.assertIn('id="mode-clear"', HTML)
        self.assertIn('data-agent-mode="ultra"', HTML)
        self.assertIn('data-agent-mode="deep-research"', HTML)
        self.assertIn('<span id="mode-label">Fast</span>', HTML)
        self.assertIn('<strong>Fast</strong>', HTML)
        self.assertNotIn('<strong>Normal</strong>', HTML)
        self.assertIn("function setAgentMode(mode)", APP)
        self.assertIn('agent_mode: "normal"', APP)
        self.assertIn('normal: { label: "Fast"', APP)
        self.assertIn('{ command: "/fast"', APP)
        self.assertIn('{ command: "/ultrathink"', APP)
        self.assertIn('{ command: "/deepresearch"', APP)
        self.assertIn("await settingsWriteChain;", APP)
        self.assertIn('appendStatus(event.message || "", event.activity_mode || "")', APP)
        self.assertIn("function appendModeActivity(text, activityMode)", APP)
        self.assertIn('activity.className = "mode-activity-inline running"', APP)
        self.assertIn('querySelector(".block-title")?.appendChild(activity)', APP)
        self.assertIn('runtime_profile: "manual"', APP)
        self.assertIn(".mode-clear[hidden]", STYLE)
        self.assertIn(".mode-menu[hidden]", STYLE)
        self.assertIn(".mode-activity-inline", STYLE)
        self.assertIn("@keyframes mode-text-shine", STYLE)
        self.assertIn("@keyframes mode-text-flash", STYLE)
        self.assertIn("background: color-mix(in srgb, var(--elevated) 96%, var(--bg))", STYLE)
        self.assertIn("background: color-mix(in srgb, var(--option-tone) 14%, var(--surface))", STYLE)
        self.assertNotIn("rgba(10, 14, 27, 0.98)", STYLE)
        self.assertNotIn("var(--accent-2)", STYLE)


if __name__ == "__main__":
    unittest.main()
