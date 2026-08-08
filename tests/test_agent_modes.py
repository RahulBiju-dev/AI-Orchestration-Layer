from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.runtime_config import RuntimeConfigurationError
from agent.modes import (
    AGENT_MODE_SLASH_COMMANDS,
    DEEP_RESEARCH_COMPACT_MARKER,
    compact_deep_research_messages,
    extract_research_sources,
    force_hard_web_search_schema,
    force_high_tool_difficulty,
    merge_research_sources,
    normalize_agent_mode,
    parse_research_queries,
    research_query_count,
    research_sources_summary,
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


class ResearchCitationTests(unittest.TestCase):
    def test_search_results_become_sources_with_query_provenance(self):
        payload = json.dumps([
            {
                "title": "Ignition report",
                "url": "https://example.com/report",
                "snippet": "The facility reached ignition.",
                "content": {"description": "full page"},
            },
            {"title": "Commentary", "url": "https://blog.test/post", "snippet": "opinion"},
        ])
        sources = extract_research_sources(
            "web_search",
            {"query": "fusion ignition evidence", "difficulty": "hard"},
            payload,
        )

        self.assertEqual([source["url"] for source in sources], [
            "https://example.com/report",
            "https://blog.test/post",
        ])
        self.assertEqual(sources[0]["host"], "example.com")
        self.assertEqual(sources[0]["query"], "fusion ignition evidence")
        self.assertTrue(sources[0]["fetched"])
        self.assertFalse(sources[1]["fetched"])

    def test_only_public_http_sources_are_citable(self):
        payload = json.dumps({"results": [
            {"title": "script", "url": "javascript:alert(1)"},
            {"title": "file", "url": "file:///etc/passwd"},
            {"title": "credentials", "url": "https://user:pass@example.com/x"},
            {"title": "ok", "url": "https://example.org/a"},
        ]})
        sources = extract_research_sources("web_search", {"query": "q"}, payload)
        self.assertEqual([source["url"] for source in sources], ["https://example.org/a"])

    def test_non_research_tools_and_unparsable_results_cite_nothing(self):
        self.assertEqual(extract_research_sources("read_file", {}, "{}"), [])
        self.assertEqual(extract_research_sources("web_search", {}, "not json"), [])
        self.assertEqual(
            extract_research_sources("web_scrape", {"url": "https://x.io"}, '{"error":"blocked"}'),
            [],
        )

    def test_repeat_sources_merge_and_a_scrape_upgrades_the_search_hit(self):
        collected: list[dict] = []
        merge_research_sources(collected, extract_research_sources(
            "web_search",
            {"query": "first"},
            json.dumps([{"title": "Report", "url": "https://example.com/report/"}]),
        ))
        merge_research_sources(collected, extract_research_sources(
            "web_search",
            {"query": "second"},
            json.dumps([{"title": "Report", "url": "https://example.com/report"}]),
        ))
        merge_research_sources(collected, extract_research_sources(
            "web_scrape",
            {"url": "https://example.com/report"},
            json.dumps({
                "url": "https://example.com/report",
                "title": "Report",
                "description": "Full text of the report",
            }),
        ))

        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["query"], "first")
        self.assertTrue(collected[0]["fetched"])
        self.assertEqual(collected[0]["snippet"], "Full text of the report")

    def test_citation_list_is_bounded(self):
        collected: list[dict] = []
        merge_research_sources(
            collected,
            [
                {"url": f"https://example.com/{index}", "title": str(index)}
                for index in range(200)
            ],
            limit=5,
        )
        self.assertEqual(len(collected), 5)

    def test_summary_counts_sources_and_distinct_sites(self):
        self.assertEqual(research_sources_summary([]), "no sources")
        self.assertEqual(
            research_sources_summary([{"url": "https://a.io/1", "host": "a.io"}]),
            "1 link",
        )
        self.assertEqual(
            research_sources_summary([
                {"url": "https://a.io/1", "host": "a.io"},
                {"url": "https://a.io/2", "host": "a.io"},
            ]),
            "2 links · 1 site",
        )

    def test_terminal_deep_research_shows_sources_and_keeps_them_on_the_answer(self):
        from agent import core

        session = core._new_session_state()
        session["agent_mode"] = "deep-research"
        history: list[dict] = []
        searching = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "web_search",
                    "arguments": {"query": "ignition evidence", "difficulty": "hard"},
                },
            }],
        }
        answer = {"role": "assistant", "content": "Here is the synthesis."}
        tool_result = [{
            "role": "tool",
            "tool_name": "web_search",
            "name": "web_search",
            "content": json.dumps([{
                "title": "Ignition report",
                "url": "https://example.com/report",
                "snippet": "evidence",
            }]),
        }]
        with (
            patch.object(core, "_stream_complete_response", side_effect=[searching, answer]),
            patch.object(
                core,
                "_process_tool_calls_with_turn_guard",
                MagicMock(return_value=tool_result),
            ),
            patch.object(core, "_check_and_compact_history"),
            patch.object(core, "print_research_sources") as show_sources,
        ):
            core.process_user_turn("compare the evidence", session, history, "system")

        shown = show_sources.call_args.args[0]
        self.assertEqual([source["url"] for source in shown], ["https://example.com/report"])
        self.assertEqual(shown[0]["query"], "ignition evidence")
        self.assertEqual(
            history[-1]["research_sources"][0]["url"],
            "https://example.com/report",
        )

    def test_normal_mode_terminal_turn_reports_no_sources(self):
        from agent import core

        session = core._new_session_state()
        history: list[dict] = []
        searching = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {"name": "web_search", "arguments": {"query": "topic"}},
            }],
        }
        answer = {"role": "assistant", "content": "done"}
        tool_result = [{
            "role": "tool",
            "tool_name": "web_search",
            "name": "web_search",
            "content": json.dumps([{"title": "T", "url": "https://example.com/t"}]),
        }]
        with (
            patch.object(core, "_stream_complete_response", side_effect=[searching, answer]),
            patch.object(
                core,
                "_process_tool_calls_with_turn_guard",
                MagicMock(return_value=tool_result),
            ),
            patch.object(core, "_check_and_compact_history"),
            patch.object(core, "print_research_sources") as show_sources,
        ):
            core.process_user_turn("what happened?", session, history, "system")

        show_sources.assert_not_called()
        self.assertNotIn("research_sources", history[-1])

    def test_web_deep_research_streams_sources_and_stores_them_on_the_answer(self):
        from agent import web
        from agent.cancellation import CancellationToken
        from agent.runtime_config import get_runtime_config
        from agent.tool_runner import ToolCallResult, normalize_tool_calls

        search_payload = json.dumps([{
            "title": "Ignition report",
            "url": "https://example.com/report",
            "snippet": "The facility reached ignition.",
        }])

        def fake_execute(calls, **kwargs):
            return [
                ToolCallResult(spec=spec, content=search_payload)
                for spec in normalize_tool_calls(calls)
            ]

        def fake_plan(*args, **kwargs):
            yield {"type": "status", "message": "planning", "color": "blue"}
            return ["ignition evidence"]

        answer = SimpleNamespace(
            message=SimpleNamespace(
                content="Synthesis with citations.",
                thinking="",
                planning="",
                tool_calls=[],
            ),
            prompt_eval_count=0,
            eval_count=0,
            done_reason="stop",
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(web, "_SESSIONS_DIR", temporary),
            patch.object(web, "_deep_research_plan_events", fake_plan),
            patch.object(web, "execute_tool_calls", fake_execute),
            patch.object(web, "_model_chat", return_value=iter([answer])),
            patch.object(web, "title_temporary_session", return_value=None),
        ):
            runtime = get_runtime_config()
            session = {
                **web._session_from_runtime(runtime),
                "agent_mode": "deep-research",
            }
            history: list[dict] = []
            events = list(web._generate_chat_events_impl(
                "compare the evidence",
                session,
                history,
                "Active Session",
                cancellation_token=CancellationToken(),
                publish_global=False,
                client_id="client-one",
            ))

        citation_events = [event for event in events if event.get("type") == "research_sources"]
        self.assertTrue(citation_events)
        self.assertEqual(citation_events[-1]["mode"], "deep-research")
        self.assertEqual(
            [source["url"] for source in citation_events[-1]["sources"]],
            ["https://example.com/report"],
        )
        self.assertEqual(citation_events[-1]["sources"][0]["query"], "ignition evidence")
        answers = [
            message for message in history
            if message.get("role") == "assistant" and message.get("content")
        ]
        self.assertEqual(
            answers[-1]["research_sources"][0]["url"],
            "https://example.com/report",
        )


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

    def test_sources_dropdown_is_wired_in_the_web_ui(self):
        self.assertIn('case "research_sources":', APP)
        self.assertIn("function sourcesBlock(sources)", APP)
        self.assertIn("function upsertStreamSources(sources)", APP)
        self.assertIn("function safeSourceURL(value)", APP)
        self.assertIn('detailBlock("Sources", sourcesSummary(cited)', APP)
        self.assertIn('link.rel = "noopener noreferrer"', APP)
        self.assertIn("normalizeSources(message.research_sources)", APP)
        self.assertIn(".sources-block", STYLE)
        self.assertIn(".source-list", STYLE)
        self.assertIn(".source-snippet", STYLE)

    def test_sources_fold_is_wired_in_the_tui(self):
        tui_source = (ROOT / "agent" / "tui.py").read_text(encoding="utf-8")
        terminal_source = (ROOT / "agent" / "terminal.py").read_text(encoding="utf-8")
        self.assertIn("class SourcesFold(Static):", tui_source)
        self.assertIn("def ui_research_sources(self, sources: list[dict]) -> None:", tui_source)
        self.assertIn("def research_sources(self, sources: list[dict]) -> None:", tui_source)
        self.assertIn("def print_research_sources(sources: list[dict]) -> None:", terminal_source)


if __name__ == "__main__":
    unittest.main()
