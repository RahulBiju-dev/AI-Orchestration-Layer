from __future__ import annotations

import unittest
from pathlib import Path

from agent.runtime_config import RuntimeConfigurationError
from agent.modes import (
    DEEP_RESEARCH_COMPACT_MARKER,
    compact_deep_research_messages,
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
        self.assertIn("Completed web searches: 3", checkpoints[0])
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
        self.assertIn('appendStatus(event.message || "", event.activity_mode || "")', APP)
        self.assertIn('status.classList.add("mode-activity", "running")', APP)
        self.assertIn('runtime_profile: "manual"', APP)
        self.assertIn(".mode-clear[hidden]", STYLE)
        self.assertIn(".mode-menu[hidden]", STYLE)
        self.assertIn("@keyframes mode-text-shine", STYLE)
        self.assertIn("@keyframes mode-text-flash", STYLE)


if __name__ == "__main__":
    unittest.main()
