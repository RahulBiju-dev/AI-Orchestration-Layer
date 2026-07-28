"""Request-specific tool schema selection coverage."""

from __future__ import annotations

import unittest

from agent.core import compact_tool_schemas, select_tool_schemas
from tools.registry import TOOL_SCHEMAS


class ToolSchemaSelectionTests(unittest.TestCase):
    def setUp(self):
        self.session = {"options": {"num_ctx": 4096}}

    def _names(self, user_text: str) -> set[str]:
        messages = [{"role": "user", "content": user_text}]
        selected = select_tool_schemas(messages, self.session, TOOL_SCHEMAS)
        self.assertIsNotNone(selected)
        self.assertLessEqual(len(selected), 10)
        return {
            schema["function"]["name"]
            for schema in selected
            if schema.get("function", {}).get("name")
        }

    def test_vault_phrasing_selects_vault_tools(self):
        names = self._names("search my vault notes for the meeting summary")
        self.assertIn("vault_search", names)

    def test_compact_index_schema_retains_handwriting_and_resume_contract(self):
        compact = compact_tool_schemas(TOOL_SCHEMAS)
        schema = next(
            item for item in compact
            if item.get("function", {}).get("name") == "index_vault"
        )
        description = schema["function"]["description"]
        self.assertIn("vision_mode=all", description)
        self.assertIn("next_page", description)
        self.assertIn("resume_page", description)
        self.assertIn("complete", description)

    def test_spotify_phrasing_selects_spotify(self):
        names = self._names("play a song on spotify")
        self.assertIn("spotify_play", names)

    def test_spreadsheet_phrasing_selects_spreadsheet(self):
        names = self._names("open the excel spreadsheet and update cells")
        self.assertIn("spreadsheet", names)

    def test_csv_and_legacy_xls_write_requests_select_spreadsheet(self):
        for request in (
            "write these rows to a CSV file",
            "export this table as a legacy XLS spreadsheet",
        ):
            with self.subTest(request=request):
                self.assertIn("spreadsheet", self._names(request))

    def test_routine_definition_request_selects_routine_executor(self):
        names = self._names("save a routine that exports my report")
        self.assertIn("automated_routine_executor", names)

    def test_natural_what_if_and_projection_requests_select_simulation(self):
        requests = (
            "What happens if demand varies randomly for the next 30 days?",
            "Project inventory growth over 12 months",
            "Compare best case, base case, and worst case cash flow",
        )
        for request in requests:
            with self.subTest(request=request):
                self.assertIn("run_simulation", self._names(request))

    def test_required_simulation_is_prioritized_ahead_of_generic_defaults(self):
        selected = select_tool_schemas(
            [{
                "role": "user",
                "content": "If demand grows each month, project inventory over 12 months",
            }],
            self.session,
            TOOL_SCHEMAS,
        )
        self.assertEqual(selected[0]["function"]["name"], "run_simulation")

    def test_natural_relationship_mapping_requests_select_knowledge_graph(self):
        requests = (
            "Map how these services depend on each other",
            "Trace the causal path through these supplied factors",
            "Find feedback loops and central concepts in these relationships",
            "Build a graph connecting these entities",
        )
        for request in requests:
            with self.subTest(request=request):
                self.assertIn("knowledge_graph_builder", self._names(request))

    def test_required_knowledge_graph_is_prioritized(self):
        selected = select_tool_schemas(
            [{
                "role": "user",
                "content": "Map how these services depend on each other",
            }],
            self.session,
            TOOL_SCHEMAS,
        )
        self.assertEqual(
            selected[0]["function"]["name"],
            "knowledge_graph_builder",
        )

    def test_browser_phrasing_selects_browser(self):
        names = self._names("open this website in the browser")
        self.assertIn("open_browser", names)

    def test_datetime_preflight_included_with_web_search(self):
        names = self._names("search the web for today's latest news headlines")
        self.assertIn("web_search", names)
        self.assertIn("web_scrape", names)
        self.assertIn("get_current_datetime", names)

    def test_public_url_requires_page_scraper_without_exact_tool_name(self):
        names = self._names("Summarize the claims in https://example.com/article")
        self.assertIn("web_scrape", names)

    def test_natural_argument_audit_selects_reasoning_debugger(self):
        names = self._names("Check this argument and its evidence for unsupported conclusions")
        self.assertIn("reasoning_chain_debugger", names)

    def test_natural_history_compaction_selects_memory_optimizer(self):
        names = self._names("Compress this chat history to reduce the context window")
        self.assertIn("context_memory_optimizer", names)

    def test_each_model_exposed_tool_has_recall_phrasing(self):
        phrases = {
            "get_current_datetime": "what is the current date and time today",
            "spreadsheet": "read the xlsx spreadsheet worksheet cells",
            "web_search": "search the internet for latest research news",
            "web_scrape": "scrape this webpage article url",
            "read_document": "extract text from this pdf document",
            "read_file": "read the text file lines on this path",
            "create_file": "create a new file and write content",
            "create_pdf": "create a pdf document with these notes",
            "export_vault_pdf": "export the complete entire vault to a reference pdf",
            "build_vault_notes_pdf": "generate refined lecture notes pdf from my entire vault",
            "spotify_play": "play music on spotify playlist",
            "open_browser": "open the browser to a website url",
            "view_code": "inspect the source code function class",
            "describe_image": "describe this screenshot image photo",
            "open_terminal_at_path": "open a terminal console at this folder directory",
            "launch_apps": "launch the desktop application app",
            "google_workspace": "list my google calendar events and tasks",
            "codebase_indexer": "index this repository codebase architecture",
            "index_vault": "index vault document folder embeddings",
            "vault_search": "semantic search vault knowledge documents",
            "vault_read": "read all vault chunks exhaustively in order",
            "delete_vault_item": "delete vault collection chunks from index",
            "list_vaults": "list vault collections indexes",
            "list_vault_aliases": "list vault aliases friendly names",
            "create_structured_note": "create an obsidian markdown note with tags",
            "knowledge_graph_builder": "build knowledge graph concepts relationships",
            "run_simulation": "run monte carlo simulation scenario probability",
            "api_orchestrator": "call api http endpoint request integration",
            "context_memory_optimizer": "optimize conversation context memory compact",
            "reasoning_chain_debugger": "audit claim evidence reasoning confidence",
            "automated_routine_executor": "run automated routine workflow trigger",
        }
        for tool_name, phrase in phrases.items():
            with self.subTest(tool=tool_name):
                names = self._names(phrase)
                self.assertIn(tool_name, names, f"{tool_name} not selected for: {phrase}")


if __name__ == "__main__":
    unittest.main()
