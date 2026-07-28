"""Regression coverage for the recursive reliability pass over legacy tools."""

from __future__ import annotations

import json
import importlib
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.ollama_runtime import OllamaUnavailableError
from tools import api_orchestrator as api_module
from tools import obsi_vault_writer
from tools import vault_embeddings as embedding_module
from tools import web_scraper as scraper_module
from tools.api_orchestrator import api_orchestrator
from tools.app_launcher import _is_safe_desktop_entry
from tools.automated_routine_executor import _validate as validate_routine
from tools.code import view_code
from tools.codebase_indexer import codebase_indexer
from tools.context_memory_optimizer import context_memory_optimizer
from tools.current_datetime import get_current_datetime
from tools.document import MAX_PAGE_SELECTION, _parse_page_spec, read_document
from tools.file import read_file
from tools.knowledge_graph_builder import knowledge_graph_builder
from tools.pdf_writer import _parse_notes_cursor
from tools.reasoning_chain_debugger import reasoning_chain_debugger
from tools.run_simulation import run_simulation
from tools.search import web_search
from tools.spreadsheet import _Sheet, _parse_range
from tools.vault_embeddings import embed_texts
from tools.vault_search import ordered_vault_records
from agent.cancellation import CancellationToken, OperationCancelled
from tools.vision_describer import describe_image
from tools.web_scraper import web_scrape


class _StreamingResponse:
    def __init__(self, chunks: list[str], *, status: int = 200) -> None:
        self._chunks = chunks
        self.status_code = status
        self.ok = 200 <= status < 400
        self.encoding = "utf-8"
        self.headers = {
            "Content-Type": "text/plain",
            "Set-Cookie": "session=secret",
            "X-Token": "secret",
            "X-Request-ID": "safe",
        }
        self.closed = False

    def iter_content(self, **_kwargs):
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


class RemainingToolReliabilityTests(unittest.TestCase):
    def test_context_optimizer_rejects_malformed_nested_values(self):
        payload = json.loads(context_memory_optimizer(["not-an-object"]))
        self.assertIn("messages[0]", payload["error"])
        payload = json.loads(context_memory_optimizer([], critical_terms="wrong"))
        self.assertIn("critical_terms", payload["error"])
        payload = json.loads(context_memory_optimizer([], target_tokens=1))
        self.assertIn("target_tokens", payload["error"])

    def test_context_optimizer_reports_unavoidable_budget_and_preserves_tool_protocol(self):
        oversized = json.loads(context_memory_optimizer(
            [
                {"role": "system", "content": "policy " * 500},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "web_scrape", "arguments": {"url": "https://example.com"}}}],
                },
                {"role": "tool", "tool_name": "web_scrape", "content": "page evidence"},
            ],
            target_tokens=256,
            preserve_recent=1,
        ))
        self.assertFalse(oversized["stats"]["target_met"])
        self.assertEqual(oversized["messages"][-2]["role"], "assistant")
        self.assertEqual(oversized["messages"][-1]["role"], "tool")
        self.assertEqual(oversized["stats"]["preserved_recent_messages"], 2)

    def test_simulation_rejects_non_finite_and_expensive_inputs(self):
        payload = json.loads(run_simulation({"x": math.inf}, {"x": "x"}))
        self.assertIn("finite", payload["error"])
        payload = json.loads(run_simulation({"x": 2}, {"x": "x ** 101"}))
        self.assertIn("Power exponent", payload["error"])
        payload = json.loads(run_simulation(
            {"x": 1}, {"x": "x + 1"}, scenarios=[{"overrides": {"missing": 2}}]
        ))
        self.assertIn("unknown variables", payload["error"])
        payload = json.loads(run_simulation(
            {"x": 1}, {"x": "x"}, steps=10_000, trials=50,
            scenarios=[{"name": "a"}, {"name": "b"}],
        ))
        self.assertIn("across all scenarios", payload["error"])

    def test_simulation_still_runs_valid_recurrence(self):
        payload = json.loads(run_simulation({"x": 1}, {"x": "x + 1"}, steps=2))
        self.assertEqual(payload["scenarios"][0]["sample_trajectory"][-1]["x"], 3.0)

    def test_simulation_rejects_silent_clamping_and_ambiguous_scenarios(self):
        self.assertIn(
            "steps must be between",
            json.loads(run_simulation({"x": 1}, {"x": "x"}, steps=0))["error"],
        )
        self.assertIn(
            "trials must be an integer",
            json.loads(run_simulation({"x": 1}, {"x": "x"}, trials=True))["error"],
        )
        duplicate = json.loads(run_simulation(
            {"x": 1},
            {"x": "x"},
            scenarios=[
                {"name": "Baseline", "overrides": {}},
                {"name": "baseline", "overrides": {}},
            ],
        ))
        self.assertIn("Duplicate scenario name", duplicate["error"])

    def test_simulation_compiles_each_equation_once_and_seed_is_reproducible(self):
        simulation_module = importlib.import_module("tools.run_simulation")
        real_parse = simulation_module.ast.parse
        arguments = {
            "variables": {"x": 0, "y": 1},
            "equations": {
                "x": "x + normal(0, 1)",
                "y": "max(0, y + uniform(-0.1, 0.1))",
            },
            "steps": 20,
            "trials": 10,
            "seed": 42,
        }
        with patch.object(
            simulation_module.ast, "parse", wraps=real_parse
        ) as parse:
            first = json.loads(run_simulation(**arguments))
        second = json.loads(run_simulation(**arguments))

        self.assertEqual(parse.call_count, 2)
        self.assertEqual(first, second)
        self.assertEqual(first["evaluations"], 400)

    def test_simulation_bounds_result_shape_without_losing_final_summary(self):
        variables = {f"x{index}": float(index) for index in range(20)}
        equations = {name: f"{name} + 1" for name in variables}
        payload = json.loads(run_simulation(
            variables,
            equations,
            steps=1_000,
            scenarios=[
                {"name": f"case-{index}", "overrides": {}}
                for index in range(10)
            ],
        ))

        self.assertNotIn("error", payload)
        self.assertTrue(all(
            scenario["trajectory_truncated"] for scenario in payload["scenarios"]
        ))
        self.assertTrue(all(
            len(scenario["final_distribution"]) == 20
            for scenario in payload["scenarios"]
        ))
        trajectory_values = sum(
            scenario["trajectory_points"] * 20
            for scenario in payload["scenarios"]
        )
        self.assertLessEqual(trajectory_values, 1_000)

    def test_simulation_rejects_function_keywords_before_running(self):
        payload = json.loads(run_simulation(
            {"x": 0},
            {"x": "normal(mu=0, sigma=1)"},
            steps=100,
            trials=100,
        ))
        self.assertIn("positional arguments only", payload["error"])

    def test_simulation_honors_cancellation_before_work_starts(self):
        token = CancellationToken()
        token.cancel("stop simulation")
        with self.assertRaises(OperationCancelled):
            run_simulation(
                {"x": 0},
                {"x": "x + 1"},
                steps=10_000,
                cancellation_token=token,
            )

    def test_reasoning_debugger_normalizes_references_and_mermaid_ids(self):
        payload = json.loads(reasoning_chain_debugger(
            "done",
            [{"id": 'bad] --> injected["', "claim": "done", "depends_on": "wrong"}],
        ))
        self.assertFalse(payload["valid"])
        self.assertIn("depends_on must be an array", [item["issue"] for item in payload["issues"]])
        self.assertIn("n1[", payload["mermaid"])
        self.assertNotIn("bad] --> injected", payload["mermaid"])

    def test_reasoning_debugger_reports_duplicate_evidence(self):
        payload = json.loads(reasoning_chain_debugger(
            "claim",
            [{"claim": "claim", "evidence_ids": ["e1"]}],
            [{"id": "e1"}, {"id": "e1"}],
        ))
        self.assertFalse(payload["valid"])
        self.assertIn("Duplicate evidence id", [item["issue"] for item in payload["issues"]])

    def test_reasoning_debugger_normalizes_links_and_reports_graph_coverage(self):
        payload = json.loads(reasoning_chain_debugger(
            "final claim",
            [
                {"id": "root", "claim": "supported fact", "evidence_ids": [" e1 "]},
                {"id": "final", "claim": "final claim", "depends_on": [" root "]},
                {"id": "unused", "claim": "separate assumption", "assumption": True},
            ],
            [{"id": "e1", "source": "https://example.com", "quality": 0.9}],
        ))
        self.assertTrue(payload["valid"])
        self.assertTrue(payload["conclusion_established"])
        self.assertEqual(payload["evidence_coverage"]["referenced"], 1)
        self.assertEqual(payload["graph_coverage"]["disconnected_steps"], 1)

    def test_reasoning_debugger_rejects_an_empty_graph(self):
        payload = json.loads(reasoning_chain_debugger("claim", []))
        self.assertIn("at least one", payload["error"])

    def test_reasoning_debugger_bounds_large_diagnostic_output(self):
        steps = [
            {
                "id": f"step-{index}",
                "claim": "final" if index == 0 else f"claim {index}",
                "depends_on": [f"missing-{value}-{'x' * 80}" for value in range(100)],
            }
            for index in range(200)
        ]
        raw = reasoning_chain_debugger("final", steps)
        payload = json.loads(raw)
        self.assertLessEqual(len(raw), 24_000)
        self.assertTrue(payload["issues_truncated"])
        self.assertGreater(payload["total_issues"], len(payload["issues"]))

    def test_web_scraper_rejects_ambiguous_options_and_extraction_failures(self):
        self.assertIn("max_chars", json.loads(web_scrape("https://8.8.8.8", max_chars="many"))["error"])
        self.assertIn("include_links", json.loads(web_scrape("https://8.8.8.8", include_links="false"))["error"])
        with (
            patch.object(scraper_module, "_validate_public_http_url", return_value="https://example.com"),
            patch.object(scraper_module, "_fetch", return_value={
                "url": "https://example.com",
                "status_code": 200,
                "content_type": "text/html",
                "text": "<p>text</p>",
            }),
            patch.object(scraper_module, "_extract", side_effect=RuntimeError("bad html")),
        ):
            payload = json.loads(web_scrape("https://example.com"))
        self.assertIn("page extraction failed", payload["error"])

    def test_web_scraper_retries_transient_status_and_reports_attempts(self):
        class Response:
            def __init__(self, status, body):
                self.status_code = status
                self.headers = {"Content-Type": "text/plain"}
                self.url = "https://example.com"
                self.encoding = "utf-8"
                self.is_redirect = False
                self.is_permanent_redirect = False
                self.body = body
                self.closed = False

            def iter_content(self, **_kwargs):
                yield self.body

            def close(self):
                self.closed = True

        first = Response(503, b"temporary")
        second = Response(200, b"recovered")
        session = SimpleNamespace(
            get=Mock(side_effect=[first, second]),
            close=Mock(),
        )
        with (
            patch("requests.Session", return_value=session),
            patch.object(
                scraper_module,
                "_validate_public_http_url",
                return_value="https://example.com",
            ),
            patch.object(scraper_module.time, "sleep"),
        ):
            payload = scraper_module._fetch("https://example.com")

        self.assertEqual(payload["text"], "recovered")
        self.assertEqual(payload["request_attempts"], 2)
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)
        session.close.assert_called_once()

    def test_knowledge_graph_rejects_bad_query_weight_and_edge_ids(self):
        concepts = [{"id": "a"}, {"id": "b"}]
        payload = json.loads(knowledge_graph_builder(concepts, [], query="wrong"))
        self.assertIn("query must be", payload["error"])
        payload = json.loads(knowledge_graph_builder(
            concepts, [{"source": "a", "target": "b", "weight": float("nan")}]
        ))
        self.assertIn("Invalid graph", payload["error"])
        payload = json.loads(knowledge_graph_builder(concepts, [
            {"id": "same", "source": "a", "target": "b"},
            {"id": "same", "source": "a", "target": "b"},
        ]))
        self.assertIn("Duplicate relationship id", " ".join(payload["details"]))

    def test_api_orchestrator_validates_urls_and_literal_secrets(self):
        payload = json.loads(api_orchestrator({"url": "https://user:pass@example.com"}))
        self.assertIn("embedded credentials", payload["error"])
        payload = json.loads(api_orchestrator({
            "url": "https://example.com", "headers": {"Authorization": "Bearer literal"}
        }))
        self.assertIn("environment-variable", payload["error"])

    def test_api_orchestrator_streams_bounds_and_redacts_response(self):
        response = _StreamingResponse(["a" * 800, "b" * 800])
        with patch.object(api_module.requests, "request", return_value=response) as request:
            payload = json.loads(api_orchestrator({
                "url": "https://example.com/path?api_key=secret",
                "max_response_chars": 1000,
            }))
        self.assertEqual(len(payload["body"]), 1000)
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["endpoint"], "https://example.com/path")
        self.assertNotIn("Set-Cookie", payload["headers"])
        self.assertNotIn("X-Token", payload["headers"])
        self.assertEqual(payload["headers"]["X-Request-ID"], "safe")
        self.assertTrue(response.closed)
        self.assertTrue(request.call_args.kwargs["stream"])

    def test_web_search_skips_malformed_backend_records(self):
        class FakeDDGS:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def text(self, *_args, **_kwargs):
                return [None, {"href": "javascript:alert(1)"}, {
                    "href": "https://example.com", "title": "ok", "body": "snippet"
                }]

        fake_module = SimpleNamespace(DDGS=FakeDDGS)
        with patch.dict("sys.modules", {"ddgs": fake_module}):
            payload = json.loads(web_search("query", difficulty="easy"))
        self.assertEqual(payload["skipped_invalid_results"], 2)
        self.assertEqual(payload["results"][0]["url"], "https://example.com")

    def test_web_search_page_limit_bounds_attempts_not_only_successes(self):
        class FakeDDGS:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def text(self, *_args, **_kwargs):
                return [
                    {"href": f"https://example.com/{index}", "title": str(index), "body": ""}
                    for index in range(3)
                ]

        with (
            patch.dict("sys.modules", {"ddgs": SimpleNamespace(DDGS=FakeDDGS)}),
            patch("tools.search.web_scrape", return_value='{"error":"failed"}') as scrape,
        ):
            web_search("query", include_content=True, max_pages=1)
        self.assertEqual(scrape.call_count, 1)

    def test_native_and_file_tools_reject_malformed_paths_without_crashing(self):
        self.assertTrue(describe_image(None).startswith("Error:"))
        self.assertIn("error", json.loads(read_file(None)))
        self.assertIn("error", json.loads(read_document(None)))
        self.assertIn("error", json.loads(view_code(None)))
        self.assertIn("error", json.loads(get_current_datetime("x" * 300)))
        self.assertIn("error", json.loads(web_scrape("https://" + "a" * 5000)))

    def test_embedding_helper_rejects_single_string_and_bad_timeout(self):
        with self.assertRaises(TypeError):
            embed_texts("not-a-sequence-of-documents")
        with self.assertRaises(ValueError):
            embed_texts(["document"], timeout="invalid")
        with self.assertRaises(TypeError):
            embed_texts([123])
        with self.assertRaises(ValueError):
            embed_texts(["  "])
        with self.assertRaises(ValueError):
            embed_texts(["document"], timeout=1.5)
        with self.assertRaises(ValueError):
            embed_texts(["document"], model="bad model")

    def test_embedding_helper_batches_under_one_owner_and_preserves_text(self):
        service = SimpleNamespace(
            coordinator=SimpleNamespace(current_context_owner=lambda: "outer-operation"),
        )

        def embed(values, **_kwargs):
            return {
                "embeddings": [
                    [float(index + 1), 1.0]
                    for index, _value in enumerate(values)
                ]
            }

        service.embed = Mock(side_effect=embed)
        texts = [" leading and trailing "] + [
            f"document {index}" for index in range(128)
        ]
        with (
            patch.object(embedding_module, "_EMBED_SERVICE", service),
            patch.dict(embedding_module._MODEL_DIMENSIONS, {}, clear=True),
        ):
            result = embed_texts(texts, model="batch-model")

        self.assertEqual(len(result), 129)
        self.assertEqual(service.embed.call_count, 2)
        self.assertEqual(len(service.embed.call_args_list[0].args[0]), 128)
        self.assertEqual(len(service.embed.call_args_list[1].args[0]), 1)
        self.assertEqual(
            service.embed.call_args_list[0].args[0][0],
            " leading and trailing ",
        )
        self.assertTrue(all(
            call.kwargs["owner"] == "outer-operation"
            for call in service.embed.call_args_list
        ))

    def test_embedding_helper_retries_transient_failure_and_rejects_dimension_drift(self):
        service = SimpleNamespace(
            coordinator=SimpleNamespace(current_context_owner=lambda: None),
            embed=Mock(side_effect=[
                OllamaUnavailableError("starting"),
                {"embeddings": [[1.0, 2.0]]},
                {"embeddings": [[1.0, 2.0, 3.0]]},
            ]),
        )
        with (
            patch.object(embedding_module, "_EMBED_SERVICE", service),
            patch.object(embedding_module, "_retry_delay", return_value=True),
            patch.dict(embedding_module._MODEL_DIMENSIONS, {}, clear=True),
        ):
            first = embed_texts(["document"], model="stable-model")
            with self.assertRaisesRegex(RuntimeError, "changed from 2 to 3"):
                embed_texts(["other"], model="stable-model")

        self.assertEqual(first, [[1.0, 2.0]])
        self.assertEqual(service.embed.call_count, 3)

    def test_embedding_helper_rejects_malformed_and_zero_vectors(self):
        malformed_responses = (
            {"embeddings": [1.0, 2.0]},
            {"embeddings": [[True, 1.0]]},
            {"embeddings": [[0.0, 0.0]]},
            {"embeddings": [[1.0], [1.0, 2.0]]},
        )
        for index, response in enumerate(malformed_responses):
            with self.subTest(response=index):
                service = SimpleNamespace(
                    coordinator=SimpleNamespace(current_context_owner=lambda: None),
                    embed=Mock(return_value=response),
                )
                with (
                    patch.object(embedding_module, "_EMBED_SERVICE", service),
                    patch.dict(embedding_module._MODEL_DIMENSIONS, {}, clear=True),
                ):
                    with self.assertRaises(RuntimeError):
                        embed_texts(
                            ["one"] if index < 3 else ["one", "two"],
                            model=f"malformed-{index}",
                        )

    def test_linux_env_desktop_wrapper_allows_assignments_but_blocks_flags(self):
        base = {"name": "Example", "terminal": False}
        self.assertTrue(_is_safe_desktop_entry({**base, "exec": "env FOO=bar /usr/bin/example"}))
        self.assertFalse(_is_safe_desktop_entry({**base, "exec": "env -S 'bash -c id'"}))

    def test_routine_validator_rejects_invalid_runtime_values(self):
        routine = {
            "description": "bad delay",
            "triggers": ["run it"],
            "actions": [{"type": "delay", "seconds": float("nan")}],
        }
        self.assertIn("seconds must be between", " ".join(validate_routine(routine)))
        routine["actions"] = [{"type": "open_url", "url": "https://user:pass@example.com"}]
        self.assertIn("embedded credentials", " ".join(validate_routine(routine)))

    def test_note_writer_bounds_content_before_writing(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            obsi_vault_writer, "VAULTS_DIR", directory
        ):
            payload = json.loads(obsi_vault_writer.create_structured_note(
                "title", "x" * (obsi_vault_writer.MAX_NOTE_CHARS + 1)
            ))
        self.assertIn("limit", payload["error"])

    def test_pdf_page_range_and_cursor_are_bounded(self):
        with self.assertRaisesRegex(ValueError, "at most"):
            _parse_page_spec(f"1-{MAX_PAGE_SELECTION + 1}", MAX_PAGE_SELECTION + 1)
        with self.assertRaisesRegex(ValueError, "100-character"):
            _parse_notes_cursor("1" * 101)

    def test_spreadsheet_cell_range_length_is_bounded(self):
        sheet = _Sheet("Sheet1", 1, 1, lambda _row, _column: None)
        with self.assertRaisesRegex(ValueError, "64-character"):
            _parse_range("A" * 65 + "1", sheet)

    def test_vault_ordering_tolerates_malformed_legacy_metadata(self):
        collection = SimpleNamespace(get=lambda **_kwargs: {
            "ids": ["a", "b"],
            "metadatas": [
                {"source": "file", "page": "not-a-number", "chunk_index": None},
                None,
            ],
        })
        client = SimpleNamespace(get_collection=lambda **_kwargs: collection)
        with patch("tools.vault_search.get_chroma_client", return_value=client):
            records = ordered_vault_records("vault")
        self.assertEqual({item["id"] for item in records}, {"a", "b"})

    def test_codebase_model_name_is_bounded_before_index_access(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = json.loads(codebase_indexer(directory, action="status", model="x" * 201))
        self.assertIn("model", payload["error"])


if __name__ == "__main__":
    unittest.main()
