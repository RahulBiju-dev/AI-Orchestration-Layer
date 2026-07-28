"""Focused correctness tests for the persistent codebase indexer."""

from __future__ import annotations

import importlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


indexer = importlib.import_module("tools.codebase_indexer")


class FakeCollection:
    def __init__(self, records: dict[str, tuple[str, dict]] | None = None) -> None:
        self.records = dict(records or {})
        self.deleted: list[str] = []
        self.query_result: dict | None = None

    def get(self, **_kwargs):
        ids = list(self.records)
        return {
            "ids": ids,
            "metadatas": [self.records[item_id][1] for item_id in ids],
        }

    def upsert(self, *, ids, documents, embeddings, metadatas):
        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
            raise AssertionError("upsert payload lengths differ")
        for item_id, document, metadata in zip(ids, documents, metadatas):
            self.records[item_id] = (document, metadata)

    def delete(self, *, ids):
        self.deleted.extend(ids)
        for item_id in ids:
            self.records.pop(item_id, None)

    def count(self):
        return len(self.records)

    def query(self, **_kwargs):
        if self.query_result is None:
            raise AssertionError("query_result was not configured")
        return self.query_result


class FakeClient:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection

    def get_or_create_collection(self, *, name):
        return self.collection

    def get_collection(self, *, name):
        return self.collection


class CodebaseIndexerTests(unittest.TestCase):
    def test_discovery_does_not_silently_cap_large_supported_files(self):
        with tempfile.TemporaryDirectory() as directory:
            large = Path(directory, "large.py")
            large.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
            files, skipped = indexer._discover_files(directory)

        self.assertEqual(files, [large])
        self.assertEqual(skipped, [])

    def test_non_utf8_source_is_decoded_instead_of_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "legacy.py")
            source.write_bytes("price = '€'".encode("cp1252"))
            self.assertEqual(indexer._read_source(source), "price = '€'")

    def test_empty_source_removes_its_previous_stale_chunks(self):
        old_id = "old-empty-chunk"
        collection = FakeCollection({
            old_id: ("old implementation", {"source": "empty.py"}),
        })
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "empty.py").write_text("", encoding="utf-8")
            with (
                patch.object(indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(
                    indexer,
                    "embed_texts",
                    side_effect=lambda documents, **_kwargs: [[1.0] for _ in documents],
                ),
                patch.object(indexer, "_load_state", return_value={}),
                patch.object(indexer, "_save_state") as save_state,
            ):
                result = indexer._index_repository(
                    directory,
                    "collection",
                    "embedding-model",
                    1000.0,
                )

        self.assertNotIn("error", result)
        self.assertEqual(result["indexed_files"], 1)
        self.assertEqual(result["removed_chunks"], 1)
        self.assertIn(old_id, collection.deleted)
        self.assertNotIn(old_id, collection.records)
        save_state.assert_called_once()

    def test_failed_stale_cleanup_does_not_record_successful_refresh(self):
        class DeleteFailingCollection(FakeCollection):
            def delete(self, *, ids):
                raise RuntimeError("database is read-only")

        collection = DeleteFailingCollection({
            "deleted-file-chunk": ("stale", {"source": "deleted.py"}),
        })
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "current.py").write_text("value = 1", encoding="utf-8")
            with (
                patch.object(indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(
                    indexer,
                    "embed_texts",
                    side_effect=lambda documents, **_kwargs: [[1.0] for _ in documents],
                ),
                patch.object(indexer, "_load_state", return_value={}),
                patch.object(indexer, "_save_state") as save_state,
            ):
                result = indexer._index_repository(
                    directory,
                    "collection",
                    "embedding-model",
                    1000.0,
                )

        self.assertIn("stale chunks could not be removed", result["error"])
        self.assertTrue(result["preserved"])
        save_state.assert_not_called()

    def test_existing_index_inspection_failure_is_controlled_before_embedding(self):
        class GetFailingCollection(FakeCollection):
            def get(self, **_kwargs):
                raise RuntimeError("collection metadata unavailable")

        collection = GetFailingCollection()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "current.py").write_text("value = 1", encoding="utf-8")
            with (
                patch.object(indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(indexer, "embed_texts") as embed,
            ):
                result = indexer._index_repository(
                    directory,
                    "collection",
                    "embedding-model",
                    1000.0,
                )

        self.assertIn("Could not inspect the existing codebase index", result["error"])
        self.assertTrue(result["preserved"])
        embed.assert_not_called()

    def test_refresh_status_derived_fields_override_stale_state(self):
        now = 10_000.0
        state = {
            "/repo": {
                "collection": "collection",
                "last_indexed_at": now - 60,
                "needs_refresh": False,
                "reason": "stale persisted value",
            }
        }
        with patch.object(indexer, "_load_state", return_value=state):
            status = indexer._refresh_status(
                "/repo",
                now,
                index_available=False,
            )

        self.assertTrue(status["needs_refresh"])
        self.assertFalse(status["index_available"])
        self.assertEqual(status["reason"], "index collection is missing or unavailable")
        self.assertIsInstance(status["last_indexed_at"], str)
        self.assertEqual(status["age_seconds"], 60)

    def test_refresh_status_rejects_non_finite_timestamp(self):
        with patch.object(
            indexer,
            "_load_state",
            return_value={"/repo": {"last_indexed_at": math.inf}},
        ):
            status = indexer._refresh_status("/repo", 100.0, index_available=True)
        self.assertTrue(status["needs_refresh"])
        self.assertEqual(status["reason"], "index timestamp is invalid")

    def test_incomplete_attempt_never_enters_refresh_cooldown(self):
        with patch.object(
            indexer,
            "_load_state",
            return_value={
                "/repo": {
                    "last_indexed_at": 99.0,
                    "last_attempted_at": 100.0,
                    "complete": False,
                }
            },
        ):
            status = indexer._refresh_status("/repo", 101.0, index_available=True)

        self.assertTrue(status["needs_refresh"])
        self.assertEqual(status["reason"], "previous indexing attempt was incomplete")

        with patch.object(
            indexer,
            "_load_state",
            return_value={"/repo": {"last_indexed_at": 1e300}},
        ):
            status = indexer._refresh_status("/repo", 100.0, index_available=True)
        self.assertTrue(status["needs_refresh"])
        self.assertEqual(status["reason"], "index timestamp is invalid")

    def test_status_reports_missing_collection_as_refresh_required(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(indexer, "_index_available", return_value=False),
            patch.object(indexer, "time") as clock,
            patch.object(indexer, "_load_state", return_value={
                str(Path(directory).resolve()): {
                    "collection": "old-collection",
                    "last_indexed_at": 900.0,
                }
            }),
        ):
            clock.time.return_value = 1000.0
            payload = json.loads(indexer.codebase_indexer(directory, action="status"))

        self.assertTrue(payload["needs_refresh"])
        self.assertFalse(payload["index_available"])
        self.assertEqual(payload["reason"], "index collection is missing or unavailable")
        self.assertEqual(payload["collection"], indexer._collection_name(str(Path(directory).resolve())))

    def test_search_context_never_exceeds_requested_character_limit(self):
        collection = FakeCollection({"one": ("", {}), "two": ("", {})})
        collection.query_result = {
            "documents": [["a" * 600, "b" * 600]],
            "metadatas": [[
                {"source": "a.py", "line_start": 1, "line_end": 10},
                {"source": "b.py", "line_start": 1, "line_end": 10},
            ]],
            "distances": [[0.1, 0.2]],
        }
        with (
            patch.object(indexer, "get_chroma_client", return_value=FakeClient(collection)),
            patch.object(indexer, "embed_query", return_value=[1.0]),
        ):
            result = indexer._search(
                "/repo",
                "collection",
                "query",
                "embedding-model",
                2,
                1000,
            )

        self.assertLessEqual(len(result["context"]), 1000)
        self.assertEqual(result["match_count"], 2)

        collection.query_result = "malformed"
        with (
            patch.object(indexer, "get_chroma_client", return_value=FakeClient(collection)),
            patch.object(indexer, "embed_query", return_value=[1.0]),
        ):
            malformed = indexer._search(
                "/repo",
                "collection",
                "query",
                "embedding-model",
                2,
                1000,
            )
        self.assertIn("malformed", malformed["error"])

    def test_generated_desktop_bundle_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Path(root, "src").mkdir()
            Path(root, "src", "app.py").write_text("print('source')", encoding="utf-8")
            Path(root, "dist-electron").mkdir()
            Path(root, "dist-electron", "generated.py").write_text(
                "print('generated')",
                encoding="utf-8",
            )
            files, _ = indexer._discover_files(directory)

        relative = {path.relative_to(root).as_posix() for path in files}
        self.assertEqual(relative, {"src/app.py"})

    def test_repository_metadata_and_notebooks_are_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Path(root, ".env.example").write_text("FEATURE=true", encoding="utf-8")
            Path(root, "LICENSE").write_text("license terms", encoding="utf-8")
            Path(root, "analysis.ipynb").write_text("{}", encoding="utf-8")
            files, skipped = indexer._discover_files(directory)

        self.assertEqual(
            {path.name for path in files},
            {".env.example", "LICENSE", "analysis.ipynb"},
        )
        self.assertEqual(skipped, [])

    def test_refresh_state_is_written_privately(self):
        with patch.object(indexer, "atomic_write_json") as write_json:
            indexer._save_state({"/repo": {"last_indexed_at": 100.0}})
        write_json.assert_called_once_with(
            indexer.STATE_FILE,
            {"/repo": {"last_indexed_at": 100.0}},
            private=True,
        )

    def test_direct_call_normalizes_non_finite_limits_and_rejects_bad_model_name(self):
        with tempfile.TemporaryDirectory() as directory:
            bad_model = json.loads(indexer.codebase_indexer(
                directory,
                action="status",
                model="invalid model",
            ))
            self.assertIn("without whitespace", bad_model["error"])

            with (
                patch.object(indexer, "_index_available", return_value=True),
                patch.object(indexer, "_load_state", return_value={
                    str(Path(directory).resolve()): {
                        "collection": indexer._collection_name(str(Path(directory).resolve())),
                        "last_indexed_at": 1000.0,
                    }
                }),
                patch.object(indexer, "_search", return_value={"match_count": 0}) as search,
                patch.object(indexer.time, "time", return_value=1001.0),
            ):
                payload = json.loads(indexer.codebase_indexer(
                    directory,
                    query="query",
                    top_k=math.inf,
                    max_chars=math.inf,
                ))

        self.assertEqual(payload["match_count"], 0)
        self.assertEqual(search.call_args.args[4:6], (
            indexer.DEFAULT_TOP_K,
            indexer.DEFAULT_MAX_CHARS,
        ))


if __name__ == "__main__":
    unittest.main()
