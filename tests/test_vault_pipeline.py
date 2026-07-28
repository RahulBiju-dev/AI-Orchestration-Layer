import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import pdf_writer, vault_indexer, vault_search


class FakeCollection:
    def __init__(self, documents):
        self.documents = dict(documents)
        self.deleted = []

    def count(self):
        return len(self.documents)

    def get(self, ids=None, where=None, include=None, **kwargs):
        if ids is None:
            selected = list(self.documents)
        else:
            selected = [item_id for item_id in ids if item_id in self.documents]
        if where and "source" in where:
            selected = [
                item_id for item_id in selected
                if self.documents[item_id][1].get("source") == where["source"]
            ]
        return {
            "ids": selected,
            "documents": [self.documents[item_id][0] for item_id in selected],
            "metadatas": [self.documents[item_id][1] for item_id in selected],
        }

    def upsert(self, ids, documents, embeddings, metadatas):
        for item_id, document, metadata in zip(ids, documents, metadatas):
            self.documents[item_id] = (document, metadata)

    def delete(self, ids):
        self.deleted.extend(ids)
        for item_id in ids:
            self.documents.pop(item_id, None)


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def get_collection(self, name):
        return self.collection

    def get_or_create_collection(self, name):
        return self.collection


class VaultReadTests(unittest.TestCase):
    def test_misaligned_collection_rows_are_rejected_instead_of_dropped(self):
        class MisalignedCollection(FakeCollection):
            def get(self, **kwargs):
                return {
                    "ids": ["one", "two"],
                    "documents": ["first"],
                    "metadatas": [{}, {}],
                }

        with patch.object(
            vault_search,
            "get_chroma_client",
            return_value=FakeClient(MisalignedCollection({})),
        ):
            payload = json.loads(vault_search.read_vault(collection="lecture"))

        self.assertIn("misaligned", payload["error"])

    def test_search_context_bound_includes_join_separators(self):
        results = {
            "documents": [["a" * 800, "b" * 800]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, float("nan")]],
        }
        matches, context = vault_search._flatten_results(results, max_chars=1000)

        self.assertLessEqual(len(context), 1000)
        self.assertIsNone(matches[1]["distance"])

    def test_cursor_beyond_collection_is_rejected_instead_of_claiming_complete(self):
        collection = FakeCollection({
            "one": ("content", {"source": "notes.txt", "chunk_index": 0}),
        })
        with patch.object(vault_search, "get_chroma_client", return_value=FakeClient(collection)):
            payload = json.loads(vault_search.read_vault(
                collection="lecture",
                cursor=5,
            ))

        self.assertIn("beyond", payload["error"])
        self.assertEqual(payload["total_chunks"], 1)

    def test_partial_cursor_reads_oversized_chunk_without_gaps(self):
        document = "abcdefghijklmnopqrstuvwxyz" * 80
        collection = FakeCollection({
            "one": (document, {
                "source": "slides.pdf",
                "source_path": "/tmp/slides.pdf",
                "page": 1,
                "chunk_index": 0,
                "char_start": 0,
                "char_end": len(document),
                "content_kind": "text+vision",
            })
        })
        cursor = 0
        recovered = []
        with patch.object(vault_search, "get_chroma_client", return_value=FakeClient(collection)):
            for _ in range(10):
                payload = json.loads(vault_search.read_vault(
                    collection="lecture",
                    cursor=cursor,
                    max_chunks=1,
                    max_chars=500,
                ))
                if payload["content"]:
                    recovered.append(payload["content"].split("\n", 1)[1])
                if payload["complete"]:
                    break
                cursor = payload["next_cursor"]

        self.assertEqual("".join(recovered), document)
        self.assertTrue(payload["complete"])

    def test_ordered_records_sort_by_source_page_and_chunk(self):
        collection = FakeCollection({
            "b": ("second", {"source": "slides.pdf", "page": 2, "chunk_index": 0}),
            "a2": ("later", {"source": "slides.pdf", "page": 1, "chunk_index": 1}),
            "a1": ("first", {"source": "slides.pdf", "page": 1, "chunk_index": 0}),
        })
        with patch.object(vault_search, "get_chroma_client", return_value=FakeClient(collection)):
            records = vault_search.ordered_vault_records("lecture")
        self.assertEqual([item["id"] for item in records], ["a1", "a2", "b"])


class IncrementalPdfIndexTests(unittest.TestCase):
    def test_chunking_and_text_read_preserve_frontmatter_and_whitespace(self):
        text = "---\r\ntitle: Notes\r\n---\r\n    indented detail\r\n"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "notes.md")
            path.write_text(text, encoding="utf-8", newline="")
            extracted, info = vault_indexer._read_text_for_index(str(path))
            chunks = vault_indexer.chunk_text_with_offsets(
                extracted,
                chunk_size=500,
                chunk_overlap=0,
            )

        self.assertEqual(extracted, text)
        self.assertEqual(chunks[0]["text"], text)
        self.assertEqual(chunks[0]["char_end"], len(text))
        self.assertEqual(info["encoding"], "utf-8")

    def test_empty_text_file_removes_previous_chunks(self):
        collection = FakeCollection({
            "old": ("obsolete", {
                "source": "notes.txt",
            }),
        })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "notes.txt")
            path.write_text("", encoding="utf-8")
            with (
                patch.object(vault_indexer, "VAULTS_DIR", str(Path(temporary, "vaults"))),
                patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(vault_indexer, "register_vault_alias"),
            ):
                payload = json.loads(vault_indexer.index_vault(
                    file_path=str(path),
                    collection="lecture",
                ))

        self.assertTrue(payload["complete"])
        self.assertEqual(payload["indexed_files"], 1)
        self.assertNotIn("old", collection.documents)

    def test_partial_text_generation_is_rolled_back_on_write_failure(self):
        class FailingCollection(FakeCollection):
            def __init__(self, documents):
                super().__init__(documents)
                self.upsert_count = 0

            def upsert(self, ids, documents, embeddings, metadatas):
                self.upsert_count += 1
                if self.upsert_count == 2:
                    raise RuntimeError("disk write failed")
                super().upsert(ids, documents, embeddings, metadatas)

        collection = FailingCollection({
            "old": ("previous complete content", {
                "source": "notes.txt",
            }),
        })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "notes.txt")
            path.write_text(("section\n" + "x" * 600 + "\n") * 3, encoding="utf-8")
            with (
                patch.object(vault_indexer, "VAULTS_DIR", str(Path(temporary, "vaults"))),
                patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(
                    vault_indexer,
                    "embed_texts",
                    side_effect=lambda docs, **kwargs: [[1.0] for _ in docs],
                ),
                patch.object(vault_indexer, "register_vault_alias"),
            ):
                payload = json.loads(vault_indexer.index_vault(
                    file_path=str(path),
                    collection="lecture",
                    chunk_size=500,
                    chunk_overlap=0,
                    batch_size=1,
                ))

        self.assertFalse(payload["complete"])
        self.assertIn("disk write failed", payload["error"])
        self.assertEqual(set(collection.documents), {"old"})

    def test_same_relative_name_from_another_path_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current_dir = root / "current"
            other_dir = root / "other"
            current_dir.mkdir()
            other_dir.mkdir()
            current = current_dir / "notes.txt"
            other = other_dir / "notes.txt"
            current.write_text("new current notes", encoding="utf-8")
            collection = FakeCollection({
                "other": ("other notes", {
                    "source": "notes.txt",
                    "source_path": str(other),
                }),
            })
            with (
                patch.object(vault_indexer, "VAULTS_DIR", str(root / "vaults")),
                patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(
                    vault_indexer,
                    "embed_texts",
                    side_effect=lambda docs, **kwargs: [[1.0] for _ in docs],
                ),
                patch.object(vault_indexer, "register_vault_alias"),
            ):
                payload = json.loads(vault_indexer.index_vault(
                    file_path=str(current),
                    collection="lecture",
                ))

        self.assertTrue(payload["complete"])
        self.assertIn("other", collection.documents)
        self.assertEqual(len(collection.documents), 2)

    def test_completed_folder_index_removes_deleted_sources_only_inside_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            current = source_root / "current.txt"
            current.write_text("current notes", encoding="utf-8")
            collection = FakeCollection({
                "removed": ("stale", {
                    "source": "removed.txt",
                    "source_path": str(source_root / "removed.txt"),
                }),
                "external": ("keep", {
                    "source": "external.txt",
                    "source_path": str(root / "elsewhere" / "external.txt"),
                }),
            })
            with (
                patch.object(vault_indexer, "VAULTS_DIR", str(root / "vaults")),
                patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(
                    vault_indexer,
                    "embed_texts",
                    side_effect=lambda docs, **kwargs: [[1.0] for _ in docs],
                ),
            ):
                payload = json.loads(vault_indexer.index_vault(
                    vault_path=str(source_root),
                    collection="lecture",
                ))

        self.assertTrue(payload["complete"])
        self.assertEqual(payload["removed_chunks"], 1)
        self.assertNotIn("removed", collection.documents)
        self.assertIn("external", collection.documents)

    def test_pdf_cleanup_failure_keeps_checkpoint_incomplete(self):
        class FakePage:
            def extract_text(self):
                return "complete page text"

            def get(self, key):
                return {}

        class FakeReader:
            is_encrypted = False

            def __init__(self, stream):
                self.pages = [FakePage()]

        class CleanupFailingCollection(FakeCollection):
            def delete(self, ids):
                if "old-generation" in ids:
                    raise RuntimeError("cleanup unavailable")
                super().delete(ids)

        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary, "slides.pdf")
            pdf_path.write_bytes(b"fake pdf")
            collection = CleanupFailingCollection({
                "old-generation": ("old", {
                    "source": "slides.pdf",
                    "source_path": str(pdf_path),
                }),
            })
            with (
                patch.dict(sys.modules, {"pypdf": SimpleNamespace(PdfReader=FakeReader)}),
                patch.object(vault_indexer, "VAULTS_DIR", temporary),
                patch.object(vault_indexer, "_pdf_fingerprint", return_value="new-generation"),
                patch.object(
                    vault_indexer,
                    "embed_texts",
                    side_effect=lambda docs, **kwargs: [[1.0] for _ in docs],
                ),
            ):
                result = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path),
                    rel="slides.pdf",
                    collection_name="lecture",
                    collection_obj=collection,
                    chunk_size=500,
                    chunk_overlap=0,
                    model="embed",
                    vision_mode="off",
                    max_pages=1,
                    resume_page=None,
                    cancellation_token=None,
                )

        self.assertFalse(result["complete"])
        self.assertTrue(result["cleanup_pending"])
        self.assertEqual(result["next_page"], 2)
        self.assertIn("cleanup failed", result["warnings"][-1])

    def test_chroma_directory_preserves_legacy_store_without_migration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vaults = root / "vaults"
            managed = vaults / ".chroma"
            legacy = root / ".chroma"

            self.assertEqual(
                vault_indexer._select_chroma_directory(str(root), str(vaults)),
                str(managed),
            )
            legacy.mkdir()
            self.assertEqual(
                vault_indexer._select_chroma_directory(str(root), str(vaults)),
                str(legacy),
            )

    def test_single_file_ignores_stale_vault_path_and_creates_managed_directory(self):
        collection = FakeCollection({})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_directory = root / "external source"
            source_directory.mkdir()
            source_file = source_directory / "notes.txt"
            source_file.write_text("searchable notes", encoding="utf-8")
            vaults_directory = root / "runtime" / "vaults"
            stale_chroma_path = root / "missing legacy" / ".chroma"
            with (
                patch.object(vault_indexer, "VAULTS_DIR", str(vaults_directory)),
                patch.object(vault_indexer, "CHROMA_DIR", str(vaults_directory / ".chroma")),
                patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(
                    vault_indexer,
                    "embed_texts",
                    side_effect=lambda docs, **kwargs: [[1.0] for _ in docs],
                ),
                patch.object(vault_indexer, "register_vault_alias"),
            ):
                payload = json.loads(vault_indexer.index_vault(
                    collection="lecture",
                    file_path=str(source_file),
                    vault_path=str(stale_chroma_path),
                ))

            self.assertTrue(payload["complete"])
            self.assertNotIn("error", payload)
            self.assertEqual(payload["source_root"], str(source_directory))
            self.assertNotIn("vault_directory", payload)
            self.assertNotIn("persist_directory", payload)
            self.assertTrue((vaults_directory / "lecture").is_dir())
            metadata = next(iter(collection.documents.values()))[1]
            self.assertEqual(metadata["source"], "notes.txt")

    def test_empty_source_folder_is_not_reported_as_completed_indexing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_directory = root / "empty source"
            source_directory.mkdir()
            vaults_directory = root / "runtime" / "vaults"
            with (
                patch.object(vault_indexer, "VAULTS_DIR", str(vaults_directory)),
                patch.object(vault_indexer, "CHROMA_DIR", str(vaults_directory / ".chroma")),
                patch.object(vault_indexer, "get_chroma_client") as get_client,
            ):
                payload = json.loads(vault_indexer.index_vault(
                    collection="empty",
                    vault_path=str(source_directory),
                ))

            self.assertFalse(payload["complete"])
            self.assertIn("no supported files", payload["error"])
            self.assertTrue((vaults_directory / "empty").is_dir())
            get_client.assert_not_called()

    def test_status_uses_checkpoint_when_source_and_stale_vault_path_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_source = root / "unmounted" / "notes.pdf"
            vaults_directory = root / "runtime" / "vaults"
            state = {
                "collection": "lecture",
                "fingerprint": "fingerprint",
                "source": "notes.pdf",
                "source_path": str(missing_source),
                "next_page": 21,
                "page_count": 100,
                "indexed_pages": 20,
                "indexed_chunks": 20,
                "complete": False,
                "config": {
                    "chunk_size": 500,
                    "chunk_overlap": 0,
                    "embedding_model": "embed",
                    "vision_mode": "all",
                },
            }
            with patch.object(vault_indexer, "VAULTS_DIR", str(vaults_directory)):
                vault_indexer._save_index_job("lecture", str(missing_source), state)
                status = json.loads(vault_indexer.index_vault(
                    action="status",
                    collection="lecture",
                    file_path=str(missing_source),
                    vault_path=str(root / "missing legacy" / ".chroma"),
                ))
                collection_status = json.loads(vault_indexer.index_vault(
                    action="status",
                    collection="lecture",
                ))
                failed_index = json.loads(vault_indexer.index_vault(
                    action="index",
                    collection="lecture",
                    file_path=str(missing_source),
                    vault_path=str(root / "missing legacy" / ".chroma"),
                ))

        self.assertEqual(status["job_count"], 1)
        self.assertEqual(status["jobs"][0]["next_page"], 21)
        self.assertTrue(status["jobs"][0]["checkpoint_exists"])
        self.assertEqual(collection_status["job_count"], 1)
        self.assertEqual(collection_status["jobs"][0]["source_path"], str(missing_source))
        self.assertIsNone(collection_status["source_root"])
        self.assertIn("source file not found", failed_index["error"])
        self.assertNotIn("vault path", failed_index["error"])

    def test_collection_status_on_fresh_runtime_is_empty_not_path_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            vaults_directory = Path(temporary) / "not-created" / "vaults"
            with patch.object(vault_indexer, "VAULTS_DIR", str(vaults_directory)):
                status = json.loads(vault_indexer.index_vault(
                    action="status",
                    collection="lecture",
                ))

        self.assertFalse(status["complete"])
        self.assertEqual(status["jobs"], [])
        self.assertEqual(status["job_count"], 0)
        self.assertNotIn("error", status)

    def test_resume_keeps_checkpoint_source_label_when_caller_root_changes(self):
        class FakePage:
            def __init__(self, number):
                self.number = number

            def extract_text(self):
                return f"page {self.number} text"

            def get(self, key):
                return {}

        class FakeReader:
            is_encrypted = False

            def __init__(self, stream):
                self.pages = [FakePage(1), FakePage(2)]

        collection = FakeCollection({})
        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "slides.pdf"
            pdf_path.write_bytes(b"fake pdf")
            with (
                patch.dict(sys.modules, {"pypdf": SimpleNamespace(PdfReader=FakeReader)}),
                patch.object(vault_indexer, "VAULTS_DIR", temporary),
                patch.object(vault_indexer, "_pdf_fingerprint", return_value="fingerprint"),
                patch.object(
                    vault_indexer,
                    "embed_texts",
                    side_effect=lambda docs, **kwargs: [[1.0] for _ in docs],
                ),
            ):
                first = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="legacy/root/slides.pdf",
                    collection_name="lecture", collection_obj=collection,
                    chunk_size=500, chunk_overlap=0, model="embed",
                    vision_mode="off", max_pages=1, resume_page=None,
                    cancellation_token=None,
                )
                second = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="slides.pdf",
                    collection_name="lecture", collection_obj=collection,
                    chunk_size=500, chunk_overlap=0, model="embed",
                    vision_mode="off", max_pages=1, resume_page=2,
                    cancellation_token=None,
                )

        self.assertFalse(first["complete"])
        self.assertTrue(second["complete"])
        self.assertEqual(second["source"], "legacy/root/slides.pdf")
        self.assertTrue(all(
            item_id.startswith("legacy/root/slides.pdf::pdf::fingerprint::")
            for item_id in collection.documents
        ))

    def test_pdf_index_resumes_from_atomic_page_checkpoint(self):
        class FakePage:
            def __init__(self, number):
                self.number = number

            def extract_text(self):
                return f"page {self.number} text"

            def get(self, key):
                return {}

        class FakeReader:
            is_encrypted = False

            def __init__(self, stream):
                self.pages = [FakePage(1), FakePage(2), FakePage(3)]

        collection = FakeCollection({
            "old-generation": ("old", {"source": "slides.pdf"})
        })
        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "slides.pdf"
            pdf_path.write_bytes(b"fake pdf")
            fake_pypdf = SimpleNamespace(PdfReader=FakeReader)
            with (
                patch.dict(sys.modules, {"pypdf": fake_pypdf}),
                patch.object(vault_indexer, "VAULTS_DIR", temporary),
                patch.object(vault_indexer, "_pdf_fingerprint", return_value="fingerprint"),
                patch.object(vault_indexer, "_describe_pdf_page", side_effect=lambda path, page, *args: f"visual {page}"),
                patch.object(vault_indexer, "embed_texts", side_effect=lambda docs, **kwargs: [[float(i)] for i, _ in enumerate(docs)]),
            ):
                first = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="slides.pdf", collection_name="lecture",
                    collection_obj=collection, chunk_size=500, chunk_overlap=0,
                    model="embed", vision_mode="all", max_pages=1,
                    resume_page=None,
                    cancellation_token=None,
                )
                second = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="slides.pdf", collection_name="lecture",
                    collection_obj=collection, chunk_size=500, chunk_overlap=0,
                    model="embed", vision_mode="all", max_pages=5,
                    resume_page=2,
                    cancellation_token=None,
                )

        self.assertFalse(first["complete"])
        self.assertEqual(first["next_page"], 2)
        self.assertTrue(second["complete"])
        self.assertEqual(second["indexed_pages"], 3)
        self.assertEqual(second["vision_pages"], 3)
        self.assertIn("old-generation", collection.deleted)
        page_one_ids = [item_id for item_id in collection.documents if "page::1::" in item_id]
        self.assertEqual(len(page_one_ids), 1)

    def test_visual_failures_remain_resumable_and_block_complete_claim(self):
        class FakePage:
            def extract_text(self):
                return "slide text"

            def get(self, key):
                return {}

        class FakeReader:
            is_encrypted = False

            def __init__(self, stream):
                self.pages = [FakePage()]

        collection = FakeCollection({})
        attempts = iter((RuntimeError("moondream unavailable"), "visual details"))

        def describe(*args):
            result = next(attempts)
            if isinstance(result, Exception):
                raise result
            return result

        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "slides.pdf"
            pdf_path.write_bytes(b"fake pdf")
            with (
                patch.dict(sys.modules, {"pypdf": SimpleNamespace(PdfReader=FakeReader)}),
                patch.object(vault_indexer, "VAULTS_DIR", temporary),
                patch.object(vault_indexer, "_pdf_fingerprint", return_value="fingerprint"),
                patch.object(vault_indexer, "_describe_pdf_page", side_effect=describe),
                patch.object(vault_indexer, "embed_texts", side_effect=lambda docs, **kwargs: [[1.0] for _ in docs]),
            ):
                first = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="slides.pdf", collection_name="lecture",
                    collection_obj=collection, chunk_size=500, chunk_overlap=0,
                    model="embed", vision_mode="all", max_pages=1,
                    resume_page=None, cancellation_token=None,
                )
                second = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="slides.pdf", collection_name="lecture",
                    collection_obj=collection, chunk_size=500, chunk_overlap=0,
                    model="embed", vision_mode="all", max_pages=1,
                    resume_page=1, cancellation_token=None,
                )

        self.assertFalse(first["complete"])
        self.assertEqual(first["vision_failed_pages"], [1])
        self.assertEqual(first["next_page"], 1)
        self.assertFalse(first["vision_complete"])
        self.assertTrue(second["complete"])
        self.assertTrue(second["vision_complete"])

    def test_empty_visual_output_remains_an_incomplete_all_vision_job(self):
        class FakePage:
            def extract_text(self):
                return "handwritten page"

            def get(self, key):
                return {}

        class FakeReader:
            is_encrypted = False

            def __init__(self, stream):
                self.pages = [FakePage()]

        collection = FakeCollection({})
        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "notes.pdf"
            pdf_path.write_bytes(b"fake pdf")
            with (
                patch.dict(sys.modules, {"pypdf": SimpleNamespace(PdfReader=FakeReader)}),
                patch.object(vault_indexer, "VAULTS_DIR", temporary),
                patch.object(vault_indexer, "_pdf_fingerprint", return_value="fingerprint"),
                patch.object(vault_indexer, "_describe_pdf_page", return_value=""),
                patch.object(
                    vault_indexer,
                    "embed_texts",
                    side_effect=lambda docs, **kwargs: [[1.0] for _ in docs],
                ),
            ):
                result = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="notes.pdf", collection_name="notes",
                    collection_obj=collection, chunk_size=500, chunk_overlap=0,
                    model="embed", vision_mode="all", max_pages=1,
                    resume_page=None, cancellation_token=None,
                )

        self.assertFalse(result["complete"])
        self.assertEqual(result["next_page"], 1)
        self.assertEqual(result["vision_pages"], 0)
        self.assertEqual(result["vision_failed_pages"], [1])
        self.assertFalse(result["vision_complete"])

    def test_empty_moondream_response_uses_short_handwriting_fallback(self):
        fake_pdf2image = SimpleNamespace(
            convert_from_path=lambda *args, **kwargs: ["/tmp/rendered-page.jpg"]
        )
        with (
            patch.dict(sys.modules, {"pdf2image": fake_pdf2image}),
            patch(
                "tools.vision_describer.describe_image",
                side_effect=(
                    "Error describing image: Ollama returned an empty response",
                    "Recovered handwritten content",
                ),
            ) as describe,
        ):
            result = vault_indexer._describe_pdf_page(
                "/tmp/notes.pdf",
                1,
                "/tmp",
                None,
            )

        self.assertEqual(result, "Recovered handwritten content")
        self.assertEqual(describe.call_count, 2)
        self.assertEqual(
            describe.call_args_list[1].kwargs["prompt"],
            "Describe this page in detail.",
        )

    def test_vision_retry_queue_skips_pages_that_already_succeeded(self):
        class FakePage:
            def __init__(self, number):
                self.number = number

            def extract_text(self):
                return f"page {self.number}"

            def get(self, key):
                return {}

        class FakeReader:
            is_encrypted = False

            def __init__(self, stream):
                self.pages = [FakePage(number) for number in range(1, 7)]

        collection = FakeCollection({})
        described_pages = []
        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "slides.pdf"
            pdf_path.write_bytes(b"fake pdf")
            state = {
                "version": 2,
                "collection": "lecture",
                "source": "slides.pdf",
                "source_path": str(pdf_path),
                "fingerprint": "fingerprint",
                "revision": 0,
                "next_page": 1,
                "indexed_pages": 6,
                "indexed_chunks": 4,
                "vision_pages": 4,
                "page_count": 6,
                "page_chunk_counts": {str(page): 1 for page in (2, 3, 4, 6)},
                "vision_completed_pages": [2, 3, 4, 6],
                "vision_failed_pages": [1, 5],
                "warning_count": 2,
                "warnings": ["page 1 failed", "page 5 failed"],
                "complete": False,
                "config": {
                    "chunk_size": 500,
                    "chunk_overlap": 0,
                    "embedding_model": "embed",
                    "vision_mode": "all",
                },
            }
            with (
                patch.dict(sys.modules, {"pypdf": SimpleNamespace(PdfReader=FakeReader)}),
                patch.object(vault_indexer, "VAULTS_DIR", temporary),
                patch.object(vault_indexer, "_pdf_fingerprint", return_value="fingerprint"),
                patch.object(
                    vault_indexer,
                    "_describe_pdf_page",
                    side_effect=lambda path, page, *args: described_pages.append(page) or f"visual {page}",
                ),
                patch.object(vault_indexer, "embed_texts", side_effect=lambda docs, **kwargs: [[1.0] for _ in docs]),
            ):
                vault_indexer._save_index_job("lecture", str(pdf_path), state)
                first = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="slides.pdf", collection_name="lecture",
                    collection_obj=collection, chunk_size=500, chunk_overlap=0,
                    model="embed", vision_mode="all", max_pages=1,
                    resume_page=1, cancellation_token=None,
                )
                second = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="slides.pdf", collection_name="lecture",
                    collection_obj=collection, chunk_size=500, chunk_overlap=0,
                    model="embed", vision_mode="all", max_pages=1,
                    resume_page=5, cancellation_token=None,
                )

        self.assertEqual(described_pages, [1, 5])
        self.assertEqual(first["next_page"], 5)
        self.assertTrue(second["complete"])

    def test_resume_inherits_durable_all_vision_policy(self):
        class FakePage:
            def __init__(self, number):
                self.number = number

            def extract_text(self):
                return "text " * 150

            def get(self, key):
                return {}

        class FakeReader:
            is_encrypted = False

            def __init__(self, stream):
                self.pages = [FakePage(1), FakePage(2)]

        collection = FakeCollection({})
        described_pages = []
        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "slides.pdf"
            pdf_path.write_bytes(b"fake pdf")
            with (
                patch.dict(sys.modules, {"pypdf": SimpleNamespace(PdfReader=FakeReader)}),
                patch.object(vault_indexer, "VAULTS_DIR", temporary),
                patch.object(vault_indexer, "_pdf_fingerprint", return_value="fingerprint"),
                patch.object(
                    vault_indexer,
                    "_describe_pdf_page",
                    side_effect=lambda path, page, *args: described_pages.append(page) or f"visual {page}",
                ),
                patch.object(vault_indexer, "embed_texts", side_effect=lambda docs, **kwargs: [[1.0] for _ in docs]),
            ):
                first = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="slides.pdf", collection_name="lecture",
                    collection_obj=collection, chunk_size=500, chunk_overlap=0,
                    model="embed", vision_mode="all", max_pages=1,
                    resume_page=None, cancellation_token=None,
                )
                second = vault_indexer._index_pdf_incremental(
                    path=str(pdf_path), rel="slides.pdf", collection_name="lecture",
                    collection_obj=collection, chunk_size=500, chunk_overlap=0,
                    model="embed", vision_mode="auto", max_pages=1,
                    resume_page=2, cancellation_token=None,
                    vision_mode_explicit=False,
                )
                with self.assertRaisesRegex(ValueError, "durable job setting"):
                    vault_indexer._index_pdf_incremental(
                        path=str(pdf_path), rel="slides.pdf", collection_name="lecture",
                        collection_obj=collection, chunk_size=500, chunk_overlap=0,
                        model="embed", vision_mode="auto", max_pages=1,
                        resume_page=None, cancellation_token=None,
                        vision_mode_explicit=True,
                    )

        self.assertFalse(first["complete"])
        self.assertTrue(second["complete"])
        self.assertEqual(second["vision_mode"], "all")
        self.assertEqual(described_pages, [1, 2])

    def test_public_pdf_failure_returns_incomplete_checkpoint_not_false_success(self):
        class FakePage:
            def __init__(self, number):
                self.number = number

            def extract_text(self):
                return f"page {self.number} text"

            def get(self, key):
                return {}

        class FakeReader:
            is_encrypted = False

            def __init__(self, stream):
                self.pages = [FakePage(1), FakePage(2)]

        collection = FakeCollection({})
        embed_attempt = 0

        def embed(documents, **kwargs):
            nonlocal embed_attempt
            embed_attempt += 1
            if embed_attempt == 2:
                raise RuntimeError("embedding model stopped")
            return [[1.0] for _ in documents]

        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "slides.pdf"
            pdf_path.write_bytes(b"fake pdf")
            with (
                patch.dict(sys.modules, {"pypdf": SimpleNamespace(PdfReader=FakeReader)}),
                patch.object(vault_indexer, "VAULTS_DIR", temporary),
                patch.object(vault_indexer, "CHROMA_DIR", str(Path(temporary) / ".chroma")),
                patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(vault_indexer, "_pdf_fingerprint", return_value="fingerprint"),
                patch.object(vault_indexer, "embed_texts", side_effect=embed),
                patch.object(vault_indexer, "register_vault_alias"),
            ):
                payload = json.loads(vault_indexer.index_vault(
                    file_path=str(pdf_path),
                    collection="lecture",
                    include_vision=False,
                    vision_mode="off",
                    max_pages=2,
                    chunk_size=500,
                    chunk_overlap=0,
                    model="embed",
                ))

        self.assertFalse(payload["complete"])
        self.assertIn("embedding model stopped", payload["error"])
        self.assertEqual(payload["incomplete_pdf_count"], 1)
        self.assertEqual(payload["failed_pdf_count"], 1)
        self.assertEqual(payload["pdf_jobs"][0]["indexed_pages"], 1)
        self.assertEqual(payload["pdf_jobs"][0]["next_page"], 2)
        self.assertEqual(payload["indexed_chunks"], payload["pdf_jobs"][0]["indexed_chunks"])

    def test_public_continuation_round_trip_preserves_custom_chunk_settings(self):
        class FakePage:
            def __init__(self, number):
                self.number = number

            def extract_text(self):
                return f"page {self.number} text"

            def get(self, key):
                return {}

        class FakeReader:
            is_encrypted = False

            def __init__(self, stream):
                self.pages = [FakePage(1), FakePage(2)]

        collection = FakeCollection({})
        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "slides.pdf"
            pdf_path.write_bytes(b"fake pdf")
            with (
                patch.dict(sys.modules, {"pypdf": SimpleNamespace(PdfReader=FakeReader)}),
                patch.object(vault_indexer, "VAULTS_DIR", temporary),
                patch.object(vault_indexer, "CHROMA_DIR", str(Path(temporary) / ".chroma")),
                patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(vault_indexer, "_pdf_fingerprint", return_value="fingerprint"),
                patch.object(
                    vault_indexer,
                    "embed_texts",
                    side_effect=lambda docs, **kwargs: [[1.0] for _ in docs],
                ),
                patch.object(vault_indexer, "register_vault_alias"),
            ):
                first = json.loads(vault_indexer.index_vault(
                    file_path=str(pdf_path), collection="lecture",
                    vault_path=str(Path(temporary) / "missing" / ".chroma"),
                    vision_mode="off", max_pages=1,
                    chunk_size=700, chunk_overlap=0, model="embed",
                ))
                continuation = first["continuation"]["arguments"]
                second = json.loads(vault_indexer.index_vault(**continuation))

        self.assertEqual(continuation["chunk_size"], 700)
        self.assertEqual(continuation["chunk_overlap"], 0)
        self.assertNotIn("vault_path", continuation)
        self.assertNotIn("persist_directory", first)
        self.assertNotIn("vault_directory", first)
        self.assertTrue(second["complete"])
        self.assertEqual(second["chunk_size"], 700)
        self.assertEqual(second["chunk_overlap"], 0)

    def test_first_page_failure_saves_a_truthful_checkpoint_visible_to_status(self):
        class FakePage:
            def extract_text(self):
                return "page text"

            def get(self, key):
                return {}

        class FakeReader:
            is_encrypted = False

            def __init__(self, stream):
                self.pages = [FakePage(), FakePage()]

        collection = FakeCollection({})
        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "slides.pdf"
            pdf_path.write_bytes(b"fake pdf")
            with (
                patch.dict(sys.modules, {"pypdf": SimpleNamespace(PdfReader=FakeReader)}),
                patch.object(vault_indexer, "VAULTS_DIR", temporary),
                patch.object(vault_indexer, "CHROMA_DIR", str(Path(temporary) / ".chroma")),
                patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(vault_indexer, "_pdf_fingerprint", return_value="fingerprint"),
                patch.object(vault_indexer, "embed_texts", side_effect=RuntimeError("embed offline")),
                patch.object(vault_indexer, "register_vault_alias"),
            ):
                failed = json.loads(vault_indexer.index_vault(
                    file_path=str(pdf_path), collection="lecture",
                    vision_mode="off", max_pages=2,
                    chunk_size=500, chunk_overlap=0, model="embed",
                ))
                status = json.loads(vault_indexer.index_vault(
                    action="status", file_path=str(pdf_path), collection="lecture"
                ))

        job = failed["pdf_jobs"][0]
        self.assertFalse(failed["complete"])
        self.assertIn("checkpoint", failed["error"])
        self.assertEqual(job["page_count"], 2)
        self.assertEqual(job["next_page"], 1)
        self.assertEqual(job["fingerprint"], "fingerprint")
        self.assertEqual(job["vision_mode"], "off")
        self.assertTrue(job["retryable"])
        self.assertEqual(status["job_count"], 1)
        self.assertTrue(status["jobs"][0]["checkpoint_exists"])
        self.assertEqual(status["jobs"][0]["next_page"], 1)

    def test_folder_status_includes_pdfs_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            indexed = root / "indexed.pdf"
            missing = root / "missing.pdf"
            indexed.write_bytes(b"one")
            missing.write_bytes(b"two")
            state = {
                "fingerprint": "fingerprint",
                "next_page": 2,
                "page_count": 1,
                "indexed_pages": 1,
                "indexed_chunks": 1,
                "complete": True,
                "config": {
                    "chunk_size": 500,
                    "chunk_overlap": 0,
                    "embedding_model": "embed",
                    "vision_mode": "off",
                },
            }
            with patch.object(vault_indexer, "VAULTS_DIR", temporary):
                vault_indexer._save_index_job("lecture", str(indexed), state)
                status = json.loads(vault_indexer.index_vault(
                    action="status", vault_path=temporary, collection="lecture"
                ))

        self.assertFalse(status["complete"])
        self.assertEqual(status["job_count"], 2)
        self.assertEqual(
            sorted(job["checkpoint_exists"] for job in status["jobs"]),
            [False, True],
        )


class PdfToolTests(unittest.TestCase):
    def test_incomplete_slide_index_blocks_notes_and_returns_resume_call(self):
        records = [{
            "id": "one",
            "metadata": {
                "source": "slides.pdf",
                "source_path": "/tmp/slides.pdf",
                "extension": ".pdf",
                "page": 1,
            },
        }]
        job = {
            "complete": False,
            "next_page": 2,
            "source": "slides.pdf",
            "source_path": "/tmp/slides.pdf",
            "vision_mode": "all",
            "chunk_size": 500,
            "chunk_overlap": 0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(pdf_writer, "EXPORTS_DIR", root / "exports"),
                patch.object(pdf_writer, "PDF_JOBS_DIR", root / "jobs"),
                patch.object(vault_search, "ordered_vault_records", return_value=records),
                patch.object(
                    vault_indexer,
                    "_collection_index_job_summaries",
                    return_value=[job],
                ),
                patch.object(pdf_writer, "_generate_note_section") as generate,
            ):
                payload = json.loads(pdf_writer.build_vault_notes_pdf(
                    collection="lecture",
                    file_path="notes.pdf",
                    require_vision=True,
                ))

        self.assertEqual(payload["error_code"], "vault_index_incomplete")
        self.assertEqual(
            payload["continuations"][0]["arguments"]["resume_page"],
            2,
        )
        generate.assert_not_called()

    def test_slide_notes_require_verified_all_page_vision(self):
        records = [{
            "id": "one",
            "metadata": {
                "source": "slides.pdf",
                "source_path": "/tmp/slides.pdf",
                "extension": ".pdf",
                "page": 1,
            },
        }]
        job = {
            "complete": True,
            "source": "slides.pdf",
            "source_path": "/tmp/slides.pdf",
            "page_count": 10,
            "vision_mode": "auto",
            "vision_pages": 4,
            "vision_complete": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(pdf_writer, "EXPORTS_DIR", root / "exports"),
                patch.object(pdf_writer, "PDF_JOBS_DIR", root / "jobs"),
                patch.object(vault_search, "ordered_vault_records", return_value=records),
                patch.object(
                    vault_indexer,
                    "_collection_index_job_summaries",
                    return_value=[job],
                ),
            ):
                payload = json.loads(pdf_writer.build_vault_notes_pdf(
                    collection="lecture",
                    file_path="notes.pdf",
                    require_vision=True,
                ))

        self.assertEqual(payload["error_code"], "vision_coverage_incomplete")
        self.assertIn("vision_mode=all", payload["guidance"])

    def test_slide_notes_require_a_checkpoint_for_every_selected_pdf(self):
        records = [
            {
                "id": "one",
                "metadata": {
                    "source": "first.pdf",
                    "source_path": "/tmp/first.pdf",
                    "extension": ".pdf",
                    "page": 1,
                },
            },
            {
                "id": "two",
                "metadata": {
                    "source": "second.pdf",
                    "source_path": "/tmp/second.pdf",
                    "extension": ".pdf",
                    "page": 1,
                },
            },
        ]
        job = {
            "complete": True,
            "source": "first.pdf",
            "source_path": "/tmp/first.pdf",
            "page_count": 1,
            "vision_mode": "all",
            "vision_pages": 1,
            "vision_complete": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(pdf_writer, "EXPORTS_DIR", root / "exports"),
                patch.object(pdf_writer, "PDF_JOBS_DIR", root / "jobs"),
                patch.object(vault_search, "ordered_vault_records", return_value=records),
                patch.object(
                    vault_indexer,
                    "_collection_index_job_summaries",
                    return_value=[job],
                ),
            ):
                payload = json.loads(pdf_writer.build_vault_notes_pdf(
                    collection="lecture",
                    file_path="notes.pdf",
                    require_vision=True,
                ))

        self.assertEqual(payload["error_code"], "vision_source_unverified")
        self.assertEqual(
            payload["unverified_sources"],
            [{"identity_kind": "path", "identity": "/tmp/second.pdf"}],
        )

    def test_verified_vision_content_flows_into_slide_notes(self):
        visual_content = (
            "[Page 1 extracted text]\nGradient descent\n\n"
            "[Page 1 visual analysis]\nChart falls from loss 10 to loss 2."
        )
        collection = FakeCollection({
            "one": (visual_content, {
                "source": "slides.pdf",
                "source_path": "/tmp/slides.pdf",
                "extension": ".pdf",
                "page": 1,
                "page_count": 1,
                "chunk_index": 0,
                "char_start": 0,
                "char_end": len(visual_content),
                "content_kind": "text+vision",
            }),
        })
        records = [{"id": "one", "metadata": collection.documents["one"][1]}]
        job = {
            "complete": True,
            "source": "slides.pdf",
            "source_path": "/tmp/slides.pdf",
            "page_count": 1,
            "vision_mode": "all",
            "vision_pages": 1,
            "vision_complete": True,
        }
        captured = {}

        def generate(source_text, title, cancellation_token):
            captured["source_text"] = source_text
            return "# Notes\n\nLoss decreases from 10 to 2."

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fake_render(path, title, content):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"%PDF-notes")

            with (
                patch.object(pdf_writer, "EXPORTS_DIR", root / "exports"),
                patch.object(pdf_writer, "PDF_JOBS_DIR", root / "jobs"),
                patch.object(vault_search, "ordered_vault_records", return_value=records),
                patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(
                    vault_indexer,
                    "_collection_index_job_summaries",
                    return_value=[job],
                ),
                patch.object(pdf_writer, "_generate_note_section", side_effect=generate),
                patch.object(pdf_writer, "_render_pdf", side_effect=fake_render),
            ):
                payload = json.loads(pdf_writer.build_vault_notes_pdf(
                    collection="lecture",
                    file_path="notes.pdf",
                    require_vision=True,
                ))

        self.assertTrue(payload["created"])
        self.assertTrue(payload["vision_verified"])
        self.assertIn("Kind: text+vision", captured["source_text"])
        self.assertIn("Chart falls from loss 10 to loss 2", captured["source_text"])

    def test_tampered_committed_note_section_is_rejected(self):
        collection = FakeCollection({
            "one": ("A" * 7000, {
                "source": "notes.txt",
                "chunk_index": 0,
                "char_start": 0,
                "char_end": 7000,
            }),
            "two": ("tail", {
                "source": "notes.txt",
                "chunk_index": 1,
                "char_start": 7000,
                "char_end": 7004,
            }),
        })
        records = [
            {"id": "one", "metadata": collection.documents["one"][1]},
            {"id": "two", "metadata": collection.documents["two"][1]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = root / "jobs"
            with (
                patch.object(pdf_writer, "EXPORTS_DIR", root / "exports"),
                patch.object(pdf_writer, "PDF_JOBS_DIR", jobs),
                patch.object(vault_search, "ordered_vault_records", return_value=records),
                patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(pdf_writer, "_generate_note_section", return_value="# Valid notes"),
            ):
                first = json.loads(pdf_writer.build_vault_notes_pdf(
                    collection="lecture",
                    file_path="notes.pdf",
                    sections_per_run=1,
                ))
                section = next(Path(first["job_directory"]).glob("section-*.md"))
                section.write_text("tampered", encoding="utf-8")
                second = json.loads(pdf_writer.build_vault_notes_pdf(
                    collection="lecture",
                    file_path="notes.pdf",
                    cursor=first["next_cursor"],
                    sections_per_run=1,
                ))

        self.assertIn("integrity", second["error"])
        self.assertTrue(second["preserved"])

    def test_export_rejects_missing_chroma_rows(self):
        class MissingCollection(FakeCollection):
            def get(self, ids=None, **kwargs):
                return {"ids": [], "documents": [], "metadatas": []}

        records = [{
            "id": "missing",
            "metadata": {"source": "notes.txt", "chunk_index": 0},
        }]
        with (
            patch.object(vault_search, "ordered_vault_records", return_value=records),
            patch.object(
                vault_indexer,
                "get_chroma_client",
                return_value=FakeClient(MissingCollection({})),
            ),
            patch.object(pdf_writer, "create_pdf") as create,
        ):
            payload = json.loads(pdf_writer.export_vault_pdf(
                collection="lecture",
                file_path="export.pdf",
            ))

        self.assertIn("omitted", payload["error"])
        create.assert_not_called()

    def test_finalizing_job_recovers_after_pdf_was_atomically_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exports = root / "exports"
            jobs = root / "jobs"
            output = exports / "notes.pdf"
            output.parent.mkdir(parents=True)
            output.write_bytes(b"%PDF-recovered")
            with (
                patch.object(pdf_writer, "EXPORTS_DIR", exports),
                patch.object(pdf_writer, "PDF_JOBS_DIR", jobs),
            ):
                job_dir, state_path = pdf_writer._notes_job_paths(
                    "lecture",
                    None,
                    output,
                )
                job_dir.mkdir(parents=True)
                pdf_writer.atomic_write_json(state_path, {
                    "version": 2,
                    "collection": "lecture",
                    "source": None,
                    "file_path": str(output),
                    "require_vision": False,
                    "finalizing": True,
                    "complete": False,
                    "next_cursor": "1:0",
                    "total_chunks": 1,
                    "completed_sections": 1,
                    "section_files": ["section.md"],
                })
                payload = json.loads(pdf_writer.build_vault_notes_pdf(
                    collection="lecture",
                    file_path="notes.pdf",
                ))

        self.assertTrue(payload["complete"])
        self.assertTrue(payload["already_complete"])
        self.assertEqual(payload["bytes"], len(b"%PDF-recovered"))

    def test_finalizing_job_does_not_claim_an_unchanged_old_pdf(self):
        collection = FakeCollection({
            "one": ("source facts", {
                "source": "notes.txt",
                "chunk_index": 0,
                "char_start": 0,
                "char_end": 12,
            }),
        })
        records = [{"id": "one", "metadata": collection.documents["one"][1]}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exports = root / "exports"
            jobs = root / "jobs"
            output = exports / "notes.pdf"
            output.parent.mkdir(parents=True)
            output.write_bytes(b"%PDF-old")
            old_hash = pdf_writer._file_sha256(output)
            with (
                patch.object(pdf_writer, "EXPORTS_DIR", exports),
                patch.object(pdf_writer, "PDF_JOBS_DIR", jobs),
            ):
                job_dir, state_path = pdf_writer._notes_job_paths(
                    "lecture",
                    None,
                    output,
                )
                job_dir.mkdir(parents=True)
                section_name = "section-00000000-00000000--00000001-00000000.md"
                section_text = "# Notes\n\nsource facts"
                (job_dir / section_name).write_text(section_text, encoding="utf-8")
                selection_digest = hashlib.sha256(
                    json.dumps(
                        [{"id": "one", "metadata": records[0]["metadata"]}],
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                pdf_writer.atomic_write_json(state_path, {
                    "version": 2,
                    "collection": "lecture",
                    "source": None,
                    "file_path": str(output),
                    "title": "Notes",
                    "require_vision": False,
                    "vision_verified": False,
                    "selection_digest": selection_digest,
                    "finalizing": True,
                    "complete": False,
                    "next_cursor": "1:0",
                    "total_chunks": 1,
                    "completed_sections": 1,
                    "section_files": [section_name],
                    "section_manifest": {
                        section_name: pdf_writer._section_integrity(section_text),
                    },
                    "output_existed_before_finalizing": True,
                    "previous_output_sha256": old_hash,
                })
                with (
                    patch.object(vault_search, "ordered_vault_records", return_value=records),
                    patch.object(
                        vault_indexer,
                        "get_chroma_client",
                        return_value=FakeClient(collection),
                    ),
                ):
                    payload = json.loads(pdf_writer.build_vault_notes_pdf(
                        collection="lecture",
                        file_path="notes.pdf",
                    ))

        self.assertFalse(payload.get("complete", False))
        self.assertTrue(payload["overwrite_required"])

    def test_create_pdf_is_non_overwriting_and_uses_export_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fake_render(path, title, content):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"%PDF-test")

            with (
                patch.object(pdf_writer, "EXPORTS_DIR", root),
                patch.object(pdf_writer, "_render_pdf", side_effect=fake_render),
            ):
                first = json.loads(pdf_writer.create_pdf("notes.pdf", "# Notes", "Lecture"))
                second = json.loads(pdf_writer.create_pdf("notes.pdf", "replacement", "Lecture"))
                unconfirmed = json.loads(pdf_writer.create_pdf(
                    "notes.pdf", "replacement", "Lecture", overwrite=True
                ))

            self.assertTrue(first["created"])
            self.assertEqual(Path(first["file_path"]), root / "notes.pdf")
            self.assertTrue(second["overwrite_required"])
            self.assertIn("confirmed=true", unconfirmed["error"])

    def test_refined_notes_job_resumes_and_finalizes(self):
        long_document = "A" * 7000
        collection = FakeCollection({
            "one": (long_document, {"source": "slides.pdf", "page": 1, "chunk_index": 0}),
            "two": ("final facts", {"source": "slides.pdf", "page": 2, "chunk_index": 0}),
        })
        records = [
            {"id": "one", "metadata": collection.documents["one"][1]},
            {"id": "two", "metadata": collection.documents["two"][1]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fake_render(path, title, content):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"%PDF-notes")

            with (
                patch.object(pdf_writer, "EXPORTS_DIR", root / "exports"),
                patch.object(pdf_writer, "PDF_JOBS_DIR", root / "jobs"),
                patch.object(vault_search, "ordered_vault_records", return_value=records),
                patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
                patch.object(pdf_writer, "_generate_note_section", return_value="# Refined notes"),
                patch.object(pdf_writer, "_render_pdf", side_effect=fake_render),
            ):
                first = json.loads(pdf_writer.build_vault_notes_pdf(
                    collection="lecture", file_path="notes.pdf", sections_per_run=1
                ))
                second = json.loads(pdf_writer.build_vault_notes_pdf(
                    collection="lecture", file_path="notes.pdf",
                    cursor=first["next_cursor"], sections_per_run=1,
                ))

            self.assertFalse(first["complete"])
            self.assertTrue(second["created"])
            self.assertTrue(second["refined"])
            self.assertTrue((root / "exports" / "notes.pdf").is_file())

    def test_vault_export_removes_chunk_overlap_without_dropping_tail(self):
        collection = FakeCollection({
            "one": ("abcdefghij", {
                "source": "slides.pdf", "page": 1, "chunk_index": 0,
                "char_start": 0, "char_end": 10,
            }),
            "two": ("ijklmnop", {
                "source": "slides.pdf", "page": 1, "chunk_index": 1,
                "char_start": 8, "char_end": 16,
            }),
        })
        records = [
            {"id": "one", "metadata": collection.documents["one"][1]},
            {"id": "two", "metadata": collection.documents["two"][1]},
        ]
        captured = {}

        def fake_create_pdf(**kwargs):
            captured.update(kwargs)
            return json.dumps({"created": True, "file_path": "/tmp/export.pdf", "bytes": 10})

        with (
            patch.object(vault_search, "ordered_vault_records", return_value=records),
            patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
            patch.object(pdf_writer, "create_pdf", side_effect=fake_create_pdf),
        ):
            result = json.loads(pdf_writer.export_vault_pdf(
                collection="lecture", file_path="export.pdf"
            ))

        normalized = captured["content"].replace("\n", "")
        self.assertIn("abcdefghijklmnop", normalized)
        self.assertNotIn("abcdefghijijklmnop", normalized)
        self.assertEqual(result["exported_chunks"], 2)

    def test_vault_export_does_not_deduplicate_across_different_files(self):
        collection = FakeCollection({
            "one": ("abcdefghij", {
                "source": "slides.pdf", "source_path": "/tmp/a/slides.pdf",
                "page": 1, "chunk_index": 0, "char_start": 0, "char_end": 10,
            }),
            "two": ("ijklmnop", {
                "source": "slides.pdf", "source_path": "/tmp/b/slides.pdf",
                "page": 1, "chunk_index": 0, "char_start": 8, "char_end": 16,
            }),
        })
        records = [
            {"id": "one", "metadata": collection.documents["one"][1]},
            {"id": "two", "metadata": collection.documents["two"][1]},
        ]
        captured = {}

        def fake_create_pdf(**kwargs):
            captured.update(kwargs)
            return json.dumps({"created": True, "file_path": "/tmp/export.pdf", "bytes": 10})

        with (
            patch.object(vault_search, "ordered_vault_records", return_value=records),
            patch.object(vault_indexer, "get_chroma_client", return_value=FakeClient(collection)),
            patch.object(
                vault_indexer,
                "_collection_index_job_summaries",
                return_value=[],
            ),
            patch.object(pdf_writer, "create_pdf", side_effect=fake_create_pdf),
        ):
            payload = json.loads(pdf_writer.export_vault_pdf(
                collection="lecture",
                file_path="export.pdf",
            ))

        normalized = captured["content"].replace("\n", "")
        self.assertIn("abcdefghij", normalized)
        self.assertIn("ijklmnop", normalized)
        self.assertEqual(payload["exported_chunks"], 2)


if __name__ == "__main__":
    unittest.main()
