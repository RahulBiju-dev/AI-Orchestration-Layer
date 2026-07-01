import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import app_launcher
from tools.browser import open_browser
from tools.code import _read_single_file, view_code
from tools.document import _chunk_text, _iter_docx_blocks, _parse_page_spec
from tools.file import create_file, read_file
from tools.obsi_vault_writer import create_structured_note
from tools.spotify import _url_to_uri, spotify_play
from tools.vault_embeddings import _validate_embedding_count, normalize_embeddings
from tools.vault_indexer import chunk_text_with_offsets, index_vault, resolve_vault_alias
from tools.vault_search import _query_collection
from tools.vision_describer import describe_image


class LegacyToolTests(unittest.TestCase):
    def test_desktop_exec_codes_are_removed_without_shell_parsing(self):
        self.assertEqual(app_launcher._clean_exec_line('viewer --name "A B" %U %%'), ["viewer", "--name", "A B", "%"])

    def test_linux_app_matcher_resolves_common_name_to_desktop_entry(self):
        apps = [{
            "filename": "code.desktop",
            "name": "Visual Studio Code",
            "generic_name": "Text Editor",
            "keywords": ["vscode"],
            "exec": "/usr/share/code/code %F",
            "terminal": False,
        }]
        matched, suggestions = app_launcher._find_matching_app("VS Code", apps)
        self.assertIs(matched, apps[0])
        self.assertEqual(suggestions, [])

    def test_linux_app_matcher_uses_name_abbreviation_without_known_alias(self):
        apps = [{
            "filename": "example.desktop",
            "name": "Visual Studio Code",
            "generic_name": "",
            "keywords": [],
            "exec": "/opt/editor/bin/code %F",
            "terminal": False,
        }]
        matched, _ = app_launcher._find_matching_app("VS Code", apps)
        self.assertIs(matched, apps[0])

    def test_linux_app_matcher_does_not_choose_an_ambiguous_alias(self):
        apps = [
            {"filename": "one.desktop", "name": "One", "generic_name": "File Manager", "keywords": [], "exec": "one", "terminal": False},
            {"filename": "two.desktop", "name": "Two", "generic_name": "File Manager", "keywords": [], "exec": "two", "terminal": False},
        ]
        matched, suggestions = app_launcher._find_matching_app("file manager", apps)
        self.assertIsNone(matched)
        self.assertEqual(suggestions, ["One", "Two"])

    def test_windows_app_launch_uses_startfile_not_shell(self):
        with patch.object(app_launcher.sys, "platform", "win32"), patch.object(app_launcher.os, "startfile", create=True) as start:
            result = json.loads(app_launcher.open_app('safe & literal'))
        self.assertTrue(result["success"])
        start.assert_called_once_with('safe & literal')

    def test_browser_distinguishes_domain_from_search(self):
        with patch("tools.browser.webbrowser.open", return_value=True) as opened:
            self.assertIn("Successfully", open_browser("example.com/path"))
            self.assertEqual(opened.call_args.args[0], "https://example.com/path")
            open_browser("release notes please")
            self.assertTrue(opened.call_args.args[0].startswith("https://duckduckgo.com/?q="))

    def test_browser_resolves_common_web_app_names(self):
        with patch("tools.browser.webbrowser.open", return_value=True) as opened:
            open_browser("Google Mail")
            self.assertEqual(opened.call_args.args[0], "https://mail.google.com/")
            open_browser("google-docs")
            self.assertEqual(opened.call_args.args[0], "https://docs.google.com/")
            open_browser("Figma")
            self.assertEqual(opened.call_args.args[0], "https://www.figma.com/")

    def test_browser_does_not_treat_generic_app_words_as_aliases(self):
        with patch("tools.browser.webbrowser.open", return_value=True) as opened:
            open_browser("documents")
            self.assertTrue(opened.call_args.args[0].startswith("https://duckduckgo.com/?q="))

    def test_code_reader_streams_ranges_and_scanner_skips_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
            (root / "node_modules").mkdir()
            (root / "node_modules" / "hidden.py").write_text("hidden", encoding="utf-8")
            result = _read_single_file(str(root / "main.py"), "3-2")
            self.assertEqual(result["displayed_lines"], "2-3")
            scan = json.loads(view_code(str(root), extension="py"))
            self.assertEqual(scan["files_found"], 1)

    def test_document_helpers_preserve_boundaries_and_block_order(self):
        text = ("A" * 600) + ". " + ("B" * 600)
        chunks = _chunk_text(text, 1000)
        self.assertEqual("".join(chunks), text)
        self.assertEqual(_parse_page_spec("3-2,1", 3), [0, 1, 2])
        self.assertEqual(_parse_page_spec("1-999999999", 3), [0, 1, 2])
        paragraph = types.SimpleNamespace(text="first")
        cell = types.SimpleNamespace(text="cell")
        table = types.SimpleNamespace(rows=[types.SimpleNamespace(cells=[cell])])
        document = types.SimpleNamespace(iter_inner_content=lambda: iter([paragraph, table]))
        self.assertEqual(list(_iter_docx_blocks(document)), ["first", "cell"])

    def test_read_file_large_range_does_not_require_full_read(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as stream:
            stream.write("a\nb\nc\n")
            path = stream.name
        try:
            result = json.loads(read_file(path, lines="2-3"))
            self.assertEqual(result["mode"], "lines")
            self.assertIn("b", result["text"])
        finally:
            os.unlink(path)

    def test_create_file_never_overwrites_and_reports_index_state(self):
        import tools.vault_indexer as indexer
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(indexer, "VAULTS_DIR", directory), \
             patch.object(indexer, "index_vault", return_value='{"indexed_files":1}'), \
             patch.object(indexer, "register_vault_alias"):
            first = json.loads(create_file("note.md", "first"))
            second = json.loads(create_file("note.md", "second"))
            self.assertTrue(first["success"])
            self.assertIn("error", second)
            self.assertEqual(Path(directory, "note.md").read_text(encoding="utf-8"), "first")

    def test_obsidian_writer_quotes_tags_sanitizes_links_and_versions_atomically(self):
        import tools.obsi_vault_writer as writer
        with tempfile.TemporaryDirectory() as directory, patch.object(writer, "VAULTS_DIR", directory):
            first = json.loads(create_structured_note("Test\nTitle", "body", incoming_links=["bad]]\nlink"], tags=["a: b"]))
            second = json.loads(create_structured_note("Test\nTitle", "body"))
            content = Path(first["filepath"]).read_text(encoding="utf-8")
            self.assertIn('  - "a: b"', content)
            self.assertIn("[[bad] ] link]]", content)
            self.assertNotIn("# link", content)
            self.assertNotEqual(first["filepath"], second["filepath"])

    def test_search_keeps_urls_and_deduplicates(self):
        class FakeDDGS:
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def text(self, query, max_results):
                return [{"title": "A", "href": "https://a.test", "body": "one"}, {"title": "A2", "href": "https://a.test", "body": "two"}]
        fake_module = types.SimpleNamespace(DDGS=FakeDDGS)
        with patch.dict(sys.modules, {"ddgs": fake_module}):
            from tools.search import web_search
            results = json.loads(web_search("topic", "easy"))
        self.assertEqual(results, [{"title": "A", "url": "https://a.test", "snippet": "one"}])

    def test_spotify_url_conversion_and_macos_argument_separation(self):
        import tools.spotify as spotify
        self.assertEqual(_url_to_uri("https://open.spotify.com/track/abc123?si=x"), "spotify:track:abc123")
        completed = Mock(returncode=0)
        with patch.object(spotify.sys, "platform", "darwin"), patch.object(spotify, "_launch_spotify", return_value=True), patch.object(spotify.subprocess, "run", return_value=completed) as run:
            result = json.loads(spotify_play("spotify:track:abc123"))
        self.assertTrue(result["success"])
        self.assertEqual(run.call_args.args[0][-1], "spotify:track:abc123")

    def test_embedding_validation_rejects_bad_vectors(self):
        self.assertEqual(normalize_embeddings({"embedding": [1, 2]}), [[1, 2]])
        with self.assertRaises(RuntimeError):
            _validate_embedding_count([[float("nan")]], 1)

    def test_chunk_offsets_are_monotonic(self):
        chunks = chunk_text_with_offsets("paragraph one. " * 100, chunk_size=500, chunk_overlap=100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(current["char_start"] < following["char_start"] for current, following in zip(chunks, chunks[1:])))

    def test_index_embedding_failure_preserves_existing_collection(self):
        collection = Mock()
        client = Mock()
        client.get_or_create_collection.return_value = collection
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "note.txt")
            path.write_text("knowledge", encoding="utf-8")
            with patch("tools.vault_indexer.get_chroma_client", return_value=client), patch("tools.vault_indexer.embed_texts", side_effect=RuntimeError("offline")):
                result = json.loads(index_vault(vault_path=directory, file_path=str(path), collection="notes"))
        self.assertEqual(result["indexed_files"], 0)
        collection.delete.assert_not_called()
        collection.upsert.assert_not_called()

    def test_alias_substring_requires_unique_collection(self):
        import tools.vault_indexer as indexer
        aliases = {"project alpha": {"collection": "alpha"}, "project beta": {"collection": "beta"}}
        with patch.object(indexer, "_load_aliases", return_value=aliases):
            self.assertEqual(resolve_vault_alias("project"), "project")

    def test_empty_vault_search_skips_embedding(self):
        collection = Mock()
        collection.count.return_value = 0
        client = Mock()
        client.get_collection.return_value = collection
        with patch("tools.vault_search.get_chroma_client", return_value=client), patch("tools.vault_search._embed_query") as embed:
            result = _query_collection("q", "vault", "model", 5)
        self.assertEqual(result["documents"], [[]])
        embed.assert_not_called()

    def test_vision_rejects_non_image_before_model_call(self):
        with tempfile.NamedTemporaryFile(suffix=".txt") as stream:
            self.assertIn("Unsupported image", describe_image(stream.name))


if __name__ == "__main__":
    unittest.main()
