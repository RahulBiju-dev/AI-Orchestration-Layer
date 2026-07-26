"""Tests for the non-destructive ``--doctor`` diagnostics command."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agent import diagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_run_diagnostics_returns_structured_report(self):
        report = diagnostics.run_diagnostics(include_network=False)
        self.assertIn("ok", report)
        self.assertIn("checks", report)
        self.assertIn("python", report["checks"])
        self.assertIn("tool_registry", report["checks"])
        self.assertTrue(report["checks"]["python"]["ok"])
        self.assertTrue(report["checks"]["tool_registry"]["ok"])
        self.assertTrue(report["checks"]["runtime_paths"]["ok"])

    def test_format_diagnostics_report_is_readable(self):
        report = diagnostics.run_diagnostics(include_network=False)
        text = diagnostics.format_diagnostics_report(report)
        self.assertIn("Selene diagnostics", text)
        self.assertIn("tool_registry", text)

    def test_tool_registry_check_surfaces_errors(self):
        with patch("agent.diagnostics.validate_tool_registry", return_value=["broken"]):
            payload = diagnostics._check_tool_registry()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["errors"], ["broken"])

    def test_optional_dependencies_skips_dbus_off_linux(self):
        with patch("agent.diagnostics.platform_family", return_value="windows"):
            payload = diagnostics._check_optional_dependencies()
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["dependencies"]["dbus-python"])

    def test_optional_dependencies_reports_complete_google_oauth_stack(self):
        payload = diagnostics._check_optional_dependencies()

        self.assertIn("cryptography", payload["dependencies"])
        self.assertIn("google-api-python-client", payload["dependencies"])
        self.assertIn("google-auth-oauthlib", payload["dependencies"])
        self.assertIn("keyring", payload["dependencies"])

    def test_optional_dependencies_reports_all_spreadsheet_backends(self):
        payload = diagnostics._check_optional_dependencies()

        self.assertIn("openpyxl", payload["dependencies"])
        self.assertIn("xlrd", payload["dependencies"])
        self.assertIn("xlwt", payload["dependencies"])

    def test_chromadb_runtime_checks_dynamic_packaged_modules(self):
        payload = diagnostics._check_chromadb_runtime()

        self.assertTrue(payload["ok"])
        self.assertIn("chromadb.api.rust", payload["modules"])
        self.assertIn("chromadb.telemetry.product.posthog", payload["modules"])
        self.assertIn("chromadb_rust_bindings", payload["modules"])

    def test_chromadb_runtime_reports_incomplete_bundle(self):
        real_import = diagnostics.importlib.import_module

        def import_module(name):
            if name == "chromadb.api.rust":
                raise ModuleNotFoundError(name)
            return real_import(name)

        with patch.object(diagnostics.importlib, "import_module", side_effect=import_module):
            payload = diagnostics._check_chromadb_runtime()

        self.assertFalse(payload["ok"])
        self.assertIn("chromadb.api.rust", payload["errors"])
        self.assertIn("complete ChromaDB package", payload["remedy"])

    def test_main_doctor_exit_codes(self):
        with patch.object(diagnostics, "run_diagnostics", return_value={"ok": True, "checks": {}, "failed_checks": [], "elapsed_seconds": 0}):
            self.assertEqual(diagnostics.main_doctor(as_json=True), 0)
        with patch.object(diagnostics, "run_diagnostics", return_value={"ok": False, "checks": {}, "failed_checks": ["x"], "elapsed_seconds": 0}):
            self.assertEqual(diagnostics.main_doctor(as_json=False), 1)


if __name__ == "__main__":
    unittest.main()
