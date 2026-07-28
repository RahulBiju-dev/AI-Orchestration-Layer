"""Reliability tests for CSV and Excel spreadsheet creation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.spreadsheet import MAX_EXCEL_CELL_CHARS, spreadsheet


class SpreadsheetReliabilityTests(unittest.TestCase):
    def test_csv_create_is_verified_and_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            created = json.loads(spreadsheet(
                action="create",
                file_path=str(path),
                rows=[
                    ["Name", "Score", "Note"],
                    ["Ada", 10, "Δ"],
                    ["Formula", 0, "=1+1"],
                ],
                delimiter=";",
                confirmed=True,
            ))
            read_back = json.loads(spreadsheet(
                action="read",
                file_path=str(path),
                delimiter=";",
            ))

        self.assertTrue(created["ok"])
        self.assertTrue(created["verified"])
        self.assertEqual(created["format"], "csv")
        self.assertEqual(created["escaped_formula_cells"], 1)
        self.assertGreater(created["size_bytes"], 0)
        self.assertEqual(read_back["rows"][1], ["Ada", "10", "Δ"])
        self.assertEqual(read_back["rows"][2][2], "'=1+1")

    def test_csv_reader_preserves_fields_larger_than_the_python_default(self):
        large_value = "x" * 200_000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large-field.csv"
            created = json.loads(spreadsheet(
                action="create",
                file_path=str(path),
                rows=[[large_value]],
                confirmed=True,
            ))
            read_back = json.loads(spreadsheet(
                action="read",
                file_path=str(path),
            ))

        self.assertTrue(created["ok"])
        self.assertEqual(read_back["rows"], [[large_value]])

    def test_legacy_xls_create_round_trips_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.xls"
            created = json.loads(spreadsheet(
                action="create",
                file_path=str(path),
                sheets=[{
                    "name": "Scores",
                    "rows": [["Name", "Score"], ["Ada", 10], ["Text", "=1+1"]],
                }],
                confirmed=True,
            ))
            read_back = json.loads(spreadsheet(
                action="read",
                file_path=str(path),
                sheet="Scores",
            ))

        self.assertTrue(created["ok"])
        self.assertTrue(created["verified"])
        self.assertEqual(created["format"], "xls")
        self.assertEqual(read_back["rows"][0], ["Name", "Score"])
        self.assertEqual(read_back["rows"][1], ["Ada", 10.0])
        self.assertEqual(read_back["rows"][2], ["Text", "=1+1"])

    def test_failed_verification_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.csv"
            path.write_bytes(b"original\n")
            with patch(
                "tools.spreadsheet._verify_created_workbook",
                side_effect=RuntimeError("verification failed"),
            ):
                result = json.loads(spreadsheet(
                    action="create",
                    file_path=str(path),
                    rows=[["replacement"]],
                    overwrite=True,
                    confirmed=True,
                ))

            self.assertIn("verification failed", result["error"])
            self.assertEqual(path.read_bytes(), b"original\n")

    def test_create_rejects_ambiguous_or_mistyped_options(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ambiguous.csv"
            ambiguous = json.loads(spreadsheet(
                action="create",
                file_path=str(path),
                sheets=[{"name": "CSV", "rows": [["one"]]}],
                rows=[["two"]],
                confirmed=True,
            ))
            mistyped = json.loads(spreadsheet(
                action="create",
                file_path=str(path),
                rows=[["one"]],
                overwrite="false",
                confirmed=True,
            ))

        self.assertIn("either sheets or rows", ambiguous["error"])
        self.assertEqual(mistyped["error"], "overwrite must be boolean")

    def test_excel_cell_limit_is_reported_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "too-long.xls"
            result = json.loads(spreadsheet(
                action="create",
                file_path=str(path),
                rows=[["x" * (MAX_EXCEL_CELL_CHARS + 1)]],
                confirmed=True,
            ))

        self.assertIn("32767-character XLS cell limit", result["error"])
        self.assertFalse(path.exists())

    def test_read_rejects_workbooks_with_silently_omitted_sheets(self):
        import openpyxl

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "many-sheets.xlsx"
            workbook = openpyxl.Workbook()
            for index in range(20):
                workbook.create_sheet(f"Sheet{index + 2}")
            workbook.save(path)
            workbook.close()
            result = json.loads(spreadsheet(action="view", file_path=str(path)))

        self.assertIn("contains 21 worksheets", result["error"])
        self.assertIn("supported limit is 20", result["error"])


if __name__ == "__main__":
    unittest.main()
