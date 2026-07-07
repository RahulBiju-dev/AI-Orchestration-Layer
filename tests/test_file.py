import json
import unittest
from unittest.mock import patch
import tempfile
import tools.vault_indexer
from tools.file import read_file, create_file

class TestReadFile(unittest.TestCase):
    @patch('os.path.exists')
    def test_read_file_not_found(self, mock_exists):
        # Setup mock
        mock_exists.return_value = False

        # Call function
        result = read_file('/fake/path/that/does/not/exist.txt')

        # Assertions
        mock_exists.assert_called_once_with('/fake/path/that/does/not/exist.txt')
        expected_json = json.dumps({"error": "File not found: /fake/path/that/does/not/exist.txt"})
        self.assertEqual(result, expected_json)

    @patch('os.path.isfile')
    @patch('os.path.exists')
    def test_read_file_not_a_file(self, mock_exists, mock_isfile):
        # Setup mocks
        mock_exists.return_value = True
        mock_isfile.return_value = False

        # Call function
        result = read_file('/fake/path/that/is/a/directory')

        # Assertions
        mock_exists.assert_called_once_with('/fake/path/that/is/a/directory')
        mock_isfile.assert_called_once_with('/fake/path/that/is/a/directory')
        expected_json = json.dumps({"error": "Not a file: /fake/path/that/is/a/directory"})
        self.assertEqual(result, expected_json)


class TestCreateFile(unittest.TestCase):
    def test_create_file_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tools.vault_indexer.VAULTS_DIR", temp_dir):
                with patch("tools.vault_indexer.index_vault") as mock_index:
                    with patch("tools.vault_indexer.register_vault_alias") as mock_alias:
                        mock_index.return_value = '{"success": true}'
                        result_str = create_file("test.txt", "hello world")
                        result = json.loads(result_str)

                        self.assertTrue(result.get("success"))
                        self.assertTrue(result.get("indexed"))
                        self.assertEqual(result.get("collection"), "test")
                        self.assertEqual(result.get("alias"), "test")

                        mock_index.assert_called_once()
                        mock_alias.assert_called_once()

    def test_create_file_invalid_basename(self):
        invalid_names = ["", ".", "..", ".vault_aliases.json"]
        for name in invalid_names:
            with self.subTest(name=name):
                result_str = create_file(name, "content")
                result = json.loads(result_str)
                self.assertIn("error", result)
                self.assertEqual(result["error"], "file_path must contain a valid filename")

    def test_create_file_content_too_large(self):
        large_content = "a" * (10 * 1024 * 1024 + 1)
        result_str = create_file("test.txt", large_content)
        result = json.loads(result_str)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "File content exceeds the 10 MiB creation limit")

    def test_create_file_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tools.vault_indexer.VAULTS_DIR", temp_dir):
                # Pre-create the file to trigger FileExistsError
                with open(f"{temp_dir}/test.txt", "w") as f:
                    f.write("existing")
                result_str = create_file("test.txt", "new content")
                result = json.loads(result_str)
                self.assertIn("error", result)
                self.assertIn("File already exists", result["error"])

    def test_create_file_index_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tools.vault_indexer.VAULTS_DIR", temp_dir):
                with patch("tools.vault_indexer.index_vault") as mock_index:
                    mock_index.side_effect = Exception("Indexing failed")
                    result_str = create_file("test.txt", "content")
                    result = json.loads(result_str)
                    self.assertTrue(result.get("success"))
                    self.assertFalse(result.get("indexed"))
                    self.assertIn("Indexing failed", result["index_result"]["error"])

    def test_create_file_alias_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tools.vault_indexer.VAULTS_DIR", temp_dir):
                with patch("tools.vault_indexer.index_vault") as mock_index:
                    with patch("tools.vault_indexer.register_vault_alias") as mock_alias:
                        mock_index.return_value = '{"success": true}'
                        mock_alias.side_effect = Exception("Alias failed")
                        result_str = create_file("test.txt", "content")
                        result = json.loads(result_str)
                        self.assertTrue(result.get("success"))
                        self.assertTrue(result.get("indexed"))
                        self.assertIn("Alias failed", result["index_result"]["alias_warning"])

if __name__ == '__main__':
    unittest.main()
