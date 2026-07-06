import json
import unittest
from unittest.mock import patch
from tools.file import read_file

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

if __name__ == '__main__':
    unittest.main()
