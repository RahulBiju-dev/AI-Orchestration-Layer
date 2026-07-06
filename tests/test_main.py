import sys
import os
import unittest
from unittest.mock import patch

# Mock modules to prevent import errors during test collection
sys.modules['ollama'] = unittest.mock.MagicMock()
sys.modules['agent'] = unittest.mock.MagicMock()
sys.modules['agent.core'] = unittest.mock.MagicMock()
sys.modules['agent.web'] = unittest.mock.MagicMock()
sys.modules['rich'] = unittest.mock.MagicMock()
sys.modules['rich.console'] = unittest.mock.MagicMock()

import main
from main import _get_modelfile_path

class TestMain(unittest.TestCase):
    def test_get_modelfile_path_default(self):
        # Remove _MEIPASS if it exists to test default behavior
        if hasattr(sys, '_MEIPASS'):
            del sys._MEIPASS

        path = _get_modelfile_path()
        expected = os.path.join(os.path.dirname(os.path.abspath(main.__file__)), 'Modelfile')
        self.assertEqual(path, expected)

    def test_get_modelfile_path_meipass(self):
        with patch.object(sys, '_MEIPASS', '/tmp/fake_meipass', create=True):
            path = _get_modelfile_path()
            expected = os.path.join('/tmp/fake_meipass', 'Modelfile')
            self.assertEqual(path, expected)

if __name__ == '__main__':
    unittest.main()
