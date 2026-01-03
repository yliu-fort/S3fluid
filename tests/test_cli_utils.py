import os
import shutil
import unittest
from cli_utils import ensure_directory, progress_bar

class TestCliUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_results_temp"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_ensure_directory(self):
        ensure_directory(self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertTrue(os.path.isdir(self.test_dir))

        # Test idempotency
        ensure_directory(self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))

    def test_progress_bar_fallback(self):
        # Since we know rich/tqdm are likely missing, this tests the fallback
        # or whatever is installed.
        items = range(10)
        result = []
        for item in progress_bar(items, desc="Testing"):
            result.append(item)

        self.assertEqual(result, list(items))

if __name__ == '__main__':
    unittest.main()
