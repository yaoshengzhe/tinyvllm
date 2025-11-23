import unittest
import tinyvllm

class TestTinyVllm(unittest.TestCase):
    def test_import(self):
        """Test that the module can be imported."""
        self.assertIsNotNone(tinyvllm)

if __name__ == '__main__':
    unittest.main()
