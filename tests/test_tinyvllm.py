import unittest
import tinyvllm

class TestTinyVllm(unittest.TestCase):
    def test_import(self):
        """Test that the module can be imported."""
        self.assertIsNotNone(tinyvllm)

    def test_main(self):
        """Test that main function exists."""
        self.assertTrue(hasattr(tinyvllm, 'main'))

if __name__ == '__main__':
    unittest.main()
