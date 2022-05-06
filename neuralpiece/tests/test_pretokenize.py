import unittest

from neuralpiece.pretokenize import pretokenize


class TestPretokenize(unittest.TestCase):

    def test_pretokenize(self):
        sentence = "This is a sentence (with a bracket). Walrus."
        output = [
            '▁This', '▁is', '▁a', '▁sentence', '▁(',
            'with', '▁a', '▁bracket', ')', '.', '▁Walrus', '.']
        self.assertEqual(pretokenize(sentence), output)


if __name__ == "__main__":
    unittest.main()
