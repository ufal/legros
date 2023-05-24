import unittest

from neuralpiece.vocab import Vocab
from neuralpiece.model import Model
from neuralpiece.estimators import UniformEstimator, TableEstimator


class TestModel(unittest.TestCase):

    def test_basic_usage(self):
        vocab = Vocab([
            "a", "ab", "b", "c", "abc", "abd", "d",
            "abcaababbdbabcd", "dddd"])

        estimator = UniformEstimator(vocab.size)
        segm = Model(vocab, estimator)

        s = list(segm.segment("ddddabcaababbdbabcd"))
        self.assertEqual(s, ['dddd', 'abcaababbdbabcd'])

        s = list(segm.segment("abc"))
        self.assertEqual(s, ['abc'])

        s = list(segm.segment("bc"))
        self.assertEqual(s, ["b", "c"])


    def test_table_estimator(self):
        vocab = Vocab(["a", "b", "c", "ab", "bc", "abc"])

        table = {
            "###": {"a": -5, "abc": -1},
            "a": {"b": -5, "bc": -1},
            "b": {"c": -5, "ca": -5},
            "c": {"a": 0},
            "ab": {"c": -5, "ca": -2},
            "bc": {"abc": -1},
            "abc": {"a": -10}
            }

        estimator = TableEstimator(table)
        segm = Model(vocab, estimator)
        token = "abcabca"
        s = list(segm.segment(token))
        self.assertEqual(s, ["a", "bc", "abc", "a"])

        token = "abcabc"
        s = list(segm.segment(token))
        self.assertEqual(s, ["a", "bc", "abc"])

    def test_sampling(self):
        vocab = Vocab(["a", "b", "ab", "ba"])
        table = {"###": {"a": -1, "b": -1},
                 "a": {"a": -5, "b": -5, "ba": -1, "ab": -1},
                 "b": {"a": -5, "b": -5, "ba": -1, "ab": -1},
                 "ab": {"a": -1, "b": -1},
                 "ba": {"a": -1, "b": -1}
                 }

        estimator = TableEstimator(table)
        segm = Model(vocab, estimator)
        token = "abababaa"

        for _ in range(10):
            s = list(segm.segment(token, sample=True))
            self.assertEqual("".join(s), token)


if __name__ == "__main__":
    unittest.main()
