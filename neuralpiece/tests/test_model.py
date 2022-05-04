import unittest

from neuralpiece.vocab import Vocab
from neuralpiece.model import Model
from neuralpiece.estimators import UniformEstimator



class TestModel(unittest.TestCase):


    def test_basic_usage(self):
        vocab = Vocab(["a", "ab", "b", "c", "abc", "abd", "d", "abcaababbdbabcd", "dddd"])

        estimator = UniformEstimator(vocab.size)
        segm = Model(vocab, estimator)

        s = list(segm.segment("ddddabcaababbdbabcd"))

        self.assertEqual(s, ['dddd', 'abcaababbdbabcd'])



if __name__ == "__main__":
    unittest.main()
