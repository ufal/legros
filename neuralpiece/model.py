import math
from typing import List, Callable, Any

from neuralpiece.vocab import Vocab
from neuralpiece.estimators import UniformEstimator


class Model:


    def __init__(self, vocab: Vocab, estimator: Callable[Any, float]) -> None:
        self.vocab = vocab
        self.estimator = estimator

    def segment(self, token: str, sample: bool = False) -> List[str]:

        assert token

        scores = [0] # this list holds at index i the score of the best segmentation ending at index i
        prevs = [None] # at index i, this contains the index to the start of the previous subword in the best segmentation that ends at index i - 1
        # keys are ends, values are starts

        for i in range(len(token)):
            # the index i marks the end of the currently considered subwords

            best_score = -math.inf
            best_prev = None

            for j in range(max(0, i + 1 - self.vocab.max_subword_length), i + 1):
                # the current subword begins at index 'j' and ends at index 'i' inclusive
                # previous subword (if any) begins at prevs[j] and ends at j-1

                subword = token[j:i + 1]
                assert len(subword) >= 1

                if subword not in self.vocab:
                    if len(subword) == 1:
                        raise ValueError(f"character '{subword}' not in vocab")
                    continue

                if j == 0:
                    # this means this subword begins at the start of the token
                    score = self.estimator(subword, None)

                else:
                    # there are subwords before this one
                    prev_subword = token[prevs[j]:j]
                    score = self.estimator(subword, prev_subword) + scores[j]

                if score > best_score:
                    best_score = score
                    best_prev = j

            assert best_prev is not None
            assert len(scores) == i + 1

            scores.append(best_score)
            prevs.append(best_prev)

        assert prevs[1] == 0

        index = len(token)
        segmentation = []

        while index is not None:
            subword_end = index
            subword_begin = prevs[index]

            subword = token[subword_begin:subword_end]
            segmentation.append(subword)

            index = subword_begin

        return reversed(segmentation[:-1])
