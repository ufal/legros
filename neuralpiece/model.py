from typing import List, Callable, Any
import numpy as np

from neuralpiece.vocab import Vocab
from neuralpiece.estimators import UniformEstimator


class Model:
    def __init__(self, vocab: Vocab, estimator: Callable[[Any], float]) -> None:
        self.vocab = vocab
        self.estimator = estimator

    def segment(self, token: str) -> List[str]:
        assert token

        score_table = np.full([len(token), len(token)], -np.inf)
        prev_rows = np.zeros([len(token), len(token)], dtype=np.int32)

        # on a given row there are all subwords beginning at index i
        for row in range(len(token)):

            for col in range(row, len(token)):

                subword = token[row:col + 1]

                if subword not in self.vocab:
                    if len(subword) == 1:
                        raise ValueError(f"character '{subword}' not in vocab")
                    continue

                if row == 0:
                    score_table[row, col] = self.estimator(subword)
                    continue

                best_predecesor = (-np.inf, -1)

                for prev_row in range(row):
                    prev_subword = token[prev_row:row]
                    bigram_score = (
                        self.estimator(subword, prev_subword) +
                        score_table[prev_row, row - 1])

                    if bigram_score > best_predecesor[0]:
                        best_predecesor = (bigram_score, prev_row)

                prev_rows[row, col] = best_predecesor[1]
                score_table[row, col] = best_predecesor[0]

        subword_end = len(token)
        row = score_table[:, -1].argmax()
        segmentation = []

        while subword_end > 0:
            subword_begin = row
            segmentation.append(token[subword_begin:subword_end])

            row = prev_rows[row, subword_end - 1]
            subword_end = subword_begin

        return reversed(segmentation)
