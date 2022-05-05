import math
from typing import Dict


class UniformEstimator:

    def __init__(self, vocab_size: int) -> None:
        self.logprob = -math.log(vocab_size)

    def __call__(self, *whatever, **kwhatever) -> float:
        return self.logprob


class TableEstimator:

    def __init__(self, table: Dict[str, Dict[str, float]]) -> None:
        self.table = table
        assert "###" in self.table


    def __call__(self, subword: str, prev_subword: str = "###") -> float:

        if prev_subword not in self.table:
            return -math.inf

        return self.table[prev_subword].get(subword, -math.inf)
