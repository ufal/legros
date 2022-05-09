import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralpiece.vocab import Vocab


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


class DotProdEstimator(nn.Module):
    def __init__(self, vocab: Vocab, dim: int) -> None:
        super().__init__()

        self.vocab = vocab
        self.embeddings = nn.Embedding(vocab.size, dim)
        self.output_layer = nn.Linear(dim, vocab.size)

    def forward(self, subword: str, prev_subword: str = "###") -> float:
        input_idx = torch.tensor([self.vocab.word2idx[s] for s in prev_subword])
        # TODO if cuda, do something
        embedded = self.embeddings(input_idx)
        distribution = F.log_softmax(self.output_layer(embedded), 1)

        output_idx = torch.tensor([self.vocab.word2idx[s] for s in subword])
        return distribution[torch.arange(len(subword)), output_idx]
