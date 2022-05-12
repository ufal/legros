import math
from typing import List, Dict

import numpy as np
import scipy.special
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


class NumpyDotProdEstimator:
    def __init__(
            self, vocab: Vocab, embeddings: np.array,
            output_proj: np.array, bias: np.array) -> None:
        self.vocab = vocab
        self.embeddings = embeddings
        self.output_proj = output_proj
        self.bias = bias

        self._cache = {}

    def __call__(self, subword: str, prev_subword: str = "###") -> float:
        if (prev_subword, subword) in self._cache:
            return self._cache[(prev_subword, subword)]

        embedded = self.embeddings[self.vocab.word2idx[prev_subword]]
        logprobs = scipy.special.log_softmax(
            embedded.dot(self.output_proj) + self.bias)
        tgt_idx = self.vocab.word2idx[subword]
        output = logprobs[tgt_idx]

        self._cache[(prev_subword, subword)] = output
        return output


class DotProdEstimator(nn.Module):
    def __init__(self, vocab: Vocab, dim: int) -> None:
        super().__init__()

        self.vocab = vocab
        self.embeddings = nn.Embedding(vocab.size, dim)
        self.output_layer = nn.Linear(dim, vocab.size)

    @torch.no_grad()
    def forward(self, subword: str, prev_subword: str = "###") -> float:
        logprobs = self.batch([subword], [prev_subword])
        return logprobs[0]

    def batch(self, subword: List[str], prev_subword: List[str] = "###") -> float:
        input_idx = torch.tensor([
            self.vocab.word2idx[s] for s in prev_subword])
        # TODO if cuda, do something
        embedded = self.embeddings(input_idx)
        distribution = F.log_softmax(self.output_layer(embedded), 1)

        output_idx = torch.tensor([self.vocab.word2idx[s] for s in subword])

        logprobs = distribution[torch.arange(len(subword)), output_idx]
        return logprobs

    def to_numpy(self) -> NumpyDotProdEstimator:
        return NumpyDotProdEstimator(
            self.vocab,
            self.embeddings.weight.detach().numpy(),
            self.output_layer.weight.detach().numpy().T,
            self.output_layer.bias.detach().numpy())
