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
        self.output_layer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.1),
            nn.Linear(dim, vocab.size))
        self.device = "cpu"

    def set_fixed_embeddings(self, subword_embeddings: np.array) -> None:
        self.embeddings.weights = nn.Parameter(
            torch.tensor(subword_embeddings).to(self.device))
        self.embeddings.weights.requires_grad = False

    @torch.no_grad()
    def forward(self, subword: str, prev_subword: str = "###") -> float:
        logprobs = self.batch([subword], [prev_subword])
        return logprobs[0]

    def to(self, device):
        self.device = device
        return super().to(device)

    def batch(self, subword: List[str], prev_subword: List[str] = "###") -> float:
        input_idx = torch.tensor([
            self.vocab.word2idx.get(s, 0) for s in prev_subword]).to(self.device)
        embedded = self.embeddings(input_idx)
        distribution = F.log_softmax(self.output_layer(embedded), 1)

        output_idx = torch.tensor([self.vocab.word2idx.get(s, 0) for s in subword]).to(self.device)

        logprobs = distribution[torch.arange(len(subword)), output_idx]
        return logprobs

    def to_numpy(self) -> NumpyDotProdEstimator:
        return NumpyDotProdEstimator(
            self.vocab,
            self.embeddings.weight.cpu().detach().numpy(),
            self.output_layer[1].cpu().weight.detach().numpy().T,
            self.output_layer[1].cpu().bias.detach().numpy())

    @torch.no_grad()
    def to_table(self) -> TableEstimator:
        import ipdb; ipdb.set_trace()
        all_words = torch.arange(self.vocab.size).to(self.device)
        all_embedded = self.embeddings(all_words)
        distributions = F.log_softmax(
            self.output_layer(all_embedded), 1)
        table = {}
        for i, word1 in enumerate(self.vocab.wordlist):
            table[word1] = {}
            for j, word2 in enumerate(self.vocab.wordlist):
                table[word1][word2] = distributions[i, j]

        return TableEstimator(table)

    @torch.no_grad()
    def estimate_posteriors(self, batch_size: int) -> List[float]:
        all_words = torch.arange(self.vocab.size)
        logsum = torch.full(
            (1, self.vocab.size), float('-inf')).to(self.device)
        for i in range(self.vocab.size // batch_size):
            batch = all_words[
                i * batch_size:(i + 1) * batch_size].to(self.device)
            embedded = self.embeddings(batch)
            distribution = F.log_softmax(self.output_layer(embedded), 1)
            logsum = torch.logsumexp(torch.cat((logsum, distribution), dim=0),
                                     keepdim=True, dim=0)

        return logsum.squeeze(0).cpu().numpy().tolist()

