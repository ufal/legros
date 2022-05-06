#!/usr/bin/env python3

import argparse

from neuralpiece.estimators import UniformEstimator
from neuralpiece.model import Model
from neuralpiece.vocab import Vocab


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "vocab", type=argparse.FileType("r"),
        help="Initial vocabulary.")
    parser.add_argument(
        "token_counts", type=argparse.FileType("r"),
        help="Tab-separated file with token counts.")
    args = parser.parse_args()

    vocab = Vocab([line.strip() for line in args.vocab])
    args.vocab.close()

    token_counts = {}
    for line in args.token_counts:
        token, count_str = line.strip().split()
        token_counts[token] = int(count_str)
    args.token_counts.close()

    estimator = UniformEstimator(vocab.size)
    model = Model(vocab, estimator)

    bigrams = []
    for token, count in token_counts.items():
        for _ in range(count):
            segmentation = list(model.segment(token, sample=True))
            for i in range(len(segmentation) - 1):
                bigrams.append(segmentation[i:i + 2])

    for bigram in bigrams:
        print(bigram)


if __name__ == "__main__":
    main()
