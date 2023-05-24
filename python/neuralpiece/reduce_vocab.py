#!/usr/bin/env python3

import argparse
import logging

import torch
import torch.nn as nn

from neuralpiece.estimators import DotProdEstimator
from neuralpiece.vocab import Vocab


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "vocab", type=argparse.FileType("r"),
        help="Initial vocabulary.")
    parser.add_argument(
        "estimator", type=str, help="Saved estimator.")
    parser.add_argument(
        "keep_size", type=int, help="Target vocabulary size.")
    parser.add_argument(
        "--batch-size", type=int, default=1024,
        help="Batch size.")
    args = parser.parse_args()

    logging.info("Load vocabulary from '%s'.", args.vocab)
    vocab = Vocab([line.rstrip() for line in args.vocab])
    args.vocab.close()
    logging.info("Vocab size %d", vocab.size)

    logging.info("Load model from '%s'.", args.estimator)
    estimator = torch.load(args.estimator)
    assert isinstance(estimator, DotProdEstimator)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    estimator.to(device)
    estimator.eval()

    word_scores = estimator.estimate_posteriors(args.batch_size)
    assert len(word_scores) == vocab.size

    scored_words = [
        (word, score) for word, score in zip(vocab.wordlist, word_scores)
        if len(word) > 1 and word != "###"]

    remove_words_count = vocab.size - args.keep_size

    scored_words.sort(key=lambda x: x[1])
    to_be_removed = set(word for word, _ in scored_words[:remove_words_count])

    for word in vocab.wordlist:
        if word not in to_be_removed:
            print(word)


if __name__ == "__main__":
    main()
