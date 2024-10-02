#!/usr/bin/env python3

import argparse
import logging
import sys

import torch
import torch.nn as nn

from legros.estimators import UniformEstimator#, DotProdEstimator
from legros.model import Model
from legros.pretokenize import pretokenize
from legros.vocab import Vocab


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "vocab", type=argparse.FileType("r"),
        help="Initial vocabulary.")
    parser.add_argument(
        "estimator", type=str, default=None,
        help="Saved estimator.")
    parser.add_argument(
        "input", type=argparse.FileType("r"),
        default=sys.stdin, nargs="?",
        help="Plain text input, default is stdin.")
    args = parser.parse_args()

    logging.info("Load vocabulary from '%s'.", args.vocab)
    vocab = Vocab([line.rstrip() for line in args.vocab])
    args.vocab.close()
    logging.info("Vocab size %d", vocab.size)

    if args.estimator == "uniform":
        estimator = UniformEstimator(vocab.size)
    else:
        logging.info("Load model from '%s'.", args.estimator)
        estimator = torch.load(args.estimator)
    model = Model(vocab, estimator)

    for line in args.input:
        tokenized = []
        for token in pretokenize(line.rstrip()):
            segmentation = list(model.segment(token, sample=False))
            tokenized.extend(segmentation)
        print(" ".join(tokenized))


if __name__ == "__main__":
    main()
