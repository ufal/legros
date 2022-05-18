#!/usr/bin/env python3

import argparse
import logging
import sys

import torch
import torch.nn as nn

from neuralpiece.estimators import UniformEstimator#, DotProdEstimator
from neuralpiece.model import Model
from neuralpiece.pretokenize import pretokenize
from neuralpiece.vocab import Vocab


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
        for token in pretokenize(line.rstrip()):
            segmentation = list(model.segment(token, sample=True))
            print(f"###\t{segmentation[0]}")
            for i in range(len(segmentation) - 1):
                print(f"{segmentation[i]}\t{segmentation[i + 1]}")


if __name__ == "__main__":
    main()
