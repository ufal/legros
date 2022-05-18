#!/usr/bin/env python3

import argparse
import copy
import logging
import random
import multiprocessing
import sys

import torch
import torch.nn as nn

from neuralpiece.estimators import UniformEstimator, DotProdEstimator
from neuralpiece.model import Model
from neuralpiece.vocab import Vocab


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "vocab", type=argparse.FileType("r"),
        help="Vocabulary.")
    parser.add_argument(
        "bigrams", type=argparse.FileType("r"),
        help="List of tab separated bigrams.")
    parser.add_argument(
        "save_estimator", type=str, default=None,
        help="Path to save estimator.")
    parser.add_argument(
        "--batch-size", type=int, default=1024,
        help="Batch size for the neural model.")
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Batch size for the neural model.")
    parser.add_argument(
        "--embedding-dim", type=int, default=300,
        help="Embedding dim in the neural model.")
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Embedding dim in the neural model.")
    parser.add_argument(
        "--load-estimator", type=str, default=None,
        help="Saved estimator.")
    args = parser.parse_args()

    logging.info("Load vocabulary from '%s'.", args.vocab)
    vocab = Vocab([line.strip() for line in args.vocab])
    args.vocab.close()
    logging.info("Vocab size %d", vocab.size)

    if args.load_estimator is not None:
        logging.info("Initialize model from '%s'.", args.load_estimator)
        estimator = torch.load(args.load_estimator)
    else:
        logging.info("Initialize new estimator.")
        estimator = DotProdEstimator(vocab, args.embedding_dim)

    bigrams = [line.strip().split() for line in args.bigrams]
    args.bigrams.close()

    random.shuffle(bigrams)

    logging.info("Start training neural model.")
    val_set = bigrams[:2000]
    val_prev_subwords, val_subwords = zip(*val_set)
    train_set = bigrams[2000:]
    optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-4)
    stalled = 0
    best_loss = 1e9
    best_params = None

    @torch.no_grad()
    def validate():
        return -estimator.batch(
            val_subwords, val_prev_subwords).mean()

    for epoch in range(args.epochs):
        for i in range(0, len(train_set) // args.batch_size):
            prev_subwords, subwords = zip(
                *train_set[i * args.batch_size:(i + 1) * args.batch_size])
            loss = -estimator.batch(subwords, prev_subwords).mean()
            loss.backward()
            optimizer.step()

            if i % 10 == 9:
                val_loss = validate()
                logging.info(
                    "Epoch %d, batch %d, val loss: %.3g",
                    epoch + 1, i + 1, val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = copy.deepcopy(estimator.state_dict())
                    stalled = 0
                else:
                    stalled += 1

            if stalled > args.patience:
                break
        if stalled > args.patience:
            break

    estimator.load_state_dict(best_params)
    torch.save(estimator, args.save_estimator)

    logging.info("Done.")


if __name__ == "__main__":
    main()
