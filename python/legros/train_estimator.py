#!/usr/bin/env python3

import argparse
import copy
import logging
import random
import sys

import torch
import torch.nn as nn

from legros.estimators import DotProdEstimator
from legros.vocab import Vocab


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def load_bigrams(fh):
    bigrams = []
    for line in fh:
        tokens = line.strip().split("\t")
        if len(tokens) != 2:
            continue
        bigrams.append(tokens)
    return bigrams


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
        "--epochs", type=int, default=50,
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
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="Learning rate.")
    args = parser.parse_args()

    logging.info("Load vocabulary from '%s'.", args.vocab)
    vocab = Vocab([line.rstrip() for line in args.vocab])
    args.vocab.close()
    logging.info("Vocab size %d", vocab.size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.load_estimator is not None:
        logging.info("Initialize model from '%s'.", args.load_estimator)
        estimator = torch.load(args.load_estimator).to(device)
        assert isinstance(estimator, DotProdEstimator)
    else:
        logging.info("Initialize new estimator.")
        estimator = DotProdEstimator(vocab, args.embedding_dim).to(device)

    logging.info("Load bigrams from file.")
    bigrams = load_bigrams(args.bigrams)
    args.bigrams.close()

    logging.info("Shuffle data.")
    random.shuffle(bigrams)

    logging.info("Start training the model.")
    val_set = bigrams[:2000]
    val_prev_subwords, val_subwords = zip(*val_set)
    train_set = bigrams[2000:]
    optimizer = torch.optim.Adam(estimator.parameters(), lr=args.learning_rate)
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
            optimizer.zero_grad()

            if i % 10 == 9:
                estimator.eval()
                val_loss = validate()
                estimator.train()
                logging.info(
                    "Epoch %d, batch %d, val loss: %.5g",
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

    logging.info("Resume best checkpoint.")
    estimator.load_state_dict(best_params)
    estimator.eval()

    logging.info("Cache values for bigrams that appeared in the data.")
    unique_bigrams = list(set([(x1, x2) for x1, x2 in bigrams]))
    cached_bigrams = {}
    for i in range(0, len(unique_bigrams) // args.batch_size):
        batch = unique_bigrams[i * args.batch_size:(i + 1) * args.batch_size]
        prev_subwords, subwords = zip(*batch)
        with torch.no_grad():
            scores = estimator.batch(subwords, prev_subwords)
        for bigram, score in zip(batch, scores):
            cached_bigrams[bigram] = score.item()

    torch.save(estimator.to("cpu"), args.save_estimator)

    numpy_estimator = estimator.to_numpy()
    numpy_estimator._cache = cached_bigrams
    torch.save(numpy_estimator, args.save_estimator + ".numpy")

    logging.info("Done.")


if __name__ == "__main__":
    main()
