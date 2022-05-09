#!/usr/bin/env python3

import argparse
import logging
import multiprocessing

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

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
    parser.add_argument(
        "--num-threads", type=int, default=4,
        help="Number of threads.")
    args = parser.parse_args()

    logging.info("Load vocabulary from '%s'.", args.vocab)
    vocab = Vocab([line.strip() for line in args.vocab])
    args.vocab.close()
    logging.info("Vocab size %d", vocab.size)

    logging.info("Load token counts from '%s'.", args.token_counts)
    token_counts = []
    for line in args.token_counts:
        token, count_str = line.strip().split()
        token_counts.append((token, int(count_str)))
    args.token_counts.close()
    split_size = len(token_counts) // args.num_threads + 1
    split_token_counts = [
        token_counts[i * split_size:(i + 1) * split_size]
        for i in range(args.num_threads)]
    logging.info(
        "Load %d tokens, %d per thread.", len(token_counts), split_size)

    logging.info("Initialize model.")
    estimator = UniformEstimator(vocab.size)
    model = Model(vocab, estimator)

    logging.info("Sample and count bigrams.")
    pool = multiprocessing.Pool(processes=args.num_threads)
    bigrams = []
    for thread_bigrams in pool.map(model.extract_bigrams, split_token_counts):
        bigrams.extend(thread_bigrams)

    for bigram in bigrams:
        print(bigram)

    logging.info("Done.")


if __name__ == "__main__":
    main()
