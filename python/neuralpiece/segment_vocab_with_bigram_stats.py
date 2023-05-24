#!/usr/bin/env python3

import argparse
import logging
import pickle
from typing import Dict, List, Tuple
import sys

from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
import numpy as np
from scipy.spatial import distance

from neuralpiece.model import Model
from neuralpiece.vocab import Vocab

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

SUBSTR_CACHE = {}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "bigram_stats", type=argparse.FileType("rb"),
        help="Subword vocab, subword per line.")
    parser.add_argument(
        "input", nargs="?", default=sys.stdin, type=argparse.FileType("r"),
        help="Input words to segment: word per line.")
    parser.add_argument(
        "--bert-wordpiece", default=False, action="store_true",
        help="Flag is the model uses BERT-like wordpiece.")
    parser.add_argument("--model-type", choices=["counts", "neural"], default="counts")
    args = parser.parse_args()

    if args.model_type == "counts":
        logging.info("Load bigram stats.")
        bigram_stats = pickle.load(args.bigram_stats)
        args.bigram_stats.close()
        vocab = Vocab(list(bigram_stats.keys()))
        estimator = lambda subword, prev_subword: bigram_stats[prev_subword][subword]
    elif args.model_type == "neural":
        logging.info("Load the saved estimator.")
        estimator = pickle.load(args.bigram_stats)
        args.bigram_stats.close()
        vocab = estimator.vocab
    else:
        raise ValueError("Unknown model type.")

    model = Model(
        vocab=vocab,
        estimator=estimator,
        is_bert_wordpiece=args.bert_wordpiece)

    logging.info("Segment words.")
    for line in args.input:
        segmentation = model.segment(line.strip())
        print(" ".join(segmentation))

    args.input.close()
    logging.info("Done.")


if __name__ == "__main__":
    main()
