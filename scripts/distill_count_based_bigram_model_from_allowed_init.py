#!/usr/bin/env python3

import argparse
from collections import defaultdict
from functools import partial
import logging
import math
import pickle
from typing import Dict, List, Tuple
import sys

from gensim.models.fasttext import FastText
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from scipy.spatial import distance

from neuralpiece.unigram_segment import viterbi_segment
from neuralpiece.segment_vocab_with_subword_embeddings import get_substrings

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "allowed_init_file", type=argparse.FileType("r"),
        help="Segmented vocab to initialize experiment.")
    parser.add_argument(
        "vocab", type=argparse.FileType("r"),
        help="Word vocab: <word>\\t<count>")
    parser.add_argument(
        "output", type=str, help="Output table.")
    parser.add_argument(
        "--bert-wordpiece", default=False, action="store_true",
        help="Flag is the model uses BERT-like wordpiece.")
    args = parser.parse_args()

    logging.info("Load initial vocab segmentation.")
    allowed_init = {}
    subwords = set()
    for line in args.allowed_init_file:
        tokens = line.strip().split()
        word = tokens[0]
        segments = tokens[1:]
        if args.bert_wordpiece:
            segments = seg[0] + ["##" + s for s in seg[1:]]
        segments = ["###"] + segments + ["###"]
        for seg in segments:
            subwords.add(seg)
        allowed_init[word] = segments
    args.allowed_init_file.close()

    counts = defaultdict(lambda: defaultdict(int))
    logging.info("Compute bigram stats for word vocab with counts.")
    for i, line in enumerate(args.vocab):
        word, count_str = line.strip().split("\t")
        if word not in allowed_init:
            continue
        segmentation = allowed_init[word]
        for seg1, seg2 in zip(segmentation, segmentation[1:]):
            counts[seg1][seg2] += int(count_str)
        if i % 10000 == 0:
            logging.info("Processed %d lines.", i)
    args.vocab.close()

    logging.info("Normalize counts.")
    norm_counts = {}
    for seg1, seg2_counts in counts.items():
        denominator = sum(seg2_counts.values()) + len(subwords)
        norm_counts[seg1] = defaultdict(partial(float, -math.log(denominator)))
        for seg2, count in seg2_counts.items():
            norm_counts[seg1][seg2] = math.log(count + 1) - math.log(denominator)

    logging.info("Save the counts into a file.")
    with open(args.output, "wb") as f_save:
        pickle.dump(norm_counts, f_save)
    logging.info("Done.")


if __name__ == "__main__":
    main()
