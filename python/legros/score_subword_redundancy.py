#!/usr/bin/env python3

import argparse
import logging
import math
from typing import Dict, List, Tuple
import sys

import numpy as np
from scipy.spatial import distance

from legros.unigram_segment import viterbi_segment
from legros.segment_vocab_with_subword_embeddings import try_segment

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "subword_vocab", type=argparse.FileType("r"),
        help="Subword vocab, subword per line.")
    parser.add_argument(
        "subword_embeddings", type=argparse.FileType("r"),
        help="Subword embeddings as txt.")
    args = parser.parse_args()

    logging.info(
        "Load subword vocabulary from %s.", args.subword_vocab)
    subwords = []
    for line in args.subword_vocab:
        subwords.append(line.strip())
    args.subword_vocab.close()

    logging.info("Load subword embeddings from '%s'.", args.subword_embeddings)
    subword_embeddings = np.loadtxt(args.subword_embeddings)
    args.subword_embeddings.close()

    if len(subwords) != subword_embeddings.shape[0]:
        raise ValueError(
            "The number of subwords does not match the embedding count.")

    subwrd2idx = {s: i for i, s in enumerate(subwords)}
    logging.info("Loaded %d subwords with embeddings.", len(subwords))

    logging.info("Scoring subwords.")
    scored_subwords = []
    for idx, subword in enumerate(subwords):
        emb = subword_embeddings[idx]
        subword_segmented = try_segment(
            subword, emb, subwrd2idx, subword_embeddings,
            exclude_subwords=[subword])[0][0].split()

        if len(subword_segmented) == 1:
            continue

        avg_seg_embedding = np.mean(
            [subword_embeddings[subwrd2idx[seg]] for seg in subword_segmented
             if seg in subwrd2idx], axis=0)

        scored_subwords.append(
            (subword, subword_segmented, distance.cosine(emb, avg_seg_embedding)))

    logging.info("Sort and print subwords.")
    scored_subwords.sort(key=lambda x: x[2])

    for subword, seg, score in scored_subwords:
        print(f"{score:.5}\t{subword}\t{' '.join(seg)}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
