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
from gensim.models import Word2Vec
import numpy as np
from scipy.spatial import distance

from neuralpiece.unigram_segment import viterbi_segment
from neuralpiece.segment_vocab_with_subword_embeddings import get_substrings

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

SUBSTR_CACHE = {}


def segment(
        word: str,
        fasttext: FastText,
        subwrd2idx: Dict[str, int],
        subword_embeddings: np.array) -> List[Tuple[List[str], float]]:
    try:
        vector = fasttext.wv[word]
    except KeyError:
        vector = fasttext.wv.vectors.mean(0)
    subwords = get_substrings(word, subwrd2idx)
    subword_scores = {
        swrd: -distance.cosine(vector, subword_embeddings[idx])
        for swrd, idx in subwords}

    seg, _ = viterbi_segment(word, subword_scores, sample=False)
    return seg


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("fasttext", help="W2V/FastText model from Gensim.")
    parser.add_argument(
        "subword_vocab", type=argparse.FileType("r"),
        help="Subword vocab, subword per line.")
    parser.add_argument(
        "subword_embeddings", type=argparse.FileType("r"),
        help="Subword embeddings as txt.")
    parser.add_argument(
        "input", type=argparse.FileType("r"),
        help="Word vocab: <word>\\t<count>")
    parser.add_argument(
        "output", type=argparse.FileType("w"),
        nargs="?", default=sys.stdout)
    parser.add_argument(
        "--embeddings-type", default="fasttext", choices=["fasttext", "w2v"])
    parser.add_argument(
        "--remove-count", default=None, type=int,
        help="If set, it lists the worse subwrords which will get removed.")
    args = parser.parse_args()

    logging.info("Load word embeddings model from %s.", args.fasttext)
    if args.embeddings_type == "fasttext":
        fasttext = FastText.load(args.fasttext)
    else:
        fasttext = Word2Vec.load(args.fasttext)

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

    logging.info("Loaded %d subwords with embeddings.", len(subwords))
    subwrd2idx = {s: i for i, s in enumerate(subwords)}

    counts = defaultdict(int)
    logging.info("Segment words and keep stats.")
    for line in args.input:
        word, count_str = line.strip().split("\t")
        for seg in segment(
                word, fasttext, subwrd2idx, subword_embeddings):
            counts[seg] += int(count_str)
    args.input.close()

    logging.info("Sort and print.")
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    if args.remove_count is None:
        for sbwrd, count in sorted_counts:
            print(f"{sbwrd}\t{count}", file=args.output)
        logging.info("Done.")
    else:
        to_remove = []
        for sbwrd, _ in reversed(sorted_counts):
            if len(sbwrd) > 1:
                to_remove.append(sbwrd)
            if len(to_remove) >= args.remove_count:
                break
        for sbwrd in to_remove:
            print(sbwrd, file=args.output)


if __name__ == "__main__":
    main()
