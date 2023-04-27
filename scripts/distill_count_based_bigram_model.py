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

SUBSTR_CACHE = {}


def segment(
        word: str,
        fasttext: KeyedVectors,
        subwrd2idx: Dict[str, int],
        subword_embeddings: np.array,
        is_bert_wordpiece: bool = False) -> List[Tuple[List[str], float]]:
    try:
        vector = fasttext[word]
    except KeyError:
        vector = fasttext.vectors.mean(0)
    subwords = get_substrings(
        word, subwrd2idx, is_bert_wordpiece=is_bert_wordpiece)
    subword_scores = {
        swrd: -distance.cosine(vector, subword_embeddings[idx])
        for swrd, idx in subwords}

    seg, _ = viterbi_segment(
        word, subword_scores, sample=False,
        is_bert_wordpiece=is_bert_wordpiece)
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
        "output", type=str, help="Output table.")
    parser.add_argument(
        "--embeddings-type", default="fasttext",
        choices=["fasttext", "w2v", "txt"])
    parser.add_argument(
        "--bert-wordpiece", default=False, action="store_true",
        help="Flag is the model uses BERT-like wordpiece.")
    args = parser.parse_args()

    logging.info("Load word embeddings model from %s.", args.fasttext)
    if args.embeddings_type == "fasttext":
        fasttext = FastText.load(args.fasttext).wv
    elif args.embeddings_type == "w2v":
        fasttext = Word2Vec.load(args.fasttext).wv
    elif args.embeddings_type == "txt":
        fasttext = KeyedVectors.load_word2vec_format(args.fasttext)
    else:
        raise ValueError("Unknown embeddings type: %s" % args.embeddings_type)

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

    counts = defaultdict(lambda: defaultdict(int))
    logging.info("Compute bigram stats.")
    for i, line in enumerate(args.input):
        word, count_str = line.strip().split("\t")
        segmentation = ["###"] + segment(
            word, fasttext, subwrd2idx,
            subword_embeddings, is_bert_wordpiece=args.bert_wordpiece) + ["###"]

        for seg1, seg2 in zip(segmentation, segmentation[1:]):
            counts[seg1][seg2] += int(count_str)
        if i % 10000 == 0:
            logging.info("Processed %d lines.", i)
    args.input.close()

    logging.info("Normalize counts.")
    norm_counts = {}
    for seg1, seg2_counts in counts.items():
        denominator = sum(seg2_counts.values()) + len(subwords)
        norm_counts[seg1] = defaultdict(partial(float, -math.log(denominator)))
        for seg2, count in seg2_counts.items():
            norm_counts[seg1][seg2] = math.log(count) - math.log(denominator)

    logging.info("Save the counts into a file.")
    with open(args.output, "wb") as f_save:
        pickle.dump(norm_counts, f_save)
    logging.info("Done.")


if __name__ == "__main__":
    main()
