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

from neuralpiece.estimators import NumpyDotProdEstimator
from neuralpiece.vocab import Vocab
from neuralpiece.unigram_segment import viterbi_segment
from neuralpiece.segment_vocab_with_subword_embeddings import get_substrings

import torch

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


def sample_batch(
        bigrams: List[Tuple[str, str]],
        probs: np.array,
        batch_size: int) -> Tuple[List[str], List[str], List[float]]:

    first, second, weights = [], [], []

    for i in np.random.choice(len(probs), 1000, p=probs):
        seg1, seg2 = bigrams[i]
        first.append(seg1)
        second.append(seg2)
        weights.append(probs[i])

    return first, second, weights


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
        "--embeddings-type", default="fasttext", choices=["fasttext", "w2v"])
    parser.add_argument(
        "--batch-size", default=100000, type=int)
    parser.add_argument(
        "--max-steps", default=100000, type=int)
    parser.add_argument(
        "--learning-rate", default=1e-4, type=float)
    parser.add_argument(
        "--patience", default=20, type=int)
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
    subwords.append("###")

    logging.info("Load subword embeddings from '%s'.", args.subword_embeddings)
    subword_embeddings = np.loadtxt(args.subword_embeddings)
    args.subword_embeddings.close()
    subword_embeddings = np.concatenate((
        subword_embeddings,
        subword_embeddings.mean(axis=0, keepdims=True)))

    if len(subwords) != subword_embeddings.shape[0]:
        raise ValueError(
            "The number of subwords does not match the embedding count.")

    logging.info("Loaded %d subwords with embeddings.", len(subwords))
    subwrd2idx = {s: i for i, s in enumerate(subwords)}

    logging.info("Initialize count matrix.")
    stats = np.full((len(subwords), len(subwords)), 1.)

    logging.info("Compute bigram stats.")
    for line in args.input:
        word, count_str = line.strip().split("\t")
        segmentation = ["###"] + segment(
            word, fasttext, subwrd2idx,
            subword_embeddings) + ["###"]

        for seg1, seg2 in zip(segmentation, segmentation[1:]):
            seg1_idx = subwrd2idx[seg1]
            seg2_idx = subwrd2idx[seg2]
            stats[seg1_idx, seg2_idx] += float(count_str)
    args.input.close()

    logging.info("Normalize and log the matrix.")
    stats = np.log(stats)
    stats -= stats.sum(axis=1, keepdims=True)

    logging.info("Pseudo inverse of the embeeding matrix.")
    embeddings_inv = np.linalg.pinv(subword_embeddings)

    logging.info("Compute the weight matrix.")
    weights = np.dot(embeddings_inv, stats)

    estimator = NumpyDotProdEstimator(
        Vocab(subwords), subword_embeddings, weights,
        np.zeros(weights.shape[1]))

    logging.info("Save estimator.")
    with open(args.output, "wb") as f_np:
        pickle.dump(estimator, f_np)

    logging.info("Done.")


if __name__ == "__main__":
    main()
