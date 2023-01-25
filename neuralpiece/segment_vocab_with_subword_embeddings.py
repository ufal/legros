#!/usr/bin/env python3

import argparse
import logging
import math
from typing import Dict, List, Tuple, Union
import sys

from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
import numpy as np
from scipy.spatial import distance

from neuralpiece.unigram_segment import viterbi_segment

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def get_substrings(
        word: str,
        subwrd2idx: Dict[str, int],
        max_len: int = 10,
        exclude_subwords: List[str] = None) -> List[Tuple[str, int]]:
    substrings = []
    for sub_len in range(1, min(len(word), max_len) + 1):
        for i in range(0, len(word) - sub_len + 1):
            substr = word[i:i + sub_len]
            if (substr in subwrd2idx and
                (exclude_subwords is not None and
                 substr not in exclude_subwords)):
                substrings.append((substr, subwrd2idx[substr]))
    return substrings


def try_segment(
        word: str,
        fasttext: Union[FastText, np.ndarray],
        subwrd2idx: Dict[str, int],
        subword_embeddings: np.array,
        inference_mode: str = "sum",
        sample: bool = False,
        exclude_subwords: List[str] = None) -> List[Tuple[List[str], float]]:
    if isinstance(fasttext ,np.ndarray):
        vector = fasttext
    else:
        try:
            vector = fasttext.wv[word]
        except KeyError:
            vector = fasttext.wv.vectors.mean(0)
    subwords = get_substrings(
        word, subwrd2idx, exclude_subwords=exclude_subwords)
    subword_scores = {
        swrd: -distance.cosine(vector, subword_embeddings[idx]) #+ 1e-9
        for swrd, idx in subwords}
    #if subword_scores:
    #    denominator = math.log(sum(subword_scores.values()))
    #    subword_scores = {
    #        swrd: math.log(val) - denominator
    #        for swrd, val in subword_scores.items()
    #    }

    def seg():
        seg, score = viterbi_segment(
            word, subword_scores, inference_mode=inference_mode, sample=True)
        return " ".join(seg), -score

    if sample:
        return list(sorted({seg() for _ in range(10)}, key=lambda x: x[1]))

    seg, score = viterbi_segment(
        word, subword_scores, inference_mode=inference_mode, sample=False)
    return [(" ".join(seg), -score)]


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
        "input", nargs="?", default=sys.stdin, type=argparse.FileType("r"),
        help="Input words to segment: word per line.")
    parser.add_argument(
        "--sample", default=None, type=int,
        help="Sample N segmentations and consider all their subwords.")
    parser.add_argument(
        "--limit-vocab", type=int, default=None,
        help="Limit subword vocabulary to N items.")
    parser.add_argument(
        "--embeddings-type", default="fasttext",
        choices=["fasttext", "w2v"])
    parser.add_argument(
        "--inference-mode", default="sum", choices=["sum", "maxmin"])
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
    if args.limit_vocab is not None:
        subwords = subwords[:args.limit_vocab]
        subword_embeddings = subword_embeddings[:args.limit_vocab]
        logging.info("Only using %d subwords.", len(subwords))

    subwrd2idx = {s: i for i, s in enumerate(subwords)}

    logging.info("Segment words.")
    for line in args.input:
        segmentations = try_segment(
            line.strip(), fasttext, subwrd2idx,
            subword_embeddings,
            inference_mode=args.inference_mode,
            sample=args.sample)
        #print(segmentations[0][0])
        if args.sample:
            segmnetation_options = {
                sbw for seg, _ in segmentations[:args.sample] for sbw in seg.split()}
            print(" ".join(segmnetation_options))
        else:
            print(segmentations[0][0])
        #for seg in segmentations[:10]:
        #    print(seg)

    args.input.close()
    logging.info("Done.")


if __name__ == "__main__":
    main()
