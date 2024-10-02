#i!/usr/bin/env python3

import argparse
import logging
import math
from typing import Dict, List, Tuple, Union
import sys

from gensim.models.fasttext import FastText
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from scipy.spatial import distance
from scipy.special import logsumexp

from legros.unigram_segment import viterbi_segment, expected_counts

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def get_substrings(
        word: str,
        subwrd2idx: Dict[str, int],
        max_len: int = 10,
        is_bert_wordpiece: bool = False,
        exclude_subwords: List[str] = None) -> List[Tuple[str, int]]:
    substrings = []
    for sub_len in range(1, min(len(word), max_len) + 1):
        for i in range(0, len(word) - sub_len + 1):
            substr = word[i:i + sub_len]
            if is_bert_wordpiece and i > 0:
                substr = "##" + substr
            if (substr in subwrd2idx and
                (exclude_subwords is None or substr not in exclude_subwords)):
                substrings.append((substr, subwrd2idx[substr]))
    return substrings


def try_segment(
        word: str,
        fasttext: Union[FastText, np.ndarray],
        subwrd2idx: Dict[str, int],
        subword_embeddings: np.array,
        inference_mode: str = "sum",
        sample: bool = False,
        is_bert_wordpiece: bool = False,
        exclude_subwords: List[str] = None) -> List[Tuple[List[str], float]]:
    if isinstance(fasttext, np.ndarray):
        vector = fasttext
    else:
        try:
            vector = fasttext[word]
        except KeyError:
            vector = fasttext.vectors.mean(0)
    subwords = get_substrings(
        word, subwrd2idx, exclude_subwords=exclude_subwords,
        is_bert_wordpiece=is_bert_wordpiece)
    subword_scores = {
            swrd: -distance.cosine(vector, subword_embeddings[idx]) #+ 1e-9
            #swrd: 2 - distance.cosine(vector, subword_embeddings[idx]) #+ 1e-9
            #swrd: np.dot(vector, subword_embeddings[idx])
        for swrd, idx in subwords}
    #if subword_scores:
    #    #denominator = math.log(sum(subword_scores.values()))
    #    denominator = logsumexp(list(subword_scores.values()))
    #    subword_scores = {
    #        #swrd: math.log(val) - denominator
    #        swrd: val - denominator
    #        for swrd, val in subword_scores.items()
    #    }

    def seg():
        seg, score = viterbi_segment(
            word, subword_scores, inference_mode=inference_mode,
            sample=True, is_bert_wordpiece=is_bert_wordpiece)
        return " ".join(seg), -score

    if sample:
        return list(sorted({seg() for _ in range(10)}, key=lambda x: x[1]))

    seg, score = viterbi_segment(
        word, subword_scores, inference_mode=inference_mode,
        sample=False, is_bert_wordpiece=is_bert_wordpiece)
    return [(" ".join(seg), -score)]


def expected_counts_segment(
        word: str,
        fasttext: Union[FastText, np.ndarray],
        subwrd2idx: Dict[str, int],
        subword_embeddings: np.array,
        exclude_subwords: List[str] = None) -> Dict[str, float]:
    if isinstance(fasttext, np.ndarray):
        vector = fasttext
    else:
        try:
            vector = fasttext[word]
        except KeyError:
            vector = fasttext.vectors.mean(0)
    subwords = get_substrings(
        word, subwrd2idx, exclude_subwords=exclude_subwords)
    subword_scores = {
            #swrd: -distance.cosine(vector, subword_embeddings[idx]) #+ 1e-9
            #swrd: 2 - distance.cosine(vector, subword_embeddings[idx]) #+ 1e-9
        swrd: np.dot(vector, subword_embeddings[idx])
        for swrd, idx in subwords}
    if subword_scores:
        #denominator = math.log(sum(subword_scores.values()))
        denominator = logsumexp(list(subword_scores.values()))
        subword_scores = {
            #swrd: math.log(val) - denominator
            swrd: val - denominator
            for swrd, val in subword_scores.items()
        }

    counts = expected_counts(word, subword_scores)
    counts_list = [(swrd, math.exp(x)) for swrd, x in counts.items()]
    counts_list.sort(key=lambda x: -x[1])
    return counts_list


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
        choices=["fasttext", "w2v", "txt"])
    parser.add_argument(
        "--inference-mode", default="sum", choices=["sum", "maxmin"])
    parser.add_argument(
        "--expected-counts", default=False, action="store_true",
        help="Return expected counts.")
    parser.add_argument(
        "--excluded", default=None, type=argparse.FileType("r"),
        help="File with a list of forbidden subwords.",
        required=False)
    parser.add_argument(
        "--bert-wordpiece", default=False, action="store_true",
        help="Set for tokenizer based BER's WordPiece.",
        required=False)
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

    exclude_subwords = None
    if args.excluded is not None:
        logging.info("Loading excluded subwords from args.excluded.")
        exclude_subwords = {line.rstrip() for line in args.excluded}
        args.excluded.close()

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
        if args.expected_counts:
            expected_counts = expected_counts_segment(
                line.strip(), fasttext, subwrd2idx,
                subword_embeddings)
            print(" ".join(f"{s} {v}" for s, v in expected_counts))
        else:
            segmentations = try_segment(
                line.strip(), fasttext, subwrd2idx,
                subword_embeddings,
                inference_mode=args.inference_mode,
                sample=args.sample,
                is_bert_wordpiece=args.bert_wordpiece,
                exclude_subwords=exclude_subwords)

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
