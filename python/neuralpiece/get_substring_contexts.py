#!/usr/bin/env python3

"""Get subword count in the context of other words."""


from typing import List

import argparse
from collections import defaultdict
import logging
import pickle
import sys

import numpy as np


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


SUBSTR_CACHE = {}


def get_all_substrings(word: str, max_len: int = 10) -> List[str]:
    if word in SUBSTR_CACHE:
        return SUBSTR_CACHE[word]

    substrings = []
    for sub_len in range(1, min(len(word), max_len) + 1):
        for i in range(0, len(word) - sub_len + 1):
            substrings.append(word[i:i + sub_len])
    SUBSTR_CACHE[word] = substrings
    return substrings


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "subword_vocabulary", type=argparse.FileType("r"),
        help="Subword vocabulary, subword per line.")
    parser.add_argument(
        "word_vocabulary", type=argparse.FileType("r"),
        help="Word vocabulary, word per line.")
    parser.add_argument(
        "output", type=str, help="Output file.")
    parser.add_argument(
        "input", help="Tokenized text.",
        type=argparse.FileType("r"), nargs="?", default=sys.stdin)
    parser.add_argument("--window-size", default=3, type=int)
    parser.add_argument("--max-subword", default=10, type=int)
    parser.add_argument(
        "--min-freq", type=int, default=1,
        help="Minimum frequence for a subword to be included.")
    parser.add_argument(
        "--min-forms", type=int, default=2,
        help="Minimum number of forms to be in for a subword to be included.")
    args = parser.parse_args()

    word2idx = {wrd.strip(): i for i, wrd in enumerate(args.word_vocabulary)}
    args.word_vocabulary.close()
    logging.info("Loaded word vocab from '%s'.", args.word_vocabulary)

    substr2idx = {
        wrd.strip(): i for i, wrd in enumerate(args.subword_vocabulary)}
    args.word_vocabulary.close()
    logging.info("Loaded subword vocab from '%s'.", args.subword_vocabulary)

    # JH: Proc je tady ten int? to ma bejt defaultni hodnota
    stats = [defaultdict(int) for _ in substr2idx]
    #stats = np.zeros(
    #    (len(substr2idx), len(word2idx)), dtype=int)

    def try_add_to_stats(token: str, substrings: List[str]) -> None:
        if token not in word2idx:
            return
        wrd_idx = word2idx[token]
        for substr in substrings:
            if substr not in substr2idx:
                continue
            stats[substr2idx[substr]][wrd_idx] += 1

    logging.info("Iterate over sentences from '%s'.", args.input)
    ln_n = 0
    for line in args.input:
        ln_n += 1
        tokens = line.strip().split()
        for i, tok in enumerate(tokens):
            substrings = get_all_substrings(tok, max_len=args.max_subword)
            for j in range(max(0, i - args.window_size), i):
                try_add_to_stats(tokens[j], substrings)

            # bug
            #if i == len(tokens) - 2:
            #    continue
            
            for j in range(i + 1, min(i + 1 + args.window_size, len(tokens))):
                try_add_to_stats(tokens[j], substrings)
        print(ln_n, file=sys.stderr, end="\r")
    args.input.close()
    logging.info("Read %d lines in total.", ln_n)

    logging.info("Transform stats to a dense matrix.")
    stats_np = np.zeros(
        (len(substr2idx), len(word2idx)), dtype=np.int32)
    for i, subword_stat in enumerate(stats):
        for key, val in subword_stat.items():
            stats_np[i, key] = val

    logging.info("Save result, total %d substrings.", len(stats))
    np.save(args.output, stats_np)
    logging.info("Done.")


if __name__ == "__main__":
    main()
