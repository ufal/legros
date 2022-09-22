#!/usr/bin/env python3

"""Get subword count in the context of other words."""


from typing import List

import argparse
from collections import defaultdict
import json
import logging
import sys

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
        "word_vocabulary", type=argparse.FileType("r"),
        help="Word vocabulary, word per line.")
    parser.add_argument(
        "input", help="Tokenized text.",
        type=argparse.FileType("r"), nargs="?", default=sys.stdin)
    parser.add_argument("--window-size", default=3, type=int)
    parser.add_argument("--max-subword", default=10, type=int)
    args = parser.parse_args()

    word2idx = {wrd.strip(): i for i, wrd in enumerate(args.word_vocabulary)}
    args.word_vocabulary.close()

    stats = defaultdict(lambda: defaultdict(int))
    logging.info("Loaded word vocab from '%s'.", args.word_vocabulary)

    def try_add_to_stats(token: str, substrings: List[str]) -> None:
        if token not in word2idx:
            return
        wrd_idx = word2idx[token]
        for substr in substrings:
            stats[substr][wrd_idx] += 1

    logging.info("Iterate over sentences from '%s'.", args.input)
    for line in args.input:
        tokens = line.strip().split()
        for i, tok in enumerate(tokens):
            substrings = get_all_substrings(tok)
            for j in range(max(0, i - args.window_size), i):
                try_add_to_stats(tokens[j], substrings)

            if i == len(tokens) - 2:
                continue
            for j in range(i + 1, min(i + 1 + args.window_size, len(tokens))):
                try_add_to_stats(tokens[j], substrings)

    logging.info("Print result, total %d substrings.", len(stats))
    print(json.dumps(stats))
    logging.info("Done.")


if __name__ == "__main__":
    main()
