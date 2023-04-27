#!/usr/bin/env python3

"""Extract all character n-grams form dictionary.

This is meant for experimenting with an initial vocabulary consisting of
character ngrams.
"""

import argparse
from collections import defaultdict
import logging
import sys


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "input", nargs="?", type=argparse.FileType("r"),
        default=sys.stdin)
    parser.add_argument(
        "--max-ngram", default=8, type=int)
    args = parser.parse_args()

    logging.info("Reading vocabulary from %s.", args.input)
    ngrams = defaultdict(int)
    for ln, line in enumerate(args.input):
        print(ln, end="\r", file=sys.stderr)
        token, count_str = line.rstrip().split("\t")
        count = int(count_str)
        for i in range(len(token)):
            for j in range(args.max_ngram):
                ngrams[token[i:i + j + 1]] += count

    logging.info("Reading vocab file finished. Sort and print.")

    for ngram, count in sorted(ngrams.items(), key=lambda x: -x[1]):
        print(f"{ngram}\t{count}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
