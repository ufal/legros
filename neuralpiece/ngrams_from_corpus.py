#!/usr/bin/env python3

"""Extract all character n-grams form dictionary.

This is meant for experimenting with an initial vocabulary consisting of
character ngrams.
"""

from typing import List, Dict
import argparse
from collections import defaultdict
import multiprocessing
import sys

from neuralpiece.pretokenize import pretokenize


class NgramsCounter:
    def __init__(self, order: int) -> None:
        self.order = order

    def get_ngram_counts(self, lines: List[str]) -> Dict[str, int]:
        ngrams = defaultdict(int)

        for line in lines:
            for token in pretokenize(line.rstrip()):
                for i in range(len(token)):
                    for j in range(self.order):
                        ngrams[token[i:i + j + 1]] += 1
        return ngrams


def merge_in_vocab(orig, new):
    for token, count in new.items():
        if token not in orig:
            orig[token] = count
        else:
            orig[token] += count


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "input", nargs="?", type=argparse.FileType("r"),
        default=sys.stdin)
    parser.add_argument(
        "--max-ngram", default=4, type=int)
    parser.add_argument(
        "--num-threads", default=4, type=int)
    args = parser.parse_args()

    counter = NgramsCounter(args.max_ngram)

    pool = multiprocessing.Pool(processes=args.num_threads)
    ngrams = {}

    for i, input_file in enumerate(args.input):
        print(i, end="\r", file=sys.stderr)
        line_buffers = []
        current_buffer = []

        for line in input_file:
            current_buffer.append(line)

            if len(current_buffer) > 200:
                line_buffers.append(current_buffer)
                current_buffer = []

            if len(line_buffers) > args.num_threads:
                for vocab in pool.map(counter.get_ngram_counts, line_buffers):
                    merge_in_vocab(ngrams, vocab)
                line_buffers = []

        line_buffers.append(current_buffer)
        for vocab in pool.map(counter.get_ngram_counts, line_buffers):
            merge_in_vocab(ngrams, vocab)

    for ngram in ngrams:
        print(ngram)


if __name__ == "__main__":
    main()
