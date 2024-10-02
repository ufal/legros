#!/usr/bin/evn python3

"""Get pretokenized vocabulary for a corpus."""

import argparse
import multiprocessing
import sys

from legros.pretokenize import pretokenize


def get_vocab(lines):
    vocabulary = {}

    for line in lines:
        for token in pretokenize(line):
            if token not in vocabulary:
                vocabulary[token] = 0
            vocabulary[token] += 1
    return vocabulary


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
        "--num-threads", type=int, default=12,
        help="Number of threads")
    args = parser.parse_args()

    pool = multiprocessing.Pool(processes=args.num_threads)
    vocabulary = {}

    line_buffers = []
    current_buffer = []

    for i, line in enumerate(args.input):
        print(i, end="\r", file=sys.stderr)
        current_buffer.append(line.rstrip())

        if len(current_buffer) > 200:
            line_buffers.append(current_buffer)
            current_buffer = []

        if len(line_buffers) > args.num_threads:
            for vocab in pool.map(get_vocab, line_buffers):
                merge_in_vocab(vocabulary, vocab)
            line_buffers = []

    line_buffers.append(current_buffer)
    for vocab in pool.map(get_vocab, line_buffers):
        merge_in_vocab(vocabulary, vocab)

    for token, count in sorted(vocabulary.items(), key=lambda x: -x[1]):
        print(f"{token}\t{count}")


if __name__ == "__main__":
    main()
