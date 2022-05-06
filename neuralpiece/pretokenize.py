#!/usr/bin/env python3

from typing import List
import argparse
from collections import defaultdict
import sys


def pretokenize(text: str) -> List[str]:
    text = "▁" + text.replace(" ", "▁")
    tokens = []
    token_start = 0
    for i, char in enumerate(text):
        if i == 0:
            continue

        if text[i - 1] != '▁' and (
                not char.isalnum() or not text[i - 1].isalnum()):
            tokens.append(text[token_start:i])
            token_start = i
    tokens.append(text[token_start:])
    return tokens


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "input", nargs="?", type=argparse.FileType("r"),
        default=sys.stdin)
    args = parser.parse_args()

    token_counts = defaultdict(int)
    for line in args.input:
        for token in pretokenize(line.strip()):
            if "\t" in token:
                continue
            token_counts[token] += 1

    for token, count in token_counts.items():
        print(f"{token}\t{count}")


if __name__ == "__main__":
    main()
