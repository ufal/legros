#!/usr/bin/env python3

import argparse
import logging
import sys

from gensim.models.fasttext import FastText
from gensim.models.fasttext_inner import ft_hash_bytes
import numpy as np


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "fasttext", type=str,
        help="Path to saved model.")
    parser.add_argument(
        "vocab", type=argparse.FileType('r'), nargs="?", default=sys.stdin,
        help="File with subword vocabulary.")
    parser.add_argument(
        "output", type=argparse.FileType('w'), nargs="?", default=sys.stdout,
        help="Output -- saved substring embeddings.")
    args = parser.parse_args()

    logging.info("Load FastText.")
    fasttext = FastText.load(args.fasttext)

    logging.info("Iterate over vocabulary.")
    vectors = []
    for line in args.vocab:
        hash_id = ft_hash_bytes(str.encode(line.strip())) % fasttext.wv.bucket
        vectors.append(fasttext.wv.vectors_ngrams[hash_id])

    logging.info("Print embeddings.")
    np.savetxt(args.output, np.stack(vectors))

    logging.info("Done.")


if __name__ == "__main__":
    main()
