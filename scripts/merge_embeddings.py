#!/usr/bin/env python3

import argparse
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "output", type=str, help="Output NPZ file.")
    parser.add_argument(
        "input", type=str, nargs="+", help="Output NPZ file.")
    args = parser.parse_args()

    logging.info("Merge %d embeddings files.", len(args.input))

    logging.info("Procees '%s'.", args.input[0])
    data = np.load(args.input[0])
    embeddings = data["embeddings"]
    vocab = data["vocab"].tolist()
    subword_counts = data["subword_counts"]

    for file_name in args.input[1:]:
        logging.info("Procees '%s'.", file_name)
        data = np.load(file_name)
        vocab.extend(data["vocab"].tolist())
        embeddings = np.concatenate(
            (embeddings, data["embeddings"]))
        subword_counts = np.concatenate(
            (subword_counts, data["subword_counts"]))

    logging.info("Save output to '%s'.", args.output)
    np.savez(
        args.output,
        embeddings=embeddings,
        vocab=vocab,
        subword_counts=subword_counts)


if __name__ == "__main__":
    main()
