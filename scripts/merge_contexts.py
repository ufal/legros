#!/usr/bin/env python3

"""Merge the context stats."""

import argparse
import logging

import numpy as np

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=str)
    parser.add_argument("input", type=str, nargs="+")
    args = parser.parse_args()

    logging.info("Merge %d numpy files.", len(args.input))
    if len(args.input) < 2:
        raise ValueError("No input provided.")

    logging.info("Processing 1/%d.", len(args.input))
    stats = np.load(args.input[0])
    for i, stat_file in enumerate(args.input[1:]):
        logging.info("Processing %d/%d.", i + 2, len(args.input))
        stats += np.load(stat_file)

    logging.info("All files collected. Save.")
    np.save(args.output, stats)

    logging.info("Done.")


if __name__ == "__main__":
    main()
