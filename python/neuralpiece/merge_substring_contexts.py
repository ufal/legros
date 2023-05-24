#!/usr/bin/env python3

"""Merge subword context counts."""


import argparse
import pickle
import logging
import gc

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "stats_1", type=argparse.FileType("rb"),
        help="Pickle file with context stats.")
    parser.add_argument(
        "stats_2", type=argparse.FileType("rb"),
        help="Pickle file with context stats.")
    parser.add_argument(
        "output", type=argparse.FileType("wb"),
        help="Pickle file with merged stats.")
    parser.add_argument(
        "--min-freq", type=int, default=1,
        help="Minimum frequence for a subword to be included.")
    parser.add_argument(
        "--min-forms", type=int, default=2,
        help="Minimum number of forms to be in for a subword to be included.")
    args = parser.parse_args()

    logging.info(
        "Merging stat files %s and %s.", args.stats_1, args.stats_2)

    logging.info("Loading %s.", args.stats_1)
    stats = pickle.load(args.stats_1)
    args.stats_1.close()

    logging.info("Loading %s.", args.stats_2)
    stats_2 = pickle.load(args.stats_2)
    args.stats_2.close()

    logging.info("Tables loaded, now merge.")

    stats2_subwords = list(stats_2.keys())

    for subword in stats2_subwords:
        freqs = stats_2[subword]
        if subword not in stats:
            stats[subword] = freqs
            continue
        for word_id, count in freqs.items():
            stats[subword][word_id] += count
        del stats_2[subword]
        gc.collect()
    del stats_2
    gc.collect()

    logging.info("In total %d subwords. Filter.", len(stats))
    to_delete = []
    for subword, freqs in stats.items():
        if (len(subword) > 1 and
                (len(freqs) < args.min_forms or
                 sum(freqs.values()) < args.min_freq)):
            to_delete.append(subword)
            continue

    for subword in to_delete:
        del stats[subword]

    logging.info("After filtering %d subwords. Save output to %s.", len(stats), args.output)
    pickle.dump(stats, args.output)
    logging.info("Done.")


if __name__ == "__main__":
    main()
