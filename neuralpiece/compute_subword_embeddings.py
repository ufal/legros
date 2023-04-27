#!/usr/bin/evn python3

import argparse
import io
import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "output_matrix", type=str,
        help="File with the ouput projection matrix from W2V.")
    parser.add_argument(
        "subword_vocab", type=argparse.FileType("r"),
        help="Subword vocab.")
    parser.add_argument(
        "output", type=str,
        help="NumPy file with subword freqnecies and embeddings.")
    parser.add_argument(
        "subword_stats", type=argparse.FileType("r"),
        nargs="?", default=sys.stdin,
        help="Pickled subword stats.")
    parser.add_argument("--smoothing", default=1e-5, type=float)
    parser.add_argument("--line-buffer", default=1000, type=int)
    args = parser.parse_args()

    logging.info("Load word embeddings output matrix.")
    out_mat = np.load(args.output_matrix)
    vocab_size, _ = out_mat.shape
    logging.info("Ouput matix shape: %s.", out_mat.shape)
    logging.info("Inverse the output matrix.")
    out_mat_inv = np.linalg.pinv(out_mat)

    logging.info("Load subword vocab.")
    vocab = []
    for line in args.subword_vocab:
        vocab.append(line.strip())
    args.subword_vocab.close()

    logging.info("Load subword stats.")
    line_buffer = []
    embedding_list = []
    subword_counts = []

    def process_buffer():
        #subword_data = np.loadtxt(args.subword_stats)
        subword_data = pd.read_csv(
            io.StringIO("\n".join(line_buffer)),
            delimiter=" ", header=None, engine="pyarrow").values
        logging.info("Buffer turned into a numpy matrix.")
        subword_counts.extend(subword_data.sum(1).tolist())

        assert subword_data.shape[1] == vocab_size

        logging.info("Total %d subwords.", len(subword_data))

        logging.info("Normalize the subword count to log distribution.")
        subword_data = subword_data.astype(float) + args.smoothing
        subword_data = (
            np.log(subword_data) - np.log(subword_data.sum(1, keepdims=True)))

        logging.info("Compute the subword embeddings.")
        embedding_list.append(subword_data.dot(out_mat_inv.T))

    for line in args.subword_stats:
        line_buffer.append(line.strip())
        if len(line_buffer) < args.line_buffer:
            continue
        logging.info("Processing a buffer of %d lines.", len(line_buffer))
        process_buffer()

        logging.info("In total processed %d stats line.", len(subword_counts))
        line_buffer = []

    if line_buffer:
        process_buffer()

    #assert len(vocab) == len(subword_counts)
    embeddings = np.concatenate(embedding_list, axis=0)
    #assert len(vocab) == embeddings.shape[0]

    logging.info("Save the subword embeddings.")
    np.savez(
        args.output,
        vocab=vocab,
        embeddings=embeddings,
        subword_counts=subword_counts)
    logging.info("Done.")


if __name__ == "__main__":
    main()
