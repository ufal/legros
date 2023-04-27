#!/usr/bin/env python3

import argparse
import logging

from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
import numpy as np


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "corpus", type=str,
        help="Tokenized plain-text corpus.")
    parser.add_argument(
        "output", type=str,
        help="Path to saved model.")
    parser.add_argument("--num-threads", type=int, default=64)
    parser.add_argument("--dimension", type=int, default=200)
    parser.add_argument("--vocab-size", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--type", default="fasttext", choices=["fasttext", "w2v"])
    args = parser.parse_args()

    logging.info("Initialize model with vocab.")
    if args.type == "fasttext":
        model = FastText(
            vector_size=args.dimension,
            workers=args.num_threads,
            sg=1,
            max_final_vocab=args.vocab_size)
    else:
        model = Word2Vec(
            vector_size=args.dimension,
            workers=args.num_threads,
            sg=1,
            max_final_vocab=args.vocab_size)

    model.build_vocab(
        corpus_file=args.corpus,
        progress_per=2000000)

    logging.info("Start training the model.")
    model.train(
        corpus_file=args.corpus,
        epochs=args.epochs,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words)

    logging.info("Trained, save the model to '%s'.", args.output)
    vocab = list(model.wv.key_to_index.keys())
    with open(f"{args.output}.vocab", "w") as f_vocab:
        for word in vocab:
            print(word, file=f_vocab)
    model.save(args.output)

    logging.info("Transpose output matrix.")
    out_inv = np.linalg.pinv(model.syn1neg)
    logging.info("Save output matrix as text file..")
    np.savetxt(f"{args.output}.out_inv.txt", out_inv)

    logging.info("Save embeddings in the text format.")
    model.wv.save_word2vec_format(f"{args.output}.txt")

    logging.info("Done.")


if __name__ == "__main__":
    main()
