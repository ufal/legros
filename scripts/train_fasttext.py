#!/usr/bin/env python3

import argparse
import logging

from gensim.models.fasttext import FastText


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "corpus", type=str,
        help="Tab-separated dictionary with frequencies.")
    parser.add_argument(
        "output", type=str,
        help="Path to saved model.")
    parser.add_argument("--num-threads", type=int, default=40)
    parser.add_argument("--dimension", type=int, default=200)
    parser.add_argument("--vocab-size", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    logging.info("Initialize model with vocab.")
    model = FastText(
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

    logging.info("Trained, save the model.")
    vocab = list(model.wv.key_to_index.keys())
    with open(f"{args.output}.vocab", "w") as f_vocab:
        for word in vocab:
            print(word, file=f_vocab)
    model.save(args.output)
    logging.info("Done.")


if __name__ == "__main__":
    main()
