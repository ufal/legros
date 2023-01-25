#!/usr/bin/env python3

import argparse
import logging

import numpy as np
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix
import sentencepiece as spm
import pandas as pd


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("fasttext", help="W2V/FastText model from Gensim.")
    parser.add_argument("word_cooccurrence", type=argparse.FileType('r'))
    #parser.add_argument(
    #    "subword_vocab", type=argparse.FileType("r"),
    #    help="Subword vocab, subword per line.")
    parser.add_argument("sentencepiece", type=str)
    parser.add_argument(
        "--sample", default=None, type=int,
        help="Sample N segmentations and consider all their subwords.")
    parser.add_argument(
        "--limit-vocab", type=int, default=None,
        help="Limit subword vocabulary to N items.")
    parser.add_argument(
        "--embeddings-type", default="fasttext",
        choices=["fasttext", "w2v"])
    parser.add_argument(
        "--inference-mode", default="maxmin", choices=["sum", "maxmin"])
    args = parser.parse_args()

    logging.info("Load word embeddings model from %s.", args.fasttext)
    if args.embeddings_type == "fasttext":
        fasttext = FastText.load(args.fasttext)
    else:
        fasttext = Word2Vec.load(args.fasttext)
    word_vocab = fasttext.wv.index_to_key
    word2idx = {wrd: i for i, wrd in enumerate(word_vocab)}

    logging.info("Load word cooccurrence matrix.")
    word_cooccurrence = np.zeros((len(word_vocab), len(word_vocab)), dtype=np.int32)
    for ln_n, line in enumerate(args.word_cooccurrence):
        i_str, j_str, val_str = line.strip().split()
        word_cooccurrence[int(i_str), int(j_str)] = int(val_str)
        if ln_n % 1000 == 999:
            print(ln_n)


    #logging.info(
    #    "Load subword vocabulary from %s.", args.subword_vocab)
    #subwords = []
    #for line in args.subword_vocab:
    #    subwords.append(line.strip())
    #args.subword_vocab.close()


    logging.info("Load SentencePiece model.")
    spm_model = spm.SentencePieceProcessor(model_file=args.sentencepiece)
    subword_vocab = [
        spm_model.id_to_piece(id) for id in range(spm_model.get_piece_size())]
    subword2idx = {swrd: i for i, swrd in enumerate(subword_vocab)}

    #subword_word_cooc = csr_matrix((len(subword_vocab), len(word_vocab)))
    subword_word_cooc = np.zeros((len(subword_vocab), len(word_vocab)))

    logging.info("Iterate over vocab.")
    for wrd_id, word in enumerate(word_vocab):
        print(f"{wrd_id}/{len(word_vocab)}", end="\r")
        for _ in range(10):
            for swrd in spm_model.encode(word, out_type=str,
                    enable_sampling=True, alpha=0.1, nbest_size=-1):
                if swrd not in subword2idx:
                    continue
                swrd_idx = subword2idx[swrd]
                subword_word_cooc[swrd_idx, wrd_id] += 1
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
