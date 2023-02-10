#!/usr/bin/env python3

import argparse
import logging
import os

import sentencepiece as spm

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


SP_SPACE = "▁"
SPECIALS = [SP_SPACE, '<unk>', '<s>', '</s>']


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("sp_model", type=str)
    parser.add_argument(
        "vocabulary", type=argparse.FileType("r"),
        help="FastText vocabulary used during training.")
    parser.add_argument("target_dir", type=str)
    args = parser.parse_args()

    logging.info("Create target dir.")
    os.mkdir(args.target_dir)

    logging.info("Load word vocabulary from '%s'.", args.vocabulary)
    word_vocab = []
    for line in args.vocabulary:
        word_vocab.append(line.strip())
    args.vocabulary.close()

    logging.info("Load SentencePiece model from '%s'.", args.sp_model)
    model = spm.SentencePieceProcessor(model_file=args.sp_model)

    logging.info("Get SentencePiece vocab.")
    sp_vocab = set()
    for id in range(model.get_piece_size()):
        piece = model.id_to_piece(id)
        if piece in SPECIALS:
            continue
        if piece.startswith(SP_SPACE):
            piece = piece[1:]
        sp_vocab.add(piece)
    with open(os.path.join(args.target_dir, "spvocab"), "w") as f_voc:
        for piece in sp_vocab:
            print(piece, file=f_voc)

    logging.info("Segment vocabulary.")
    with open(os.path.join(args.target_dir, "allowed.init"), "w") as f_seg:
        for word in word_vocab:
            segmentation = []
            for piece in model.encode(word, out_type=str):
                if piece == SP_SPACE:
                    continue
                if piece.startswith(SP_SPACE):
                    piece = piece[1:]
                segmentation.append(piece)
            print(f"{word} {' '.join(segmentation)}", file=f_seg)

    logging.info("Done.")


if __name__ == "__main__":
    main()
