#include <string>
#include <vector>
#include <iostream>
#include "CLI11.hpp"

#include "vocabs.h"
#include "unigram_model.h"

struct opt {
  std::string embeddings_file;
  std::string subword_vocab_file;
  std::string inverse_embeddings_file;
  std::string saved_model_file;
} opt;

void get_options(CLI::App& app) {
  app.add_option(
      "embeddings_file", opt.embeddings_file, "Word embeddings.")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option(
      "subword_vocab_file", opt.subword_vocab_file, "List of subwords")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option(
      "pseudo_inverse_embeddings", opt.inverse_embeddings_file,
      "File with a pseudo-inverse matrix of word embeddings")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option(
      "saved_model_file", opt.saved_model_file, "Model file (the W_s matrix)")
      ->required()
      ->check(CLI::ExistingFile);
}

int main(int argc, char* argv[]) {
  CLI::App app{
    "Byte-based Forward-backward EM estimation of subword embeddings."};
  get_options(app);
  CLI11_PARSE(app, argc, argv);

  std::cerr << "Loading subword vocab: " << opt.subword_vocab_file
            << std::endl;
  Vocab subword_vocab(opt.subword_vocab_file);

  std::cerr << "Loading embedding matrix from " << opt.embeddings_file
            << std::endl;
  Embeddings words(opt.embeddings_file);

  int embedding_dim = words.embedding_dim;
  int word_count = words.size();

  std::cerr << "Loading the pseudo-inverse embedding matrix from "
            << opt.inverse_embeddings_file << std::endl;
  std::ifstream inv_embedding_fh(opt.inverse_embeddings_file);
  Eigen::MatrixXf inverse_emb(embedding_dim, word_count);

  int i = 0;
  for(std::string line; std::getline(inv_embedding_fh, line); ++i) {
    std::stringstream ss(line);
    for(int j = 0; j < word_count; ++j) {
      ss >> inverse_emb(i, j);
    }
  }

  UnigramModel model(words, subword_vocab, inverse_emb);
  std::cerr << "Loading model from " << opt.saved_model_file << std::endl;
  model.load(opt.saved_model_file);

  for(std::string word; std::getline(std::cin, word);) {
    std::vector<std::string> segm;
    try {
      model.viterbi_decode(segm, word);

      std::string sep = "";
      for(auto it = segm.rbegin(); it != segm.rend(); ++it) {
        std::cout << sep << *it;
        sep = " ";
      }
      std::cout << std::endl;
    }
    catch (...) {
      std::cout << "OOV" << std::endl;
    }

  }

  return 0;
}
