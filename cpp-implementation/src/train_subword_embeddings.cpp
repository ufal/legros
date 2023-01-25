/**
 * Train subword embeddings using a pseudoEM algorithm.
 *
 * Word vocabulary
 * Subword vocabulary
 * Word cooccurrence matrix
 *   - (obtained from data using the word_cooccurrence_matrix binary)
 *
 * Fasttext word embeddings (pseudo-inverse)
 *
 * Output file (will contain the trained subword embeddings)
 */

#include <string>
#include <iostream>

#include "CLI11.hpp"

#include "vocabs.h"
#include "word_cooccurrence_matrix.h"
#include "substring_stats.h"

struct opt {
  std::string subword_vocab;
  std::string word_vocab;
  //std::string cooccurrence_matrix;
  std::string allowed_substrings;
  std::string fasttext_output_pseudoinverse;
  std::string output;
  std::string train_data;

  int window_size = 3;
} opt;

void get_options(CLI::App& app) {
  app.add_option("word_vocabulary", opt.word_vocab,
                 "Word vocabulary, word per line.")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option("subword_vocabulary", opt.subword_vocab,
                 "Subword vocabulary, subword per line.")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option("train_data", opt.train_data,
                 "Training data for cooccurrence matrix.")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option("--allowed-substrings", opt.allowed_substrings,
                 "List of words accompanied with allowed substrings.")
      ->check(CLI::ExistingFile);

  // app.add_option(
  //   "word_cooccurrence_matrix", opt.cooccurrence_matrix,
  //   "Word cooccurrence matrix.")
  //   ->required()
  //   ->check(CLI::ExistingFile);

  app.add_option("--window-size",
                 opt.window_size, "Window size.");

}

int main(int argc, char* argv[]) {
  CLI::App app{"Train subword embeddings using a pseudoEM algorithm."};
  get_options(app);
  CLI11_PARSE(app, argc, argv);

  // compute word cooccurrence matrix for data C_v (dim. V x V)
  // implemented in word_cooccurrence_matrix.cpp
  std::cerr << "Loading word vocab: " << opt.word_vocab << std::endl;
  Vocab word_vocab(opt.word_vocab);

  std::string test_word = "včelař";
  std::cerr << "Index of '" << test_word << "': " << word_vocab[test_word]
            << std::endl;

  std::cerr << "Loading subword vocab: " << opt.subword_vocab << std::endl;
  Vocab subword_vocab(opt.subword_vocab);

  int word_count = word_vocab.size();
  //int subword_count = subword_vocab.size();

  Eigen::MatrixXi c_v(word_count, word_count);
  std::cerr << "Populating word cooccurrence stats (" << word_count
            << " words)" << std::endl;

  populate_word_stats<Eigen::MatrixXi>(c_v, word_vocab, opt.train_data, opt.window_size);

  std::cerr << "Done, here are some stats:" << std::endl;
  std::cerr << c_v.topLeftCorner<5,5>() << std::endl;



  // construct allowed substring matrix A (dim. S x V)



  // compute subword-word cooccurrence matrix AC_v (dim. S x V)


  // get subword embeddings from AC_v

  // segment data using subword embeddings - unigram model

  // recompute matrix A




  return 0;
}
