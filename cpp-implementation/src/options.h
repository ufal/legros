#pragma once

#include <string>
#include "CLI11.hpp"

struct opt {
  std::string subword_vocab_file;
  std::string word_vocab_file;
  std::string training_data_file;
  std::string fasttext_output_matrix;
  std::string output_file;
  std::string allowed_substrings;
  int max_subword = 10;
  int window_size = 3;
  int fasttext_dim = 200;
  size_t buffer_size = 1000000;
  int shard_size = 1000;
};

extern struct opt opt;

void get_options(CLI::App& app, int argc, char* argv[]);