/**
  get_substring_contexts.cpp
*/

#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include "options.h"
#include "vocabs.h"
#include "substring_stats.h"
#include "CLI11.hpp"



int main(int argc, char* argv[]) {
  CLI::App app{"Compute subword embeddings."};
  get_options(app, argc, argv);
  CLI11_PARSE(app, argc, argv);

  std::cerr << "Loading subword vocab: " << opt.subword_vocab_file << std::endl;
  std::unordered_map<std::string, int> subword_to_index;
  get_word_to_index(subword_to_index, opt.subword_vocab_file);

  std::cerr << "Loading word vocab: " << opt.word_vocab_file << std::endl;
  std::unordered_map<std::string, int> word_to_index;
  get_word_to_index(word_to_index, opt.word_vocab_file);

  std::cerr << "Index of 'společenství': " << word_to_index["společenství"] << std::endl;

  int subword_count = subword_to_index.size();
  int word_count = word_to_index.size();

  std::vector<std::vector<int>> stats(
    subword_count, std::vector<int>(word_count));

  populate_substring_stats<std::vector<std::vector<int>>>(stats, word_to_index, subword_to_index);

  std::cerr << "Dumping stats to " << opt.output_file << std::endl;
  std::ofstream output_fh(opt.output_file);

  // how many buffers
  int output_buffer_size = 1000;
  int howmany = stats.size() / output_buffer_size;
  int remainder = stats.size() % output_buffer_size;

  std::vector<std::string> partial_strings(output_buffer_size);

  for(int b = 0; b <= howmany; ++b) {
    int begin = b * output_buffer_size;
    int end = std::min((int)stats.size(), begin + output_buffer_size);
    int diff = end - begin;

    #pragma omp parallel for
    for(int i = begin; i < end; ++i) {
      std::ostringstream ss;
      for(const auto &e : stats[i]) {
        ss << e << " ";
      }
      ss << "\n";
      partial_strings[i - begin] = ss.str();
    }

    for(int i = begin; i < end; ++i) {
      std::cerr << "Line: " << i << "\r";
      output_fh << partial_strings[i - begin];
    }
  }

  output_fh.close();

  std::cerr << std::endl << "Done." << std::endl;

  return 0;
}