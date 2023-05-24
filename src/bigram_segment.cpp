/**
 * Bigram segment -- using subword bigram statistics for subword segmentation.
 * Input:
 * - STDIN which gets segmented (tokenized text)
 * - bigram counts
 * - unigram counts
 *
 * Output:
 * - STDOUT
 */

#include <cassert>
#include <string>
#include <unordered_map>
#include "CLI11.hpp"
#include "vocabs.h"

typedef std::unordered_map<std::string, int> unigram_table;
typedef std::unordered_map<std::string,
                           std::unordered_map<std::string, int>> bigram_table;
typedef std::vector<std::vector<float>> matrix;

const std::string sub_sep = "@@ ";

struct opt {
  std::string bigram_stats;
  std::string unigram_stats;

} opt;

void get_options(CLI::App& app) {
  app.add_option(
      "bigram_stats", opt.bigram_stats, "Bigram statistics.")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option(
      "unigram_stats", opt.unigram_stats, "Unigram statistics.")
      ->required()
      ->check(CLI::ExistingFile);
}


int load_unigrams(unigram_table& unigram_frequencies,
                  const std::string& path) {
  int total_count = 0;
  std::ifstream ifs(path);
  for(std::string line; std::getline(ifs, line);) {
    std::istringstream iss(line);
    std::string subword;
    iss >> subword;
    int frequency;
    iss >> frequency;
    total_count += frequency;

    unigram_frequencies.insert({subword, frequency});
  }
  return total_count;
}

void load_bigrams(bigram_table& bigram_frequencies,
                  const std::string& path) {

  std::ifstream ifs(path);
  for(std::string line; std::getline(ifs, line);) {
    std::istringstream iss(line);
    std::string subword1, subword2;
    iss >> subword1;
    iss >> subword2;
    int frequency;
    iss >> frequency;

    bigram_frequencies[subword1][subword2] = frequency;
  }
}

int score_table_column_argmax(const std::vector<std::vector<float>>& table,
                              int col) {
  int best_index = -1;
  float best_value = -std::numeric_limits<float>::infinity();

  for(int row = 0; row < table.size(); ++row) {
    if(table[row][col] > best_value) {
      best_value = table[row][col];
      best_index = row;
    }
  }

  assert(best_index != -1);
  return best_index;
}

int score_table_row_argmax(const std::vector<std::vector<float>>& table,
                           int row) {
  int best_index = -1;
  float best_value = -std::numeric_limits<float>::infinity();

  for(int col = 0; col < table[row].size(); ++col) {
    if(table[row][col] > best_value) {
      best_value = table[row][col];
      best_index = col;
    }
  }

  return best_index;
}

template<typename T>
int argmax(const std::vector<T>& array) {
  int best_index = -1;
  T best_value = std::numeric_limits<T>::lowest();

  for(int i = 0; i < array.size(); ++i) {
    if(array[i] > best_value) {
      best_value = array[i];
      best_index = i;
    }
  }

  return best_index;
}

float score_bigram(const std::string& subword,
                   const std::string& prev,
                   unigram_table& unigrams,
                   bigram_table& bigrams,
                   int unigram_count) {

  // in case everything is OOV, return log uniform prob
  if(unigrams.count(prev) == 0 && unigrams.count(subword) == 0)
    return -std::log(unigram_count); // technically this should be vocab size

  // for prev OOVs, return log unigram prob
  if(unigrams.count(prev) == 0)
    return std::log((float)unigrams[subword] / (float)unigram_count);

  // for subword OOVs, use trivial add-one smoothing
  return std::log((float)(bigrams[prev][subword] + 1) / (float)unigrams[prev]);
}


void segment_token(std::vector<std::string>& segmentation,
                   const std::string& token,
                   unigram_table& unigrams,
                   bigram_table& bigrams,
                   int unigram_count,
                   int max_subword_length) {

  // todo pridat cache

  std::vector<std::vector<float>> score_table(
      token.size(), std::vector<float>(
          token.size(), -std::numeric_limits<float>::infinity()));

  std::vector<std::vector<int>> prev_rows(
      token.size(), std::vector<int>(token.size(), -1));

  for(int row = 0; row < token.size(); ++row) {
    int max_column = std::min((int)token.size(), row + max_subword_length);

    for(int col = row; col < max_column; ++col) {
      std::string subword = token.substr(row, col + 1 - row);

      if(unigrams.count(subword) == 0 && col > row)
        // we want to allow single-byte OOVs
        continue;

      if(row == 0) {
        float sc = score_bigram(
            subword, bow, unigrams, bigrams, unigram_count);
        score_table[row][col] = sc;
        continue;
      }

      float best_prev_score = -std::numeric_limits<float>::infinity();
      int best_prev_index = -1;

      int min_prev_row = std::max(0, row - max_subword_length);
      for(int prev_row = min_prev_row; prev_row < row; ++prev_row) {
        std::string prev_subword = token.substr(prev_row, row - prev_row);

        if(unigrams.count(prev_subword) == 0 && row - prev_row > 1)
          // if previous one was a single-byte, proceed even if it was an OOV
          continue;

        if(score_table[prev_row][row - 1] ==
           -std::numeric_limits<float>::infinity())
          continue;

        float bigram_score = score_bigram(subword, prev_subword, unigrams,
                                          bigrams, unigram_count)
                             + score_table[prev_row][row - 1];

        if(bigram_score > best_prev_score) {
          best_prev_score = bigram_score;
          best_prev_index = prev_row;
        }
      } // for prev_row

      assert(best_prev_index != -1);
      prev_rows[row][col] = best_prev_index;
      score_table[row][col] = best_prev_score;
    } // for col
  } // for row

  int subword_end = token.size();
  int row = score_table_column_argmax(score_table, token.size() - 1);

  // for(int i=0; i < score_table.size(); ++i) {
  //   for(int j = 0; j < score_table.size(); ++j) {
  //     std::cerr << score_table[i][j] << " ";
  //   }
  //   std::cerr << "\n";
  // }

  while(subword_end > 0) {
    int subword_begin = row;
    std::string subword = token.substr(subword_begin, subword_end - subword_begin);
    segmentation.push_back(subword);
    row = prev_rows[row][subword_end - 1];
    subword_end = subword_begin;
  }

  std::reverse(segmentation.begin(), segmentation.end());
}


int main(int argc, char* argv[]) {
  CLI::App app{"Bigram segment -- using subword bigram statistics for subword segmentation."};
  get_options(app);
  CLI11_PARSE(app, argc, argv);

  unigram_table unigram_frequencies;
  bigram_table bigram_frequencies;

  std::cerr << "loading bigrams and unigrams" << std::endl;

  // je potreba si pamatovat ze tohle neni vocab size ale data size
  int unigram_count = load_unigrams(unigram_frequencies, opt.unigram_stats);
  load_bigrams(bigram_frequencies, opt.bigram_stats);

  std::cerr << "done" << std::endl;

  int max_unigram_length = 0;
  for(auto unigram_f : unigram_frequencies) {
    int len = unigram_f.first.size();
    if(len > max_unigram_length)
      max_unigram_length = len;
  }

  std::cerr << "max unigram length: " << max_unigram_length << std::endl;

  for(std::string line; std::getline(std::cin, line);) {
    std::istringstream ss(line);

    std::string wordsep = "";
    for(std::string word; std::getline(ss, word, ' ');) {
      std::vector<std::string> segm;

      std::cout << wordsep;
      wordsep = " ";

      segment_token(segm, word, unigram_frequencies, bigram_frequencies,
                    unigram_count, max_unigram_length);

      for(auto it = segm.begin(); it != segm.end() - 1; ++it) {
        std::cout << *it << sub_sep;
      }

      std::cout << *(segm.end() - 1);
    }

    std::cout << std::endl;
  }

  return 0;
}
