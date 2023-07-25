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
#include "counters.h"

#define BUFFER_SIZE 10000

typedef std::vector<std::vector<float>> matrix;

const std::string sub_sep = "@@ ";

struct opt {
  std::string bigram_stats;
  std::string unigram_stats;
  int num_threads = 1;

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

  app.add_option(
      "num_threads", opt.num_threads, "Number of threads to use.")
      ->required();
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
                   const unigram_counter& unigrams,
                   const bigram_counter_n& bigrams,
                   int unigram_count) {

  // in case everything is OOV, return log uniform prob
  if(unigrams.count(prev) == 0 && unigrams.count(subword) == 0)
    return -std::log(unigram_count); // technically this should be vocab size

  // for prev OOVs, return log unigram prob
  if(unigrams.count(prev) == 0)
    return std::log((float)unigrams.at(subword) / (float)unigram_count);

  // for bigram OOVs, use trivial add-one smoothing
  if(bigrams.count(prev) == 0 || bigrams.at(prev).count(subword) == 0)
    return -std::log(unigrams.at(prev));

  // if everything is known, return the smoothed logprob
  return std::log((float)(bigrams.at(prev).at(subword) + 1)
                  / (float)unigrams.at(prev));
}


void segment_token(std::vector<std::string>& segmentation,
                   const std::string& token,
                   const unigram_counter& unigrams,
                   const bigram_counter_n& bigrams,
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


void segment_buffer(
    const std::vector<std::string> &buffer,
    std::vector<std::string> &segmented,
    int length,
    const unigram_counter& unigrams,
    const bigram_counter_n& bigrams,
    int unigram_count,
    int max_unigram_length) {
  // segment tokens in the buffer

  #pragma omp parallel for
  for(int i = 0; i < length; i++) {
    std::string line = buffer[i];
    std::istringstream ss(line);
    std::ostringstream segmented_line;

    std::string wordsep = "";
    for(std::string word; std::getline(ss, word, ' ');) {
      std::vector<std::string> segm;

      segmented_line << wordsep;
      wordsep = " ";

      segment_token(segm, word, unigrams, bigrams, unigram_count,
                    max_unigram_length);

      for(auto it = segm.begin(); it != segm.end() - 1; ++it)
        segmented_line << *it << sub_sep;

      segmented_line << *(segm.end() - 1);
    }

    segmented[i] = segmented_line.str();
  }
}


int main(int argc, char* argv[]) {
  CLI::App app{"Bigram segment -- using subword bigram statistics for subword segmentation."};
  get_options(app);
  CLI11_PARSE(app, argc, argv);

  unigram_counter unigram_frequencies;
  bigram_counter_n bigram_frequencies;

  std::cerr << "loading bigrams and unigrams" << std::endl;

  // je potreba si pamatovat ze tohle neni vocab size ale data size
  int unigram_count = load_unigrams_from_vocab(
      unigram_frequencies, opt.unigram_stats);
  load_bigrams_from_vocab(bigram_frequencies, opt.bigram_stats);

  std::cerr << "done" << std::endl;

  int max_unigram_length = 0;
  for(auto unigram_f : unigram_frequencies) {
    int len = unigram_f.first.size();
    if(len > max_unigram_length)
      max_unigram_length = len;
  }

  std::cerr << "max unigram length: " << max_unigram_length << std::endl;

  int lineno = 0;
  int buffer_pos = 0;

  std::vector<std::string> buffer(BUFFER_SIZE);
  std::vector<std::string> segmented_buffer(BUFFER_SIZE);

  while(std::getline(std::cin, buffer[buffer_pos])) {
    lineno++;
    buffer_pos++;

    if(buffer_pos == BUFFER_SIZE) {
      segment_buffer(
          buffer, segmented_buffer, buffer_pos, unigram_frequencies,
          bigram_frequencies, unigram_count, max_unigram_length);

      for(int i = 0; i < buffer_pos; i++)
        std::cout << segmented_buffer[i] << std::endl;

      buffer_pos = 0;
    }

  }

  if(buffer_pos > 0) {
    segment_buffer(buffer, segmented_buffer, buffer_pos, unigram_frequencies,
                   bigram_frequencies, unigram_count, max_unigram_length);
    for(int i = 0; i < buffer_pos; i++)
      std::cout << segmented_buffer[i] << std::endl;
  }

  return 0;
}
