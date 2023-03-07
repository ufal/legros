#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "vocabs.h"

typedef Eigen::MatrixXi CooccurrenceMatrix;
//typedef std::unordered_map<int, std::unordered_map<int, int>> CooccurrenceMatrix;
typedef std::unordered_map<std::string, std::vector<std::pair<std::string, float>>> AllowedSubstringMap;
typedef std::unordered_map<std::string, std::vector<std::pair<std::string, float>>> InverseAllowedSubstringMap;

#define BUFFER_SIZE 1000000

void load_weighted_allowed_substrings(
    std::unordered_map<std::string, std::vector<std::pair<std::string, float>>>& allowed_substrings,
    const std::string& file);

void load_allowed_substrings(
    std::unordered_map<std::string,std::vector<std::pair<std::string, float>>>& allowed_substrings,
    const std::string& file);

void load_allowed_substrings(
    std::unordered_map<std::string,std::vector<std::pair<std::string, float>>>& allowed_substrings,
    std::unordered_map<std::string,std::vector<std::pair<std::string, float>>>& inverse_allowed_substrings,
    const std::string& file);

void load_allowed_substrings(
    Eigen::MatrixXi& allowed_substrings,
    const Vocab& word_vocab,
    const Vocab& subword_vocab,
    const std::string& file);

void load_allowed_substrings_sparse(
    Eigen::SparseMatrix<int, Eigen::RowMajor>& allowed_substrings,
    const Vocab& word_vocab,
    const Vocab& subword_vocab,
    const std::string& file);

void get_all_substrings(
    std::vector<std::pair<std::string, float>> &substrings,
    const Vocab &subword_to_index,
    const std::string &word,
    int max_len);

template<typename U>
U& get_2d(
    std::vector<std::vector<U>>& stats,
    int stat_index, int word_index) {
  return stats[stat_index][word_index];
}

template<typename Derived>
typename Derived::Scalar& get_2d(
    Eigen::MatrixBase<Derived> &stats,
    int stat_index, int word_index) {
  return stats(stat_index, word_index);
}

inline int& get_2d(
    Eigen::SparseMatrix<int, Eigen::RowMajor>& stats,
    int stat_index, int word_index) {
  return stats.coeffRef(stat_index, word_index);
}

// template<typename Derived>
// typename Derived::Scalar& get_2d(
//     Eigen::SparseMatrix<Derived>& stats,
//     int stat_index, int word_index) {
//   return stats.coeffRef(stat_index, word_index);
// }

// inline int& get_2d(
//     Eigen::MatrixXi &stats,
//     int stat_index, int word_index) {
//   return stats(stat_index, word_index);
// }

template<typename T>
void try_add_to_stats(
    T& stats,
    const std::string& token,
    const std::vector<std::pair<std::string, float>>& substrings,
    const Vocab& words,
    const Vocab& subwords) {

  if(!words.contains(token))
    return;

  int word_index = words[token];

  for(std::pair<std::string, float> substring_pair : substrings) {
    if(!subwords.contains(substring_pair.first))
      continue;

    // add to stats
    int stat_index = subwords[substring_pair.first];

#pragma omp atomic
    get_2d(stats, stat_index, word_index) += substring_pair.second;
  }
}


template<typename T>
void try_add_word_to_stats(T& stats,
                           const Vocab& words,
                           const std::string& target_token,
                           const std::string& window_token) {

  if(!words.contains(target_token))
    return;

  if(!words.contains(window_token))
    return;

  int word_index = words[target_token];
  int stat_index = words[window_token];

#pragma omp atomic
  get_2d(stats, stat_index, word_index) += 1;
}


template<>
inline void try_add_word_to_stats<CooccurrenceMatrix>(CooccurrenceMatrix& stats,
                                                      const Vocab& words,
                                                      const std::string& target_token,
                                                      const std::string& window_token) {
  if(!words.contains(target_token))
    return;

  if(!words.contains(window_token))
    return;

  int target_index = words[target_token];
  int window_index = words[window_token];

//   if(stats.count(target_index) == 0)
// #pragma omp critical
//     stats.insert({target_index, std::unordered_map<int, int>()});


//   if(stats[target_index].count(window_index) == 0) {
// #pragma omp critical
//     stats[target_index].insert({window_index, 1});
//   }
//   else {

#pragma omp atomic
  get_2d(stats, target_index, window_index) += 1;

}



template<typename T>
void process_buffer(
    const std::vector<std::string> &buffer,
    int end,
    int max_subword,
    int window_size,
    T &stats,
    const Vocab &words,
    const Vocab &subwords,
    bool use_allowed_substrings,
    std::unordered_map<std::string,std::vector<std::pair<std::string, float>>> &allowed_substrings) {

#pragma omp parallel for
  for(int i = 0; i < end; ++i) {
    std::string line = buffer[i];
    std::istringstream iss(line);

    std::vector<std::string> tokens(
        (std::istream_iterator<std::string>(iss)),
        std::istream_iterator<std::string>());

    int t = 0;
    for(auto token: tokens) {

      std::vector<std::pair<std::string, float>> substrings;

      if(use_allowed_substrings) {
        if(allowed_substrings.count(token))  // use only this if you want to use all substrings instead of none
          substrings = allowed_substrings[token];
        else
          continue;
      }
      else
        get_all_substrings(substrings, subwords, token, max_subword);

      for(int j = std::max(0, t - window_size); j < t; ++j) {
        try_add_to_stats<T>(stats, tokens[j], substrings, words, subwords);
      }

      for(int k = t + 1; k < std::min(t + 1 + window_size, (int)tokens.size()); ++k) {
        try_add_to_stats<T>(stats, tokens[k], substrings, words, subwords);
      }
      ++t;
    }
  }
}

template<typename T>
void populate_substring_stats(
    T &stats,
    const Vocab &words,
    const Vocab &subwords,
    const std::string &training_data_file,
    const std::string &allowed_substrings_file,
    int window_size,
    int max_subword,
    bool use_weighted_substrings) {

  std::cerr << "Iterating over sentences from " << training_data_file << std::endl;
  std::ifstream input_fh(training_data_file);

  std::unordered_map<std::string,std::vector<std::pair<std::string, float>>> allowed_substrings;
  if (!allowed_substrings_file.empty()) {
    if(use_weighted_substrings) {
      std::cerr << "Loading list of weighted allowed substrings from " << allowed_substrings_file << std::endl;
      load_weighted_allowed_substrings(allowed_substrings, allowed_substrings_file);
    } else {
      std::cerr << "Loading list of allowed substrings from " << allowed_substrings_file << std::endl;
      load_allowed_substrings(allowed_substrings, allowed_substrings_file);
    }
  }

  int lineno = 0;
  int buffer_pos = 0;
  std::vector<std::string> buffer(BUFFER_SIZE);

  while(std::getline(input_fh, buffer[buffer_pos])) {
    ++lineno;
    ++buffer_pos;
    if(lineno % 1000 == 0)
      std::cerr << "Lineno: " << lineno << "\r";

    // full buffer -> process
    if(buffer_pos == BUFFER_SIZE) {
      process_buffer<T>(buffer, buffer_pos, max_subword, window_size,
                        stats, words, subwords, !allowed_substrings_file.empty(),
                        allowed_substrings);
      buffer_pos = 0;
    }
  }

  // process the rest of the buffer
  if(buffer_pos > 0) {
    process_buffer<T>(buffer, buffer_pos, max_subword, window_size, stats,
                      words, subwords, !allowed_substrings_file.empty(),
                      allowed_substrings);
  }

  std::cerr << "Read " << lineno << " lines in total." << std::endl;
}



template<typename T>
void process_word_buffer(T& stats,
                         const std::vector<std::string>& buffer,
                         const Vocab& words,
                         int length,
                         int window_size) {
#pragma omp parallel for
  for(int i = 0; i < length; ++i) {
    std::string line = buffer[i];
    std::istringstream iss(line);

    std::vector<std::string> tokens(
        (std::istream_iterator<std::string>(iss)),
        std::istream_iterator<std::string>());

    int t = 0;
    for(auto token: tokens) {

      for(int j = std::max(0, t - window_size); j < t; ++j) {
        try_add_word_to_stats<T>(stats, words, tokens[j], token);
      }

      for(int k = t + 1; k < std::min(t + 1 + window_size, (int)tokens.size()); ++k) {
        try_add_word_to_stats<T>(stats, words, tokens[k], token);
      }
      ++t;
    }
  }
}


template<typename T>
void populate_word_stats(T& stats,
                         const Vocab& words,
                         const std::string &training_data_file,
                         int window_size) {

  std::cerr << "Iterating over sentences from " << training_data_file
            << std::endl;
  std::ifstream input_fh(training_data_file);

  int lineno = 0;
  int buffer_pos = 0;
  std::vector<std::string> buffer(BUFFER_SIZE);

  while(std::getline(input_fh, buffer[buffer_pos])) {
    ++lineno;
    ++buffer_pos;

    // full buffer -> process
    if(buffer_pos == BUFFER_SIZE) {
      std::cerr << "Processing buffer; lineno: " << lineno << "\r";

      process_word_buffer<T>(stats, buffer, words, buffer_pos, window_size);
      buffer_pos = 0;
    }
  }

  std::cerr << "Processing last buffer; lineno: " << lineno << std::endl;

  // process the rest of the buffer
  if(buffer_pos > 0) {
    process_word_buffer<T>(stats, buffer, words, buffer_pos, window_size);
  }

  std::cerr << "Read " << lineno << " lines in total." << std::endl;
}
