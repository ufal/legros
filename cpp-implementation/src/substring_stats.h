#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "options.h"
#include <Eigen/Dense>

void load_allowed_substrings(
  std::unordered_map<std::string,std::unordered_set<std::string>> &allowed_substrings,
  const std::string &file);

void get_all_substrings(
    std::unordered_set<std::string> &substrings,
    const std::unordered_map<std::string,int> &subword_to_index,
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

// inline int& get_2d(
//     Eigen::MatrixXi &stats,
//     int stat_index, int word_index) {
//   return stats(stat_index, word_index);
// }

template<typename T>
void try_add_to_stats(
    T &stats,
    const std::string &token,
    const std::unordered_set<std::string> &substrings,
    const std::unordered_map<std::string,int> &word_to_index,
    const std::unordered_map<std::string,int> &subword_to_index) {

  if(word_to_index.count(token) == 0) {
    return;
  }

  int word_index = word_to_index.at(token);

  for(std::string substring : substrings) {
    if(subword_to_index.count(substring) == 0) {
      continue;
    }

    // add to stats
    int stat_index = subword_to_index.at(substring);

    #pragma omp atomic
      get_2d(stats, stat_index, word_index) += 1;
  }
}

template<typename T>
void process_buffer(
    const std::vector<std::string> &buffer,
    int end,
    int max_subword,
    int window_size,
    T &stats,
    const std::unordered_map<std::string,int> &word_to_index,
    const std::unordered_map<std::string,int> &subword_to_index,
    std::unordered_map<std::string,std::unordered_set<std::string>> &allowed_substrings) {

  #pragma omp parallel for
  for(int i = 0; i < end; ++i) {
    std::string line = buffer[i];
    std::istringstream iss(line);

    std::vector<std::string> tokens(
      (std::istream_iterator<std::string>(iss)),
      std::istream_iterator<std::string>());

    int t = 0;
    for(auto token: tokens) {

      std::unordered_set<std::string> substrings;

      if(!opt.allowed_substrings.empty()) {
        if(allowed_substrings.count(token)) {  // use only this if if you want to use all substrings instead of none
          substrings = allowed_substrings[token];
        }
        else
          continue;
      }
      else
        get_all_substrings(substrings, subword_to_index, token, max_subword);

      for(int j = std::max(0, t - window_size); j < t; ++j) {
        try_add_to_stats<T>(stats, tokens[j], substrings, word_to_index, subword_to_index);
      }

      for(int k = t + 1; k < std::min(t + 1 + window_size, (int)tokens.size()); ++k) {
        try_add_to_stats<T>(stats, tokens[k], substrings, word_to_index, subword_to_index);
      }
      ++t;
    }
  }
}

template<typename T>
void populate_substring_stats(
    T &stats,
    std::unordered_map<std::string, int> word_to_index,
    std::unordered_map<std::string, int> subword_to_index) {

  std::cerr << "Iterating over sentences from " << opt.training_data_file << std::endl;
  std::ifstream input_fh(opt.training_data_file);

  std::unordered_map<std::string,std::unordered_set<std::string>> allowed_substrings;
  if (!opt.allowed_substrings.empty()) {
    std::cerr << "Loading list of allowed substrings from " << opt.allowed_substrings << std::endl;
    load_allowed_substrings(allowed_substrings, opt.allowed_substrings);
  }

  int lineno = 0;
  int buffer_pos = 0;
  std::vector<std::string> buffer(opt.buffer_size);

  while(std::getline(input_fh, buffer[buffer_pos])) {
    ++lineno;
    ++buffer_pos;
    if(lineno % 1000 == 0)
      std::cerr << "Lineno: " << lineno << "\r";

    // full buffer -> process
    if(buffer_pos == opt.buffer_size) {
      process_buffer<T>(buffer, buffer_pos, opt.max_subword, opt.window_size, stats, word_to_index, subword_to_index, allowed_substrings);
      buffer_pos = 0;
    }
  }

  // process the rest of the buffer
  if(buffer_pos > 0) {
    process_buffer<T>(buffer, buffer_pos, opt.max_subword, opt.window_size, stats, word_to_index, subword_to_index, allowed_substrings);
  }

  std::cerr << "Read " << lineno << " lines in total." << std::endl;
}