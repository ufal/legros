#include "counters.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <ranges>
#include <cassert>

int load_unigrams_from_vocab(
    unigram_counter& counts, const std::string& path) {

  int total_count = 0;
  std::ifstream ifs(path);
  for(std::string line; std::getline(ifs, line);) {
    std::istringstream iss(line);
    std::string subword;
    iss >> subword;
    int frequency;
    iss >> frequency;
    total_count += frequency;

    counts.insert({subword, frequency});
  }

  return total_count;
}

void load_bigrams_from_vocab(
    bigram_counter& counts, const std::string& path) {

  std::ifstream ifs(path);
  for(std::string line; std::getline(ifs, line);) {
    std::istringstream iss(line);
    std::string subword1, subword2;
    iss >> subword1;
    iss >> subword2;
    int frequency;
    iss >> frequency;

    counts[subword1][subword2] = frequency;
  }
}

brown_counter::brown_counter(const std::string& path, int limit) {

  std::cerr << "loading ngrams from " << path << std::endl;
  std::ifstream ifs(path);
  std::string prev;

  data_size_ = 0;

  for(std::string line; std::getline(ifs, line);) {
    std::istringstream iss(line);

    for(std::string word; std::getline(iss, word, ' ');) {
      unigrams[word]++;

      if(data_size_ == 0)
        first_token = word;
      else
        bigrams[prev][word]++;

      prev = word;
      data_size_++;
    }

    if(limit > 0 && data_size_ >= limit)
      break;
  }

  last_token = prev;

  vocab.reserve(unigrams.size());
  for(const auto& unigram: std::views::keys(unigrams)) {
    vocab.push_back(unigram);
  }
}

int brown_counter::unigram_count(const std::string& token) const {
  if(!unigrams.contains(token))
    return 0;

  return unigrams.at(token);
}

int brown_counter::unigram_count_left(const std::string& token) const {
  if(!unigrams.contains(token))
    return 0;

  if(token == last_token)
    return unigrams.at(token) - 1;

  return unigrams.at(token);
}

int brown_counter::unigram_count_right(const std::string& token) const {
  if(!unigrams.contains(token))
    return 0;

  if(token == first_token)
    return unigrams.at(token) - 1;

  return unigrams.at(token);
}

int brown_counter::bigram_count(const std::string& left,
                                const std::string& right) const {
  if(!bigrams.contains(left) || !bigrams.at(left).contains(right))
    return 0;

  return bigrams.at(left).at(right);
}

void brown_counter::merge_tokens(const std::string& dest,
                                 const std::string& src) {
  // this assumes that dest and src are both present in unigrams.

  // update bigrams
  // TODO think about having two maps (left-right and right-left) to make this
  // faster

  std::vector<std::string>::iterator src_pt;

  for(auto it = vocab.begin(); it != vocab.end(); it++ ) {
    auto ctx = *it;
    if(ctx == src) {
      src_pt = it;
      continue; // do this later
    }

    // bigram (ctx, src), add to (ctx, dest), erase src from the counts
    // this may also include (dest, src) adding that to (dest, dest)
    // also go ahead and erase the src token as right context
    int left_count = bigram_count(ctx, src);
    if(left_count > 0) {
      bigrams[ctx][dest] += left_count;
      bigrams[ctx].erase(src);
    }

    // bigram (src, ctx), add to (dest, ctx)
    // similarly here, this might include (src, dest) -> (dest, dest)
    int right_count = bigram_count(src, ctx);
    if(right_count > 0)
      bigrams[dest][ctx] += right_count;
  }

  // now add cooccurrences of (src, src)
  int join_count = bigram_count(src, src); //+ bigram_count(dest, src) + bigram_count(src, dest);
  if(join_count > 0)
    bigrams[dest][dest] += join_count;

  // erase src from all left contexts (already added to dest)
  bigrams.erase(src);

  // update unigrams. everything from src just goes to dest.  we assume that
  // src is a seen unigram so this should fail if the count is zero
  unigrams.at(dest) += unigrams.at(src);
  unigrams.erase(src);

  // update vocab. we need to be sure that src was indeed found in the vocab
  assert(src_pt != vocab.end());
  vocab.erase(src_pt);

  // update first/last word
  // if src was first/last word, we set dest as the first/last word instead
  if(src == first_token)
    first_token = dest;
  if(src == last_token)
    last_token = src;
}
