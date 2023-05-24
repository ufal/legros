#ifndef __COUNTERS_H_
#define __COUNTERS_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <iterator>


typedef std::unordered_map<std::string, int> unigram_counter;
typedef std::unordered_map<
  std::string, std::unordered_map<std::string, int>> bigram_counter;

int load_unigrams_from_vocab(unigram_counter& counts, const std::string& path);
void load_bigrams_from_vocab(bigram_counter& counts, const std::string& path);

int count_ngrams_from_file(unigram_counter& unigrams, bigram_counter& bigrams,
                           const std::string& path, int limit);


class brown_counter {

 private:
  int data_size_;

  std::string first_token;
  std::string last_token;

  std::vector<std::string> vocab;
  std::unordered_map<std::string, int> unigrams;
  bigram_counter bigrams;

 public:
  brown_counter(const std::string& path, int limit);

  inline std::vector<std::string>::const_iterator vocab_begin() const {
    return vocab.begin();
  }

  inline std::vector<std::string>::const_iterator vocab_end() const {
    return vocab.end();
  }

  inline int vocab_size() const { return vocab.size(); }
  inline int data_size() const { return data_size_; }

  int unigram_count_left(const std::string& token) const;
  int unigram_count_right(const std::string& token) const;
  int unigram_count(const std::string& token) const;

  int bigram_count(const std::string& left, const std::string& right) const;

  void merge_tokens(const std::string& dest, const std::string& src);

  inline const bigram_counter& get_bigrams() { return bigrams; }
  inline const std::vector<std::string>& get_vocab() { return vocab; }

};

#endif
