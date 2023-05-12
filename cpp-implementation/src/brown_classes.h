#ifndef __BROWN_CLASSES_H_
#define __BROWN_CLASSES_H_

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "counters.h"

typedef std::vector<std::vector<std::string>> class_list;
typedef std::unordered_map<
  std::string, std::unordered_map<std::string, float>> bigram_floats;

//typedef std::vector<std::pair<std::string, float>> unigram_floats_dense;
typedef std::unordered_map<std::string, float> unigram_floats;
typedef std::tuple<std::string, std::string, float> merge_triplet;

template<typename K, typename V>
inline const V& get(
    const std::unordered_map<K, std::unordered_map<K, V>>& data,
    const K& key1, const K& key2, const V& def_val) {

  if(data.contains(key1) && data.at(key1).contains(key2))
    return data.at(key1).at(key2);
  return def_val;
}


class brown_classes {

private:

  int data_size;
  float T;
  int k;
  unigram_counter counts;
  bigram_counter bigram_counts;

  unigram_counter unigram_counts_left;
  unigram_counter unigram_counts_right;

  class_list classes;
  std::unordered_map<std::string, int> inv_classes;

  bigram_floats mutual_information_terms;
  bigram_floats loss_table;
  unigram_floats cross_sums;

  void compute_mutual_information_terms();
  void compute_cross_sums();
  float merge_loss_manual(const std::string& a, const std::string& b) const;
  void initialize_loss_table();
  void update_loss_table(const std::string& a, const std::string& b,
                         const bigram_floats& old_mis,
                         const unigram_floats& old_xsums);

public:
  brown_classes(const std::string& path, int min_freq, int limit);
  float mutual_information();
  inline int size() const {return k;}
  inline const std::vector<std::string>& get_class(int i) const {return classes[i];}

  merge_triplet find_best_merge() const;
  void merge_classes(const std::string& a, const std::string& b);
  float merge_loss_cached(const std::string& a, const std::string& b) const;

};


#endif
