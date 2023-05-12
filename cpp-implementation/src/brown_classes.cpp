#include "brown_classes.h"

#include <ranges>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <numeric>
#include <iostream>

// void brown_classes::compute_cross_sums() {
//   cross_sums.clear();

//   for(const auto& [a, map]: mutual_information_terms) {
//     if(!cross_sums.contains(a))
//       cross_sums.insert({a, 0});

//     for(const auto& [b, mi]: map) {
//       if(!cross_sums.contains(b))
//         cross_sums.insert({b, 0});

//       cross_sums.at(a) += mi;
//       cross_sums.at(b) += mi;
//     }

//     if(map.contains(a)) {
//       cross_sums.at(a) -= map.at(a);
//     }
//   }
// }

void brown_classes::compute_cross_sums() {
  cross_sums.clear();
  for(const std::string& a : std::views::keys(counts)) {
    float cross_sum = 0;
    bool valid_a = mutual_information_terms.contains(a);

    for(const std::string& b: std::views::keys(counts)) {
      if(valid_a && mutual_information_terms.at(a).contains(b))
        cross_sum += mutual_information_terms.at(a).at(b);

      if(mutual_information_terms.contains(b)
         && mutual_information_terms.at(b).contains(a))
        cross_sum += mutual_information_terms.at(b).at(a);
    }

    if(valid_a && mutual_information_terms.at(a).contains(a))
      cross_sum -= mutual_information_terms.at(a).at(a);

    cross_sums.insert({a, cross_sum});
  }
}

float brown_classes::merge_loss_manual(
    const std::string& a, const std::string& b) const {

  float initial_loss = cross_sums.at(a) + cross_sums.at(b)
                       - get(mutual_information_terms, a, b, 0.0f)
                       - get(mutual_information_terms, b, a, 0.0f);

  float merged_loss = 0;

  float unigram_left =
      (unigram_counts_left.at(a) + unigram_counts_left.at(b)) / T;

  float unigram_right =
      (unigram_counts_right.at(a) + unigram_counts_right.at(b)) / T;

  float bigram = (get(bigram_counts, a, a, 0)
                  + get(bigram_counts, a, b, 0)
                  + get(bigram_counts, b, a, 0)
                  + get(bigram_counts, b, b, 0)) / T;

  for(const auto& lr : std::views::keys(counts)) {
    if(lr == a || lr == b)
      continue;

    float merged_on_right = (get(bigram_counts, lr, a, 0)
                             + get(bigram_counts, lr, b, 0)) / T;

    float merged_on_left = (get(bigram_counts, a, lr, 0)
                            + get(bigram_counts, b, lr, 0)) / T;

    float other_unigram_right = unigram_counts_right.at(lr) / T;
    float other_unigram_left = unigram_counts_left.at(lr) / T;

    if(merged_on_right > 0)
      merged_loss +=
          merged_on_right
          * std::log2(merged_on_right / (other_unigram_left * unigram_right));

    if(merged_on_left > 0)
      merged_loss +=
          merged_on_left
          * std::log2(merged_on_left / (unigram_left * other_unigram_right));
  }

  if(bigram > 0)
    merged_loss += bigram * std::log2(bigram / (unigram_left * unigram_right));

  return initial_loss - merged_loss;
}

float brown_classes::merge_loss_cached(
    const std::string& a, const std::string& b) const {
  return loss_table.at(a).at(b);
}

void brown_classes::initialize_loss_table() {
  loss_table.clear();
  for(const auto& clslist1: classes) {
    const auto& cls1 = clslist1[0];

    if(!loss_table.contains(cls1))
      loss_table.insert({cls1, std::unordered_map<std::string, float>()});

    for(const auto& clslist2: classes) {
      const auto& cls2 = clslist2[0];
      if(cls1.compare(cls2) < 0)
        loss_table.at(cls1).insert({cls2, merge_loss_manual(cls1, cls2)});
    }
  }
}

merge_triplet brown_classes::find_best_merge() const {
  float min_loss = std::numeric_limits<float>::infinity();
  std::string bm_left;
  std::string bm_right;

  for(const auto& clslist1: classes) {
    const auto& cls1 = clslist1[0];

    for(const auto& clslist2: classes) {
      const auto& cls2 = clslist2[0];

      if(cls1.compare(cls2) < 0) {
        float potential_loss = merge_loss_cached(cls1, cls2);
        if(potential_loss < min_loss) {
          min_loss = potential_loss;
          bm_left = cls1;
          bm_right = cls2;
        }
      }
    }
  }

  return {bm_left, bm_right, min_loss};
}

void brown_classes::merge_classes(
    const std::string& cls1, const std::string& cls2) {
  int merged_index = inv_classes.at(cls1);
  int old_index = inv_classes.at(cls2);

  for(const auto& w : classes[old_index])
    classes[merged_index].push_back(w);

  classes.erase(classes.begin() + old_index);
  inv_classes.clear();

  int index = 0;

  for(const auto& clslist: classes)
    inv_classes.insert({clslist[0], index++});

  k--;

  for(const auto& left: std::views::keys(unigram_counts_left)) {
    if(left == cls2)
      continue;

    if(!bigram_counts.contains(left))
      continue;

    if(bigram_counts.at(left).contains(cls2))
      // here the [] is used because we actually want to create it if it does
      // not exist yet.
      bigram_counts.at(left)[cls1] += bigram_counts.at(left).at(cls2);
    bigram_counts.at(left).erase(cls2);
  }

  for(const auto& right: std::views::keys(unigram_counts_right)) {
    if(right == cls2)
      continue;

    if(!bigram_counts.contains(cls2))
      continue;

    if(bigram_counts.at(cls2).contains(right))
      // again, note that the [] is used
      bigram_counts.at(cls1)[right] += bigram_counts.at(cls2).at(right);
    //bigram_counts.at(cls2).erase(right);
  }

  float join_increment = get(bigram_counts, cls1, cls2, 0)
                         + get(bigram_counts, cls2, cls1, 0)
                         + get(bigram_counts, cls2, cls2, 0);

  if(join_increment > 0) {
    if(!bigram_counts.contains(cls1))
      bigram_counts.insert({cls1, std::unordered_map<std::string, int>()});
    bigram_counts.at(cls1)[cls1] += join_increment;
  }

  bigram_counts.at(cls1).erase(cls2);
  bigram_counts.erase(cls2);

  unigram_counts_left.at(cls1) += unigram_counts_left.at(cls2);
  unigram_counts_left.erase(cls2);

  unigram_counts_right.at(cls1) += unigram_counts_right.at(cls2);
  unigram_counts_right.erase(cls2);

  counts.at(cls1) += counts.at(cls2);
  counts.erase(cls2);

  bigram_floats old_mis = mutual_information_terms;
  compute_mutual_information_terms();

  unigram_floats old_xsums = cross_sums;
  compute_cross_sums();

  update_loss_table(cls1, cls2, old_mis, old_xsums);
}

void brown_classes::update_loss_table(
    const std::string& a, const std::string& b,
    const bigram_floats& old_mis, const unigram_floats& old_xsums) {


  for(const auto& clslist1: classes) {
    const auto& cls1 = clslist1[0];
    if(cls1 == a || cls1 == b)
      continue;

    for(const auto& clslist2: classes) {
      const auto& cls2 = clslist2[0];

      if(cls2 == a || cls2 == b || cls1.compare(cls2) >= 0)
        continue;

      loss_table.at(cls1).at(cls2) +=
          - old_xsums.at(cls1) + cross_sums.at(cls1)
          - old_xsums.at(cls2) + cross_sums.at(cls2)
          + (get(old_mis, cls1, a, 0.0f) + get(old_mis, cls2, a, 0.0f))
          + (get(old_mis, a, cls1, 0.0f) + get(old_mis, a, cls2, 0.0f))
          + (get(old_mis, cls1, b, 0.0f) + get(old_mis, cls2, b, 0.0f))
          + (get(old_mis, b, cls1, 0.0f) + get(old_mis, b, cls2, 0.0f))
          - (get(mutual_information_terms, cls1, a, 0.0f) +
             get(mutual_information_terms, cls2, a, 0.0f))
          - (get(mutual_information_terms, a, cls1, 0.0f) +
             get(mutual_information_terms, a, cls2, 0.0f));
    }

    if(a.compare(cls1) < 0) {
      loss_table.at(a).at(cls1) = merge_loss_manual(a, cls1);
    } else {
      loss_table.at(cls1).at(a) = merge_loss_manual(cls1, a);
    }
  }
}

brown_classes::brown_classes(
    const std::string& path, int min_freq, int limit) {

  std::cerr << "loading ngrams from " << path << std::endl;
  data_size = count_ngrams_from_file(counts, bigram_counts, path, limit);
  T = data_size - 1;

  std::cerr << "initializing unigram left and right counts" << std::endl;

  for(const auto& [left, map]: bigram_counts) {
    for(const auto& [right, freq]: map) {

      if(!unigram_counts_left.contains(left))
        unigram_counts_left.insert({left, 0});
      if(!unigram_counts_right.contains(right))
        unigram_counts_right.insert({right, 0});

      unigram_counts_left.at(left) += freq;
      unigram_counts_right.at(right) += freq;
    }
  }

  k = 0;

  std::cerr << "initializing classes and inverse classes" << std::endl;
  for(const auto& [unigram, freq]: counts) {
    if(freq < min_freq)
      continue;

    classes.push_back({unigram});
    inv_classes.insert({unigram, k});
    k++;
  }

  std::cerr << "initializing MI terms" << std::endl;
  compute_mutual_information_terms();

  std::cerr << "initializing cross sums" << std::endl;
  compute_cross_sums();

  std::cerr << "initializing loss table" << std::endl;
  initialize_loss_table();
}

float brown_classes::mutual_information() {
  float sum = 0;
  for(const auto& map: mutual_information_terms | std::views::values) {
    auto v = map | std::views::values | std::views::common;
    sum += std::accumulate(v.begin(), v.end(), 0.0f);
  }
  return sum;
}

void brown_classes::compute_mutual_information_terms() {
  mutual_information_terms.clear();

  for(const auto& [left, map]: bigram_counts) {
    if(!mutual_information_terms.contains(left))
      mutual_information_terms.insert(
          {left, std::unordered_map<std::string, float>()});

    for(const auto& [right, freq]: map) {

      int lf = unigram_counts_left.at(left);
      int rf = unigram_counts_right.at(right);

      float mi = freq / T * std::log2(freq * T / (lf * rf));
      mutual_information_terms.at(left).insert({right, mi});
    }
  }
}
