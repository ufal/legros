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

  for(auto it1 = counter.vocab_begin(); it1 != counter.vocab_end(); it1++) {

    auto a = *it1;
    //for(const std::string& a : std::views::keys(counts)) {
    double cross_sum = 0;
    bool valid_a = mutual_information_terms.contains(a);

#pragma omp parallel for reduction(+:cross_sum)
    for(auto it2 = counter.vocab_begin(); it2 != counter.vocab_end(); it2++) {
      auto b = *it2;

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

double brown_classes::merge_loss_manual(
    const std::string& a, const std::string& b) const {

  double initial_loss = cross_sums.at(a) + cross_sums.at(b)
                       - get(mutual_information_terms, a, b, 0.0d)
                       - get(mutual_information_terms, b, a, 0.0d);

  double merged_loss = 0;

  double unigram_left =
      (counter.unigram_count_left(a) + counter.unigram_count_left(b)) / T();

  double unigram_right =
      (counter.unigram_count_right(a) + counter.unigram_count_right(b)) / T();

  double bigram = (counter.bigram_count(a, a)
                  + counter.bigram_count(a, b)
                  + counter.bigram_count(b, a)
                  + counter.bigram_count(b, b)) / T();

#pragma omp parallel for reduction(+:merged_loss)
  for(auto it = counter.vocab_begin(); it != counter.vocab_end(); it++) {
    auto lr = *it;
    if(lr == a || lr == b)
      continue;

    double merged_on_right =
        (counter.bigram_count(lr, a) + counter.bigram_count(lr, b)) / T();

    double merged_on_left =
        (counter.bigram_count(a, lr) + counter.bigram_count(b, lr)) / T();

    double other_unigram_right = counter.unigram_count_right(lr) / T();
    double other_unigram_left = counter.unigram_count_left(lr) / T();

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

double brown_classes::merge_loss_cached(
    const std::string& a, const std::string& b) const {
  return loss_table.at(a).at(b);
}

void brown_classes::initialize_loss_table() {
  loss_table.clear();

#pragma omp parallel for
  for(auto it1 = classes.cbegin(); it1 != classes.cend(); it1++) {
    const auto& cls1 = (*it1)[0];

    for(auto it2 = classes.cbegin(); it2 < it1; it2++) {
      const auto& cls2 = (*it2)[0];

      double loss = merge_loss_manual(cls1, cls2);

#pragma omp critical
      loss_table[cls1][cls2] = loss;
    }
  }
}

merge_triplet brown_classes::find_best_merge() const {
  double min_loss = std::numeric_limits<double>::infinity();
  std::string bm_left;
  std::string bm_right;

  for(auto it1 = classes.cbegin(); it1 != classes.cend(); it1++) {
    const auto& cls1 = (*it1)[0];

    for(auto it2 = classes.cbegin(); it2 < it1; it2++) {
      const auto& cls2 = (*it2)[0];

      double potential_loss = merge_loss_cached(cls1, cls2);
      if(potential_loss < min_loss) {
        min_loss = potential_loss;
        bm_left = cls1;
        bm_right = cls2;
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

  // update counter
  counter.merge_tokens(cls1, cls2);

  bigram_floats old_mis = mutual_information_terms;
  compute_mutual_information_terms();

  unigram_floats old_xsums = cross_sums;
  compute_cross_sums();

  update_loss_table(cls1, cls2, old_mis, old_xsums);
}

void brown_classes::update_loss_table(
    const std::string& a, const std::string& b,
    const bigram_floats& old_mis, const unigram_floats& old_xsums) {


  for(auto it1 = classes.cbegin(); it1 != classes.cend(); it1++) {
    const auto& cls1 = (*it1)[0];
    bool afirst = true;

    if(cls1 == a || cls1 == b)
      continue;

    for(auto it2 = classes.cbegin(); it2 < it1; it2++) {
      const auto& cls2 = (*it2)[0];
      if(cls2 == a)
        afirst = false; // this is here so we know whether cls1 comes before or
                        // after a - if we encounter 'a' here, it means cls1
                        // goes before a.

      if(cls2 == a || cls2 == b)
        continue;

      loss_table.at(cls1).at(cls2) +=
          - old_xsums.at(cls1) + cross_sums.at(cls1)
          - old_xsums.at(cls2) + cross_sums.at(cls2)
          + (get(old_mis, cls1, a, 0.0d) + get(old_mis, cls2, a, 0.0d))
          + (get(old_mis, a, cls1, 0.0d) + get(old_mis, a, cls2, 0.0d))
          + (get(old_mis, cls1, b, 0.0d) + get(old_mis, cls2, b, 0.0d))
          + (get(old_mis, b, cls1, 0.0d) + get(old_mis, b, cls2, 0.0d))
          - (get(mutual_information_terms, cls1, a, 0.0d) +
             get(mutual_information_terms, cls2, a, 0.0d))
          - (get(mutual_information_terms, a, cls1, 0.0d) +
             get(mutual_information_terms, a, cls2, 0.0d));
    }

    if(afirst) {
      loss_table.at(a).at(cls1) = merge_loss_manual(a, cls1);
    } else {
      loss_table.at(cls1).at(a) = merge_loss_manual(cls1, a);
    }
  }
}

brown_classes::brown_classes(
    const std::string& path, int min_freq, int limit) : counter(path, limit) {

  k = 0;
  std::cerr << "initializing classes and inverse classes" << std::endl;

  for(auto it = counter.vocab_begin(); it != counter.vocab_end(); it++) {
    auto unigram = *it;
    int freq = counter.unigram_count(unigram);
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

double brown_classes::mutual_information() {
  double sum = 0;
  for(const auto& [left, map]: counter.get_bigrams()) {
    for(const auto& [right, freq]: map) {
      sum += mutual_information_terms[left][right];
    }
  }
  return sum;
}

void brown_classes::compute_mutual_information_terms() {
  mutual_information_terms.clear();

  for(const auto& [left, map]: counter.get_bigrams()) {
    for(const auto& [right, freq]: map) {
      int lf = counter.unigram_count_left(left);
      int rf = counter.unigram_count_right(right);

      double mi = freq / T() * std::log2(freq * T() / (lf * rf));
      mutual_information_terms[left][right] = mi;
    }
  }
}
