#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>
#include "vocabs.h"

class UnigramModel {
private:
  Embeddings words_;
  Vocab subwords_;
  Eigen::MatrixXf inverse_emb_;

  Eigen::MatrixXf ws_;

  void forward_costs(std::vector<float>& costs, const std::string& word, const Eigen::ArrayXf& subword_logprobs) const;
  void backward_costs(std::vector<float>& costs, const std::string& word, const Eigen::ArrayXf& subword_logprobs) const;
  float compute_expected_counts(Eigen::MatrixXf& exp_counts, const std::string& word, int word_index) const;
  Eigen::ArrayXf subword_scores(const std::string& word) const;

  Eigen::ArrayXf scores_as_logprobs(const std::string& word) const;
  // Eigen::ArrayXf scores_as_cosine_distance_minus_one(const std::string& word) const;

public:
  void save(const std::string& filename) const;

  void viterbi_decode(std::vector<std::string>& reversed_segmentation,
                      const std::string& word) const;

  void load(const std::string& filename);
  void estimate_parameters(int epochs, float base_logprob);

  UnigramModel(const Embeddings& words, const Vocab& subwords, const Eigen::MatrixXf& inverse_emb);
};
