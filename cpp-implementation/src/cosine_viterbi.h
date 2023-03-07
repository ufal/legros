#pragma once

#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "vocabs.h"

class CosineViterbi {

private:
  Embeddings words_;
  Vocab subwords_;
  Eigen::MatrixXf inverse_emb_;

  void subword_cosine_distances(
      std::map<int, float>& distances,
      const Eigen::MatrixXf& subword_embeddings,
      const std::string& word) const;

public:
  void save(const std::string& filename) const;

  void viterbi_decode(std::vector<std::string>& segmentation,
                      const Eigen::MatrixXf& subword_embeddings,
                      const std::string& word) const;

  void load(const std::string& filename);

  CosineViterbi(const Embeddings& words, const Vocab& subwords);
};
