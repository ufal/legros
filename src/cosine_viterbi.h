#pragma once

#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "vocabs.h"

void subword_cosine_distances(
    std::map<int, float>& distances,
    const Embeddings& words,
    const Vocab& subwords,
    const Eigen::MatrixXf& subword_embeddings,
    const std::string& word);

void viterbi_decode(
    std::vector<std::string>& segmentation,
    const Embeddings& words,
    const Vocab& subwords,
    const Eigen::MatrixXf& subword_embeddings,
    const std::string& word);
