#ifndef SSEG_COSINE_VITERBI_H_
#define SSEG_COSINE_VITERBI_H_

#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "vocabs.h"

// Pre-computes the cosine similarities between a word and all subwords
// contained in it. Similarity(x, y) = dot(x, y) / (norm(x) * norm(y)).
void subword_cosine_similarities(
    std::map<int, float>& similarities,
    const std::string& word,
    const Eigen::VectorXf& word_embedding,
    const Vocab& subwords,
    const Eigen::MatrixXf& subword_embeddings);


// Segments a single word using the viterbi algorithm to find path with highest
// score, according to cosine similarities of the word embedding with the
// subword embeddings. Fills `segmentation` with the resulting segments.
void viterbi_decode(
    std::vector<std::string>& segmentation,
    const std::string& word,
    const Eigen::VectorXf& word_embedding,
    const Vocab& subwords,
    const Eigen::MatrixXf& subword_embeddings);

#endif  // SSEG_COSINE_VITERBI_H_
