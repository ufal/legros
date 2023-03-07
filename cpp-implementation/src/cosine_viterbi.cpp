#include "cosine_viterbi.h"

#include <limits>
#include <algorithm>

CosineViterbi::CosineViterbi(const Embeddings& words, const Vocab& subwords) : words_(words), subwords_(subwords) {}


void CosineViterbi::subword_cosine_distances(
    std::map<int, float>& distances,
    const Eigen::MatrixXf& subword_embeddings,
    const std::string& word) const {

  Eigen::VectorXf word_embedding = words_.emb.row(words_[word]);
  float emb_norm = word_embedding.norm();

  for(size_t begin = 0; begin < word.size(); ++begin) {
    for(size_t end = begin + 1; end < word.size() + 1; ++end) {
      std::string subword = word.substr(begin, end - begin);

      if(!subwords_.contains(subword))
        continue;

      int subword_index = subwords_[subword];

      // retrieve subword embedding - corresponding row in subw.emb matrix
      Eigen::VectorXf subword_embedding = subword_embeddings.row(subword_index);
      float subw_norm = subword_embedding.norm();
      float dotprod = word_embedding.dot(subword_embedding);
      float cosminusone = dotprod / (emb_norm * subw_norm) - 1;

      distances.insert({subword_index, cosminusone});
    }
  }
}

void CosineViterbi::viterbi_decode(
    std::vector<std::string>& segmentation,
    const Eigen::MatrixXf& subword_embeddings,
    const std::string& word) const {

  std::map<int, float> distances;
  subword_cosine_distances(distances, subword_embeddings, word);

  // segment word using viterbi algorithm to find path with highest score
  std::vector<int> predecesors(word.size(), 0);
  std::vector<float> costs(word.size() + 1, -std::numeric_limits<float>::infinity());
  std::vector<std::string> sw_predecesors(word.size());
  costs[0] = 0.0f;

  // iterate from after the first letter (costs array begins before the word)
  for(size_t i = 1; i < word.size() + 1; ++i) {
    float max_score = -std::numeric_limits<float>::infinity();
    int best_pred = 0;
    std::string sw_best_pred;

    for(size_t j = 0; j < i; ++j) {
      // going from j to i.
      std::string subword_candidate = word.substr(j, i - j);
      if(!subwords_.contains(subword_candidate))
        continue;

      auto sub_index = subwords_[subword_candidate];
      float path_score = costs[j] + distances.at(sub_index);

      if(path_score > max_score) {
        max_score = path_score;
        best_pred = j;
        sw_best_pred = subword_candidate;
      }
    }

    costs[i] = max_score;
    predecesors[i - 1] = best_pred; // these are one-off because first has no pred but has score
    sw_predecesors[i - 1] = sw_best_pred;
  }

  int index = word.size() - 1;
  while(index >= 0) {
    segmentation.push_back(sw_predecesors[index]);
    index = predecesors[index] - 1;
  }

  std::reverse(segmentation.begin(), segmentation.end());
}
