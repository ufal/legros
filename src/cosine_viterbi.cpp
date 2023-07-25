#include "cosine_viterbi.h"

#include <limits>
#include <algorithm>
#include <cassert>


void subword_cosine_similarities(
    std::map<int, float>& similarities,
    const std::string& word,
    const Eigen::VectorXf& word_embedding,
    const Vocab& subwords,
    const Eigen::MatrixXf& subword_embeddings) {

  float emb_norm = word_embedding.norm();

  // iterate over all non-empty substrings of the word
  for(size_t begin = 0; begin < word.size(); ++begin) {
    for(size_t end = begin + 1; end < word.size() + 1; ++end) {
      std::string subword = word.substr(begin, end - begin);

      // if the substring is not in the vocabulary, skip it
      if(!subwords.contains(subword))
        continue;

      int subw_index = subwords[subword];

      Eigen::VectorXf subw_embedding = subword_embeddings.row(subw_index);
      float subw_norm = subw_embedding.norm();
      float dotprod = word_embedding.dot(subw_embedding);
      float sim = dotprod / (emb_norm * subw_norm);

      similarities.insert({subw_index, sim});
    }
  }
}


void viterbi_decode(
    std::vector<std::string>& segmentation,
    const std::string& word,
    const Eigen::VectorXf& word_embedding,
    const Vocab& subwords,
    const Eigen::MatrixXf& subword_embeddings) {

  // pre-compute cosine similarities between the word and subwords in the
  // vocabulary which are contained in the word
  std::map<int, float> similarities;
  subword_cosine_similarities(similarities, word, word_embedding, subwords,
                              subword_embeddings);

  // If the path goes through index i, then predecesors[i] is the index of the
  // previous subword on the path. The vector `sw_predecesors` contains the
  // actual corresponding subwords, not the indices.
  std::vector<int> predecesors(word.size(), 0);
  std::vector<std::string> sw_predecesors(word.size());

  // scores[i] is the score of the best path-prefix which ends at index i. The
  // vector starts *before* the first letter, so scores[0] is the score of the
  // empty prefix, initialized to zero. Note that the scores are always
  // negative.
  std::vector<float> scores(word.size() + 1,
                            -std::numeric_limits<float>::infinity());
  scores[0] = 0.0f;

  // iterate from after the first letter (scores array begins before the word)
  for(size_t i = 1; i < word.size() + 1; ++i) {

    // one iteration choses the best predecessor for the i-th position
    float max_score = -std::numeric_limits<float>::infinity();
    int best_pred = 0;
    std::string sw_best_pred;

    // Going from j to i (every possible preceding subword, aka. `candidate`):
    for(size_t j = 0; j < i; ++j) {

      std::string candidate = word.substr(j, i - j);
      float candidate_similarity;

      // if the candidate is not in the vocabulary, skip it, unless it is
      // length 1 - in that case assign it with the lowest similarity of -1.
      if(!subwords.contains(candidate)) {
        if(j == i - 1) {
          candidate_similarity = -1;
        }
        else {
          continue;
        }
      }
      else {
        int subw_index = subwords[candidate];
        candidate_similarity = similarities.at(subw_index);
      }

      // we subtract one from the similarity because we want it to be less than
      // zero, so the word does not get segmented into individual letters.
      float path_score = scores[j] + candidate_similarity - 1;

      if(path_score > max_score) {
        max_score = path_score;
        best_pred = j;
        sw_best_pred = candidate;
      }
    }

    assert(sw_best_pred.size() > 0);
    predecesors[i - 1] = best_pred; // these are off by one because first has
                                    // no pred but has score
    sw_predecesors[i - 1] = sw_best_pred;
    scores[i] = max_score;
  }

  int index = word.size() - 1;
  while(index >= 0) {
    segmentation.push_back(sw_predecesors[index]);
    index = predecesors[index] - 1;
  }

  std::reverse(segmentation.begin(), segmentation.end());
}
