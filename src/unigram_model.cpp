#include "unigram_model.h"

#include <limits>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "math_utils.h"

// CAUTION
// CAUTION !
// THIS IS BYTE-BASED, not character(grapheme)-based implementation!!!

UnigramModel::UnigramModel(const Embeddings& words, const Vocab& subwords, const Eigen::MatrixXf& inverse_emb)
    : words_(words), subwords_(subwords), inverse_emb_(inverse_emb), ws_(words.embedding_dim, subwords.size()) {
  // initialize W_s with uniform, then call forward_backward for each word, then upadte W_s with E^{-1} log P^\hat_w (the expected counts)
  // Ws uniform means all zeros
}

// UnigramModel::UnigramModel(const Embeddings& words, const Vocab& subwords, const Eigen::MatrixXf& inverse_emb, const Eigen::MatrixXf& params)
//         : words_(words), subwords_(subwords), inverse_emb_(inverse_emb), ws_(params) { }

void UnigramModel::forward_costs(std::vector<float>& costs, const std::string& word, const Eigen::ArrayXf& subword_logprobs) const {
  for(int end = 1; end < word.size() + 1; ++end) {
    std::vector<float> prefix_scores;

    for(int begin = 0; begin < end; ++begin) {
      std::string subword_candidate = word.substr(begin, end - begin);

      if(!subwords_.contains(subword_candidate))
        continue;

      float cost = costs[begin] + subword_logprobs(subwords_[subword_candidate]);
      prefix_scores.push_back(cost);
    }

    if(prefix_scores.size() > 0) {
      float lse = log_sum_exp(prefix_scores);
      costs[end] = lse;
    } else {
      costs[end] = -std::numeric_limits<float>::infinity();
    }
  }
}

void UnigramModel::backward_costs(std::vector<float>& costs, const std::string& word, const Eigen::ArrayXf& subword_logprobs) const {
  for(int begin = word.size() - 1; begin >= 0; --begin) {
    std::vector<float> suffix_scores;

    for(int end = begin + 1; end < word.size() + 1; ++end) {
      std::string subword_candidate = word.substr(begin, end - begin);

      if(!subwords_.contains(subword_candidate))
        continue;

      float cost = costs[end] + subword_logprobs(subwords_[subword_candidate]);
      suffix_scores.push_back(cost);
    }

    if(suffix_scores.size() > 0) {
      costs[begin] = log_sum_exp(suffix_scores);
    } else {
      costs[begin] = -std::numeric_limits<float>::infinity();
    }
  }
}


float UnigramModel::compute_expected_counts(
    Eigen::MatrixXf& exp_counts, const std::string& word, int word_index) const {

  std::vector<float> fw_costs(word.size() + 1);
  std::vector<float> bw_costs(word.size() + 1);

  Eigen::VectorXf word_embedding = words_.emb.row(word_index);
  Eigen::VectorXf logits = word_embedding.transpose() * ws_;

  std::vector<float> relevant_logits;
  for(int begin = 0; begin < word.size(); ++begin) {
    for(int end = begin + 1; end < word.size() + 1; ++end) {
      std::string subword = word.substr(begin, end - begin);

      if(!subwords_.contains(subword))
        continue;

      int subword_index = subwords_[subword];
      relevant_logits.push_back(logits(subword_index));
    }
  }

  Eigen::ArrayXf subword_logprobs = logits.array() - log_sum_exp(relevant_logits);

  forward_costs(fw_costs, word, subword_logprobs);
  backward_costs(bw_costs, word, subword_logprobs);

  std::vector<int> subword_indices;
  std::vector<float> subword_scores;


  for(int begin = 0; begin < word.size(); ++begin) {
    for(int end = begin + 1; end < word.size() + 1; ++end) {
      std::string subword = word.substr(begin, end - begin);

      if(!subwords_.contains(subword))
        continue;

      int subword_index = subwords_[subword];
      //subword_indices.push_back(subword_index);

      float score = fw_costs[begin] + subword_logprobs(subword_index)
                    + bw_costs[end];

      std::vector<int>::iterator it = std::find(subword_indices.begin(), subword_indices.end(), subword_index);

      if(it == subword_indices.end()) {
        subword_indices.push_back(subword_index);
        subword_scores.push_back(score);
      }
      else {
        // same subword occurrs more than once in a word:

        float old_score = subword_scores[it - subword_indices.begin()];
        subword_scores[it - subword_indices.begin()] = log_sum_exp(std::vector<float>{old_score, score});

      }

      // exp_counts(word_index, subword_index) = log_sum_exp(
      //     std::vector<float>{exp_counts(word_index, subword_index), score});
    }

  }

  float nll;
  for(int i = 0; i < subword_indices.size(); ++i) {
    int index = subword_indices[i];
    float prob = std::exp(subword_scores[i]);
    float logit = subword_logprobs(index);
    nll -= prob * logit;
  }

  float lse = log_sum_exp(subword_scores);

  exp_counts.row(word_index) = logits;

  for(int i = 0; i < subword_indices.size(); ++i) {
    exp_counts(word_index, subword_indices[i]) = subword_scores[i] - lse;
  }

  return nll;
}

void UnigramModel::estimate_parameters(int epochs, float base_logprob) {

  // the expected counts as a matrix:
  // each row corresponds to a word, columns store the expected counts (log probs of subwords)

  // E^{-1} has shape (embedding_dim, word_count)
  // logPhat has shape (word_count, subword_count) --> multiplying these two gets us a new matrix of (embedding_dim, subword_count) which is exactly what is needed

  for(int i = 0; i < epochs; ++i) {
    //Eigen::MatrixXf exp_counts = Eigen::MatrixXf::Constant(words_.size(), subwords_.size(), base_logprob);
    Eigen::MatrixXf exp_counts = Eigen::MatrixXf(words_.size(), subwords_.size());

    float cummulative_sum = 0.0f;
    float cummulative_nll = 0.0f;

#pragma omp parallel for
    for(int word_index = 0; word_index < words_.size(); ++word_index) {
      const std::string& word = words_[word_index];

      float nll = compute_expected_counts(exp_counts, word, word_index);
      float row_sum = exp_counts.row(word_index).array().sum();

#pragma omp atomic update
      cummulative_sum += row_sum;

      // if(!std::isnormal(row_sum)) {
      //     std::cerr << "STOP, row sum is not normal, word index " << word_index << ", word " << word << " row sum: " << row_sum <<std::endl ;
      //     std::abort();
      // }

#pragma omp atomic update
      cummulative_nll += nll;
    }

    std::cerr << "Cummulative sum:  " << cummulative_sum << std::endl;
    std::cerr << "Cummulative nll:  " << cummulative_nll << std::endl;

    // update W_s:
    ws_ = inverse_emb_ * exp_counts;
    std::cerr << "W_s squared norm (L2): " << ws_.squaredNorm() << std::endl;
  }
}


Eigen::ArrayXf UnigramModel::scores_as_logprobs(const std::string& word) const {
  Eigen::VectorXf word_embedding = words_.emb.row(words_[word]);
  Eigen::VectorXf logits = word_embedding.transpose() * ws_;

  std::vector<float> relevant_logits;
  for(size_t begin = 0; begin < word.size(); ++begin) {
    for(size_t end = begin + 1; end < word.size() + 1; ++end) {
      std::string subword = word.substr(begin, end - begin);

      if(!subwords_.contains(subword))
        continue;

      int subword_index = subwords_[subword];
      relevant_logits.push_back(logits(subword_index));
    }
  }

  return logits.array() - log_sum_exp(relevant_logits);
}

// Eigen::ArrayXf UnigramModel::scores_as_cosine_distance_minus_one(const std::string& word) const {

//   Eigen::VectorXf word_embedding = words_.emb.row(words_[word]);
//   float emb_norm = word_embedding.norm();

//   std::vector<float> cos_dists;

//   for(size_t begin = 0; begin < word.size(); ++begin) {
//     for(size_t end = begin + 1; end < word.size() + 1; ++end) {
//       std::string subword = word.substr(begin, end - begin);

//       if(!subwords_.contains(subword))
//         continue;

//       int subword_index = subwords_[subword];





//       relevant_logits.push_back(logits(subword_index));
//     }
//   }

//   return logits.array() - log_sum_exp(relevant_logits);


//   return ;
// }


Eigen::ArrayXf UnigramModel::subword_scores(const std::string& word) const {
  return scores_as_logprobs(word);
}


void UnigramModel::viterbi_decode(std::vector<std::string>& reversed_segmentation,
                                  const std::string& word) const {

  // refactor this and work with cosine distances instead of hte WS matrix
  // instead of computing subword logprobs here, refactor to method get_score or something
  // use the bug of cosine distace - 1.

  Eigen::ArrayXf subword_logprobs = subword_scores(word);

  // segment word using viterbi algorithm to find path with highest prob
  std::vector<int> predecesors(word.size(), 0);
  std::vector<float> costs(word.size() + 1, -std::numeric_limits<float>::infinity());
  std::vector<std::string> sw_predecesors(word.size());
  costs[0] = 0.0f;

  // iterate from after the first letter (costs array begins before the word)
  for(int i = 1; i < word.size() + 1; ++i) {
    float max_score = -std::numeric_limits<float>::infinity();
    int best_pred = 0;
    std::string sw_best_pred;

    for(int j = 0; j < i; ++j) {
      // going from j to i.
      std::string subword_candidate = word.substr(j, i - j);
      if(!subwords_.contains(subword_candidate))
        continue;

      auto sub_index = subwords_[subword_candidate];
      float path_score = costs[j] + subword_logprobs(sub_index);

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
    reversed_segmentation.push_back(sw_predecesors[index]);
    index = predecesors[index] - 1;
  }
}

void UnigramModel::save(const std::string& filename) const {
  std::ofstream ofs(filename);
  ofs << ws_ << std::endl;
  ofs.close();
}

void UnigramModel::load(const std::string& filename) {
  std::ifstream ifs(filename);

  for(int i = 0; i < ws_.rows(); ++i) {
    std::string line;
    std::getline(ifs, line);
    std::istringstream iss(line);

    for(int j = 0; j < ws_.cols(); ++j) {
      iss >> ws_(i,j);
    }
  }
}
