#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "CLI11.hpp"
#include "vocabs.h"
#include <Eigen/Dense>
#include <limits>

struct opt {
    std::string embeddings_file;
    std::string subword_vocab_file;
    std::string inverse_embeddings_file;
    float base_logprob;
    int word_count;
} opt;

void get_options(CLI::App& app, int argc, char* argv[]) {
    app.add_option("embeddings_file",
        opt.embeddings_file, "Word embeddings.")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("subword_vocab_file",
        opt.subword_vocab_file, "List of subwords")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("pseudo_inverse_embeddings",
        opt.inverse_embeddings_file, "File with a pseudo-inverse matrix of word embeddings")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("--base-logprob",
        opt.base_logprob, "Logprob of unseen subwords aka smoothing");
}

// template <typename Iter>
// std::iterator_traits<Iter>::value_type log_sum_exp(Iter begin, Iter end) {
//     using VT = std::iterator_traits<Iter>::value_type{};
//     if (begin==end) return VT{};

//     auto max_elem = *std::max_element(begin, end);

//     auto sum = std::accumulate(begin, end, VT{},
//         [max_elem](VT a, VT b) { return a + std::exp(b - max_elem); });

//     return max_elem + std::log(sum);
// }

float log_sum_exp(const Eigen::VectorXf& items) {
    if(items.size() == 0)
        return 0;

    auto max_elem = *std::max_element(items.begin(), items.end());

    float sum = std::accumulate(items.begin(), items.end(), float{},
        [max_elem](float a, float b) {return a + std::exp(b - max_elem);});

    return max_elem + std::log(sum);
}

float log_sum_exp(const std::vector<float>& items) {
    if(items.size() == 0)
        return 0;

    auto max_elem = *std::max_element(items.begin(), items.end());

    float sum = std::accumulate(items.begin(), items.end(), float{},
        [max_elem](float a, float b) {return a + std::exp(b - max_elem);});

    return max_elem + std::log(sum);
}

// template <typename T>
// float log_sum_exp(const std::unordered_map<T, float>& items) {
//     if(items.size() == 0)
//         return 0;

//     auto max_elem = *std::max_element(items.begin(), items.end(),
//         [](const std::pair<const T, float>& p1, const std::pair<const T, float>& p2) {
//             return p1.second < p2.second;
//         }
//     );

//     float sum = std::accumulate(items.begin(), items.end(), float{},
//         [max_elem](float a, const std::pair<const T, float>& b) {
//             return a + std::exp(b.second - max_elem.second);
//         }
//     );

//     return max_elem.second + std::log(sum);
// }


void viterbi_decode(std::vector<std::string>& reversed_segmentation,
                    const std::string& word,
                    const Eigen::ArrayXf& subword_logprobs,
                    const std::unordered_map<std::string, int>& subword_to_index) {

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
            if(subword_to_index.count(subword_candidate) == 0)
                continue;

            auto sub_index = subword_to_index.at(subword_candidate);
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

// CAUTION THIS IS BYTE-BASED, not character(grapheme)-based
void forward_costs(std::vector<float>& costs, const std::string& word,
                   const Eigen::ArrayXf& subword_logprobs,
                   const std::unordered_map<std::string, int>& subword_to_index) {

    for(size_t end = 1; end < word.size() + 1; ++end) {
        std::vector<float> prefix_scores;

        for(size_t begin = 0; begin < end; ++begin) {
            std::string subword_candidate = word.substr(begin, end - begin);

            if(subword_to_index.count(subword_candidate) == 0)
                continue;

            float cost = costs[begin] + subword_logprobs(subword_to_index.at(subword_candidate));
            prefix_scores.push_back(cost);
        }

        if(prefix_scores.size() > 0) {
            float lse = log_sum_exp(prefix_scores);
            costs[end] = lse;
        } else {
            costs[end] = - std::numeric_limits<float>::infinity();
        }

    }
}

void backward_costs(std::vector<float>& costs, const std::string& word,
                    const Eigen::ArrayXf& subword_logprobs,
                    const std::unordered_map<std::string, int>& subword_to_index) {

    for(int begin = word.size() - 1; begin >= 0; --begin) {
        std::vector<float> suffix_scores;

        for(size_t end = begin + 1; end < word.size() + 1; ++end) {
            std::string subword_candidate = word.substr(begin, end - begin);

            if(subword_to_index.count(subword_candidate) == 0)
                continue;

            float cost = costs[end] + subword_logprobs(subword_to_index.at(subword_candidate));
            suffix_scores.push_back(cost);
        }

        if(suffix_scores.size() > 0) {
            costs[begin] = log_sum_exp(suffix_scores);
        } else {
            costs[begin] = - std::numeric_limits<float>::infinity();
        }
    }
}

float compute_expected_counts(Eigen::MatrixXf& exp_counts,
                             const std::string& word,
                             int word_index,
                             const Eigen::ArrayXf& subword_logprobs,
                             const std::unordered_map<std::string, int>& subword_to_index) {

    std::vector<float> fw_costs(word.size() + 1);
    std::vector<float> bw_costs(word.size() + 1);

    forward_costs(fw_costs, word, subword_logprobs, subword_to_index);
    backward_costs(bw_costs, word, subword_logprobs, subword_to_index);

    // std::cerr << "forward costs of word " << word << ": " << std::endl;
    // for(const auto a: fw_costs) {
    //     std::cerr << a << " ";
    // }
    // std::cerr << std::endl;

    // std::cerr << "backward costs of word " << word << ": " << std::endl;
    // for(const auto a: bw_costs) {
    //     std::cerr << a << " ";
    // }
    // std::cerr << std::endl;

    for(size_t begin = 0; begin < word.size(); ++begin) {
        for(size_t end = begin + 1; end < word.size() + 1; ++end) {
            std::string subword = word.substr(begin, end - begin);

            if(subword_to_index.count(subword) == 0)
                continue;

            int subword_index = subword_to_index.at(subword);

            float score = fw_costs[begin] + subword_logprobs(subword_index)
                          + bw_costs[end];

            exp_counts(word_index, subword_index) = log_sum_exp(std::vector<float>{exp_counts(word_index, subword_index), score});
        }
    }

    float lse = log_sum_exp(exp_counts.row(word_index).array());
    exp_counts.row(word_index).array() -= lse;

    return fw_costs.back();
}


int main(int argc, char* argv[]) {

    CLI::App app{
        "Byte-based Forward-backward EM estimation of subword embeddings."};
    get_options(app, argc, argv);
    CLI11_PARSE(app, argc, argv);

    std::cerr << "Loading subword vocab: " << opt.subword_vocab_file << std::endl;
    std::unordered_map<std::string, int> subword_to_index;
    get_word_to_index(subword_to_index, opt.subword_vocab_file);
    int subword_count = subword_to_index.size();

    std::cerr << "Loading embedding matrix from " << opt.embeddings_file
              << std::endl;

    std::ifstream embedding_fh(opt.embeddings_file);

    std::string firstline;
    std::getline(embedding_fh, firstline);

    std::stringstream ss_firstline(firstline);
    int word_count;
    int embedding_dim;
    ss_firstline >> word_count;
    ss_firstline >> embedding_dim;

    Eigen::MatrixXf emb(word_count, embedding_dim);
    std::unordered_map<std::string, int> word_to_index(word_count);
    std::vector<std::string> index_to_word(word_count);

    int i = 0;
    for(std::string line; std::getline(embedding_fh, line); ++i) {
        std::stringstream ss(line);
        std::string word;
        ss >> word;
        word_to_index.insert({word, i});
        index_to_word[i] = word;

        for(int j = 0; j < embedding_dim; ++j) {
            ss >> emb(i, j);
        }
    }

    std::cerr << "Top-left corner of the embedding matrix:" << std::endl;
    std::cerr << emb.block(0,0,5,5) << std::endl;

    std::cerr << "Bottom-right corner of the embedding matrix:" << std::endl;
    std::cerr << emb.bottomRightCorner(5, 5) << std::endl;

    std::cerr << "Loading the pseudo-inverse embedding matrix from " << opt.inverse_embeddings_file << std::endl;
    std::ifstream inv_embedding_fh(opt.inverse_embeddings_file);
    Eigen::MatrixXf inverse_emb(embedding_dim, word_count);

    i = 0;
    for(std::string line; std::getline(inv_embedding_fh, line); ++i) {
        std::stringstream ss(line);
        for(int j = 0; j < word_count; ++j) {
            ss >> inverse_emb(i, j);
        }
    }

    std::cerr << "Top-left corner of the inverse embedding matrix:" << std::endl;
    std::cerr << inverse_emb.block(0,0,5,5) << std::endl;

    std::cerr << "Bottom-right corner of the inverse embedding matrix:" << std::endl;
    std::cerr << inverse_emb.bottomRightCorner(5, 5) << std::endl;

    // now, initialize W_s with uniform, then call forward_backward for each word, then upadte W_s with E^{-1} log P^\hat_w (the expected counts)
    // Ws uniform means all zeros.
    Eigen::MatrixXf w_s(embedding_dim, subword_count);

    // the expected counts as a matrix:
    // each row corresponds to a word, columns store the expected counts (log probs of subwords)

    // E^{-1} has shape (embedding_dim, word_count)
    // logPhat has shape (word_count, subword_count) --> multiplying these two gets us a new matrix of (embedding_dim, subword_count) which is exactly what is needed


    std::vector<std::string> test_words({"včelař", "hokejista", "podpatek", "náramný", "veličenstvo"});

    for(int i = 0; i < 10; ++i) {
        std::cerr << "Iteration " << i + 1 << std::endl;
        Eigen::MatrixXf exp_counts = Eigen::MatrixXf::Constant(word_count, subword_count, opt.base_logprob);

        float cummulative_sum = 0.0f;
        float cummulative_fw_cost = 0.0f;

        #pragma omp parallel for
        for(int word_index = 0; word_index < word_count; ++word_index) {
            const std::string& word = index_to_word[word_index];

            //Eigen::ArrayXf subword_logprobs(subword_count);
            Eigen::VectorXf word_embedding = emb.row(word_index);
            Eigen::VectorXf logits = word_embedding.transpose() * w_s;
            Eigen::ArrayXf subword_logprobs = logits.array() - log_sum_exp(logits);

            float fw_cost = compute_expected_counts(exp_counts, word, word_index, subword_logprobs, subword_to_index);

            float row_sum = exp_counts.row(word_index).array().sum();

            #pragma omp atomic update
                cummulative_sum += row_sum;

            if(!std::isnormal(row_sum)) {
                std::cerr << "STOP, row sum is not normal, word index " << word_index << ", word " << word << " row sum: " << row_sum <<std::endl ;
                std::abort();
            }

            #pragma omp atomic update
                cummulative_fw_cost += fw_cost;
        }

        std::cerr << "cummulative sum: " << cummulative_sum << std::endl;
        std::cerr << "cummulative forward cost: " << cummulative_fw_cost << std::endl;

        // std::cerr << "Top-left corner of the exp counts:" << std::endl;
        // std::cerr << exp_counts.block(0,0,5,5) << std::endl;

        // std::cerr << "Bottom-right corner of the exp counts:" << std::endl;
        // std::cerr << exp_counts.bottomRightCorner(5, 5) << std::endl;

        std::cerr << "sum of exp counts: " << exp_counts.array().sum() << std::endl;
        // TODO zajima nas soucet forward costu - musi rust mezi epochama

        w_s = inverse_emb * exp_counts;

        for(auto word: test_words) {

            int word_index = word_to_index[word];

            //Eigen::ArrayXf subword_logprobs(subword_count);
            Eigen::VectorXf word_embedding = emb.row(word_index);
            Eigen::VectorXf logits = word_embedding.transpose() * w_s;
            Eigen::ArrayXf subword_logprobs = logits.array() - log_sum_exp(logits);

            std::cerr << "testing segmentation of word " << word << std::endl;
            std::vector<std::string> segm;
            viterbi_decode(segm, word, subword_logprobs, subword_to_index);

            for(auto it = segm.rbegin(); it != segm.rend(); ++it) {
                std::cerr << " " << *it;
            }

            std::cerr << std::endl;
        }
    }


    return 0;
}

