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

float log_sum_exp(const std::vector<float>& items) {
    if(items.size() == 0)
        return 0;
    
    auto max_elem = *std::max_element(items.begin(), items.end());

    float sum = std::accumulate(items.begin(), items.end(), float{},
        [max_elem](float a, float b) {return a + std::exp(b - max_elem);});

    return max_elem + std::log(sum);
}

template <typename T>
float log_sum_exp(const std::unordered_map<T, float>& items) {
    if(items.size() == 0)
        return 0;

    auto max_elem = *std::max_element(items.begin(), items.end(),
        [](const std::pair<const T, float>& p1, const std::pair<const T, float>& p2) {
            return p1.second < p2.second;
        }
    );

    float sum = std::accumulate(items.begin(), items.end(), float{},
        [max_elem](float a, const std::pair<const T, float>& b) {
            return a + std::exp(b.second - max_elem.second);
        }
    );

    return max_elem.second + std::log(sum);
}

// CAUTION THIS IS BYTE-BASED, not character(grapheme)-based
void forward_costs(std::vector<float>& costs, const std::string& word,
                   const std::unordered_map<std::string, float>& subword_vocab) {
  
    for(size_t end = 1; end < word.size() + 1; ++end) {
        std::vector<float> prefix_scores;

        for(size_t begin = 0; begin < end; ++begin) {
            std::string subword_candidate = word.substr(begin, end - begin);

            if(subword_vocab.count(subword_candidate) == 0)
                continue;

            float cost = costs[begin] + subword_vocab.at(subword_candidate);
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
                    const std::unordered_map<std::string, float>& subword_vocab) {

    for(int begin = word.size() - 1; begin >= 0; --begin) {
        std::vector<float> suffix_scores;    
        
        for(size_t end = begin + 1; end < word.size() + 1; ++end) {
            std::string subword_candidate = word.substr(begin, end - begin);

            if(subword_vocab.count(subword_candidate) == 0)
                continue;

            float cost = costs[end] + subword_vocab.at(subword_candidate);
            suffix_scores.push_back(cost);
        }

        if(suffix_scores.size() > 0) {
            costs[begin] = log_sum_exp(suffix_scores);
        } else {
            costs[begin] = - std::numeric_limits<float>::infinity();
        }
    }
}

void expected_counts(std::unordered_map<std::string, float>& exp_counts,
                     const std::string& word, 
                     const std::unordered_map<std::string, float>& subword_vocab) {
    
    std::vector<float> fw_costs(word.size() + 1);
    std::vector<float> bw_costs(word.size() + 1);

    forward_costs(fw_costs, word, subword_vocab);
    backward_costs(bw_costs, word, subword_vocab);

    std::cerr << "forward costs of word " << word << ": " << std::endl;
    for(const auto a: fw_costs) {
        std::cerr << a << " ";
    }
    std::cerr << std::endl;

    std::cerr << "backward costs of word " << word << ": " << std::endl;
    for(const auto a: bw_costs) {
        std::cerr << a << " ";
    }
    std::cerr << std::endl;

    for(size_t begin = 0; begin < word.size(); ++begin) {
        for(size_t end = begin + 1; end < word.size() + 1; ++end) {
            std::string subword = word.substr(begin, end - begin);

            if(subword_vocab.count(subword) == 0)
                continue;

            float score = fw_costs[begin] + subword_vocab.at(subword) 
                          + bw_costs[end];

            if(exp_counts.count(subword) == 0) {
                exp_counts[subword] = score;
            } else {
                exp_counts[subword] = log_sum_exp({exp_counts[subword], score});
            }            
        }
    }

    float lse = log_sum_exp<std::string>(exp_counts);
    for(std::unordered_map<std::string, float>::iterator it = exp_counts.begin(); 
            it != exp_counts.end(); ++it) {
        it->second = it->second - lse;
    }
}


int main(int argc, char* argv[]) {

    CLI::App app{
        "Byte-based Forward-backward EM estimation of subword embeddings."};
    get_options(app, argc, argv);
    CLI11_PARSE(app, argc, argv);

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

    int i = 0;
    for(std::string line; std::getline(embedding_fh, line); ++i) {
        std::stringstream ss(line);
        std::string word;
        ss >> word;
        word_to_index.insert({word, i});

        for(int j = 0; j < embedding_dim; ++j) {
            ss >> emb(i, j);
        }
    }

    std::cerr << "Top-left corner of the embedding matrix:" << std::endl;
    std::cerr << emb.block(0,0,5,5) << std::endl;

    std::cerr << "Bottom-right corner of the embedding matrix:" << std::endl;
    std::cerr << emb.bottomRightCorner(5, 5) << std::endl;

    // now, initialize W_s with uniform, then call forward_backward, then upadte W_s with E^{-1} log P^\hat_w (the expected counts)
    //Eigen::MatrixXf w_s(embedding_dim, subword_count);

    std::string test_word = "včely";
    std::cerr << test_word << " " << test_word.length() << std::endl;

    std::unordered_map<std::string, float> subword_vocab;
    subword_vocab["vč"] = -2;
    subword_vocab["el"] = -3;
    subword_vocab["ely"] = -4;
    subword_vocab["ly"] = -3;
    subword_vocab["e"] = -1.5;
    subword_vocab["v"] = -10;
    subword_vocab["č"] = -10;
    subword_vocab["l"] = -10;
    subword_vocab["y"] = -10;

    std::unordered_map<std::string, float> test_exp_counts;
    expected_counts(test_exp_counts, test_word, subword_vocab);


    std::cerr << test_exp_counts["vč"] << std::endl;

    return 0;
}

