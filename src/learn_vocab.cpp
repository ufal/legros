#include <string>
#include <iostream>
#include <filesystem>
#include <ranges>
#include <queue>

#include <Eigen/Dense>

#include "CLI11.hpp"
#include "vocabs.h"
#include "substring_stats.h"
#include "cosine_viterbi.h"

namespace fs = std::filesystem;


struct opt {
    std::string embeddings_file;
    std::string subword_vocab_file;
    std::string subword_embeddings_file;
    std::string word_counts_file;

    int fasttext_dim = 200;
    int target_vocab_size;
} opt;


void get_options(CLI::App& app) {
    app.add_option(
        "word_embeddings_file", opt.embeddings_file, "Word embeddings.")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("subword_vocabulary_file",
        opt.subword_vocab_file, "Subword vocabulary, subword per line.")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("subword_embeddings_file",
        opt.subword_embeddings_file, "Subword embeddings.")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option(
        "word_counts", opt.word_counts_file, "word counts")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option(
        "--fasttext-dim", opt.fasttext_dim,
        "Dimension of the fasttext embeddings.");

    app.add_option(
        "--target-size", opt.target_vocab_size, "Target subword vocabulary size")
        ->required();
}



int main(int argc, char* argv[]) {
    CLI::App app{"LEGROS vocabulary learning algorithm."};
    get_options(app);
    CLI11_PARSE(app, argc, argv);

#ifndef SSEG_RELEASE_BUILD
    std::cerr
        << "\n\033[31m!! WARNING !!\033[0m You are likely running a debug build"
        << "\nFor best results, use cmake with -DCMAKE_BUILD_TYPE=Release\n\n";
#endif


    std::cerr << "Loading word embeddings: " << opt.embeddings_file << std::endl;
    Embeddings word_vocab(opt.embeddings_file);
    int word_count = word_vocab.size();

    std::cerr << "Loading subword vocab: " << opt.subword_vocab_file << std::endl;
    Vocab subword_vocab(opt.subword_vocab_file);

    std::cerr << "Loading subword embeddings from "
        << opt.subword_embeddings_file << std::endl;
    Eigen::MatrixXf subword_embeddings(subword_vocab.size(), opt.fasttext_dim);
    std::ifstream swemb_fh(opt.subword_embeddings_file);
    int lineno = 0;
    for(std::string line; std::getline(swemb_fh, line); ++lineno) {
        std::stringstream linestream(line);
        for(int j = 0; j < opt.fasttext_dim; ++j) {
            linestream >> subword_embeddings(lineno, j);
        }
    }

    std::cerr << "top left corner of subword embeddings:" << std::endl;
    std::cerr << subword_embeddings.topLeftCorner<5,5>() << std::endl;

    std::cerr << "Loading word counts from " << opt.word_counts_file << std::endl;
    std::ifstream wc_fh(opt.word_counts_file);
    std::unordered_map<std::string, int> word_counts;

    for(std::string line; std::getline(wc_fh, line);) {
        std::istringstream iss(line);
        std::string word;
        int count;
        iss >> word >> count;
        word_counts[word] = count;
    }

    // check that every word from word_vocab has a count in word_counts
    bool die = false;
    for(int i = 0; i < word_count; ++i) {
        std::string word = word_vocab[i];
        if(word_counts.count(word) == 0) {
            std::cerr << "ERR: Word '" << word << "' not in word counts" << std::endl;
        }
    }
    if(die) return 1;


    std::cerr << "Computing mutual reference index for subwords and words" << std::endl;

    std::vector<std::vector<int>> subwords_in_word(word_count, std::vector<int>());
    std::vector<std::vector<int>> subword_parents(subword_vocab.size(), std::vector<int>());
    std::vector<std::vector<float>> subword_parent_diffs(subword_vocab.size(), std::vector<float>());

#pragma omp parallel for
    for(int i = 0; i < word_count; ++i) {
        std::string word = word_vocab[i];

        for(int j = 0; j < subword_vocab.size(); ++j) {
            std::string subword = subword_vocab[j];
            if(word.find(subword) != std::string::npos) {
                subwords_in_word[i].push_back(j);
                //subword_parents[j].push_back(i);
            }
        }
    }

#pragma omp parallel for
    for(int j = 0; j < subword_vocab.size(); ++j) {
        std::string subword = subword_vocab[j];

        for(int i = 0; i < word_count; ++i) {
            std::string word = word_vocab[i];

            if(word.find(subword) != std::string::npos) {
                //subwords_in_word[i].push_back(j);
                subword_parents[j].push_back(i);
                subword_parent_diffs[j].push_back(0.0);
            }
        }
    }


    std::cerr << "Initializing vocabulary with single-byte subwords" << std::endl;
    std::unordered_map<std::string, int> vocabulary; // this is what we are learning in this file
    for(int i = 0; i < subword_vocab.size(); ++i) {
        std::string subword = subword_vocab[i];
        if(subword.size() == 1) {
            vocabulary.insert({subword, i});
        }
    }

    std::cerr << "Segmenting word vocab with initial vocabulary" << std::endl;
    std::vector<float> word_segm_scores(word_count);
    float total_score = 0.0;
    long total_word_count = 0;

    for(int i = 0; i < word_count; ++i) {

        std::string word = word_vocab[i];
        std::vector<std::string> segm;
        float segm_score = viterbi_decode(
            segm, word, word_vocab.emb.row(i), vocabulary, subword_embeddings);

        word_segm_scores[i] = segm_score;

        total_score += segm_score * word_counts.at(word);
        total_word_count += word_counts.at(word);
    }

    std::cerr << "Initial score: " << total_score << std::endl;
    std::cerr << "Total word count: " << total_word_count << std::endl;
    std::cerr << "Neg. initial score (normalized): " << - total_score / total_word_count << std::endl;

    std::vector<std::pair<float, int>> subword_scores(subword_vocab.size());

#pragma omp parallel for
    for(int j = 0; j < subword_vocab.size(); ++j) {
        std::string subword = subword_vocab[j];
        if(vocabulary.count(subword) > 0) {
            subword_scores[j] = {-1, j};
            continue;
        }

        bool debug = subword == "rojař";

        float utility = 0.0;

        for(int parent_i = 0; parent_i < subword_parents[j].size(); ++parent_i) {
            int word_idx = subword_parents[j][parent_i];

            std::string word = word_vocab[word_idx];
            float current_word_segm_score = word_segm_scores[word_idx];

            std::vector<std::string> segm;
            float candidate_segm_score = viterbi_decode(
                segm, word, word_vocab.emb.row(word_idx), vocabulary, subword_embeddings,
                subword, j);

            float diff = candidate_segm_score - current_word_segm_score;
            if(diff < 0) {
                std::cerr << "Warning: negative diff for " << word << " and " << subword << std::endl;
            }
            subword_parent_diffs[j][parent_i] = diff;

            if(debug) {
                std::cerr << "přidávám rojařovi " << diff << ", což je " <<  diff * word_counts.at(word)
                    << " za slovo " << word << std::endl;
            }


            utility += diff * word_counts.at(word);
        }

        if(debug) {
            std::cerr << "Utility of subword 'rojař': " << utility << std::endl;
        }

        subword_scores[j] = {utility, j};
    }

    std::cerr << "Utility of subword 'zpěv'" << subword_scores[subword_vocab["zpěv"]].first << std::endl;
    std::cerr << "Utility of subword 'hokej'" << subword_scores[subword_vocab["hokej"]].first << std::endl;
    std::cerr << "Utility of subword 'ák'" << subword_scores[subword_vocab["ák"]].first << std::endl;
    std::cerr << "Utility of subword 'rojař'" << subword_scores[subword_vocab["rojař"]].first << std::endl;

    auto cmp = [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first < b.first; };
    std::priority_queue pq(subword_scores.begin(), subword_scores.end(), cmp);

    for (; !pq.empty(); pq.pop()) {
        float best_utility = pq.top().first;
        int best_subword_index = pq.top().second;
        std::string best_subword = subword_vocab[best_subword_index];

        if(vocabulary.count(best_subword) > 0) {
            continue;
        }

        if(best_utility != subword_scores[best_subword_index].first) {
            continue;
        }

        std::cerr << "Best subword: (utility " << best_utility << "):"
            << best_subword << std::endl;

        vocabulary.insert({best_subword, best_subword_index});
        if(vocabulary.size() >= opt.target_vocab_size)
            break;

        std::unordered_set<int> subwords_to_update;

        // for all words that contain best_subword, re-run segmentation
#pragma omp parallel for
        for (int word_idx: subword_parents[best_subword_index]) {
            std::string word = word_vocab[word_idx];

            std::vector<std::string> segm;

            float new_segm_score = viterbi_decode(
                segm, word, word_vocab.emb.row(word_idx), vocabulary, subword_embeddings);

            if(new_segm_score < word_segm_scores[word_idx]) {
                std::cerr << "WARNING: new segmentation score is worse for " << word << std::endl;
            }


            if(word_segm_scores[word_idx] != new_segm_score) {
#pragma omp critical
                {
                    word_segm_scores[word_idx] = new_segm_score;
                    for(int subword_idx : subwords_in_word[word_idx]) {
                        subwords_to_update.insert(subword_idx);
                    }
                }
            }
        }
        // now for all subwords contained in words with updated segmentation,
        // compute utility
        // we assume that if score does not change, the segmentation did not change

        for(int subword_idx : subwords_to_update) {
            std::string subword = subword_vocab[subword_idx];
            if(vocabulary.count(subword) > 0) {
                continue;
            }

            float utility = subword_scores[subword_idx].first;

            for(int parent_i = 0; parent_i < subword_parents[subword_idx].size(); ++parent_i) {
                int word_idx = subword_parents[subword_idx][parent_i];
                std::string word = word_vocab[word_idx];
                float current_word_segm_score = word_segm_scores[word_idx];
                float current_subword_parent_diff = subword_parent_diffs[subword_idx][parent_i];

                std::vector<std::string> segm;
                float current_candidate_segm_score = viterbi_decode(
                    segm, word, word_vocab.emb.row(word_idx), vocabulary, subword_embeddings,
                    subword, subword_idx);

                // float old_dif = old_candidate_segm_score - old_segm_score
                float new_diff = current_candidate_segm_score - current_word_segm_score;
                utility += (new_diff - current_subword_parent_diff) * word_counts.at(word);
            }

            subword_scores[subword_idx] = {utility, subword_idx};
            pq.push({utility, subword_idx});
        }
    }


    for(auto [subword, index] : vocabulary) {
        std::cout << subword << std::endl;
    }

    std::cerr << "Done. Thank you." << std::endl;

    return 0;
}