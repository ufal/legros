#include <string>
#include <iostream>
#include <filesystem>

#include <Eigen/Dense>

#include "CLI11.hpp"
#include "vocabs.h"
#include "substring_stats.h"

namespace fs = std::filesystem;



struct opt {
    std::string embeddings_file;
    std::string subword_vocab_file;
    std::string output;
    std::string fasttext_output_pseudoinverse;
    std::string train_data;
    int fasttext_dim = 200;
    int window_size= 5; // Careful this default value is different than in train_subword_embeddings.cpp
    int subword_shard_size = 100000;
} opt;


void get_options(CLI::App& app) {
    app.add_option(
        "word_embeddings_file", opt.embeddings_file, "Word embeddings.")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("subword_vocabulary",
        opt.subword_vocab_file, "Subword vocabulary, subword per line.")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option(
        "output", opt.output, "Output file.")
        ->required()
        ->check(CLI::NonexistentPath);

    app.add_option(
        "fastext-output-pseudoinverse", opt.fasttext_output_pseudoinverse,
        "Pseudo-inverse of the fasttext output matrix")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option(
        "train_data", opt.train_data, "Training data for cooccurrence matrix.")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option(
        "--fasttext-dim", opt.fasttext_dim,
        "Dimension of the fasttext embeddings.");

    app.add_option(
        "--window-size", opt.window_size, "Window size.");

    app.add_option(
        "--subword-shard-size", opt.subword_shard_size,
        "Number of subwords to process in one batch.");
}



// Populates a dense structure of cooccurrences of `word_vocab` vocabulary
// items in `train_data` within a window of size `window_size`.
//
// Converts the dense structure to sparse representation `sparse_c_v` to save
// memory.
//
// Saves unigram frequencies in `word_frequencies`.
//
// Optionally, when `compute_pseudoinverse_w` is specified, it computes the
// pseudo-inverse of the log cooccurrence matrix and stores it in `pinv`.
// This is done here because the dense structure is needed.
void sparse_cooccurrences(
    std::vector<std::unordered_map<int, int>>& sparse_c_v,
    std::vector<int>& word_frequencies,
    const Embeddings& word_vocab,
    const std::string& train_data,
    int window_size) {

  CooccurrenceMatrix c_v(word_vocab.size(), word_vocab.size());

  // --> this thing takes too long after traverse through data
  populate_word_stats<CooccurrenceMatrix>(
      c_v, word_frequencies, word_vocab, train_data, window_size);

  std::cerr << "Done, here are some stats:" << std::endl;
  std::cerr << c_v.topLeftCorner<5,5>() << std::endl;

  std::cerr << "Converting to sparse structure" << std::endl;

#pragma omp parallel for
  for(int i = 0; i < word_vocab.size(); ++i) {
    std::vector<std::pair<int, int>> row_pairs;
    for(int j = 0; j < word_vocab.size(); ++j) {
      int freq = c_v(i, j);
      if(freq > 0) {
        row_pairs.push_back({j, freq});
      }
    }

#pragma omp critical
    sparse_c_v[i] = std::unordered_map<int, int>(
        row_pairs.begin(), row_pairs.end());
  }
}


// Fills `c_sub` with word-subword cooccurrences, given word cooccurrences in
// `sparse_c_v`. Only considers subwords present in the `a_sub_inv` map
// (aka. allowed substrings)
void word_subword_cooccurrences(
    Eigen::MatrixXf& c_sub,
    const Embeddings& word_vocab,
    const Vocab& subword_vocab,
    int subw_shard_begin,
    int subw_shard_size,
    //const InverseAllowedSubstringMap& a_sub_inv,
    const std::vector<std::unordered_map<int, int>>& sparse_c_v) {

#pragma omp parallel for
  for(int i = 0; i < subw_shard_size; ++i) {
    std::string subword = subword_vocab[subw_shard_begin + i];

    for(int k = 0; k < word_vocab.size(); ++k) {
        std::string word = word_vocab[k];
        int count = 0;

        // find subword in word
        size_t pos = word.find(subword);

        while (pos != std::string::npos) {
            count++;
            pos = word.find(subword, pos + 1);  // Overlapping allowed
        }

        if(count == 0)
            continue;

        for(auto cooccurs : sparse_c_v.at(k)) {
            int j = cooccurs.first;
            int num = cooccurs.second;

#pragma omp atomic
            c_sub(i, j) += num * count;
        }
    }
  }
}


// Saves an Eigen matrix `embeddings` into a file specified by `path`.
void save_embedding_checkpoint(
    const fs::path& path,
    const Eigen::MatrixXf& embeddings) {
  std::ofstream ofs(path);
  ofs << embeddings << std::endl;
  ofs.close();
}

void append_embeddings_to_checkpoint(
    const fs::path& path,
    const Eigen::MatrixXf& embeddings) {
  std::ofstream ofs(path, std::ios::app);
  ofs << embeddings << std::endl;
  ofs.close();
}

int main(int argc, char* argv[]) {
    CLI::App app{"LEGROS subword embeddings initializer."};
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

    // std::cerr << "Loading subword embeddings: " << opt.subword_embeddings_file << std::endl;
    // Embeddings subword_vocab(opt.subword_embeddings_file);
    // int subword_count = subword_vocab.size();

    std::cerr << "Populating word cooccurrence stats (" << word_count
              << " words)" << std::endl;
    std::vector<std::unordered_map<int, int>> sparse_c_v(word_count);
    std::vector<int> word_frequencies(word_count);
    sparse_cooccurrences(
        sparse_c_v, word_frequencies, word_vocab, opt.train_data,
        opt.window_size);
    std::cerr << sparse_c_v[10].size() << std::endl;

    std::cerr << "Loading pseudo-inverse of fasttext output matrix from "
            << opt.fasttext_output_pseudoinverse << std::endl;
    Eigen::MatrixXf pinv(word_count, opt.fasttext_dim);
    std::ifstream fasttext_fh(opt.fasttext_output_pseudoinverse);
    int lineno = 0;
    for(std::string line; std::getline(fasttext_fh, line); ++lineno) {
        std::stringstream linestream(line);
        for(int j = 0; j < word_count; ++j) {
            linestream >> pinv(j, lineno);
        }
    }
    std::cerr << "Pseudo-inverse dim: " << pinv.rows() << " x " << pinv.cols()
        << std::endl;

    // sharding
    for(int i = 0; i < subword_vocab.size(); i += opt.subword_shard_size) {
        std::cerr << "Processing shard " << i << " - "
            << i + opt.subword_shard_size << std::endl;

        int shard_size = std::min(opt.subword_shard_size, subword_vocab.size() - i);

        // compute word-subword coocurrences
        Eigen::MatrixXf c_sub(shard_size, word_count);
        c_sub.setZero();
        word_subword_cooccurrences(c_sub, word_vocab, subword_vocab, i, shard_size, sparse_c_v);

        // compute subword embeddings for this shard
        std::cerr << "Computing subword embeddings" << std::endl;
        c_sub.array() += 0.00001f;
        Eigen::VectorXf sums = c_sub.rowwise().sum();
        Eigen::MatrixXf normed = c_sub.array().log().matrix().colwise() - sums.array().log().matrix();
        Eigen::MatrixXf subword_embeddings =  normed * pinv;


        std::cerr << "Saving checkpoint to " << opt.output << std::endl;
        append_embeddings_to_checkpoint(opt.output, subword_embeddings);
    }


    std::cerr << "Done." << std::endl;
    return 0;
}