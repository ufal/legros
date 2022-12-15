#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "CLI11.hpp"
#include "vocabs.h"
#include "substring_stats.h"
#include <Eigen/Dense>

struct opt {
  std::string subword_vocab_file;
  std::string word_vocab_file;
  std::string training_data_file;
  std::string fasttext_output_matrix;
  std::string output_file;
  std::string allowed_substrings;
  int max_subword = 10;
  int window_size = 3;
  int fasttext_dim = 200;
  size_t buffer_size = 1000000;
  int shard_size = 1000;
} opt;

void get_options(CLI::App& app, int argc, char* argv[]) {
  app.add_option("subword_vocabulary",
    opt.subword_vocab_file, "Subword vocabulary, subword per line.")
    ->required()
    ->check(CLI::ExistingFile);

  app.add_option("word_vocabulary",
    opt.word_vocab_file, "Word vocabulary, word per line.")
    ->required()
    ->check(CLI::ExistingFile);

  app.add_option("input",
    opt.training_data_file, "Tokenized text.")
    ->required()
    ->check(CLI::ExistingFile);

  app.add_option("fasttext",
    opt.fasttext_output_matrix, "Pseudo-inverse of the fasttext output matrix.")
    ->required()
    ->check(CLI::ExistingFile);

  app.add_option("output",
    opt.output_file, "Matrix data.")
    ->required()
    ->check(CLI::NonexistentPath);

  app.add_option("--allowed-substrings",
    opt.allowed_substrings, "List of words accompanied with allowed substrings.")
    ->check(CLI::ExistingFile);

  app.add_option("--max-subword",
    opt.max_subword, "Maximum subword length.");

  app.add_option("--window-size",
    opt.window_size, "Window size.");

  app.add_option("--fasttext-dim",
    opt.fasttext_dim, "Dimension of the fasttext embeddings.");

  app.add_option("--buffer-size",
    opt.buffer_size, "Buffer size.");

  app.add_option("--shard-size",
    opt.shard_size, "Shard size for matrix multiplication.");
}

int main(int argc, char* argv[]) {
    CLI::App app{"Compute subword embeddings."};
    get_options(app, argc, argv);
    CLI11_PARSE(app, argc, argv);

    std::cerr << "Loading subword vocab: " << opt.subword_vocab_file << std::endl;
    Vocab subword_vocab(opt.subword_vocab_file);

    std::cerr << "Loading word vocab: " << opt.word_vocab_file
              << std::endl;
    Vocab word_vocab(opt.word_vocab_file);

    std::cerr << "Index of 'společenství': " << word_vocab["společenství"] << std::endl;

    int subword_count = subword_vocab.size();
    int word_count = word_vocab.size();

    std::cerr << "Populating matrix stats (dim " << subword_count <<  " x " << word_count << ")" << std::endl;
    Eigen::MatrixXi stats(subword_count, word_count);
    populate_substring_stats<Eigen::MatrixXi>(stats, word_vocab, subword_vocab, opt.training_data_file, opt.allowed_substrings, opt.buffer_size, opt.window_size, opt.max_subword);
    std::cerr << stats.topLeftCorner<5,5>() << std::endl;

    // std::cerr << "Number of zero elements in stats: " << std::count(stats.data(), stats.data() + stats.size(), 0) << std::endl;
    // std::cerr << "Stats total size: " << stats.size() << std::endl;

    std::cerr << "Casting to float" << std::endl;
    Eigen::MatrixXf statsf = stats.cast<float>(); // TODO SPARSE this should work seamlessly

    std::cerr << "Smoothing" << std::endl;
    statsf.array() += 0.00001f;  // TODO SPARSE: just remember to add this

    std::cerr << "Computing log&norm" << std::endl;
    //np.log(subword_data) - np.log(subword_data.sum(1, keepdims=True)))
    Eigen::VectorXf sums = statsf.rowwise().sum();   // TODO SPARSE: sum sparsely, then add smoothing coeff * stats.cols() to each row.
    std::cerr << "Sums of the first five rows:" << std::endl;
    std::cerr << sums(Eigen::seq(0, 4)) << std::endl;
    std::cerr << "Sums size: " << sums.size() <<std::endl;

    Eigen::MatrixXf normed = statsf.array().log().matrix().colwise() - sums.array().log().matrix();
        // TODO SPARSE: Only compute for non-infinite coefficients, which should work seamlessly

    std::cerr << "Top-right corner of the normalized stats matrix:" << std::endl;
    std::cerr << normed.block(0,0,5,5) << std::endl;

    std::cerr << "Loading pseudo-inverse of fasttext output matrix from " << opt.fasttext_output_matrix << std::endl;
    std::ifstream fasttext_fh(opt.fasttext_output_matrix);
    Eigen::MatrixXf pinv(word_count, opt.fasttext_dim);
        // TODO SPARSE: This matrix will remain dense

    int i = 0;
    for(std::string line; std::getline(fasttext_fh, line); ++i) {
        std::stringstream linestream(line);

        for(int j = 0; j < word_count; ++j) {
            linestream >> pinv(j, i);
        }
    }

    std::cerr << pinv.block(0,0,5,5) << std::endl;
    std::cerr << "Pseudo-inverse dim: " << pinv.rows() << " x " << pinv.cols() << std::endl;

    int shard_count = subword_count / opt.shard_size;

    // TODO SPARSE
    /*
    1. Sharding the sparse matrix. How? Why? Shard embedding matrix instead?
    2. Smoothing. Negative-infinite coefficients are smoothed, so they are the log(10e-5) - log of sums per column.
        - these are constant vectors filled with different numbers according to the row sum.
        - basically a different (constant, dense) matrix added to the sparse one.
        - from distributivity, we have (NORMED + SMOOTH) x PINV = NORMED x PINV + SMOOTH x PINV.
        - NORMED x PINV can be done on sparse matrix using multiple cores
        - SMOOTH x PINV is easy? (since SMOOTH has constant columns) Can it be ignored altogether?
            - result of sparse x dense is already dense, so this would add some sort of constant noise to the embeddings.
    */

    std::cerr << "Computing product between pseudo-inverse and the normalized matrix (using "
              << (subword_count % opt.shard_size == 0 ? shard_count : shard_count + 1)
              << " shards of size " << opt.shard_size << ")" << std::endl;

    std::ofstream output_fh(opt.output_file);

    for(int shard = 0; shard < shard_count; ++shard) {
        std::cerr << "Shard " << shard << ", computing product" << "\r";

        // normed has shape (subword, vocab_size), pinv is (vocab_size, embedding_dim)
        // we want to select only some subwords, e.g. only some rows, and do the product on them
        Eigen::MatrixXf prod = normed.middleRows(shard * opt.shard_size, opt.shard_size) * pinv;

        for(int i = 0; i < prod.rows(); ++i) {
            if(i % 100 == 0)
                std::cerr << "Shard " << shard << ", writing to output, " << "line " << i << "\r";
            std::ostringstream line;
            for(int j = 0; j < prod.cols(); ++j) {
                line << prod(i,j) << " ";
            }
            output_fh << line.str() << "\n";
        }
    }

    std::cerr << std::endl;

    if(subword_count % opt.shard_size > 0) {
        std::cerr << "Last shard, computing product" << "\r";
        Eigen::MatrixXf prod = normed.bottomRows(subword_count % opt.shard_size) * pinv;

        //std::ofstream output_fh(opt.output_file);
        for(int i = 0; i < prod.rows(); ++i) {
            if(i % 100 ==0)
                std::cerr << "Last shard, writing to output, line " << i << "\r";
            std::ostringstream line;
            for(int j = 0; j < prod.cols(); ++j) {
                line << prod(i,j) << " ";
            }
            output_fh << line.str() << "\n";
        }
        std::cerr << std::endl;
    }
    return 0;
}
