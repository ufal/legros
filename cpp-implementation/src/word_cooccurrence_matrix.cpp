#include <string>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <CLI11.hpp>
#include "vocabs.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

struct opt {
    std::string word_vocab_file;
    std::string training_data_file;
    std::string output_file;
    int window_size = 3;
    int buffer_size = 1000000;
} opt;

void get_options(CLI::App& app) {
    app.add_option("word_vocabulary",
        opt.word_vocab_file, "Word vocabulary, word per line.")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("input",
        opt.training_data_file, "Tokenized text.")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("output",
        opt.output_file, "Matrix data.")
        ->required()
        ->check(CLI::NonexistentPath);

    app.add_option("--window-size",
        opt.window_size, "Window size.");

    app.add_option("--buffer-size",
        opt.buffer_size, "Buffer size.");
}

template<typename U>
U& get_2d(
        std::vector<std::vector<U>>& stats,
        int stat_index, int word_index) {
    return stats[stat_index][word_index];
}

template<typename Derived>
typename Derived::Scalar& get_2d(
    Eigen::MatrixBase<Derived> &stats,
    int stat_index, int word_index) {
  return stats(stat_index, word_index);
}

template<typename U>
U& get_2d(
        Eigen::SparseMatrix<U> &stats,
        int stat_index, int word_index) {
    U& ref;
    #pragma omp critical
    ref = stats.coeffRef(stat_index, word_index);
    return ref;
}


int& get_2d(
        std::unordered_map<int, std::unordered_map<int, int>> &stats,
        int stat_index, int word_index) {
    if(stats.count(stat_index) == 0) {
        #pragma omp critical
        stats.insert({stat_index, std::unordered_map<int, int>()});
    }
    auto w_stats = stats.at(stat_index);

    if(w_stats.count(word_index) == 0) {
        #pragma omp critical
        w_stats.insert({word_index, 0});
    }

    return w_stats.at(word_index);
}

template<typename T>
void try_add_to_stats(
    T& stats,
    const std::string &target_token,
    const std::string &window_token,
    const Vocab& word_vocab) {

    if(!word_vocab.contains(target_token))
        return;

    if(!word_vocab.contains(window_token))
        return;

    int target_index = word_vocab[target_token];
    int window_index = word_vocab[window_token];

#pragma omp atomic
        get_2d(stats, window_index, target_index) += 1;
}

template<typename T>
void process_buffer(
        const std::vector<std::string>& buffer,
        int end,
        T& stats,
        const Vocab&  word_vocab) {

    #pragma omp parallel for
    for(int i = 0; i < end; ++i) {
        std::string line = buffer[i];
        std::istringstream iss(line);

        std::vector<std::string> tokens(
            (std::istream_iterator<std::string>(iss)),
            std::istream_iterator<std::string>());

        int t = 0;
        for(auto token: tokens) {

            for(int j = std::max(0, t - opt.window_size); j < t; ++j) {
                try_add_to_stats<T>(stats, tokens[j], token, word_vocab);
            }

            for(int k = t + 1; k < std::min(t + 1 + opt.window_size, (int)tokens.size()); ++k) {
                try_add_to_stats<T>(stats, tokens[k], token, word_vocab);
            }
            ++t;
        }
    }
}

template<typename T>
void populate_word_stats(
        T& stats,
        const Vocab&  word_vocab) {

    std::cerr << "Iterating over sentences from " << opt.training_data_file << std::endl;
    std::ifstream input_fh(opt.training_data_file);

    int lineno = 0;
    int buffer_pos = 0;
    std::vector<std::string> buffer(opt.buffer_size);

    while(std::getline(input_fh, buffer[buffer_pos])) {
        ++lineno;
        ++buffer_pos;
        if(lineno % 1000 == 0)
        std::cerr << "Lineno: " << lineno << "\r";

        // full buffer -> process
        if(buffer_pos == opt.buffer_size) {
            process_buffer<T>(buffer, buffer_pos, stats, word_vocab);
            buffer_pos = 0;
        }
    }

    // process the rest of the buffer
    if(buffer_pos > 0) {
        process_buffer<T>(buffer, buffer_pos, stats, word_vocab);
    }

    std::cerr << "Read " << lineno << " lines in total." << std::endl;
}

int main(int argc, char* argv[]) {
    CLI::App app{"Compute word cooccurrences."};
    get_options(app);
    CLI11_PARSE(app, argc, argv);

    std::cerr << "Loading word vocab: " << opt.word_vocab_file << std::endl;
    Vocab word_vocab(opt.word_vocab_file);

    std::string test_word = "včelař";
    std::cerr << "Index of '" << test_word << "': " << word_vocab[test_word] << std::endl;
    int word_count = word_vocab.size();

    //Eigen::SparseMatrix<int> stats(word_count, word_count);
    Eigen::MatrixXi stats(word_count, word_count);
    //std::unordered_map<int, std::unordered_map<int, int>> stats;
    //std::vector<std::vector<int>> stats(word_count, std::vector<int>(word_count));


    //populate_word_stats<Eigen::SparseMatrix<int>>(stats, word_vocab);
    populate_word_stats<Eigen::MatrixXi>(stats, word_vocab);
    //populate_word_stats<std::unordered_map<int, std::unordered_map<int, int>>>(stats, word_vocab);
    //populate_word_stats<std::vector<std::vector<int>>>(stats, word_vocab);

    for(int i = 0 ; i < 5; ++i) {
        for(int j = 0; j < 5; ++j) {
            std::cerr << get_2d(stats, i, j) << " ";
        }
        std::cerr << std::endl;
    }

    //std::cerr << "Number of zero elements in stats: " << std::count(stats.data(), stats.data() + stats.size(), 0) << std::endl;
    //std::cerr << "Stats total size: " << stats.size() << std::endl;

    std::cerr << "Dumping stats to " << opt.output_file << std::endl;
    std::ofstream output_fh(opt.output_file);

    int output_buffer_size = 10000;
    int howmany = word_count / output_buffer_size;
    //int remainder = word_count % output_buffer_size;

    std::vector<std::string> partial_strings(output_buffer_size);

    for(int b = 0; b <= howmany; ++b) {
        int begin = b * output_buffer_size;
        int end = std::min(word_count, begin + output_buffer_size);
        //int diff = end - begin;

        #pragma omp parallel for
        for(int i = begin; i < end; ++i) {
            std::ostringstream ss;
            for(int j = i; j < word_count; ++j) {
                auto val = get_2d(stats, i, j);
                if(val != 0)
                    ss << i << " " << j << " " << val << "\n";
            }

            partial_strings[i - begin] = ss.str();
        }

        for(int i = begin; i < end; ++i) {
            if(i % 100 == 0)
                std::cerr << "Line: " << i << "\r";
            output_fh << partial_strings[i - begin];
        }
    }

    output_fh.close();
    std::cerr << std::endl;

    return 0;
}
