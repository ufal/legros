/**
 * Train subword embeddings using a pseudoEM algorithm.
 *
 * Word vocabulary
 * Subword vocabulary
 * Word cooccurrence matrix
 *   - (obtained from data using the word_cooccurrence_matrix binary)
 *
 * Fasttext word embeddings (pseudo-inverse)
 *
 * Output file (will contain the trained subword embeddings)
 */

#include <string>
#include <iostream>
#include <ranges>
#include <filesystem>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "CLI11.hpp"

#include "vocabs.h"
#include "substring_stats.h"
#include "cosine_viterbi.h"

namespace fs = std::filesystem;

struct opt {
  std::string embeddings_file;
  std::string allowed_substrings;
  std::string fasttext_output_pseudoinverse;

  std::string output;
  std::string train_data;

  std::string output_directory = ".";
  std::string segmentations_prefix = "segmentations.";
  std::string embeddings_prefix = "subword_embeddings.";
  std::string subwords_prefix = "subwords.";
  std::string unigrams_prefix = "unigram_stats.";
  std::string bigrams_prefix = "bigram_stats.";

  int fasttext_dim = 200;
  int window_size = 3;
  int epochs = 1;
} opt;

void get_options(CLI::App& app) {
  app.add_option(
      "embeddings_file", opt.embeddings_file, "Word embeddings.")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option(
      "train_data", opt.train_data, "Training data for cooccurrence matrix.")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option(
      "--allowed-substrings", opt.allowed_substrings,
      "List of words accompanied with allowed substrings.")
      ->check(CLI::ExistingFile);

  app.add_option(
      "--fastext-output-pseudoinverse", opt.fasttext_output_pseudoinverse,
      "Pseudo-inverse of the fasttext output matrix")
      ->check(CLI::ExistingFile);

  app.add_option(
      "--fasttext-dim", opt.fasttext_dim,
      "Dimension of the fasttext embeddings.");

  app.add_option(
      "--epochs", opt.epochs, "Run for this number of iterations.");

  app.add_option(
      "--window-size", opt.window_size, "Window size.");

  app.add_option(
      "--output-directory", opt.output_directory, "Output directory.");

  app.add_option(
      "--segm-prefix", opt.segmentations_prefix,
      "Prefix for segmentations checkpoints.");

  app.add_option(
      "--emb-prefix", opt.embeddings_prefix,
      "Prefix for embedding checkpoints.");

  app.add_option(
      "--subw-prefix", opt.subwords_prefix,
      "Prefix for subword vocabularies.");

  app.add_option(
      "--unigram-prefix", opt.unigrams_prefix,
      "Prefix for unigram stats.");

  app.add_option(
      "--bigram-prefix", opt.bigrams_prefix,
      "Prefix for bigram stats.");
}

// step 3: fill subword-word cooccurrence matrix
void word_subword_cooccurrences(
    Eigen::MatrixXf& c_sub,
    const Embeddings& word_vocab,
    const Vocab& subword_vocab,
    const InverseAllowedSubstringMap& a_sub_inv,
    const std::vector<std::unordered_map<int, int>>& sparse_c_v) {

#pragma omp parallel for
  for(int i = 0; i < subword_vocab.size(); ++i) {

    std::string subword = subword_vocab[i];
    if(a_sub_inv.count(subword) == 0)
      continue;

    for(std::pair<std::string, float> wordscores : a_sub_inv.at(subword)) {

      if(!word_vocab.contains(wordscores.first))
        continue;

      for(std::pair<int, int> cooccurs : sparse_c_v.at(word_vocab[wordscores.first])) {
        int num = cooccurs.second;
        int j = cooccurs.first;

#pragma omp atomic
        c_sub(i, j) += num * wordscores.second;
      }
    }
  }
}

void sparse_cooccurrences(
    std::vector<std::unordered_map<int, int>>& sparse_c_v,
    std::vector<int>& word_frequencies,
    const Embeddings& word_vocab,
    const std::string& train_data,
    int window_size,
    bool compute_pseudoinverse_w,
    Eigen::MatrixXf& pinv) {

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

  if(compute_pseudoinverse_w) {
    std::cerr << "Computing pseudoinverse of W from embeddings and word counts"
              << std::endl;
    // c_v dim: [V,V]

    // smooth:
    c_v.array() += 0.00001f;
    // smooth, log & norm:
    Eigen::VectorXf sums = c_v.rowwise().sum();
    Eigen::MatrixXf normed = c_v.array().log().matrix().colwise()
                             - sums.array().log().matrix();

    // exact inverse (still the same dim)
    normed = normed.inverse();

    // matmul with embeddings (emb dim [V,E], product dim [V,E])
    pinv = normed * word_vocab.emb;
  }

}

void save_embedding_checkpoint(
    const fs::path& path,
    const Eigen::MatrixXf& embeddings) {
  std::ofstream ofs(path);
  ofs << embeddings << std::endl;
  ofs.close();
}

void save_strings(const fs::path& path,
                  const std::vector<std::string>& segments) {
  std::ofstream ofs(path);
  for(auto w: segments) {
    ofs << w << std::endl;
  }
  ofs.close();
}


int main(int argc, char* argv[]) {
  CLI::App app{"Train subword embeddings using a pseudoEM algorithm."};
  get_options(app);
  CLI11_PARSE(app, argc, argv);

  #ifndef SSEG_RELEASE_BUILD
  std::cerr
      << "\n\033[31m!! WARNING !!\033[0m You are likely running a debug build"
      << "\nFor best results, use cmake with -DCMAKE_BUILD_TYPE=Release\n\n";
  #endif

  // compute word cooccurrence matrix for data C_v (dim. V x V)
  // implemented in word_cooccurrence_matrix.cpp
  std::cerr << "Loading word embeddings: " << opt.embeddings_file << std::endl;
  Embeddings word_vocab(opt.embeddings_file);
  int word_count = word_vocab.size();

  Eigen::MatrixXf pinv(word_count, opt.fasttext_dim);
  std::cerr << "Populating word cooccurrence stats (" << word_count
            << " words)" << std::endl;
  std::vector<std::unordered_map<int, int>> sparse_c_v(word_count);
  std::vector<int> word_frequencies(word_count);
  sparse_cooccurrences(
      sparse_c_v, word_frequencies, word_vocab, opt.train_data,
      opt.window_size, opt.fasttext_output_pseudoinverse.empty(), pinv);
  std::cerr << sparse_c_v[10].size() << std::endl;

  if(!opt.fasttext_output_pseudoinverse.empty()) {
    std::cerr << "Loading pseudo-inverse of fasttext output matrix from "
              << opt.fasttext_output_pseudoinverse << std::endl;
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
  }

  std::cerr << "Loading list of allowed substrings from "
            << opt.allowed_substrings << std::endl;

  AllowedSubstringMap a_sub;
  InverseAllowedSubstringMap a_sub_inv;
  load_allowed_substrings(a_sub, a_sub_inv, opt.allowed_substrings);

  std::cerr << "Loading subword vocab." << std::endl;
  Vocab subword_vocab(std::views::keys(a_sub_inv), true);
  std::cerr << "Initial subword vocab size: " << subword_vocab.size()
            << std::endl;

  fs::path output_dir(opt.output_directory);

  // ====== here the algorithm begins
  for(int epoch = 0; epoch < opt.epochs; ++epoch) {
    std::cerr << "Epoch " << epoch << " begins." << std::endl;

    auto subw_path = output_dir / fs::path(opt.subwords_prefix
                                           + std::to_string(epoch));

    std::cerr << "Saving subword vocabulary to " << subw_path << std::endl;
    save_strings(subw_path, subword_vocab.index_to_word);

    std::cerr << "Calculating word-subword cooccurrence matrix. " << std::endl;
    Eigen::MatrixXf c_sub(subword_vocab.size(), word_count); // = a_sub * c_v;
    word_subword_cooccurrences(
        c_sub, word_vocab, subword_vocab, a_sub_inv, sparse_c_v);

    std::cerr << "Computing subword embeddings" << std::endl;

    c_sub.array() += 0.00001f;
    Eigen::VectorXf sums = c_sub.rowwise().sum();
    Eigen::MatrixXf normed = c_sub.array().log().matrix().colwise() - sums.array().log().matrix();
    Eigen::MatrixXf subword_embeddings =  normed * pinv;

    auto checkpoint_path = output_dir / fs::path(opt.embeddings_prefix
                                                 + std::to_string(epoch));
    std::cerr << "Saving checkpoint to " << checkpoint_path << std::endl;
    save_embedding_checkpoint(checkpoint_path, subword_embeddings);

    std::cerr << "Counting new subword-word cooccurrences." << std::endl;
    InverseAllowedSubstringMap a_sub_inv_next; // maps subword to pairs word and score

    // připravit novou matici A pomocí for cyklu níže:
    std::vector<std::string> segmented_vocab(word_count);
    std::vector<int> unigram_freqs(subword_vocab.size());
    std::vector<std::unordered_map<std::string, int>> bigram_freqs(subword_vocab.size());

#pragma omp parallel for
    for(int i = 0; i < word_count; ++i) {
      std::string word = word_vocab[i];
      int w_freq = word_frequencies[i];
      std::vector<std::string> segm;

      viterbi_decode(segm, word_vocab, subword_vocab, subword_embeddings, word);

      std::pair<std::string, float> word_pair{word, 1.0};

      std::string sep = "";
      std::ostringstream oss;
      int prev_sub_index = subword_vocab[bow];
      unigram_freqs[prev_sub_index] += w_freq;

      for(auto it = segm.begin(); it != segm.end(); ++it) {
        std::string subword = *it;
        oss << sep << subword;
        sep = " ";

#pragma omp critical
        {
          int index = subword_vocab[subword];
          unigram_freqs[index] += w_freq;

          if(bigram_freqs[prev_sub_index].count(subword) == 0)
            bigram_freqs[prev_sub_index].insert({subword, 0});
          bigram_freqs[prev_sub_index].at(subword) += w_freq;
          prev_sub_index = index;

          if(a_sub_inv_next.count(subword) == 0)
            a_sub_inv_next.insert({subword, std::vector<std::pair<std::string, float>>{word_pair}});
          else
            a_sub_inv_next.at(subword).push_back(word_pair);
        }

      }
      segmented_vocab[i] = oss.str();
    } // word

    auto segmentations_path = output_dir / fs::path(opt.segmentations_prefix
                                                    + std::to_string(epoch));

    std::cerr << "Saving segmentations to " << segmentations_path << std::endl;
    save_strings(segmentations_path, segmented_vocab);

    // save unigram and bigram stats
    auto unigrams_path = output_dir / fs::path(opt.unigrams_prefix
                                               + std::to_string(epoch));

    auto bigrams_path = output_dir / fs::path(opt.bigrams_prefix
                                              + std::to_string(epoch));

    std::ofstream uniofs(unigrams_path);
    std::ofstream biofs(bigrams_path);

    for(int i = 0; i < subword_vocab.size(); ++i) {
      std::string unigram = subword_vocab[i];
      uniofs << unigram << "\t" << unigram_freqs[i] << std::endl;

      for(auto pair : bigram_freqs[i]) {
        biofs << unigram << "\t" << pair.first << "\t" << pair.second
              << std::endl;
      }
    }

    uniofs.close();
    biofs.close();


    // create new subword vocabulary -> filter subwords which are not used in
    // any segmentation
    auto filter_unused = [&a_sub_inv_next](std::string subword) {
      return a_sub_inv_next.count(subword) > 0;
    };

    auto new_subwords = subword_vocab.index_to_word | std::views::filter(filter_unused);

    // UPDATE
    subword_vocab = Vocab(new_subwords, true);
    std::cerr << "Updated subword vocabulary size: " << subword_vocab.size() << std::endl;

    a_sub_inv = a_sub_inv_next;

  } // epoch

  return 0;
}
