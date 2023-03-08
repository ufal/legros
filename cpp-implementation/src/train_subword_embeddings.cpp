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

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "CLI11.hpp"

#include "vocabs.h"
#include "substring_stats.h"
#include "cosine_viterbi.h"

struct opt {
  std::string subword_vocab;
  std::string embeddings_file;
  std::string allowed_substrings;
  std::string fasttext_output_pseudoinverse;

  std::string output;
  std::string train_data;

  std::string segmentations_prefix = "segmentations.";
  std::string embeddings_prefix = "subword_embeddings.";
  std::string subwords_prefix = "subwords.";

  int fasttext_dim = 200;
  int window_size = 3;
  int max_subword = 10;
  int epochs = 1;
} opt;

void get_options(CLI::App& app) {
  app.add_option(
      "embeddings_file", opt.embeddings_file, "Word embeddings.")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option(
      "subword_vocabulary", opt.subword_vocab,
      "Subword vocabulary, subword per line.")
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
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option(
      "--fasttext-dim", opt.fasttext_dim,
      "Dimension of the fasttext embeddings.");

  app.add_option(
      "--epochs", opt.epochs, "Run for this number of iterations.");

  app.add_option(
      "--window-size", opt.window_size, "Window size.");

  app.add_option(
      "--max-subword", opt.max_subword, "Maximum subword length.");

  app.add_option(
      "--segm-prefix", opt.segmentations_prefix, "Prefix for segmentations checkpoints.");

  app.add_option(
      "--emb-prefix", opt.embeddings_prefix, "Prefix for embedding checkpoints.");

  app.add_option(
      "--subw-prefix", opt.subwords_prefix, "Prefix for subword vocabularies.");
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

void save_embedding_checkpoint(
    const std::string& path,
    const Eigen::MatrixXf& embeddings) {
  std::ofstream ofs(path);
  ofs << embeddings << std::endl;
  ofs.close();
}

void save_strings(const std::string& path,
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

  // compute word cooccurrence matrix for data C_v (dim. V x V)
  // implemented in word_cooccurrence_matrix.cpp
  std::cerr << "Loading word embeddings: " << opt.embeddings_file << std::endl;
  Embeddings word_vocab(opt.embeddings_file);

  std::string test_word = "včelař";
  std::cerr << "Index of '" << test_word << "': " << word_vocab[test_word]
            << std::endl;

  std::cerr << "Loading subword vocab: " << opt.subword_vocab << std::endl;
  Vocab subword_vocab(opt.subword_vocab);
  std::cerr << "Initial subword vocab size: " << subword_vocab.size() << std::endl;

  int word_count = word_vocab.size();

  CooccurrenceMatrix c_v(word_count, word_count);
  std::cerr << "Populating word cooccurrence stats (" << word_count
            << " words)" << std::endl;

  // --> this thing takes too long after traverse through data
  populate_word_stats<CooccurrenceMatrix>(
      c_v, word_vocab, opt.train_data, opt.window_size);

  std::cerr << "Done, here are some stats:" << std::endl;
  std::cerr << c_v.topLeftCorner<5,5>() << std::endl;


  std::cerr << "Converting to sparse structure" << std::endl;

  //std::vector<std::tuple<int, int, int>> sparse_c_v;
  std::vector<std::unordered_map<int, int>> sparse_c_v(word_count);

#pragma omp parallel for
  for(int i = 0; i < word_count; ++i) {
    std::vector<std::pair<int, int>> row_pairs;
    for(int j = 0; j < word_count; ++j) {
      int freq = c_v(i, j);
      if(freq > 0) {
        row_pairs.push_back({j, freq});
      }
    }

#pragma omp critical
    sparse_c_v[i] = std::unordered_map<int, int>(row_pairs.begin(), row_pairs.end());

  }

  std::cerr << sparse_c_v[10].size() << std::endl;

  std::cerr << "Loading list of allowed substrings from " << opt.allowed_substrings << std::endl;
  AllowedSubstringMap a_sub;
  InverseAllowedSubstringMap a_sub_inv;
  load_allowed_substrings(a_sub, a_sub_inv, opt.allowed_substrings);

  std::cerr << "Loading pseudo-inverse of fasttext output matrix from " << opt.fasttext_output_pseudoinverse << std::endl;
  std::ifstream fasttext_fh(opt.fasttext_output_pseudoinverse);
  Eigen::MatrixXf pinv(word_count, opt.fasttext_dim);

  int i = 0;
  for(std::string line; std::getline(fasttext_fh, line); ++i) {
    std::stringstream linestream(line);

    for(int j = 0; j < word_count; ++j) {
      linestream >> pinv(j, i);
    }
  }

  std::cerr << pinv.block(0,0,5,5) << std::endl;
  std::cerr << "Pseudo-inverse dim: " << pinv.rows() << " x " << pinv.cols() << std::endl;


  // ====== here the algorithm begins
  for(int epoch = 0; epoch < opt.epochs; ++epoch) {
    std::cerr << "Epoch " << epoch << " begins." << std::endl;

    std::cerr << "Calculating word-subword cooccurrence matrix. " << std::endl;
    Eigen::MatrixXf c_sub(subword_vocab.size(), word_count); // = a_sub * c_v;
    word_subword_cooccurrences(
        c_sub, word_vocab, subword_vocab, a_sub_inv, sparse_c_v);

    std::cerr << "Computing subword embeddings" << std::endl;

    c_sub.array() += 0.00001f;
    Eigen::VectorXf sums = c_sub.rowwise().sum();
    Eigen::MatrixXf normed = c_sub.array().log().matrix().colwise() - sums.array().log().matrix();
    Eigen::MatrixXf subword_embeddings =  normed * pinv;

    std::string checkpoint_path = opt.embeddings_prefix + std::to_string(epoch);
    std::cerr << "Saving checkpoint to " << checkpoint_path << std::endl;
    save_embedding_checkpoint(checkpoint_path, subword_embeddings);

    std::cerr << "Counting new subword-word cooccurrences." << std::endl;
    InverseAllowedSubstringMap a_sub_inv_next; // maps subword to pairs word and score

    // připravit novou matici A pomocí for cyklu níže:
    std::vector<std::string> segmented_vocab(word_count);
#pragma omp parallel for
    for(i = 0; i < word_count; ++i) {
      std::string word = word_vocab[i];
      std::vector<std::string> segm;

      viterbi_decode(segm, word_vocab, subword_vocab, subword_embeddings, word);

      std::pair<std::string, float> word_pair{word, 1.0};

      std::string sep = "";
      std::ostringstream oss;

      for(auto it = segm.begin(); it != segm.end(); ++it) {
        std::string subword = *it;
        oss << sep << subword;
        sep = " ";

#pragma omp critical
        {
          if(a_sub_inv_next.count(subword) == 0)
            a_sub_inv_next.insert({subword, std::vector<std::pair<std::string, float>>{word_pair}});
          else
            a_sub_inv_next.at(subword).push_back(word_pair);
        }

      }
      segmented_vocab[i] = oss.str();
    } // word

    std::string segmentations_path = opt.segmentations_prefix + std::to_string(epoch);
    std::cerr << "Saving segmentations to " << segmentations_path << std::endl;
    save_strings(segmentations_path, segmented_vocab);

    // create new subword vocabulary -> filter subwords which are not used in
    // any segmentation
    auto filter_unused = [&a_sub_inv_next](std::string subword) {
      return a_sub_inv_next.count(subword) > 0;
    };

    auto new_subwords = subword_vocab.index_to_word | std::views::filter(filter_unused);

    // UPDATE
    subword_vocab = Vocab(new_subwords, true);
    std::cerr << "Updated subword vocabulary size: " << subword_vocab.size() << std::endl;
    std::string subwords_path = opt.subwords_prefix + std::to_string(epoch);
    std::cerr << "Saving subword vocabulary to " << subwords_path << std::endl;
    save_strings(subwords_path, subword_vocab.index_to_word);

    a_sub_inv = a_sub_inv_next;

  } // epoch

  return 0;
}
