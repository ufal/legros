#include "substring_stats.h"

#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>


void load_weighted_allowed_substrings(
    std::unordered_map<std::string, std::vector<std::pair<std::string, float>>>& allowed_substrings,
    const std::string &file) {

  // format: space-separated file, first field is the word, the rest are allowed substrings with the weights
  // example
  // word w 0.2 wo 0.1 word 0.4 rd 0.1
  std::ifstream ifs(file);
  for(std::string line; std::getline(ifs, line);) {
    std::istringstream iss(line);
    std::string word;
    iss >> word;

    std::vector<std::pair<std::string, float>> pairs;

    while(iss) {
      std::pair<std::string, float> pair;

      iss >> pair.first;
      iss >> pair.second;

      pairs.push_back(pair);
    }

    allowed_substrings.insert({word, pairs});
  }
}

void load_allowed_substrings(
    std::unordered_map<std::string,std::vector<std::pair<std::string, float>>> &allowed_substrings,
    const std::string &file) {

  // format: space-separated file, first field is the word, the rest are allowed substrings
  std::ifstream ifs(file);
  for(std::string line; std::getline(ifs, line);) {
    std::istringstream iss(line);
    std::string word;
    iss >> word;

    std::vector<std::pair<std::string, float>> pairs;

    while(iss) {
      std::pair<std::string, float> pair;

      iss >> pair.first;
      pair.second = 1.0;

      pairs.push_back(pair);
    }

    allowed_substrings.insert({word, pairs});
  }
}

void load_allowed_substrings( // THIS IS NOT WEIGHTED
    std::unordered_map<std::string,std::vector<std::pair<std::string, float>>>& allowed_substrings,
    std::unordered_map<std::string,std::vector<std::pair<std::string, float>>>& inverse_allowed_substrings,
    const std::string& file) {

  // format: space-separated file, first field is the word, the rest are allowed substrings
  std::ifstream ifs(file);
  for(std::string line; std::getline(ifs, line);) {
    std::istringstream iss(line);
    std::string word;
    iss >> word;

    std::vector<std::pair<std::string, float>> pairs;

    while(iss) {

      std::string subword;
      iss >> subword;
      float score = 1.0;

      if(inverse_allowed_substrings.count(subword) == 0) {
        inverse_allowed_substrings.insert({subword, std::vector<std::pair<std::string, float>>(
            {std::pair<std::string, float>(word, score)})});
      }
      else {
        inverse_allowed_substrings.at(subword).push_back(std::pair<std::string, float>(word, score));
      }

      pairs.push_back(std::pair<std::string, float>(subword, score));
    }

    allowed_substrings.insert({word, pairs});
  }
}

void load_allowed_substrings(
    Eigen::MatrixXi& allowed_substrings,
    const Vocab& word_vocab,
    const Vocab& subword_vocab,
    const std::string& file) {

  std::ifstream ifs(file);
  for(std::string line; std::getline(ifs, line);) {
    std::istringstream iss(line);
    std::string word;
    iss >> word;

    if(!word_vocab.contains(word)) {
      std::cerr << "ERR: Word '" << word << "' not in vocab" << std::endl;
      continue;
    }

    int word_index = word_vocab[word];

    std::string subword;
    while(iss >> subword) {

      if(!subword_vocab.contains(subword)) {
        std::cerr << "ERR: Subword '" << subword << "' of '"
                  << word << "' not in subword vocab" << std::endl;
        continue;
      }

      int subword_index = subword_vocab[subword];

      get_2d(allowed_substrings, subword_index, word_index) = 1;
    }
  }
}

void load_allowed_substrings_sparse(
    Eigen::SparseMatrix<int, Eigen::RowMajor>& allowed_substrings,
    const Vocab& word_vocab,
    const Vocab& subword_vocab,
    const std::string& file) {

  std::vector<Eigen::Triplet<int>> triplet_list;

  std::ifstream ifs(file);
  for(std::string line; std::getline(ifs, line);) {
    std::istringstream iss(line);
    std::string word;
    iss >> word;

    if(!word_vocab.contains(word)) {
      std::cerr << "ERR: Word '" << word << "' not in vocab" << std::endl;
      continue;
    }

    int word_index = word_vocab[word];

    std::string subword;
    while(iss >> subword) {
      if(!subword_vocab.contains(subword)) {
        std::cerr << "ERR: Subword '" << subword << "' of '"
                  << word << "' not in subword vocab" << std::endl;
        continue;
      }

      int subword_index = subword_vocab[subword];

      triplet_list.push_back(Eigen::Triplet<int>(subword_index, word_index, 1));
    }
  }

  // caution this is S x V as opposed to V x S in the other implementations
  // of this function
  allowed_substrings.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

/**
 * get_all_substrings
 *
 * For given word, get all its substrings (present in the subword_to_index map)
 * BE CAREFUL, for this is byte-based!!!
 */
void get_all_substrings(std::vector<std::pair<std::string, float>> &substrings,
                        const Vocab &subwords,
                        const std::string &word, int max_len) {

  for(int sub_len = 1; sub_len < std::min((int)word.size(), max_len) + 1; ++sub_len) {
    for(int i = 0; i < word.size() - sub_len + 1; ++i) {
      auto substr = word.substr(i, sub_len);
      if(!subwords.contains(substr))
        continue;

      substrings.push_back(std::pair<std::string, float>(substr, 1.0));
    }
  }
}
