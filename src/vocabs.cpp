#include "vocabs.h"

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>

void get_word_to_index(
    std::unordered_map<std::string,int> &word_to_index,
    const std::string filename) {

  std::ifstream file(filename);
  int i = 0;
  for(std::string line; std::getline(file, line); ++i) {
    if(word_to_index.count(line) != 0) {
      std::cerr << "Duplicate entry in vocabulary: '"
                << line << "' on line " << i << std::endl;
      std::abort();
    }
    word_to_index.insert({line, i});
  }

  file.close();
}

// taken from https://stackoverflow.com/a/21068014/1173884
void removeRow(Eigen::MatrixXf& matrix, unsigned int rowToRemove) {
  unsigned int numRows = matrix.rows()-1;
  unsigned int numCols = matrix.cols();

  if(rowToRemove < numRows)
    matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols)
        = matrix.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);

  matrix.conservativeResize(numRows,numCols);
}


void Vocab::insert(std::ranges::input_range auto&& words) {
  int i = index_to_word.size();
  for(std::string w: words) {

    if(word_to_index.count(w) != 0) {
      std::cerr << "Duplicate entry in vocabulary: " << w << std::endl;
      std::abort();
    }

    index_to_word.push_back(w);
    word_to_index.insert({w, i});
    i++;
  }
}


Vocab::Vocab(const std::string& filename) {
  std::ifstream file(filename);
  int i = 0;
  for(std::string line; std::getline(file, line); ++i) {
    if(word_to_index.count(w) != 0) {
      std::cerr << "Duplicate entry in vocabulary: " << w
                << ", line: " << i << std::endl;
      std::abort();
    }

    word_to_index.insert({line, i});
    index_to_word.push_back(line);
  }
}

Vocab::Vocab(std::ranges::input_range auto&& words, bool add_bow_eow) {

  if(add_bow_eow) {
    index_to_word.push_back(bow);
    index_to_word.push_back(eow);

    word_to_index.insert({bow, 0});
    word_to_index.insert({eow, 1});
  }

  insert(words);
}


Embeddings::Embeddings(const std::string& filename) {
  std::ifstream embedding_fh(filename);

  std::string firstline;
  std::getline(embedding_fh, firstline);
  std::stringstream ss_firstline(firstline);
  ss_firstline >> word_count;
  ss_firstline >> embedding_dim;

  emb.resize(word_count, embedding_dim);
  index_to_word.resize(word_count);

  int i = 0;
  for(std::string line; std::getline(embedding_fh, line); ++i) {
    std::stringstream ss(line);
    std::string word;
    ss >> word;

    if(word_to_index.count(line) != 0) {
      std::cerr << "Duplicate entry in vocabulary: '"
                << line << "' on line " << i << std::endl;
      std::abort();
    }

    word_to_index.insert({word, i});
    index_to_word[i] = word;

    for(int j = 0; j < embedding_dim; ++j) {
      ss >> emb(i, j);
    }
  }
}

WordClasses::WordClasses(const std::string& filename) {

}
