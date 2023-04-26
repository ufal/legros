#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <ranges>
#include <Eigen/Dense>

const std::string bow = "<w>";
const std::string eow = "</w>";

void get_word_to_index(
    std::unordered_map<std::string,int> &word_to_index,
    const std::string filename);


class Vocab {
 protected:
  Vocab() {}

 public:
  std::unordered_map<std::string, int> word_to_index;
  std::vector<std::string> index_to_word;

  int size() const { return word_to_index.size(); }
  bool contains(const std::string& word) const { return word_to_index.count(word) == 1; }

  int operator[](const std::string& word) const { return word_to_index.at(word); }
  const std::string& operator[](int index) const { return index_to_word[index]; }

  Vocab(const std::string& filename);

  Vocab(std::ranges::input_range auto&& words, bool unused_hack) {

    index_to_word.push_back(bow);
    index_to_word.push_back(eow);

    word_to_index.insert({bow, 0});
    word_to_index.insert({eow, 1});

    int i = 2;
    for(std::string w: words) {
      index_to_word.push_back(w);
      word_to_index.insert({w, i});
      ++i;
    }
  }

  virtual ~Vocab() {}
};


class Embeddings : public Vocab {
 private:
  int word_count;

 public:
  int embedding_dim;
  Eigen::MatrixXf emb;

  Embeddings(const std::string &filename);
  virtual ~Embeddings() {};
};
