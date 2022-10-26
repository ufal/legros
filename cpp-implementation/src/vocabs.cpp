#include "vocabs.h"

#include<fstream>
#include<string>

void get_word_to_index(
    std::unordered_map<std::string,int> &word_to_index,
    const std::string filename) {

  std::ifstream file(filename);
  int i = 0;
  for(std::string line; std::getline(file, line); ++i) {
    word_to_index.insert({line, i});
  }

  file.close();
}