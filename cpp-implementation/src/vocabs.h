#pragma once

#include<unordered_map>
#include<string>

void get_word_to_index(
    std::unordered_map<std::string,int> &word_to_index,
    const std::string filename);

