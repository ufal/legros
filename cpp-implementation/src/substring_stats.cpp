#include "substring_stats.h"

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <fstream>
#include <sstream>
#include "options.h"


void load_allowed_substrings(
        std::unordered_map<std::string,std::unordered_set<std::string>> &allowed_substrings,
        const std::string &file) {

    // format: space-separated file, first field is the word, the rest are allowed substrings
    std::ifstream ifs(file);
    for(std::string line; std::getline(ifs, line);) {
        std::istringstream iss(line);
        std::string word;
        iss >> word;

        std::unordered_set<std::string> tokens(
            (std::istream_iterator<std::string>(iss)),
             std::istream_iterator<std::string>());

        allowed_substrings.insert({word, tokens});
    }
}

/**
 * get_all_substrings
 *
 * For given word, get all its substrings (present in the subword_to_index map)
 * BE CAREFUL, for this is byte-based!!!
 */
void get_all_substrings(std::unordered_set<std::string> &substrings,
        const Vocab &subwords,
        const std::string &word, int max_len) {

    for(int sub_len = 1; sub_len < std::min((int)word.size(), max_len) + 1; ++sub_len) {
        for(int i = 0; i < word.size() - sub_len + 1; ++i) {
            auto substr = word.substr(i, sub_len);
            if(!subwords.contains(substr))
                continue;

            substrings.insert(substr);
        }
    }
}
