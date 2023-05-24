#include <iostream>
#include <iomanip>
#include "CLI11.hpp"
#include "brown_classes.h"

struct opt {
  std::string training_data_file;
  //std::string word_vocab_file;
  std::string output_file;
  int num_classes;
  int min_freq = 0;
  int limit = -1;
} opt;

void get_options(CLI::App& app) {
  app.add_option("input",
    opt.training_data_file, "Tokenized text.")
    ->required()
    ->check(CLI::ExistingFile);

  // app.add_option("word_vocabulary",
  //   opt.word_vocab_file, "Word vocabulary, word per line.")
  //   ->required()
  //   ->check(CLI::ExistingFile);

  app.add_option("output",
    opt.output_file, "Output file for the classes.")
    ->required()
    ->check(CLI::NonexistentPath);

  app.add_option("num_classes",
    opt.num_classes, "Finish merging after reaching this number of classes.");

  app.add_option("--min-freq",
    opt.min_freq, "Minimum word frequency.");

  app.add_option("--limit",
    opt.limit, "Only read this number of lines from the input.");
}


int main(int argc, char* argv[]) {
  CLI::App app{"Compute subword embeddings."};
  get_options(app);
  CLI11_PARSE(app, argc, argv);

  brown_classes classes(opt.training_data_file, opt.min_freq, opt.limit);
  std::cerr << std::setprecision(10);

  for(int i = classes.size(); i > opt.num_classes; i--) {
    merge_triplet best_merge = classes.find_best_merge();

    std::cerr << " | k = " << classes.size()
              << " | MI = " << classes.mutual_information()
              << " | merge = " << std::get<0>(best_merge)
              << " + " << std::get<1>(best_merge)
              << " | loss = " << std::get<2>(best_merge)
              << std::endl;

    classes.merge_classes(std::get<0>(best_merge), std::get<1>(best_merge));
  }

  std::cerr << "done, saving classes to " << opt.output_file << std::endl;

  std::ofstream ofs(opt.output_file);
  for(int i = 0; i < classes.size(); i++) {
    auto cls = classes.get_class(i);
    std::string sep = "";

    for(auto w: cls) {
      ofs << sep << w;
      sep = " ";
    }

    ofs << std::endl;
  }
  ofs.close();

  return 0;
}
