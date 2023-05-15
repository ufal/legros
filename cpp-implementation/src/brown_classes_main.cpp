#include <iostream>
#include <iomanip>
#include "brown_classes.h"

int main(int argc, char* argv[]) {

  brown_classes classes("TEXTEN1.txt", 10, 8000);
  std::cerr << std::setprecision(10);

  for(int i = classes.size(); i > 15; i--) {
    merge_triplet best_merge = classes.find_best_merge();

    std::cerr << " | k = " << classes.size()
              << " | MI = " << classes.mutual_information()
              << " | merge = " << std::get<0>(best_merge)
              << " + " << std::get<1>(best_merge)
              << " | loss = " << std::get<2>(best_merge)
              << std::endl;

    classes.merge_classes(std::get<0>(best_merge), std::get<1>(best_merge));
  }

  std::cerr << "done, here are the 15 classes" << std::endl;

  for(int i = 0; i < classes.size(); i++) {
    std::cerr << i << ":";
    auto cls = classes.get_class(i);
    for(auto w: cls) {
      std::cerr << " " << w;
    }
    std::cerr << std::endl;
  }

  return 0;
}
