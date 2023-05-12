#include <iostream>
#include <iomanip>
#include "brown_classes.h"

int main(int argc, char* argv[]) {

  brown_classes classes("TEXTEN1.txt", 10, 8000);
  std::cerr << std::setprecision(10);
  std::cerr << classes.mutual_information() << std::endl;
  std::cerr << classes.merge_loss_cached("case", "subject") << std::endl;


  // for i in range(w.k, 15, -1):
  // (c1, c2), loss = w.find_best_merge()
  // print(w.k, "classes, best merge: ", c1, c2, ", loss", loss)
  // w.merge_classes(c1, c2)

  for(int i = classes.size(); i > 15; i--) {
    merge_triplet best_merge = classes.find_best_merge();
    std::cerr << "best merge: " << std::get<0>(best_merge) <<
        "+" << std::get<1>(best_merge) <<
        " (loss " << std::get<2>(best_merge) << ")" << std::endl;

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
