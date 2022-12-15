#pragma once

#include <Eigen/Dense>

class SubwordEmbeddingModel {

private:
  int word_count;
  Eigen::MatrixXi stats;

public:


SubwordEmbeddingModel();

};
