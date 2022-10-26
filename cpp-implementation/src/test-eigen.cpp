#include <Eigen/Dense>
#include <iostream>

int main(int argc, char* argv[]) {

  auto m1 = Eigen::MatrixXf::Random(25000, 197000);
  auto m2 = Eigen::MatrixXf::Random(197000, 200);
  Eigen::MatrixXf prod = m1 * m2;

  std::cout << prod.block(0, 0, 50, 50);

  return 0;
}