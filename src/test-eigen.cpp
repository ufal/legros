#include <Eigen/Dense>
#include <iostream>

#define EIGEN_USE_MKL_ALL

int main(int argc, char* argv[]) {

  std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "."
            << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;
  std::cout << "Using backend: " << Eigen::SimdInstructionSetsInUse() << std::endl;

  auto m1 = Eigen::MatrixXf::Random(25000, 197000);
  auto m2 = Eigen::MatrixXf::Random(197000, 200);
  Eigen::MatrixXf prod = m1 * m2;

  std::cout << prod.block(0, 0, 50, 50);

  return 0;
}
