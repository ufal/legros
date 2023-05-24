#pragma once

#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>


// https://gist.github.com/gokhansolak/d2abaefcf3e3b767f5bc7d81cfe0b36a
// method for calculating the pseudo-Inverse as recommended by Eigen developers
template<typename _Matrix_Type_>
_Matrix_Type_ pseudoInverse(
    const _Matrix_Type_ &a,
    double epsilon = std::numeric_limits<double>::epsilon()) {
  Eigen::JacobiSVD< _Matrix_Type_ > svd(a ,Eigen::ComputeThinU | Eigen::ComputeThinV);
  double tolerance = epsilon * std::max(a.cols(), a.rows()) *svd.singularValues().array().abs()(0);
  return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}


// template <typename Iter>
// std::iterator_traits<Iter>::value_type log_sum_exp(Iter begin, Iter end) {
//     using VT = std::iterator_traits<Iter>::value_type{};
//     if (begin==end) return VT{};

//     auto max_elem = *std::max_element(begin, end);

//     auto sum = std::accumulate(begin, end, VT{},
//         [max_elem](VT a, VT b) { return a + std::exp(b - max_elem); });

//     return max_elem + std::log(sum);
// }

float log_sum_exp(const Eigen::VectorXf& items) {
    if(items.size() == 0)
        return 0;

    auto max_elem = *std::max_element(items.begin(), items.end());

    float sum = std::accumulate(items.begin(), items.end(), float{},
        [max_elem](float a, float b) {return a + std::exp(b - max_elem);});

    return max_elem + std::log(sum);
}

float log_sum_exp(const std::vector<float>& items) {
    if(items.size() == 0)
        return 0;

    auto max_elem = *std::max_element(items.begin(), items.end());

    float sum = std::accumulate(items.begin(), items.end(), float{},
        [max_elem](float a, float b) {return a + std::exp(b - max_elem);});

    return max_elem + std::log(sum);
}

// template <typename T>
// float log_sum_exp(const std::unordered_map<T, float>& items) {
//     if(items.size() == 0)
//         return 0;

//     auto max_elem = *std::max_element(items.begin(), items.end(),
//         [](const std::pair<const T, float>& p1, const std::pair<const T, float>& p2) {
//             return p1.second < p2.second;
//         }
//     );

//     float sum = std::accumulate(items.begin(), items.end(), float{},
//         [max_elem](float a, const std::pair<const T, float>& b) {
//             return a + std::exp(b.second - max_elem.second);
//         }
//     );

//     return max_elem.second + std::log(sum);
// }
