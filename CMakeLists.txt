cmake_minimum_required(VERSION 3.10)

project(NeuralPiece)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_library(MKL_CORE_LIBRARY mkl_core PATHS /opt/intel/mkl/lib/intel64)
find_library(MKL_INTEL_THREAD_LIBRARY mkl_intel_thread PATHS /opt/intel/mkl/lib/intel64)
find_library(MKL_INTEL_LP64_LIBRARY mkl_intel_lp64 PATHS /opt/intel/mkl/lib/intel64)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

add_subdirectory(3rd_party/eigen)

add_executable(brown_classes
  src/brown_classes_main.cpp
  src/brown_classes.cpp
  src/counters.cpp)

add_executable(bigram_segment
  src/bigram_segment.cpp
  src/counters.cpp
  src/vocabs.cpp)

add_executable(train_subword_embeddings
  src/train_subword_embeddings.cpp
  src/vocabs.cpp
  src/substring_stats.cpp
  src/cosine_viterbi.cpp)

add_executable(merge_contexts src/merge_contexts.cpp)

add_executable(subword_embeddings
  src/subword_embeddings.cpp
  src/substring_stats.cpp
  src/vocabs.cpp)

add_executable(test-eigen src/test-eigen.cpp)

add_executable(word_cooccurrence_matrix
  src/word_cooccurrence_matrix.cpp
  src/vocabs.cpp)

add_executable(forward_backward
  src/forward_backward.cpp
  src/vocabs.cpp
  src/unigram_model.cpp)

add_executable(test_unigram_model
  src/test_unigram_model.cpp
  src/vocabs.cpp
  src/unigram_model.cpp)


target_link_libraries (subword_embeddings Eigen3::Eigen)
target_link_libraries (brown_classes Eigen3::Eigen)
target_link_libraries (bigram_segment Eigen3::Eigen)
target_link_libraries (train_subword_embeddings Eigen3::Eigen)
target_link_libraries (word_cooccurrence_matrix Eigen3::Eigen)
target_link_libraries (forward_backward Eigen3::Eigen)
target_link_libraries (test_unigram_model Eigen3::Eigen)
target_link_libraries (test-eigen Eigen3::Eigen ${MKL_CORE_LIBRARY} ${MKL_INTEL_THREAD_LIBRARY} ${MKL_INTEL_LP64_LIBRARY})

include_directories(include)
include_directories(src)

add_subdirectory(src)
