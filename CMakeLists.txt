cmake_minimum_required(VERSION 3.10)

project(legros)

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

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_definitions(-DSSEG_RELEASE_BUILD)
endif()

include_directories(src)
include_directories(include)
include_directories(3rd_party/eigen-3.4.0)

add_executable(legros
  src/bigram_segment.cpp
  src/vocabs.cpp)

add_executable(legros-train
  src/train_subword_embeddings.cpp
  src/vocabs.cpp
  src/substring_stats.cpp
  src/cosine_viterbi.cpp)

add_subdirectory(src)
