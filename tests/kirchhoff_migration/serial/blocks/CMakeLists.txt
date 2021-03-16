cmake_minimum_required(VERSION 3.10)
project(blocks_tests)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -Wall -std=c++11 -march=native -fno-math-errno -Wno-ignored-attributes -g -fno-omit-frame-pointer -ftree-vectorize -fopt-info-all=../optimization_report.txt -fdump-ipa-all")

set(TARGET_2D run_test2D)
set(TARGET_3D run_test3D)

add_subdirectory(utils)

link_libraries(test_data_generator_lib)
include_directories(reverse_cycles)
include_directories(blocks_receivers_inner_loop)
include_directories(blocks_points_inner_loop)
include_directories(horizontaly_layered_times_arrival)

add_executable(${TARGET_2D} test2D.cpp)
add_executable(${TARGET_3D} test3D.cpp)