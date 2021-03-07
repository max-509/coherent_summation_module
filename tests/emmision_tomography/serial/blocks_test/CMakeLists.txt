cmake_minimum_required(VERSION 3.10)
project(blocks_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++11 -march=native -fno-math-errno -Wno-ignored-attributes")

set(TARGET run_test)
set(SOURCES test.cpp)

add_subdirectory(utils)
include_directories(with_blocks)
include_directories(without_blocks)

add_executable(${TARGET} ${SOURCES})
target_link_libraries(${TARGET} PRIVATE test_data_generator_lib)