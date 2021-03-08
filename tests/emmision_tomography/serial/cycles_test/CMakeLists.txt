cmake_minimum_required(VERSION 3.10)
project(cycles_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++11 -march=native -fno-math-errno -Wno-ignored-attributes -g -fno-omit-frame-pointer -ftree-vectorize -fopt-info=optimization_report.txt")
set(TARGET run_test)
set(SOURCES test.cpp)
add_subdirectory(utils)
include_directories(native_src)
include_directories(cycles_reverses_src)

add_executable(${TARGET} ${SOURCES})
target_link_libraries(${TARGET} PRIVATE test_data_generator_lib)