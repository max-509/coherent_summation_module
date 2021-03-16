cmake_minimum_required(VERSION 3.10)
project(test_data_generator)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++11 -g -fno-omit-frame-pointer")

set(TARGET test_data_generator_lib)
set(SOURCES test_data_generator2D.cpp test_data_generator3D.cpp)
set(HEADERS test_data_generator2D.h test_data_generator3D.h prof.h perf_wrapper.h)

add_library(${TARGET} SHARED ${SOURCES} ${HEADERS})
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})