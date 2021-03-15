cmake_minimum_required(VERSION 3.10)
project(test_data_generator)

find_library(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(TARGET test_data_generator_lib)
set(SOURCES test_data_generator.cpp)
set(HEADERS test_data_generator.h prof.h perf_wrapper.h)

add_library(${TARGET} SHARED ${SOURCES} ${HEADERS})
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})