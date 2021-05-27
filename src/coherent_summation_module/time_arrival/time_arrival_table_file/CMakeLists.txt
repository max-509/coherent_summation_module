cmake_minimum_required(VERSION 3.15)
project(csv)

set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message("Building CSV library using C++${CMAKE_CXX_STANDARD}")

# Defines CSV_HAS_CXX17 in compatibility.hpp
if (CMAKE_VERSION VERSION_LESS "3.12.0")
	add_definitions(-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD})
else()
	add_compile_definitions(CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD})
endif()

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads QUIET REQUIRED)

set(CSV_ROOT_DIR .)
set(CSV_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CSV_INCLUDE_DIR include)
set(CSV_SOURCE_DIR ${CSV_INCLUDE_DIR}/internal)
set(CSV_TEST_DIR ${CMAKE_CURRENT_LIST_DIR}/tests)

## Main Library
add_subdirectory(${CSV_SOURCE_DIR})

add_library(csv_header INTERFACE)

target_include_directories(csv_header INTERFACE ${CSV_INCLUDE_DIR})