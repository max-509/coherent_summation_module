cmake_minimum_required(VERSION 3.10)
project(parallel_by_samples_test)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++11 -march=native -fno-math-errno -Wno-ignored-attributes ${OpenMP_CXX_FLAGS}  -g -fno-omit-frame-pointer -ftree-vectorize -fopt-info-all=../optimization_report.txt -fdump-ipa-all")
if (AVX512) 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
elseif(AVX2)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
elseif(SSE2)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
endif()

set(TARGET run_test)
set(SOURCES test.cpp)

add_subdirectory(utils)
include_directories(src)

add_executable(${TARGET} ${SOURCES})
target_link_libraries(${TARGET} PRIVATE test_data_generator_lib Threads::Threads)