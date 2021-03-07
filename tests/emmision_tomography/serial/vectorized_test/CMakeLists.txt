cmake_minimum_required(VERSION 3.10)
project(vectorizaed_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++11 -march=native -fno-math-errno -Wno-ignored-attributes -fopenmp")
if (AVX512) 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
elseif(AVX2)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
elseif(SSE2)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
endif()

set(TARGET_NO_VECT no_vect_run_test)
set(SOURCES_NO_VECT test_no_vect.cpp)
set(TARGET_VECT vect_run_test)
set(SOURCES_VECT test_vect.cpp)

add_subdirectory(utils)

add_executable(${TARGET_VECT} ${SOURCES_VECT})
add_executable(${TARGET_NO_VECT} ${SOURCES_NO_VECT})

target_link_libraries(${TARGET_VECT} PRIVATE test_data_generator_lib)
target_include_directories(${TARGET_VECT} PRIVATE with_vectorization)

target_link_libraries(${TARGET_NO_VECT} PRIVATE test_data_generator_lib)
target_include_directories(${TARGET_NO_VECT} PRIVATE without_vectorization)