project(emmision_tomography_parallel_by_sources_samples_test)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++11 -fno-math-errno -Wno-ignored-attributes ${OpenMP_CXX_FLAGS} -g -fno-omit-frame-pointer -fopt-info-all=../optimization_report.txt -fdump-ipa-all")
if (AVX512) 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -ftree-vectorize")
elseif(AVX2)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -ftree-vectorize")
elseif(SSE2)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -ftree-vectorize")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mno-sse2 -fno-tree-vectorize")
endif()

set(TARGET_NO_VECT "${PROJECT_NAME}_no_vect_run_test")
set(SOURCES_NO_VECT test_no_vect.cpp)
set(TARGET_VECT "${PROJECT_NAME}_vect_run_test")
set(SOURCES_VECT test_vect.cpp)

add_executable(${TARGET_VECT} ${SOURCES_VECT})
add_executable(${TARGET_NO_VECT} ${SOURCES_NO_VECT})

target_link_libraries(${TARGET_VECT} PRIVATE test_data_generator_lib Threads::Threads)
target_include_directories(${TARGET_VECT} PRIVATE with_vectorization)

target_link_libraries(${TARGET_NO_VECT} PRIVATE test_data_generator_lib Threads::Threads)
target_include_directories(${TARGET_NO_VECT} PRIVATE without_vectorization)