project(emmision_tomography_cycles_test)

include(FindOpenMP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -Wall -std=c++11 -march=native -fno-math-errno -Wno-ignored-attributes -g -fno-omit-frame-pointer -ftree-vectorize -fopt-info-all=../optimization_report.txt -fdump-ipa-all")
set(TARGET "${PROJECT_NAME}_run_test")
set(SOURCES test.cpp)

add_executable(${TARGET} ${SOURCES})
target_link_libraries(${TARGET} PRIVATE test_data_generator_lib)
target_include_directories(${TARGET} PRIVATE native_src)
target_include_directories(${TARGET} PRIVATE cycles_reverses_src)