project(emission_tomography_prof)

include(FindOpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -Wall -std=c++11 -march=native -fno-math-errno -Wno-ignored-attributes -g -fno-omit-frame-pointer -ftree-vectorize -fopt-info-all=../optimization_report.txt -fdump-ipa-all")
# string(REPLACE " " ";" REPLACED_FLAGS ${CMAKE_CXX_FLAGS})

link_libraries(test_data_generator_lib)

set(NATIVE_TARGET ${PROJECT_NAME}_native)
set(NATIVE_SRC native/test_native.cpp)

add_executable(${NATIVE_TARGET} ${NATIVE_SRC})
target_include_directories(${NATIVE_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/emmision_tomography/serial/cycles_test/native_src)

set(REVERSE_TARGET ${PROJECT_NAME}_reverse)
set(REVERSE_SRC reverse_cycles/test_reverse.cpp)

add_executable(${REVERSE_TARGET} ${REVERSE_SRC})
target_include_directories(${REVERSE_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/emmision_tomography/serial/cycles_test/cycles_reverses_src)

set(OPTIMAL_SERIAL_TARGET ${PROJECT_NAME}_optimal_serial)
set(OPTIMAL_SERIAL_SRC optimal_serial/test_optimal_serial.cpp)

add_executable(${OPTIMAL_SERIAL_TARGET} ${OPTIMAL_SERIAL_SRC})
target_include_directories(${OPTIMAL_SERIAL_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/emmision_tomography/serial/vectorized_test/with_vectorization)
# target_compile_options(${OPTIMAL_SERIAL_TARGET} PRIVATE "-mfma" PRIVATE ${REPLACED_FLAGS})

set(OPTIMAL_PARALLEL_TARGET ${PROJECT_NAME}_optimal_parallel)
set(OPTIMAL_PARALLEL_SRC optimal_parallel/test_optimal_parallel.cpp)

add_executable(${OPTIMAL_PARALLEL_TARGET} ${OPTIMAL_PARALLEL_SRC})
target_include_directories(${OPTIMAL_PARALLEL_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/emmision_tomography/parallel/by_sources_samples_test/with_vectorization)
# target_compile_options(${OPTIMAL_PARALLEL_TARGET} PRIVATE "-mfma" PRIVATE ${REPLACED_FLAGS})
