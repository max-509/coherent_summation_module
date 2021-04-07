project(kirchhoff_migration_prof)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -Wall -std=c++11 -march=native -fno-math-errno -Wno-ignored-attributes -g -fno-omit-frame-pointer -ftree-vectorize -fopt-info-all=../optimization_report.txt -fdump-ipa-all")

link_libraries(test_data_generator_lib)

set(NATIVE_TARGET ${PROJECT_NAME}_native)
set(NATIVE_SRC native/test_native.cpp)

add_executable(${NATIVE_TARGET} ${NATIVE_SRC})
target_include_directories(${NATIVE_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/kirchhoff_migration/serial/reverse_cycles/native)

set(OPTIMAL_SERIAL_TARGET ${PROJECT_NAME}_optimal_serial)
set(OPTIMAL_SERIAL_SRC optimal_serial/test_optimal_serial.cpp)

set(ENV{OMP_NUM_THREADS} 16)
add_executable(${OPTIMAL_SERIAL_TARGET} ${OPTIMAL_SERIAL_SRC})
target_include_directories(${OPTIMAL_SERIAL_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/kirchhoff_migration/serial/vectorization/manual_vect)
target_compile_options(${OPTIMAL_SERIAL_TARGET} PRIVATE -mfma)

#set(OPTIMAL_PARALLEL_TARGET ${PROJECT_NAME}_optimal_parallel)
#set(OPTIMAL_PARALLEL_SRC optimal_parallel/test_optimal_parallel.cpp)

#set(ENV{OMP_NUM_THREADS} 16)
#add_executable(${OPTIMAL_PARALLEL_TARGET} ${OPTIMAL_PARALLEL_SRC})
#target_include_directories(${OPTIMAL_PARALLEL_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/kirchhoff_migration/parallel/by_sources_samples_test/with_vectorization)
#target_compile_options(${OPTIMAL_PARALLEL_TARGET} PRIVATE -mfma)