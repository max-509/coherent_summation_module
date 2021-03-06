project(kirchhoff_migration_prof)

include(FindOpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -Wall -std=c++11 -march=native -fno-math-errno -Wno-ignored-attributes -g -fno-omit-frame-pointer -ftree-vectorize -fopt-info-all=../optimization_report.txt -fdump-ipa-all")

link_libraries(test_data_generator_lib)

set(NATIVE_TARGET ${PROJECT_NAME}_native)
set(NATIVE_SRC native/test_native.cpp)

add_executable(${NATIVE_TARGET} ${NATIVE_SRC})
target_include_directories(${NATIVE_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/kirchhoff_migration/serial/reverse_cycles/native)

set(REVERSE_CYCLES_TARGET ${PROJECT_NAME}_reverse_cycles)
set(REVERSE_CYCLES_SRC reverse_cycles/test_reverse_cycles.cpp)

add_executable(${REVERSE_CYCLES_TARGET} ${REVERSE_CYCLES_SRC})
target_include_directories(${REVERSE_CYCLES_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/kirchhoff_migration/serial/reverse_cycles/reverse_cycles)

set(BLOCKS_SERIAL_TARGET ${PROJECT_NAME}_blocks_serial)
set(BLOCKS_SERIAL_SRC blocks_serial/test_blocks_serial.cpp)

add_executable(${BLOCKS_SERIAL_TARGET} ${BLOCKS_SERIAL_SRC})
target_include_directories(${BLOCKS_SERIAL_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/kirchhoff_migration/serial/blocks/blocks_points_inner_loop)

set(BLOCKS_INNER_RECEIVERS_TARGET ${PROJECT_NAME}_blocks_inner_receivers)
set(BLOCKS_INNER_RECEIVERS_SRC blocks_inner_receivers/test_blocks_inner_receivers.cpp)

add_executable(${BLOCKS_INNER_RECEIVERS_TARGET} ${BLOCKS_INNER_RECEIVERS_SRC})
target_include_directories(${BLOCKS_INNER_RECEIVERS_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/kirchhoff_migration/serial/blocks/blocks_receivers_inner_loop)

set(STRIP_MINING_TARGET ${PROJECT_NAME}_strip_mining)
set(STRIP_MINING_SRC strip_mining/test_strip_mining.cpp)

add_executable(${STRIP_MINING_TARGET} ${STRIP_MINING_SRC})
target_include_directories(${STRIP_MINING_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/kirchhoff_migration/serial/blocks/blocks_points_inner_loop)

set(OPTIMAL_SERIAL_TARGET ${PROJECT_NAME}_optimal_serial)
set(OPTIMAL_SERIAL_SRC optimal_serial/test_optimal_serial.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma -mavx2")
add_executable(${OPTIMAL_SERIAL_TARGET} ${OPTIMAL_SERIAL_SRC})
target_include_directories(${OPTIMAL_SERIAL_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/kirchhoff_migration/serial/vectorization/manual_vect)

set(OPTIMAL_PARALLEL_TARGET ${PROJECT_NAME}_optimal_parallel)
set(OPTIMAL_PARALLEL_SRC optimal_parallel/test_optimal_parallel.cpp)

set(ENV{OMP_NUM_THREADS} 32)
add_executable(${OPTIMAL_PARALLEL_TARGET} ${OPTIMAL_PARALLEL_SRC})
target_include_directories(${OPTIMAL_PARALLEL_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/kirchhoff_migration/parallel/native_blocks/src)
target_compile_options(${OPTIMAL_PARALLEL_TARGET} PRIVATE -mfma)
