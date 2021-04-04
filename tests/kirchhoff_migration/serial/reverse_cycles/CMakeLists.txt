project(kirchhoff_migration_reverse_cycles_test)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -Wall -std=c++11 -march=native -fno-math-errno -Wno-ignored-attributes -g -fno-omit-frame-pointer -ftree-vectorize -fopt-info-all=../optimization_report.txt -fdump-ipa-all")

set(TARGET_2D "${PROJECT_NAME}_run_test2D")
set(TARGET_3D "${PROJECT_NAME}_run_test3D")

add_executable(${TARGET_2D} test2D.cpp)
add_executable(${TARGET_3D} test3D.cpp)

target_link_libraries(${TARGET_2D} PRIVATE test_data_generator_lib)
target_link_libraries(${TARGET_3D} PRIVATE test_data_generator_lib)

target_include_directories(${TARGET_2D} PRIVATE native)
target_include_directories(${TARGET_2D} PRIVATE reverse_cycles)

target_include_directories(${TARGET_3D} PRIVATE native)
target_include_directories(${TARGET_3D} PRIVATE reverse_cycles)