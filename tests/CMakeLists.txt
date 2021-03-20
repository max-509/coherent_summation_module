
project(coherent_summation_tests)
cmake_minimum_required(VERSION 3.10)

find_package(OpenMP REQUIRED)

link_libraries(OpenMP::OpenMP_CXX)

include_directories(utils)

add_subdirectory(utils)
add_subdirectory(emmision_tomography)
add_subdirectory(kirchhoff_migration)