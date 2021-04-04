project(coherent_summation_tests)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

link_libraries(OpenMP::OpenMP_CXX)

include_directories(utils)

add_subdirectory(utils)
add_subdirectory(emmision_tomography)
add_subdirectory(kirchhoff_migration)