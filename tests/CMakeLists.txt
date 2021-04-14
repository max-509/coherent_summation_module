project(coherent_summation_tests)
cmake_minimum_required(VERSION 3.10)

include(FindOpenMP)

set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

link_libraries(OpenMP::OpenMP_CXX)

include_directories(utils)

add_subdirectory(utils)
add_subdirectory(emmision_tomography)
add_subdirectory(kirchhoff_migration)