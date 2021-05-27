cmake_minimum_required(VERSION 3.15)
project(UtilsLib)

set(CMAKE_POSITION_INDEPENDENT_CODE True)

set(TARGET ${PROJECT_NAME})

add_library(${TARGET} INTERFACE)

target_include_directories(${TARGET} INTERFACE .)