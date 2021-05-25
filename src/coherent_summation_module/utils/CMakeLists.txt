cmake_minimum_required(VERSION 3.15)
project(UtilsLib)

set(CMAKE_POSITION_INDEPENDENT_CODE True)

set(TARGET ${PROJECT_NAME})
set(SOURCES utils.cpp)
set(HEADERS utils.h array1D.h array2D.h py_common.h)
add_library(${TARGET} ${SOURCES} ${HEADERS})

target_include_directories(${TARGET} PUBLIC .)