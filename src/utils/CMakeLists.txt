cmake_minimum_required(VERSION 3.15)
project(UtilsLib)

set(TARGET ${PROJECT_NAME})
set(SOURCES utils.cpp)
set(HEADERS utils.h)
add_library(${TARGET} ${SOURCES} ${HEADERS})

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})