cmake_minimum_required(VERSION 3.15)
project(TimeArrivalLib)

add_subdirectory(time_arrival_NN)
add_subdirectory(time_arrival_table_file)

set(TARGET ${PROJECT_NAME})
set(SOURCES TimeArrivalBase.cpp TimeArrivalWrapperNN.cpp TimeArrivalTimesTableArray.cpp TimeArrivalTimesTableFile.cpp)
set(HEADERS TimeArrivalBase.h TimeArrivalWrapperNN.h TimeArrivalTimesTableArray.h TimeArrivalTimesTableFile.h)
add_library(${TARGET} ${SOURCES} ${HEADERS})

########################################### Внешняя загрузка, пока не работает ###########################################
#include(CheckIncludeFile)
#include(CheckIncludeFileCXX)
#include(CheckIncludeFiles)
#include(ExternalProject)
#
#set(CSV_PARSER_PREFIX csv)
#set(CSV_PARSER_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git)
#
#ExternalProject_Add(${CSV_PARSER_PREFIX}
#        PREFIX ${CSV_PARSER_PREFIX}
#        GIT_REPOSITORY ${CSV_PARSER_REPOSITORY}
#        CMAKE_CACHE_ARGS -DCSV_CXX_STANDARD:STRING=17
#        INSTALL_COMMAND "")
#
#ExternalProject_Get_Property(${CSV_PARSER_PREFIX} SOURCE_DIR)
#ExternalProject_Get_Property(${CSV_PARSER_PREFIX} BINARY_DIR)
#
#message(STATUS "Source directory of ${CSV_PARSER_PREFIX} ${CMAKE_BINARY_DIR}")
#message(STATUS "Source directory of ${CSV_PARSER_PREFIX} ${SOURCE_DIR}")
#message(STATUS "Binary directory of ${CSV_PARSER_PREFIX} ${BINARY_DIR}")
#message(STATUS "Directory with library of ${CSV_PARSER_PREFIX} is ${BINARY_DIR}/include/internal/Release")

#add_dependencies(${TARGET} TimeArrivalNNLib)
#add_dependencies(${TARGET} csv)
#
#link_directories(${BINARY_DIR}/include/internal)
#link_directories(${BINARY_DIR}/include/internal/Release)
#link_libraries(csv)

#target_include_directories(${TARGET} PRIVATE ${SOURCE_DIR}/include)
#target_link_directories(${TARGET} PRIVATE ${BINARY_DIR}/include/internal)
#target_link_directories(${TARGET} PRIVATE ${BINARY_DIR}/include/internal/Release)
########################################### Внешняя загрузка, пока не работает ###########################################

target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/time_arrival_NN)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/time_arrival_table_file/include)
target_link_libraries(${TARGET} PRIVATE TimeArrivalNNLib)
target_link_libraries(${TARGET} PRIVATE csv)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})