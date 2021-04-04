cmake_minimum_required(VERSION 3.15)
project(TimeArrivalNNLib)

set(TARGET ${PROJECT_NAME})
set(SOURCES "TimeArrivalNNBase.cpp" "TimeArrivalNNFrozen.cpp" "TimeArrivalNNException.cpp" "TimeArrivalNNModel.cpp")
set(HEADERS "TimeArrivalNNBase.h" "TimeArrivalNNFrozen.h" "TimeArrivalNNException.h" "TimeArrivalNNModel.h")
add_library(${TARGET} ${SOURCES} ${HEADERS})

message(Tensorflow_INCLUDE_DIRS = ${Tensorflow_INCLUDE_DIRS})

target_link_directories(${TARGET} PRIVATE ${Tensorflow_LIBRARY_DIRS})
target_include_directories(${TARGET} PRIVATE ${Tensorflow_INCLUDE_DIRS})
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${TARGET} PRIVATE tensorflow)