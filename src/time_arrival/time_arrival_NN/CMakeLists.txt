cmake_minimum_required(VERSION 3.15)
project(TimeArrivalNNLib)

set(TARGET ${PROJECT_NAME})
set(SOURCES "TimeArrivalNNBase.cpp" "TimeArrivalNNFrozen.cpp" "TimeArrivalNNException.cpp" "TimeArrivalNNModel.cpp")
set(HEADERS "TimeArrivalNNBase.h" "TimeArrivalNNFrozen.h" "TimeArrivalNNException.h" "TimeArrivalNNModel.h")
file(GLOB_RECURSE TENSORFLOW_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow)
add_library(${TARGET} STATIC ${HEADERS} ${SOURCES} ${TENSORFLOW_HEADERS})

set(LIBRARY_TARGET tensorflow)

message(Tensorflow_LIBRARY_DIRS=${Tensorflow_LIBRARY_DIRS})

add_library(${LIBRARY_TARGET} SHARED IMPORTED)

if (WIN32)
    set_target_properties(${LIBRARY_TARGET} PROPERTIES
            IMPORTED_LOCATION "${Tensorflow_LIBRARY_DIRS}/tensorflow.dll"
            IMPORTED_IMPLIB "${Tensorflow_LIBRARY_DIRS}/tensorflow.lib")
elseif(LINUX)
    set_target_properties(${LIBRARY_TARGET} PROPERTIES
            IMPORTED_LOCATION "${Tensorflow_LIBRARY_DIRS}/libtensorflow.so")
endif()

target_link_libraries(${TARGET} PRIVATE tensorflow)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})