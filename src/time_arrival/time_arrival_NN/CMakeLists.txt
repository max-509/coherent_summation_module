cmake_minimum_required(VERSION 3.15)
project(TimeArrivalNNLib)

set(TARGET ${PROJECT_NAME})
set(SOURCES "TimeArrivalNNBase.cpp" "TimeArrivalNNFrozen.cpp" "TimeArrivalNNException.cpp" "TimeArrivalNNModel.cpp")
set(HEADERS "TimeArrivalNNBase.h" "TimeArrivalNNFrozen.h" "TimeArrivalNNException.h" "TimeArrivalNNModel.h")
add_library(${TARGET} ${HEADERS} ${SOURCES})

########################################### Внешняя загрузка, пока не работает ###########################################
#include(CheckIncludeFile)
#include(CheckIncludeFileCXX)
#include(CheckIncludeFiles)
#include(ExternalProject)
#
#set(TENSOR_FLOW_PREFIX tensor_flow)
#if(UNIX)
#    set(TENSOR_FLOW_URL https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz)
#elseif(WIN32)
#    set(TENSOR_FLOW_URL https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-windows-x86_64-2.4.0.zip)
#elseif (MACOS)
#    set(TENSOR_FLOW_URL https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-2.4.0.tar.gz)
#endif ()
#
#ExternalProject_Add(${TENSOR_FLOW_PREFIX}
#        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${TENSOR_FLOW_PREFIX}
#        URL https://github.com/neargye-forks/tensorflow/releases/download/v1.15.0/libtensorflow-gpu-windows-x86_64-1.15.0.zip
#        BUILD_IN_SOURCE 1
#        CONFIGURE_COMMAND ""
#        BUILD_COMMAND ""
#        INSTALL_COMMAND "" ${CMAKE_COMMAND} -E copy_directory ./include ${CMAKE_CURRENT_SOURCE_DIR} && ${CMAKE_COMMAND} -E copy_directory ./lib ${CMAKE_CURRENT_SOURCE_DIR}
#        )
#
#ExternalProject_Get_Property(${TENSOR_FLOW_PREFIX} SOURCE_DIR)
#message(STATUS "Source directory of ${TENSOR_FLOW_PREFIX} ${SOURCE_DIR}")

#add_dependencies(${TARGET} ${TENSOR_FLOW_PREFIX})
########################################### Внешняя загрузка, пока не работает ###########################################

#set(TensorFlow_BUILD_DIR ${Tensorflow_LIBRARY_DIRS})
#list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow)
#find_package(TensorFlow)

#target_link_directories(${TARGET} PRIVATE ${Tensorflow_LIBRARY_DIRS})
#target_link_libraries(${TARGET} PRIVATE tensorflow)

target_link_libraries(${TARGET} PRIVATE tensorflow)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})