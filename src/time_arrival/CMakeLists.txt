cmake_minimum_required(VERSION 3.15)
project(TimeArrivalLib)

set(CMAKE_POSITION_INDEPENDENT_CODE True)

add_subdirectory(time_arrival_NN)
add_subdirectory(time_arrival_table_file)

set(TARGET ${PROJECT_NAME})
set(SOURCES TimeArrivalBase.cpp TimeArrivalWrapperNN.cpp TimeArrivalTimesTableArray.cpp TimeArrivalTimesTableFile.cpp)
set(HEADERS TimeArrivalBase.h TimeArrivalWrapperNN.h TimeArrivalTimesTableArray.h TimeArrivalTimesTableFile.h)
add_library(${TARGET} STATIC ${HEADERS} ${SOURCES})

target_link_libraries(${TARGET} PRIVATE TimeArrivalNNLib)
target_link_libraries(${TARGET} PRIVATE csv_header)
target_link_libraries(${TARGET} PRIVATE csv)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})