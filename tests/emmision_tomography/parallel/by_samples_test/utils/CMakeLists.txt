project(test_data_generator)

set(TARGET test_data_generator_lib)
set(SOURCES test_data_generator.cpp)
set(HEADERS test_data_generator.h prof.h perf_wrapper.h)

add_library(${TARGET} SHARED ${SOURCES} ${HEADERS})
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})