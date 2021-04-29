project(CoherentSummationModule LANGUAGES CXX)
cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(COMPILING_CXX_MSVC_FLAGS "-openmp:experimental" "/Ox" "/Ob2" "/DNDEBUG" "/Wall")
set(COMPILING_CXX_CLANG_GNU_FLAGS "-O3" "-lm" "-march=native" "-DNDEBUG" "-fno-math-errno" "-Wall" "-g" "-fno-omit-frame-pointer" "-ftree-vectorize" "-fPIC")
set(COMPILING_CXX_ICC_WINDOWS_FLAGS "/fast" "/Qfma" "/Oy-" "/Ox" "/Qpar" "/Wall" "/Qvec")
set(COMPILING_CXX_ICC_UNIX_FLAGS "-debug all" "-fast" "-fma" "-fno-math-errno" "-fno-omit-frame-pointer" "-g" "-parallel" "-Wall" "-vec")

include(FindOpenMP)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
exec_program(${Python_EXECUTABLE}
	ARGS "-c \"import pybind11; print(pybind11.get_cmake_dir())\""
	OUTPUT_VARIABLE pybind11_DIR
	RETURN_VALUE pybind11_NOT_FOUND)
if(Pybind11_NOT_FOUND)
	message(FATAL_ERROR "pybind11 cmake dir not found")
endif()
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

include_directories(coherent_summation_impl)
include_directories(utils)
include_directories(${Python_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})
message(STATUS "${Tensorflow_LIBRARY_DIRS}")
link_libraries(${pybind11_LIBRARIES})
link_directories("${Tensorflow_LIBRARY_DIRS}")

add_subdirectory(time_arrival)
add_subdirectory(utils)

set(COH_SUM_TARGET ${PROJECT_NAME})
set(SOURCES module.cpp CoherentSummation.cpp CoherentSummationANN.cpp CoherentSummationTableFile.cpp CoherentSummationTableArray.cpp)
set(HEADERS CoherentSummation.h CoherentSummationANN.h CoherentSummationTableFile.h CoherentSummationTableArray.h)

list(APPEND COMPILE_OPTS "${OpenMP_CXX_FLAGS}")

add_compile_options(${COMPILE_OPTS} 
		"$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:${COMPILING_CXX_MSVC_FLAGS}>"
		"$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:${COMPILING_CXX_CLANG_GNU_FLAGS}>"
		"$<$<AND:$<COMPILE_LANG_AND_ID:CXX,Intel>,$<PLATFORM_ID:WIN32>>:${COMPILING_CXX_ICC_WINDOWS_FLAGS}>"
		"$<$<AND:$<COMPILE_LANG_AND_ID:CXX,Intel>,$<PLATFORM_ID:UNIX>>:${COMPILING_CXX_ICC_UNIX_FLAGS}>")

pybind11_add_module(${COH_SUM_TARGET} ${SOURCES} ${HEADERS})

target_include_directories(${COH_SUM_TARGET} PRIVATE coherent_summation_impl/emission_tomography)
target_include_directories(${COH_SUM_TARGET} PRIVATE coherent_summation_impl/kirchhoff_migration)
target_link_libraries(${COH_SUM_TARGET} PRIVATE TimeArrivalLib)
target_link_libraries(${COH_SUM_TARGET} PRIVATE UtilsLib)

set_target_properties(${COH_SUM_TARGET}
	PROPERTIES PREFIX ""
	OUTPUT_NAME "${PROJECT_NAME}"
	LINKER_LANGUAGE CXX)
