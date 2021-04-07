project(kirchhoff_migration_parallel_by_points_with_blocks_tests)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -Wall -std=c++11 -fno-math-errno -Wno-ignored-attributes -g -fno-omit-frame-pointer -fopt-info-all=../optimization_report.txt -fdump-ipa-all")

list(APPEND VECT_INSTRUCTIONS "AVX512" "AVX2" "SSE4_2" "NO_VECT")

foreach(V_EXT ${VECT_INSTRUCTIONS})
    set(TARGET_2D "${PROJECT_NAME}_run_test2D_${V_EXT}")
    set(TARGET_3D "${PROJECT_NAME}_run_test3D_${V_EXT}")

    add_executable(${TARGET_2D} test2D.cpp)
    add_executable(${TARGET_3D} test3D.cpp)

    target_link_libraries(${TARGET_2D} PRIVATE test_data_generator_lib)
    target_link_libraries(${TARGET_3D} PRIVATE test_data_generator_lib)

    target_include_directories(${TARGET_2D} PRIVATE auto_vect)
    target_include_directories(${TARGET_2D} PRIVATE src)

    target_include_directories(${TARGET_3D} PRIVATE auto_vect)
    target_include_directories(${TARGET_3D} PRIVATE src)

    target_compile_options(${TARGET_2D}
            PRIVATE $<IF:$<STREQUAL:${V_EXT},AVX512>,-mavx512f -ftree-vectorize,
                        $<IF:$<STREQUAL:${V_EXT},AVX2>,-mavx2 -mfma -ftree-vectorize,
                            $<IF:$<STREQUAL:${V_EXT},SSE4_2>,-msse4.2 -ftree-vectorize,
                                $<IF:$<STREQUAL:${V_EXT},NO_VECT>,-mno-sse4.2 -fno-tree-vectorize,-march=native>
                            >
                        >
                    >
            )
    target_compile_options(${TARGET_3D}
            PRIVATE $<IF:$<STREQUAL:${V_EXT},AVX512>,-mavx512f -ftree-vectorize,
                        $<IF:$<STREQUAL:${V_EXT},AVX2>,-mavx2 -mfma -ftree-vectorize,
                            $<IF:$<STREQUAL:${V_EXT},SSE4_2>,-msse4.2 -ftree-vectorize,
                                $<IF:$<STREQUAL:${V_EXT},NO_VECT>,-mno-sse4.2 -fno-tree-vectorize,-march=native>
                            >
                        >
                    >
            )

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_run_script_${V_EXT}.sh "#!${BASH} \n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_run_script_${V_EXT}.sh "for i in $(seq 1 16); do \n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_run_script_${V_EXT}.sh "\texport OMP_NUM_THREADS=$i \n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_run_script_${V_EXT}.sh "\tmake clean \n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_run_script_${V_EXT}.sh "\tmake \n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_run_script_${V_EXT}.sh "\t./${TARGET_2D} \n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_run_script_${V_EXT}.sh "\t./${TARGET_3D} \n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_run_script_${V_EXT}.sh "done \n")

endforeach(V_EXT)