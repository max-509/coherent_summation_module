project(emmision_tomography_vectorized_test)

include(FindOpenMP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -Wall -std=c++11 -fno-math-errno -Wno-ignored-attributes -g -fno-omit-frame-pointer -fdump-ipa-all")

list(APPEND VECT_INSTRUCTIONS "AVX512" "AVX2" "SSE2" "NO_VECT")

foreach(V_EXT ${VECT_INSTRUCTIONS})
    set(TARGET_NO_VECT "${PROJECT_NAME}_run_test_no_vect_${V_EXT}")
    set(TARGET_VECT "${PROJECT_NAME}_run_test_vect_${V_EXT}")

    add_executable(${TARGET_NO_VECT} test_no_vect.cpp)
    add_executable(${TARGET_VECT} test_vect.cpp)

    target_link_libraries(${TARGET_NO_VECT} PRIVATE test_data_generator_lib)
    target_link_libraries(${TARGET_VECT} PRIVATE test_data_generator_lib)

    target_include_directories(${TARGET_NO_VECT} PRIVATE with_vectorization)
    target_include_directories(${TARGET_NO_VECT} PRIVATE without_vectorization)

    target_include_directories(${TARGET_VECT} PRIVATE with_vectorization)
    target_include_directories(${TARGET_VECT} PRIVATE without_vectorization)

    target_compile_options(${TARGET_NO_VECT}
            PRIVATE $<IF:$<STREQUAL:${V_EXT},AVX512>,-mavx512f -ftree-vectorize,
                        $<IF:$<STREQUAL:${V_EXT},AVX2>,-mavx2 -mfma -ftree-vectorize,
                            $<IF:$<STREQUAL:${V_EXT},SSE2>,-msse2 -ftree-vectorize,
                                $<IF:$<STREQUAL:${V_EXT},NO_VECT>,-mno-sse2 -fno-tree-vectorize,-march=native>
                            >
                        >
                    >
            )
    target_compile_options(${TARGET_VECT}
            PRIVATE $<IF:$<STREQUAL:${V_EXT},AVX512>,-mavx512f -ftree-vectorize,
                        $<IF:$<STREQUAL:${V_EXT},AVX2>,-mavx2 -mfma -ftree-vectorize,
                            $<IF:$<STREQUAL:${V_EXT},SSE2>,-msse2 -ftree-vectorize,
                                $<IF:$<STREQUAL:${V_EXT},NO_VECT>,-mno-sse2 -fno-tree-vectorize,-march=native>
                            >
                        >
                    >
            )
endforeach(V_EXT)