#ifndef _EMISSION_TOMOGRAPHY_METHOD_WITH_VECTORIZATION_H
#define _EMISSION_TOMOGRAPHY_METHOD_WITH_VECTORIZATION_H

#include "array2D.h"

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <memory>

template <typename T>
T find_min_t_to_source(const Array2D<T> &sources_receivers_times, std::ptrdiff_t i_s);

/*Selection of SIMD instructions*/
#ifdef __AVX512F__
#include "amplitudes_calculator_m512.h"

#include <immintrin.h>

template<>
double find_min_t_to_source(const Array2D<double> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 8;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        return *std::min_element(&sources_receivers_times(i_s, 0), &sources_receivers_times(i_s, n_receivers));
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers-(n_receivers%vector_dim);

    __m512d min_elements_v = _mm512_loadu_pd(&sources_receivers_times(i_s, 0));
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        min_elements_v = _mm512_min_pd(min_elements_v, _mm512_loadu_pd(&sources_receivers_times(i_s, i_r + vector_dim)));
    }
    return std::min(
        _mm512_reduce_min_pd(min_elements_v), 
        *std::min_element(&sources_receivers_times(i_s, n_receivers_multiple_vector_dim), &sources_receivers_times(i_s, n_receivers))
    );

}

template<>
float find_min_t_to_source(const Array2D<float> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 16;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        return *std::min_element(&sources_receivers_times(i_s, 0), &sources_receivers_times(i_s, n_receivers));
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers-(n_receivers%vector_dim);

    __m512 min_elements_v = _mm512_loadu_ps(&sources_receivers_times(i_s, 0));
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        min_elements_v = _mm512_min_ps(min_elements_v, _mm512_loadu_ps(&sources_receivers_times(i_s, i_r + vector_dim)));
    }

    return std::min(
        _mm512_reduce_min_ps(min_elements_v), 
        *std::min_element(&sources_receivers_times(i_s, n_receivers_multiple_vector_dim), &sources_receivers_times(i_s, n_receivers))
    );

}

template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM512<T>;

#elif __AVX2__
#include "amplitudes_calculator_m256.h"

#include <immintrin.h>

template<>
double find_min_t_to_source(const Array2D<double> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 4;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        return *std::min_element(&sources_receivers_times(i_s, 0), &sources_receivers_times(i_s, n_receivers));
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers-(n_receivers%vector_dim);

    __m256d min_elements_v = _mm256_loadu_pd(&sources_receivers_times(i_s, 0));
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        min_elements_v = _mm256_min_pd(min_elements_v, _mm256_loadu_pd(&sources_receivers_times(i_s, i_r + vector_dim)));
    }
    double min_elements[vector_dim];

    _mm256_storeu_pd(min_elements, min_elements_v);
    return std::min(
        *std::min_element(min_elements, min_elements + vector_dim), 
        *std::min_element(&sources_receivers_times(i_s, n_receivers_multiple_vector_dim), &sources_receivers_times(i_s, n_receivers))
    );

}

template<>
float find_min_t_to_source(const Array2D<float> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 8;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        return *std::min_element(&sources_receivers_times(i_s, 0), &sources_receivers_times(i_s, n_receivers));
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers-(n_receivers%vector_dim);

    __m256 min_elements_v = _mm256_loadu_ps(&sources_receivers_times(i_s, 0));
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        min_elements_v = _mm256_min_ps(min_elements_v, _mm256_loadu_ps(&sources_receivers_times(i_s, i_r + vector_dim)));
    }
    float min_elements[vector_dim];

    _mm256_storeu_ps(min_elements, min_elements_v);
    return std::min(
        *std::min_element(min_elements, min_elements + vector_dim), 
        *std::min_element(&sources_receivers_times(i_s, n_receivers_multiple_vector_dim), &sources_receivers_times(i_s, n_receivers))
    );

}

template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM256<T>;

#elif __SSE2__
#include "amplitudes_calculator_m128.h"

#include <immintrin.h>

template<>
double find_min_t_to_source(const Array2D<double> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 2;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        return sources_receivers_times(i_s, 0);
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers-(n_receivers%vector_dim);

    __m128d min_elements_v = _mm_loadu_pd(&sources_receivers_times(i_s, 0));
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        min_elements_v = _mm_min_pd(min_elements_v, _mm_loadu_pd(&sources_receivers_times(i_s, i_r + vector_dim)));
    }
    double min_elements[vector_dim];

    _mm_storeu_pd(min_elements, min_elements_v);
    return std::min(
        *std::min_element(min_elements, min_elements + vector_dim), 
        *std::min_element(&sources_receivers_times(i_s, n_receivers_multiple_vector_dim), &sources_receivers_times(i_s, n_receivers))
    );

}

template<>
float find_min_t_to_source(const Array2D<float> &sources_receivers_times, std::ptrdiff_t i_s) {
    constexpr std::ptrdiff_t vector_dim = 4;
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();

    if (n_receivers < vector_dim) {
        return *std::min_element(&sources_receivers_times(i_s, 0), &sources_receivers_times(i_s, n_receivers));
    }
    std::ptrdiff_t n_receivers_multiple_vector_dim = n_receivers-(n_receivers%vector_dim);

    __m128 min_elements_v = _mm_loadu_ps(&sources_receivers_times(i_s, 0));
    for (std::ptrdiff_t i_r = 0; i_r < n_receivers_multiple_vector_dim - vector_dim; i_r += vector_dim) {
        min_elements_v = _mm_min_ps(min_elements_v, _mm_loadu_ps(&sources_receivers_times(i_s, i_r + vector_dim)));
    }
    float min_elements[vector_dim];

    _mm_storeu_ps(min_elements, min_elements_v);
    return std::min(
        *std::min_element(min_elements, min_elements + vector_dim), 
        *std::min_element(&sources_receivers_times(i_s, n_receivers_multiple_vector_dim), &sources_receivers_times(i_s, n_receivers))
    );

}

template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM128<T>;

#else
#include "amplitudes_calculator_non_vectors.h"

template <typename T>
T find_min_t_to_source(const Array2D<T> &sources_receivers_times, std::ptrdiff_t i_s) {
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();
    return *std::min_element(&sources_receivers_times(i_s, 0), &sources_receivers_times(i_s, n_receivers));
}

template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorNonVectors<T>;

#endif /*End selection of SIMD instructions*/

template <typename T1, typename T2>
void emissionTomographyMethodWithVectorization(const Array2D<T1> &gather,
                                            const Array2D<T1> &receivers_coords,
                                            const Array2D<T1> &sources_coords,
                                            const Array2D<T2> &sources_receivers_times,
                                            double dt,
                                            const T1 *tensor_matrix,
                                            T1 *result_data,
                                            std::ptrdiff_t receivers_block_size,
                                            std::ptrdiff_t samples_block_size) {
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();
    std::ptrdiff_t n_sources = sources_receivers_times.get_y_dim();

    std::unique_ptr<T2[]> min_times_to_sources(new T2[n_sources]);

    #pragma omp parallel for simd schedule(static)
    for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
        min_times_to_sources[i_s] = find_min_t_to_source(sources_receivers_times, i_s);
    }

    double rev_dt = 1.0 / dt;

    std::unique_ptr<T1[]> amplitudes_buf(new T1[n_sources*receivers_block_size]);

    AmplitudesComputerType<T1> amplitudes_computer(sources_coords, tensor_matrix);
    
    for (std::ptrdiff_t bl_ir = 0; bl_ir < n_receivers; bl_ir += receivers_block_size) {

        std::ptrdiff_t next_receivers_block_begin = std::min(bl_ir + receivers_block_size, n_receivers);

        std::ptrdiff_t curr_receivers_block_size = next_receivers_block_begin - bl_ir;

        const Array2D<T1> receivers_coords_block(const_cast<T1*>(&receivers_coords(bl_ir, 0)), curr_receivers_block_size, 3);


        Array2D<T1> amplitudes{amplitudes_buf.get(), n_sources, curr_receivers_block_size};
        amplitudes_computer.calculate(receivers_coords_block, amplitudes);

        #pragma omp parallel for schedule(guided) shared(rev_dt)
        for (std::ptrdiff_t bl_it = 0; bl_it < n_samples; bl_it += samples_block_size) {
            for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {

                T2 min_t_to_source = min_times_to_sources[i_s];
                
                for (std::ptrdiff_t i_r = bl_ir; i_r < next_receivers_block_begin; ++i_r) {

                    T1 amplitude = amplitudes(i_s, i_r-bl_ir);
                    std::ptrdiff_t godograph_ind = static_cast<std::ptrdiff_t>((sources_receivers_times(i_s, i_r) - min_t_to_source) * rev_dt);

                    #pragma omp simd
                    for (std::ptrdiff_t i_t = bl_it; i_t < std::min(bl_it + samples_block_size, n_samples - godograph_ind); ++i_t) {
                        result_data[i_s*n_samples + i_t] += gather(i_r, godograph_ind + i_t)*amplitude;
                    }
                }        
            }

        }
    }
}


#endif //_EMISSION_TOMOGRAPHY_METHOD_WITH_VECTORIZATION_H