#ifndef _EMISSION_TOMOGRAPHY_METHOD_H
#define _EMISSION_TOMOGRAPHY_METHOD_H

#include "array2D.h"

#include <x86intrin.h>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <omp.h>

template <typename T>
T find_min_t_to_source(const Array2D<T> &sources_receivers_times, std::ptrdiff_t i_s);

/*Selection of SIMD instructions*/
#ifdef __AVX512F__
#include "amplitudes_calculator_m512.h"

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
inline T find_min_t_to_source(const Array2D<T> &sources_receivers_times, std::ptrdiff_t i_s) {
    std::ptrdiff_t n_receivers = sources_receivers_times.get_x_dim();
    return *std::min_element(&sources_receivers_times(i_s, 0), &sources_receivers_times(i_s, n_receivers));
}

template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorNonVectors<T>;

#endif /*End selection of SIMD instructions*/

template <typename T>
void emissionTomographyMethod(const Array2D<T> &gather, 
                            const Array2D<T> &receivers_coords,
                            const Array2D<T> &sources_coords,
                            const Array2D<T> &sources_receivers_times,
                            double dt,
                            const T *tensor_matrix,
                            T *result_data,
                            std::ptrdiff_t receivers_block_size,
                            std::ptrdiff_t samples_block_size) {
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();
    std::ptrdiff_t n_sources = sources_receivers_times.get_y_dim();

    T *min_times_to_sources = new T[n_sources];

    #pragma omp parallel for simd schedule(dynamic)
    for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
        min_times_to_sources[i_s] = find_min_t_to_source(sources_receivers_times, i_s);
    }

    double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t bl_ir = 0; bl_ir < n_receivers; bl_ir += receivers_block_size) {

        std::ptrdiff_t next_receivers_block_begin = std::min(bl_ir + receivers_block_size, n_receivers);

        std::ptrdiff_t curr_receivers_block_size = next_receivers_block_begin - bl_ir;

        const Array2D<T> receivers_coords_block(const_cast<T*>(&receivers_coords(bl_ir, 0)), curr_receivers_block_size, 3);

        T *amplitudes_buf = new T[n_sources*curr_receivers_block_size];
        Array2D<T> amplitudes{amplitudes_buf, n_sources, curr_receivers_block_size};
        AmplitudesComputerType<T> amplitudes_computer(sources_coords, receivers_coords_block, tensor_matrix, amplitudes);
        amplitudes_computer.calculate();
            
        for (std::ptrdiff_t bl_it = 0; bl_it < n_samples; bl_it += samples_block_size) {
            #pragma omp parallel for schedule(dynamic) shared(rev_dt)
            for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {

                T min_t_to_source = min_times_to_sources[i_s];

                for (std::ptrdiff_t i_r = bl_ir; i_r < next_receivers_block_begin; ++i_r) {

                    T amplitude = amplitudes(i_s, i_r-bl_ir);
                    std::ptrdiff_t godograph_ind = static_cast<std::ptrdiff_t>((sources_receivers_times(i_s, i_r) - min_t_to_source) * rev_dt);

                    #pragma omp simd
                    for (std::ptrdiff_t i_t = bl_it; i_t < std::min(bl_it + samples_block_size, n_samples - godograph_ind); ++i_t) {
                        result_data[i_s*n_samples + i_t] += gather(i_r, godograph_ind + i_t)*amplitude;
                    }
                }        
            }

        }

        delete [] amplitudes_buf;
    }

    delete [] min_times_to_sources;
}


#endif //_EMISSION_TOMOGRAPHY_METHOD_H