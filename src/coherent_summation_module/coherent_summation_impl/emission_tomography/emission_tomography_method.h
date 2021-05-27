#ifndef _EMISSION_TOMOGRAPHY_METHOD_H
#define _EMISSION_TOMOGRAPHY_METHOD_H

#include "array2D.h"
#include "array1D.h"
#include "minmax_t_to_source.h"

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <type_traits>

//Selection of SIMD instructions
#ifdef __AVX512F__ // AVX512 SIMD

#include "amplitudes_calculator_m512.h"

template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM512<T>;

#elif __AVX2__ // AVX2 SIMD

#include "amplitudes_calculator_m256.h"

template<typename T>
using AmplitudesComputerType = AmplitudesCalculatorM256<T>;

#elif __SSE2__ // SSE2 SIMD

#include "amplitudes_calculator_m128.h"

template <typename T>
using AmplitudesComputerType = AmplitudesCalculatorM128<T>;

#else // Without SIMD 

#include "amplitudes_calculator_non_vectors.h"

template<typename T>
using AmplitudesComputerType = AmplitudesCalculatorNonVectors<T>;

#endif //End selection of SIMD instructions

template<typename T1, typename T2>
auto emissionTomographyMethod(const Array2D<T1> &gather,
                              const Array2D<T2> &sources_receivers_times,
                              double dt,
                              std::ptrdiff_t receivers_block_size = 20,
                              std::ptrdiff_t samples_block_size = 1000) -> Array2D<typename std::remove_const<T1>::type> {
    using T1_non_const = typename std::remove_const<T1>::type;
    using T2_non_const = typename std::remove_const<T2>::type;

    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();
    std::ptrdiff_t n_sources = sources_receivers_times.get_y_dim();

    std::unique_ptr<T2_non_const[]> min_times_to_sources(new T2_non_const[n_sources]);

    double rev_dt = 1.0 / dt;

    T2_non_const max_godograph_diff_time = T2(.0);

    std::cerr << "Start finding godograph border" << std::endl;

    if (sources_receivers_times.get_x_stride() == 1) {
#pragma omp parallel
        {
            T2_non_const local_max_godograph_diff_time = max_godograph_diff_time;
            T2_non_const min_t, max_t;
#ifdef _MSC_VER
#pragma omp for
#else //_MSC_VER
#pragma omp for simd schedule(static)
#endif //_MSC_VER
            for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
                std::tie(min_t, max_t) = find_minmax_t_to_source_sequential(sources_receivers_times, i_s);
                min_times_to_sources[i_s] = min_t;

                auto godograph_diff_time = max_t - min_t;
                if (godograph_diff_time > local_max_godograph_diff_time) {
                    local_max_godograph_diff_time = godograph_diff_time;
                }
            }

#pragma omp critical
            {
                if (local_max_godograph_diff_time > max_godograph_diff_time) {
                    max_godograph_diff_time = local_max_godograph_diff_time;
                }
            }
        }
    } else {
#pragma omp parallel
        {
            T2_non_const local_max_godograph_diff_time = max_godograph_diff_time;
            T2_non_const min_t, max_t;
#ifdef _MSC_VER
#pragma omp for
#else //_MSC_VER
#pragma omp for simd schedule(static)
#endif //_MSC_VER
            for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
                std::tie(min_t, max_t) = find_minmax_t_to_source_stride(sources_receivers_times, i_s);
                min_times_to_sources[i_s] = min_t;

                auto godograph_diff_time = max_t - min_t;
                if (godograph_diff_time > local_max_godograph_diff_time) {
                    local_max_godograph_diff_time = godograph_diff_time;
                }
            }

#pragma omp critical
            {
                if (local_max_godograph_diff_time > max_godograph_diff_time) {
                    max_godograph_diff_time = local_max_godograph_diff_time;
                }
            }
        }
    }

    std::cerr << "End finding godograph border" << std::endl;

    std::cerr << "Max diff time: " << max_godograph_diff_time << std::endl;

    auto godograph_border_ind = n_samples - static_cast<std::ptrdiff_t>(max_godograph_diff_time * rev_dt);

    std::cerr << "Godograph border ind: " << godograph_border_ind << std::endl;
    std::cerr << "MAX DIFF: " << (n_samples - godograph_border_ind) << std::endl;

    std::unique_ptr<T1_non_const[]> result_data(new T1_non_const[n_sources * godograph_border_ind]);

    for (std::ptrdiff_t bl_ir = 0; bl_ir < n_receivers; bl_ir += receivers_block_size) {

        std::ptrdiff_t next_receivers_block_begin = std::min(bl_ir + receivers_block_size, n_receivers);

#pragma omp parallel for schedule(dynamic) collapse(2) shared(rev_dt)
        for (std::ptrdiff_t bl_it = 0; bl_it < godograph_border_ind; bl_it += samples_block_size) {
            for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {

                auto min_t_to_source = min_times_to_sources[i_s];

                for (std::ptrdiff_t i_r = bl_ir; i_r < next_receivers_block_begin; ++i_r) {

                    auto godograph_ind = static_cast<std::ptrdiff_t>(
                            (sources_receivers_times(i_s, i_r) - min_t_to_source) * rev_dt);

#pragma omp simd
                    for (std::ptrdiff_t i_t = bl_it;
                         i_t < std::min(bl_it + samples_block_size, godograph_border_ind); ++i_t) {
                        result_data[i_s * godograph_border_ind + i_t] += gather(i_r, godograph_ind + i_t);
                    }
                }
            }

        }

    }

    return Array2D<T1_non_const>(result_data.release(), n_sources, godograph_border_ind);
}

template<typename T1, typename T2>
auto emissionTomographyMethod(const Array2D<T1> &gather,
                              const Array2D<T1> &receivers_coords,
                              const Array2D<T1> &sources_coords,
                              const Array2D<T2> &sources_receivers_times,
                              double dt,
                              const Array1D<T1> &tensor_matrix,
                              std::ptrdiff_t receivers_block_size = 20,
                              std::ptrdiff_t samples_block_size = 1000) -> Array2D<typename std::remove_const<T1>::type> {
    using T1_non_const = typename std::remove_const<T1>::type;
    using T2_non_const = typename std::remove_const<T2>::type;

    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();
    std::ptrdiff_t n_sources = sources_receivers_times.get_y_dim();

    std::unique_ptr<T2_non_const[]> min_times_to_sources(new T2_non_const[n_sources]);

    double rev_dt = 1.0 / dt;

    T2_non_const max_godograph_diff_time = T2_non_const(.0);

    if (sources_receivers_times.get_x_stride() == 1) {
#pragma omp parallel
        {
            T2_non_const local_max_godograph_diff_time = max_godograph_diff_time;
            T2_non_const min_t, max_t;
#ifdef _MSC_VER
#pragma omp for
#else //_MSC_VER
#pragma omp for simd schedule(static)
#endif //_MSC_VER
            for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
                std::tie(min_t, max_t) = find_minmax_t_to_source_sequential(sources_receivers_times, i_s);
                min_times_to_sources[i_s] = min_t;

                auto godograph_diff_time = max_t - min_t;
                if (godograph_diff_time > local_max_godograph_diff_time) {
                    local_max_godograph_diff_time = godograph_diff_time;
                }
            }

#pragma omp critical
            {
                if (local_max_godograph_diff_time > max_godograph_diff_time) {
                    max_godograph_diff_time = local_max_godograph_diff_time;
                }
            }
        }
    } else {
#pragma omp parallel
        {
            T2_non_const local_max_godograph_diff_time = max_godograph_diff_time;
            T2_non_const min_t, max_t;
#ifdef _MSC_VER
#pragma omp for
#else //_MSC_VER
#pragma omp for simd schedule(static)
#endif //_MSC_VER
            for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
                std::tie(min_t, max_t) = find_minmax_t_to_source_stride(sources_receivers_times, i_s);
                min_times_to_sources[i_s] = min_t;

                auto godograph_diff_time = max_t - min_t;
                if (godograph_diff_time > local_max_godograph_diff_time) {
                    local_max_godograph_diff_time = godograph_diff_time;
                }
            }

#pragma omp critical
            {
                if (local_max_godograph_diff_time > max_godograph_diff_time) {
                    max_godograph_diff_time = local_max_godograph_diff_time;
                }
            }
        }
    }

    auto godograph_border_ind = n_samples - static_cast<std::ptrdiff_t>(max_godograph_diff_time * rev_dt);

    std::unique_ptr<T1_non_const[]> result_data(new T1_non_const[n_sources * godograph_border_ind]);

    std::unique_ptr<T1_non_const[]> amplitudes_buf(new T1_non_const[n_sources * receivers_block_size]);

    AmplitudesComputerType<T1> amplitudes_computer(sources_coords, tensor_matrix);

    for (std::ptrdiff_t bl_ir = 0; bl_ir < n_receivers; bl_ir += receivers_block_size) {

        std::ptrdiff_t next_receivers_block_begin = std::min(bl_ir + receivers_block_size, n_receivers);

        std::ptrdiff_t curr_receivers_block_size = next_receivers_block_begin - bl_ir;

        Array2D<T1> receivers_coords_block(receivers_coords.get(bl_ir, 0), curr_receivers_block_size, 3);

        Array2D<T1_non_const> amplitudes{amplitudes_buf.get(), n_sources, curr_receivers_block_size};
        amplitudes_computer.calculate(receivers_coords_block, amplitudes);

#pragma omp parallel for schedule(dynamic) collapse(2) shared(rev_dt)
        for (std::ptrdiff_t bl_it = 0; bl_it < godograph_border_ind; bl_it += samples_block_size) {
            for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {

                auto min_t_to_source = min_times_to_sources[i_s];

                for (std::ptrdiff_t i_r = bl_ir; i_r < next_receivers_block_begin; ++i_r) {

                    auto amplitude = amplitudes(i_s, i_r - bl_ir);
                    auto godograph_ind = static_cast<std::ptrdiff_t>(
                            (sources_receivers_times(i_s, i_r) - min_t_to_source) * rev_dt);

                    const auto samples_border = std::min(bl_it + samples_block_size, godograph_border_ind);

#pragma omp simd
                    for (std::ptrdiff_t i_t = bl_it; i_t < samples_border; ++i_t) {
                        result_data[i_s * godograph_border_ind + i_t] += gather(i_r, godograph_ind + i_t) * amplitude;
                    }
                }
            }

        }

    }

    return Array2D<T1_non_const>(result_data.release(), n_sources, godograph_border_ind);
}


#endif //_EMISSION_TOMOGRAPHY_METHOD_H