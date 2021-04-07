#ifndef _EMISSION_TOMOGRAPHY_METHOD_WITHOUT_VECTORIZATION_H
#define _EMISSION_TOMOGRAPHY_METHOD_WITHOUT_VECTORIZATION_H

#include "array2D.h"
#include "amplitudes_calculator.h"

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <omp.h>

template <typename T1, typename T2>
void emissionTomographyMethodWithoutVectorization(const Array2D<T1> &gather,
                                                const Array2D<T1> &receivers_coords,
                                                const Array2D<T1> &sources_coords,
                                                const Array2D<T2> &sources_receivers_times,
                                                double dt,
                                                const T1 *tensor_matrix,
                                                T1 *result_data,
                                                std::ptrdiff_t receivers_block_size = 20,
                                                std::ptrdiff_t samples_block_size = 1000) {
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();
    std::ptrdiff_t n_sources = sources_receivers_times.get_y_dim();

    std::unique_ptr<T2[]> min_times_to_sources(new T2[n_sources]);

    #pragma omp parallel for simd schedule(static)
    for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
        min_times_to_sources[i_s] = *std::min_element(&sources_receivers_times(i_s, 0), &sources_receivers_times(i_s, n_receivers));
    }

    double rev_dt = 1.0 / dt;

    T1 *tmp_result_data = nullptr;

    #pragma omp parallel shared(rev_dt)
    {
        //init local array for reduction
        const auto n_threads = omp_get_num_threads();
        const auto i_thread = omp_get_thread_num();

        #pragma omp single
        tmp_result_data = new T1[n_threads*(n_samples*n_sources)];

        #pragma omp for schedule(guided)
        for (std::ptrdiff_t bl_ir = 0; bl_ir < n_receivers; bl_ir += receivers_block_size) {

            std::ptrdiff_t next_receivers_block_begin = std::min(bl_ir + receivers_block_size, n_receivers);

            std::ptrdiff_t curr_receivers_block_size = next_receivers_block_begin - bl_ir;

            const Array2D<T1> receivers_coords_block(const_cast<T1*>(&receivers_coords(bl_ir, 0)), curr_receivers_block_size, 3);

            auto amplitudes_buf = std::make_unique<T1[]>(n_sources*curr_receivers_block_size);
            Array2D<T1> amplitudes{amplitudes_buf.get(), n_sources, curr_receivers_block_size};
            AmplitudesCalculator<T1> amplitudes_computer(sources_coords, receivers_coords_block, tensor_matrix, amplitudes);
            amplitudes_computer.calculate();

            for (std::ptrdiff_t bl_it = 0; bl_it < n_samples; bl_it += samples_block_size) {
                for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {

                    T2 min_t_to_source = min_times_to_sources[i_s];

                    for (std::ptrdiff_t i_r = bl_ir; i_r < next_receivers_block_begin; ++i_r) {

                        T1 amplitude = amplitudes(i_s, i_r-bl_ir);
                        std::ptrdiff_t godograph_ind = static_cast<std::ptrdiff_t>((sources_receivers_times(i_s, i_r) - min_t_to_source) * rev_dt);

                        #pragma omp simd
                        for (std::ptrdiff_t i_t = bl_it; i_t < std::min(bl_it + samples_block_size, n_samples - godograph_ind); ++i_t) {
                            tmp_result_data[i_thread*n_sources*n_samples + i_s*n_samples + i_t] += gather(i_r, godograph_ind + i_t)*amplitude;
                        }
                    }        
                }
            }
        }

        #pragma omp for simd schedule(static) collapse(2)
        for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
            for (std::ptrdiff_t i_t = 0; i_t < n_samples; ++i_t) {
                for (std::ptrdiff_t thr = 0; thr < n_threads; ++thr) {
                    result_data[i_s*n_samples + i_t] += tmp_result_data[i_thread*n_sources*n_samples + i_s*n_samples + i_t];
                }
            }
        }

    }

    delete [] tmp_result_data;
}


#endif //_EMISSION_TOMOGRAPHY_METHOD_WITHOUT_VECTORIZATION_H