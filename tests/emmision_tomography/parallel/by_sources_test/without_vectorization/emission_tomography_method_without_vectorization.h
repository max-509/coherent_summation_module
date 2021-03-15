#ifndef _EMISSION_TOMOGRAPHY_METHOD_WITHOUT_VECTORIZATION_H
#define _EMISSION_TOMOGRAPHY_METHOD_WITHOUT_VECTORIZATION_H

#include "array2D.h"
#include "amplitudes_calculator.h"

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <omp.h>

template <typename T>
void emissionTomographyMethodWithoutVectorization(const Array2D<T> &gather, 
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
        min_times_to_sources[i_s] = *std::min_element(&sources_receivers_times(i_s, 0), &sources_receivers_times(i_s, n_receivers));
    }

    double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t bl_ir = 0; bl_ir < n_receivers; bl_ir += receivers_block_size) {

        std::ptrdiff_t next_receivers_block_begin = std::min(bl_ir + receivers_block_size, n_receivers);

        std::ptrdiff_t curr_receivers_block_size = next_receivers_block_begin - bl_ir;

        const Array2D<T> receivers_coords_block(const_cast<T*>(&receivers_coords(bl_ir, 0)), curr_receivers_block_size, 3);

        T *amplitudes_buf = new T[n_sources*curr_receivers_block_size];
        Array2D<T> amplitudes{amplitudes_buf, n_sources, curr_receivers_block_size};
        AmplitudesCalculator<T> amplitudes_computer(sources_coords, receivers_coords_block, tensor_matrix, amplitudes);
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


#endif //_EMISSION_TOMOGRAPHY_METHOD_WITHOUT_VECTORIZATION_H