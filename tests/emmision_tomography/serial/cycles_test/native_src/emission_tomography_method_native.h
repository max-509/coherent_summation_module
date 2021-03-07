#ifndef _EMISSION_TOMOGRAPHY_METHOD_NATIVE_H
#define _EMISSION_TOMOGRAPHY_METHOD_NATIVE_H

#include "array2D.h"
#include "amplitudes_calculator.h"

#include <cmath>
#include <algorithm>
#include <cstddef>

template <typename T>
void emissionTomographyMethodNative(const Array2D<T> &gather, 
                                const Array2D<T> &receivers_coords,
                                const Array2D<T> &sources_coords,
                                const Array2D<T> &sources_receivers_times,
                                double dt,
                                const T *tensor_matrix,
                                T *result_data) {
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();
    std::ptrdiff_t n_sources = sources_receivers_times.get_y_dim();

    
    T *amplitudes_buf = new T[n_sources*n_receivers]();
    Array2D<T> amplitudes{amplitudes_buf, n_sources, n_receivers};
    AmplitudesCalculator<T> amplitudes_computer(sources_coords, receivers_coords, tensor_matrix, amplitudes);
    amplitudes_computer.calculate();

    double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
        T t_min_to_source = *std::min_element(&sources_receivers_times(i_s, 0), &sources_receivers_times(i_s, n_receivers));
        for (std::ptrdiff_t i_t = 0; i_t < n_samples; ++i_t) {
            for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
                std::ptrdiff_t godograph_ind = static_cast<std::ptrdiff_t>((sources_receivers_times(i_s, i_r) - t_min_to_source) * rev_dt);
                if (godograph_ind + i_t < n_samples) {
                    result_data[i_s*n_samples + i_t] += gather(i_r, godograph_ind + i_t)*amplitudes(i_s, i_r);
                }
            }   
        }
    }

    delete [] amplitudes_buf;
}


#endif //_EMISSION_TOMOGRAPHY_METHOD_NATIVE_H