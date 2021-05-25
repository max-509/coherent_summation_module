#ifndef _EMISSION_TOMOGRAPHY_METHOD_NATIVE_H
#define _EMISSION_TOMOGRAPHY_METHOD_NATIVE_H

#include "array2D.h"
#include "amplitudes_calculator.h"

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <memory>

template<typename T1, typename T2>
void emissionTomographyMethodNative(const Array2D<T1> &gather,
                                const Array2D<T1> &receivers_coords,
                                const Array2D<T1> &sources_coords,
                                const Array2D<T2> &sources_receivers_times,
                                double dt,
                                const T1 *tensor_matrix,
                                T1 *result_data) {
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();
    std::ptrdiff_t n_sources = sources_receivers_times.get_y_dim();

    
    std::unique_ptr<T1[]> amplitudes_buf(new T1[n_sources*n_receivers]);
    Array2D<T1> amplitudes{amplitudes_buf.get(), n_sources, n_receivers};
    AmplitudesCalculator<T1> amplitudes_computer(sources_coords,tensor_matrix);
    amplitudes_computer.calculate(receivers_coords, amplitudes);

    double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t i_s = 0; i_s < n_sources; ++i_s) {
        T2 t_min_to_source = *std::min_element(&sources_receivers_times(i_s, 0), &sources_receivers_times(i_s, n_receivers));
        for (std::ptrdiff_t i_t = 0; i_t < n_samples; ++i_t) {
            for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
                auto godograph_ind = static_cast<std::ptrdiff_t>((sources_receivers_times(i_s, i_r) - t_min_to_source) * rev_dt);
                if (godograph_ind + i_t < n_samples) {
                    result_data[i_s*n_samples + i_t] += gather(i_r, godograph_ind + i_t)*amplitudes(i_s, i_r);
                }
            }   
        }
    }
}


#endif //_EMISSION_TOMOGRAPHY_METHOD_NATIVE_H