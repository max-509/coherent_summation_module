#ifndef _KIRCHHOFF_MIGRATION_AUTO_VECTORIZATION_H
#define _KIRCHHOFF_MIGRATION_AUTO_VECTORIZATION_H

#include "array2D.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <numeric>
#include <omp.h>

template <typename T1, typename T2>
void kirchhoffMigrationCHG2DAutoVectorization(const Array2D<T1> &gather,
                                    const std::vector<T2> &times_to_source,
                                    const Array2D<T2> &times_to_receivers,
                                    std::ptrdiff_t z_dim, std::ptrdiff_t x_dim,
                                    double dt, 
                                    T1 *result_data) {
    const std::ptrdiff_t n_receivers = gather.get_y_dim();
    const std::ptrdiff_t n_samples = gather.get_x_dim();

    const double rev_dt = 1.0 / dt;

    const std::ptrdiff_t n_points = z_dim*x_dim;

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {

        #pragma omp simd
        for (std::ptrdiff_t i_p = 0; i_p < n_points; ++i_p) {
            const T2 t_to_s = times_to_source[i_p], t_to_r = times_to_receivers(i_r, i_p);

            const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_s + t_to_r) * rev_dt);

            if (sample_idx < n_samples) {
                result_data[i_p] += gather(i_r, sample_idx);
            }
        }
    }
}

template <typename T1, typename T2>
void kirchhoffMigrationCHG3DAutoVectorization(const Array2D<T1> &gather,
                                const std::vector<T2> &times_to_source,
                                const Array2D<T2> &times_to_receivers,
                                std::ptrdiff_t z_dim, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim,
                                double dt,
                                T1 *result_data) {
    
    const std::ptrdiff_t n_receivers = gather.get_y_dim();
    const std::ptrdiff_t n_samples = gather.get_x_dim();

    const double rev_dt = 1.0 / dt;

    const std::ptrdiff_t n_points = z_dim*y_dim*x_dim;

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {

        #pragma omp simd
        for (std::ptrdiff_t i_p = 0; i_p < n_points; ++i_p) {
            const T2 t_to_s = times_to_source[i_p], t_to_r = times_to_receivers(i_r, i_p);

            const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_s + t_to_r) * rev_dt);

            if (sample_idx < n_samples) {
                result_data[i_p] += gather(i_r, sample_idx);
            }
        }
    }
}

#endif //_KIRCHHOFF_MIGRATION_AUTO_VECTORIZATION_H