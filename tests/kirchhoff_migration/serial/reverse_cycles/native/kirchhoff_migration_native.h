#ifndef _KIRCHHOFF_MIGRATION_NATIVE_H
#define _KIRCHHOFF_MIGRATION_NATIVE_H

#include "array2D.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>

template <typename T1, typename T2>
void kirchhoffMigrationCHG2DNative(const Array2D<T1> &gather,
                                    const std::vector<T2> &times_to_source,
                                    const Array2D<T2> &times_to_receivers,
                                    std::ptrdiff_t z_dim, std::ptrdiff_t x_dim,
                                    double dt, 
                                    T1 *result_data) {
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();

    for (std::ptrdiff_t i_z = 0; i_z < z_dim; ++i_z) {
        for (std::ptrdiff_t i_x = 0; i_x < x_dim; ++i_x) {
            T1 result_sum = static_cast<T1>(0.0);

            std::ptrdiff_t i_p = i_z*x_dim + i_x;

            T2 t_to_s = times_to_source[i_p];

            for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
                T2 t_to_r = times_to_receivers(i_p, i_r);

                auto sample_idx = static_cast<std::ptrdiff_t>((t_to_s + t_to_r) / dt);

                if (sample_idx < n_samples) {
                    result_sum += gather(i_r, sample_idx);
                }
            }

            result_data[i_p] += result_sum;
        }
    }
}

template <typename T1, typename T2>
void kirchhoffMigrationCHG3DNative(const Array2D<T1> &gather,
                                const std::vector<T2> &times_to_source,
                                const Array2D<T2> &receivers_coords,
                                std::ptrdiff_t z_dim, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim,
                                double dt,
                                T1 *result_data) {
    
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();

    for (std::ptrdiff_t i_z = 0; i_z < z_dim; ++i_z) {
        for (std::ptrdiff_t i_y = 0; i_y< y_dim; ++i_y) {
            for (std::ptrdiff_t i_x = 0; i_x < x_dim; ++i_x) {
                T1 result_sum = static_cast<T1>(0.0);
                std::ptrdiff_t i_p = (i_z*y_dim + i_y)*x_dim + i_x;

                T2 t_to_s = times_to_source[i_p];

                for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {

                    T2 t_to_r = receivers_coords(i_p, i_r);

                    auto sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) / dt);

                    if (sample_idx < n_samples) {
                        result_sum += gather(i_r, sample_idx);
                    }
                }

                result_data[i_p] += result_sum;
            }
        }
    }
}

#endif //_KIRCHHOFF_MIGRATION_NATIVE_H