#ifndef _KIRCHHOFF_MIGRATION_H
#define _KIRCHHOFF_MIGRATION_H

#include "array2D.h"
#include "kirchhoff_migration_by_points_v.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <numeric>

template <typename T1, typename T2>
void kirchhoffMigrationCHG2D(const Array2D<T1> &gather,
                                    const std::vector<T2> &times_to_source,
                                    const Array2D<T2> &times_to_receivers,
                                    std::ptrdiff_t z_dim, std::ptrdiff_t x_dim,
                                    double dt, 
                                    T1 *result_data) {
    const std::ptrdiff_t n_receivers = gather.get_y_dim();
    const std::ptrdiff_t n_samples = gather.get_x_dim();

    const double rev_dt = 1.0 / dt;

    const std::ptrdiff_t n_points = z_dim * x_dim;

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {

        process_receiver_data_on_grid(&gather(i_r, 0), times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data);

    }
}

template <typename T1, typename T2>
void kirchhoffMigrationCHG3D(const Array2D<T1> &gather,
                                const std::vector<T2> &times_to_source,
                                const Array2D<T2> &times_to_receivers,
                                std::ptrdiff_t z_dim, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim,
                                double dt,
                                T1 *result_data) {
    
    const std::ptrdiff_t n_receivers = gather.get_y_dim();
    const std::ptrdiff_t n_samples = gather.get_x_dim();

    const double rev_dt = 1.0 / dt;

    const std::ptrdiff_t n_points = z_dim * y_dim * x_dim;

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {

        process_receiver_data_on_grid(&gather(i_r, 0), times_to_source, times_to_receivers, n_points, n_samples, rev_dt, i_r, result_data);
        
    }
}

#endif //_KIRCHHOFF_MIGRATION_H