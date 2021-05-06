#ifndef _KIRCHHOFF_MIGRATION_H
#define _KIRCHHOFF_MIGRATION_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>

#include "kirchhoff_migration_by_points_impl.h"
#include "array2D.h"
#include "array1D.h"

namespace {
    constexpr std::ptrdiff_t N_BLOCKS_ON_THREAD = 4;
}

template<typename T1, typename T2>
void kirchhoffMigrationCHGByPoints(const Array2D<T1> &gather,
                                   const Array1D<T2> &times_to_source,
                                   const Array2D<T2> &times_to_receivers,
                                   double dt,
                                   T1 *result_data) {

    const double rev_dt = 1.0 / dt;
    auto n_points = times_to_receivers.get_y_dim();

    process_receivers_on_points_parallel(gather, times_to_source, times_to_receivers, 0, n_points, rev_dt, result_data);
}

template<typename T1, typename T2>
void kirchhoffMigrationCHG2D(const Array2D<T1> &gather,
                             const Array1D<T2> &times_to_source,
                             const Array2D<T2> &times_to_receivers,
                             std::ptrdiff_t z_dim, std::ptrdiff_t x_dim,
                             double dt,
                             T1 *result_data,
                             std::ptrdiff_t p_block_size = 600) {

    const double rev_dt = 1.0 / dt;

    const auto n_threads = omp_get_num_threads();
    const auto n_z_blocks = (z_dim % p_block_size) == 0 ? (z_dim / p_block_size) : (z_dim / p_block_size) + 1;
    const auto n_x_blocks = (x_dim % p_block_size) == 0 ? (x_dim / p_block_size) : (x_dim / p_block_size) + 1;
    const auto n_blocks = n_x_blocks + n_z_blocks;

    if (n_threads != 1 && n_blocks < N_BLOCKS_ON_THREAD * n_threads) {
        kirchhoffMigrationCHGByPoints(gather, times_to_source, times_to_receivers, dt, result_data);
    } else {
#pragma omp parallel for schedule(static) collapse(2)
        for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
            for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {
                const auto z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
                const auto x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);
                for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {

                    const auto i_p0 = i_z * x_dim + i_b_x;
                    const auto i_pn = i_z * x_dim + x_block_upper_border;

                    process_receivers_on_points_serial(gather, times_to_source, times_to_receivers, i_p0,
                                                       i_pn, rev_dt, result_data);
                }

            }

        }
    }


}

template<typename T1, typename T2>
void kirchhoffMigrationCHG3D(const Array2D<T1> &gather,
                             const Array1D <T2> &times_to_source,
                             const Array2D<T2> &times_to_receivers,
                             std::ptrdiff_t z_dim, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim,
                             double dt,
                             T1 *result_data,
                             std::ptrdiff_t p_block_size = 80) {

    const double rev_dt = 1.0 / dt;

    const auto n_threads = omp_get_num_threads();
    const auto n_z_blocks = (z_dim % p_block_size) == 0 ? (z_dim / p_block_size) : (z_dim / p_block_size) + 1;
    const auto n_y_blocks = (y_dim % p_block_size) == 0 ? (y_dim / p_block_size) : (y_dim / p_block_size) + 1;
    const auto n_x_blocks = (x_dim % p_block_size) == 0 ? (x_dim / p_block_size) : (x_dim / p_block_size) + 1;
    const auto n_blocks = n_x_blocks + n_y_blocks + n_z_blocks;

    if (n_threads != 1 && n_blocks < N_BLOCKS_ON_THREAD * n_threads) {
        kirchhoffMigrationCHGByPoints(gather, times_to_source, times_to_receivers, dt, result_data);
    } else {
#pragma omp parallel for schedule(static) collapse(3)
        for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
            for (std::ptrdiff_t i_b_y = 0; i_b_y < y_dim; i_b_y += p_block_size) {
                for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {
                    const auto z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
                    const auto y_block_upper_border = std::min(y_dim, i_b_y + p_block_size);
                    const auto x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);
                    for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {
                        for (std::ptrdiff_t i_y = i_b_y; i_y < y_block_upper_border; ++i_y) {

                            const auto i_p0 = (i_z * y_dim + i_y) * x_dim + i_b_x;
                            const auto i_pn = (i_z * y_dim + i_y) * x_dim + x_block_upper_border;

                            process_receivers_on_points_serial(gather, times_to_source, times_to_receivers, i_p0,
                                                               i_pn, rev_dt, result_data);
                        }
                    }

                }
            }
        }
    }
}

#endif //_KIRCHHOFF_MIGRATION_H