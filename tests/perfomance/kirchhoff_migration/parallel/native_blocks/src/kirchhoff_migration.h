#ifndef _KIRCHHOFF_MIGRATION_H
#define _KIRCHHOFF_MIGRATION_H

#include "array2D.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include "kirchhoff_migration_by_points_impl.h"

template<typename T1, typename T2>
void kirchhoffMigrationCHG2D(const Array2D<T1> &gather,
                             const std::vector<T2> &times_to_source,
                             const Array2D<T2> &times_to_receivers,
                             std::ptrdiff_t z_dim, std::ptrdiff_t x_dim,
                             double dt,
                             T1 *result_data,
                             std::ptrdiff_t p_block_size = 1200) {

    const double rev_dt = 1.0 / dt;

    #pragma omp parallel for schedule(static) collapse(2)
    for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
        for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {
            const auto z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
            const auto x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);
            for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {

                const auto i_p0 = i_z * x_dim + i_b_x;
                const auto i_pn = i_z * x_dim + x_block_upper_border;

                process_receivers_on_points(gather, times_to_source, times_to_receivers, i_p0,
                                            i_pn, rev_dt, result_data);
            }

        }

    }
}

template<typename T1, typename T2>
void kirchhoffMigrationCHG3D(const Array2D<T1> &gather,
                             const std::vector<T2> &times_to_source,
                             const Array2D<T2> &times_to_receivers,
                             std::ptrdiff_t z_dim, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim,
                             double dt,
                             T1 *result_data,
                             std::ptrdiff_t p_block_size = 120) {

    const double rev_dt = 1.0 / dt;

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

                        process_receivers_on_points(gather, times_to_source, times_to_receivers, i_p0,
                                                    i_pn, rev_dt, result_data);
                    }
                }

            }
        }
    }
}

#endif //_KIRCHHOFF_MIGRATION_H