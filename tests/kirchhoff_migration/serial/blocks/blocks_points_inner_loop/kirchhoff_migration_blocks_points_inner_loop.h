#ifndef _KIRCHHOFF_MIGRATION_BLOCKS_POINTS_INNER_LOOP_H
#define _KIRCHHOFF_MIGRATION_BLOCKS_POINTS_INNER_LOOP_H

#include "array2D.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>

template <typename T1, typename T2>
void kirchhoffMigrationCHG2DBlocksPointsInnerLoop(const Array2D<T1> &gather,
                                const std::vector<T2> &times_to_source,
                                const Array2D<T2> &times_to_receivers,
                                std::ptrdiff_t z_dim, std::ptrdiff_t x_dim,
                                double dt,
                                T1 *result_data,
                                std::ptrdiff_t p_block_size) {

    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers ++i_r) {

        for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
            std::ptrdiff_t z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
            for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {
                std::ptrdiff_t x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);

                for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {
                    for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {
                        std::ptrdiff_t i_p = i_z*x_dim + i_x;

                        double t_to_s = times_to_source[i_p];
                        double t_to_r = times_to_receivers(i_r, i_p);

                        std::ptrdiff_t sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) / dt);
                        if (sample_idx < n_samples) {
                            result_data[i_p] += gather(i_r, sample_idx);
                        }
                    }
                }

            }

        }
    }
}

template <typename T1, typename T2>
void kirchhoffMigrationCHG3DBlocksPointsInnerLoop(const Array2D<T1> &gather,
                                const std::vector<T2> &times_to_source,
                                const Array2D<T2> &times_to_receivers,
                                std::ptrdiff_t z_dim, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim,
                                double dt,
                                T1 *result_data,
                                std::ptrdiff_t p_block_size) {
    
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers ++i_r) {

        for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
            std::ptrdiff_t z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
            for (std::ptrdiff_t i_b_y = 0; i_b_y < y_dim; i_b_y += p_block_size) {
                std::ptrdiff_t y_block_upper_border = std::min(y_dim, i_b_y + p_block_size);
                for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {
                    std::ptrdiff_t x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);

                    for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {
                        for (std::ptrdiff_t i_y = i_b_y; i_y < y_block_upper_border; ++i_y) {
                            for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {
                                std::ptrdiff_t i_p = (i_z*y_dim + i_y)*x_dim + i_x;

                                double t_to_s = times_to_source[i_p];
                                double t_to_r = times_to_receivers(i_r, i_p);

                                std::ptrdiff_t sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) / dt);
                                if (sample_idx < n_samples) {
                                    result_data[i_p] += gather(i_r, sample_idx);
                                }
                            }
                        }
                    }

                }
            }
        }
    }
}

#endif //_KIRCHHOFF_MIGRATION_BLOCKS_POINTS_INNER_LOOP_H