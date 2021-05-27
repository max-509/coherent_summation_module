#ifndef _KIRCHHOFF_MIGRATION_BLOCKS_RECEIVERS_INNER_LOOP_H
#define _KIRCHHOFF_MIGRATION_BLOCKS_RECEIVERS_INNER_LOOP_H

#include "array2D.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <immintrin.h>

template <typename T1, typename T2>
void kirchhoffMigrationCHG2DBlocksReceiversInnerLoop(const Array2D<T1> &gather,
                                const std::vector<T2> &times_to_source,
                                const Array2D<T2> &times_to_receivers,
                                std::ptrdiff_t z_dim, std::ptrdiff_t x_dim,
                                double dt,
                                T1 *result_data,
                                std::ptrdiff_t p_block_size) {

    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();

    double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
        std::ptrdiff_t z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
        for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {
            std::ptrdiff_t x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);
            for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {

                const std::ptrdiff_t i_z_layer = i_z*x_dim;
                const std::ptrdiff_t i_p_next = i_z_layer + x_dim + i_b_x;

                _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
                _mm_prefetch(times_to_receivers.get(i_p_next, 0), _MM_HINT_T0);

                for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {
                    const std::ptrdiff_t i_p = i_z_layer + i_x;

                    const T2 t_to_s = times_to_source[i_p];

                    T1 res_sum = static_cast<T1>(0.0);

                    for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
                        const T2 t_to_r = times_to_receivers(i_p, i_r);

                        const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) * rev_dt);
                        if (sample_idx < n_samples) {
                            res_sum += gather(i_r, sample_idx);
                        }
                    }

                    result_data[i_p] += res_sum;

                }
            }

        }

    }
}

template <typename T1, typename T2>
void kirchhoffMigrationCHG3DBlocksReceiversInnerLoop(const Array2D<T1> &gather,
                                const std::vector<T2> &times_to_source,
                                const Array2D<T2> &times_to_receivers,
                                std::ptrdiff_t z_dim, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim,
                                double dt,
                                T1 *result_data,
                                std::ptrdiff_t p_block_size) {

    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();

    double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
        std::ptrdiff_t z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
        for (std::ptrdiff_t i_b_y = 0; i_b_y < y_dim; i_b_y += p_block_size) {
            std::ptrdiff_t y_block_upper_border = std::min(y_dim, i_b_y + p_block_size);
            for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {
                std::ptrdiff_t x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);

                for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {
                    for (std::ptrdiff_t i_y = i_b_y; i_y < y_block_upper_border; ++i_y) {

                        const std::ptrdiff_t i_zy_layer = (i_z*y_dim + i_y)*x_dim;
                        const std::ptrdiff_t i_p_next = i_zy_layer + x_dim + i_b_x;

//                        _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
//                        _mm_prefetch(times_to_receivers.get(i_p_next, 0), _MM_HINT_T0);

                        for (std::ptrdiff_t i_x = i_b_x; i_x < x_block_upper_border; ++i_x) {

                            const std::ptrdiff_t i_p = i_zy_layer + i_x;

                            const T2 t_to_s = times_to_source[i_p];

                            T1 res_sum = static_cast<T1>(0.0);

                            for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
                                const T2 t_to_r = times_to_receivers(i_p, i_r);

                                const auto sample_idx = static_cast<std::ptrdiff_t>((t_to_r + t_to_s) * rev_dt);
                                if (sample_idx < n_samples) {
                                    res_sum += gather(i_r, sample_idx);
                                }
                            }

                            result_data[i_p] += res_sum;
                        }
                    }
                }

            }
        }
    }
}

#endif //_KIRCHHOFF_MIGRATION_BLOCKS_RECEIVERS_INNER_LOOP_H