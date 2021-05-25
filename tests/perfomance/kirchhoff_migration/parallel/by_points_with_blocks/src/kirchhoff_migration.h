#ifndef _KIRCHHOFF_MIGRATION_H
#define _KIRCHHOFF_MIGRATION_H

#include "array2D.h"
#include "kirchhoff_migration_by_points_v.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <memory>
#if defined(__SSE__) || defined(__KNCNI__)
#include <immintrin.h>
#endif

template <typename T1, typename T2>
void kirchhoffMigrationCHG2D(const Array2D<T1> &gather,
                                const std::vector<T2> &times_to_source,
                                const Array2D<T2> &times_to_receivers,
                                std::ptrdiff_t z_dim, std::ptrdiff_t x_dim,
                                double dt,
                                T1 *result_data,
                                std::ptrdiff_t p_block_size = 200) {

    const std::ptrdiff_t n_receivers = gather.get_y_dim();
    const std::ptrdiff_t n_samples = gather.get_x_dim();

    const double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
        for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
            const std::ptrdiff_t z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
            for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {
                const std::ptrdiff_t x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);

                #pragma omp parallel for
                for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {

                    const std::ptrdiff_t i_z_layer = i_z*x_dim;

                    #if defined(__SSE__) || defined(__KNCNI__)
                    const std::ptrdiff_t i_p_next = i_z_layer + x_dim + i_b_x;

                    _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
                    _mm_prefetch(&times_to_receivers(i_r, i_p_next), _MM_HINT_T0);
                    #endif

                    process_receiver_data_on_grid(&gather(i_r, 0), times_to_source, times_to_receivers, i_z_layer + i_b_x, i_z_layer + x_block_upper_border, n_samples, rev_dt, i_r, result_data);
                }
            }
        }
    }

}

template <typename T1, typename T2>
void kirchhoffMigrationCHG3D(const Array2D<T1> &gather,
                                const std::vector<T2> &times_to_source,
                                const Array2D<T2> &times_to_receivers,
                                std::ptrdiff_t z_dim, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim,
                                double dt,
                                T1 *result_data,
                                std::ptrdiff_t p_block_size = 32) {
    
    const std::ptrdiff_t n_receivers = gather.get_y_dim();
    const std::ptrdiff_t n_samples = gather.get_x_dim();

    const double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {
        for (std::ptrdiff_t i_b_z = 0; i_b_z < z_dim; i_b_z += p_block_size) {
            const std::ptrdiff_t z_block_upper_border = std::min(z_dim, i_b_z + p_block_size);
            for (std::ptrdiff_t i_b_y = 0; i_b_y < y_dim; i_b_y += p_block_size) {
                const std::ptrdiff_t y_block_upper_border = std::min(y_dim, i_b_y + p_block_size);
                for (std::ptrdiff_t i_b_x = 0; i_b_x < x_dim; i_b_x += p_block_size) {
                    const std::ptrdiff_t x_block_upper_border = std::min(x_dim, i_b_x + p_block_size);


                    #pragma omp parallel for collapse(2)
                    for (std::ptrdiff_t i_z = i_b_z; i_z < z_block_upper_border; ++i_z) {
                        for (std::ptrdiff_t i_y = i_b_y; i_y < y_block_upper_border; ++i_y) {

                            const std::ptrdiff_t i_zy_layer = (i_z*y_dim + i_y)*x_dim;

                            #if defined(__SSE__) || defined(__KNCNI__)
                            const std::ptrdiff_t i_p_next = i_zy_layer + x_dim + i_b_x;

                            _mm_prefetch(times_to_source.data() + i_p_next, _MM_HINT_T0);
                            _mm_prefetch(&times_to_receivers(i_r, i_p_next), _MM_HINT_T0);
                            #endif

                            process_receiver_data_on_grid(&gather(i_r, 0), times_to_source, times_to_receivers, i_zy_layer + i_b_x, i_zy_layer + x_block_upper_border, n_samples, rev_dt, i_r, result_data);
                        }
                    }

                }
            }
        }
    }

}

#endif //_KIRCHHOFF_MIGRATION_H