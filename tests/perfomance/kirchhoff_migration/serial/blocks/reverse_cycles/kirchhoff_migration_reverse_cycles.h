#ifndef _KIRCHHOFF_MIGRATION_REVERSE_CYCLES_H
#define _KIRCHHOFF_MIGRATION_REVERSE_CYCLES_H

#include "array2D.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <numeric>
#include <immintrin.h>

template <typename T1, typename T2>
void kirchhoffMigrationCHG2DReverseCycles(const Array2D<T1> &gather,
                                    const std::vector<T2> &times_to_source,
                                    const Array2D<T2> &times_to_receivers,
                                    std::ptrdiff_t z_dim, std::ptrdiff_t x_dim,
                                    double dt, 
                                    T1 *result_data) {
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();

    double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {

        for (std::ptrdiff_t i_z = 0; i_z < z_dim; ++i_z) {
            std::ptrdiff_t i_z_layer = i_z*x_dim;

            _mm_prefetch(times_to_source.data() + i_z_layer, _MM_HINT_NTA);
            _mm_prefetch(&times_to_receivers(i_r, i_z_layer), _MM_HINT_NTA);

            T2 t_to_s_first = times_to_source[i_z_layer], t_to_r_first = times_to_receivers(i_r, i_z_layer);

            auto sample_idx_first = static_cast<std::ptrdiff_t>((t_to_s_first + t_to_r_first) * rev_dt);

            if (sample_idx_first < n_samples) {
                _mm_prefetch(&gather(i_r, sample_idx_first), _MM_HINT_T0);
                result_data[i_z_layer] += gather(i_r, sample_idx_first);
            }

            for (std::ptrdiff_t i_x = 1; i_x < x_dim; ++i_x) {

                std::ptrdiff_t i_p = i_z_layer + i_x;

                T2 t_to_s = times_to_source[i_p], t_to_r = times_to_receivers(i_r, i_p);

                auto sample_idx = static_cast<std::ptrdiff_t>((t_to_s + t_to_r) * rev_dt);

                if (sample_idx < n_samples) {
                    result_data[i_p] += gather(i_r, sample_idx);
                }
            }
        }
    }
}

template <typename T1, typename T2>
void kirchhoffMigrationCHG3DReverseCycles(const Array2D<T1> &gather,
                                const std::vector<T2> &times_to_source,
                                const Array2D<T2> &times_to_receivers,
                                std::ptrdiff_t z_dim, std::ptrdiff_t y_dim, std::ptrdiff_t x_dim,
                                double dt,
                                T1 *result_data) {
    
    std::ptrdiff_t n_receivers = gather.get_y_dim();
    std::ptrdiff_t n_samples = gather.get_x_dim();

    double rev_dt = 1.0 / dt;

    for (std::ptrdiff_t i_r = 0; i_r < n_receivers; ++i_r) {

        for (std::ptrdiff_t i_z = 0; i_z < z_dim; ++i_z) {
            for (std::ptrdiff_t i_y = 0; i_y < y_dim; ++i_y) {

                std::ptrdiff_t i_zy_layer = (i_z*y_dim + i_y)*x_dim;

                for (std::ptrdiff_t i_x = 0; i_x < x_dim; ++i_x) {

                    std::ptrdiff_t i_p = i_zy_layer + i_x;

                    T2 t_to_s = times_to_source[i_p], t_to_r = times_to_receivers(i_r, i_p);

                    auto sample_idx = static_cast<std::ptrdiff_t>((t_to_s + t_to_r) * rev_dt);

                    if (sample_idx < n_samples) {
                        result_data[i_p] += gather(i_r, sample_idx);
                    }
                }
            }
        }
        
    }
}

#endif //_KIRCHHOFF_MIGRATION_REVERSE_CYCLES_H